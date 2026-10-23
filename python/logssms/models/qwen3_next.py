from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PretrainedConfig
from logssms.core import get_global_ctx
from logssms.layers import (
    BaseOP,
    GatedDeltaNet,
    GatedMLP,
    MoEMLP,
    GatedRotaryAttention,
    OPList,
    ParallelLMHead,
    RMSNormFused,
    VocabParallelEmbedding,
)
from logssms.speculator.hidden_captor import HiddenCapturePoint, with_speculator
from logssms.layers.piecewise import graph_piece
from logssms.utils import div_even, nvtx_annotate

from .base import BaseLLMModel
from .config import (
    ModelConfig,
    AffineLogCacheSpec,
    make_common_config_kwargs,
    make_moe_config_kwargs,
    unwrap_text_config,
)


@dataclass(frozen=True)
class Qwen3NextConfig(ModelConfig):
    layer_types                     : list[str]
    linear_num_key_heads            : int
    linear_num_value_heads          : int
    linear_key_head_dim             : int
    linear_value_head_dim           : int
    linear_conv_kernel_dim          : int
    intermediate_size               : int
    hidden_act                      : str
    num_experts                     : int
    num_experts_per_tok             : int
    moe_intermediate_size           : int
    shared_expert_intermediate_size : int
    norm_topk_prob                  : bool
    mlp_only_layers                 : list[int]
    decoder_sparse_step             : int

    @property
    def is_hybrid(self) -> bool:
        return True

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @property
    def num_attention_layers(self) -> int:
        return sum(layer_type == "full_attention" for layer_type in self.layer_types)

    @property
    def num_linear_layers(self) -> int:
        return sum(layer_type == "linear_attention" for layer_type in self.layer_types)

    def make_linear_cache_spec(self, tp_size: int) -> AffineLogCacheSpec:
        key_heads = div_even(self.linear_num_key_heads, tp_size)
        value_heads = div_even(self.linear_num_value_heads, tp_size)
        conv_dim = key_heads * self.linear_key_head_dim * 2 + value_heads * self.linear_value_head_dim
        return AffineLogCacheSpec(
            recurrent_type="gdn",
            num_layers=self.num_linear_layers,
            state_shape=(value_heads, self.linear_key_head_dim, self.linear_value_head_dim),
            conv_shape=(conv_dim, self.linear_conv_kernel_dim - 1),
            num_groups=key_heads,
        )

    def use_moe(self, layer_id: int) -> bool:
        return (
            self.is_moe
            and layer_id not in self.mlp_only_layers
            and (layer_id + 1) % self.decoder_sparse_step == 0
        )

    @classmethod
    def _from_pretrained_kwargs(cls, config: PretrainedConfig) -> dict[str, Any]:
        top, text = unwrap_text_config(config)
        layer_types = getattr(text, "layer_types", None)
        if layer_types is None or "linear_attention" not in layer_types:
            interval = getattr(text, "full_attention_interval", None)
            if interval is None:
                raise ValueError("Qwen3Next config does not define its layer layout")
            layer_types = [
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(text.num_hidden_layers)
            ]

        return {
            **make_common_config_kwargs(top, text),
            "layer_types"            : list(layer_types),
            "linear_num_key_heads"   : text.linear_num_key_heads,
            "linear_num_value_heads" : text.linear_num_value_heads,
            "linear_key_head_dim"    : text.linear_key_head_dim,
            "linear_value_head_dim"  : text.linear_value_head_dim,
            "linear_conv_kernel_dim" : text.linear_conv_kernel_dim,
        }

    @classmethod
    def from_pretrained(cls, config: PretrainedConfig) -> Qwen3NextConfig:
        _, text = unwrap_text_config(config)
        return cls(
            **cls._from_pretrained_kwargs(config),
            **make_moe_config_kwargs(text),
            intermediate_size=text.intermediate_size,
            hidden_act=text.hidden_act,
            shared_expert_intermediate_size=text.shared_expert_intermediate_size,
            mlp_only_layers=list(getattr(text, "mlp_only_layers", None) or []),
            decoder_sparse_step=getattr(text, "decoder_sparse_step", 1),
        )


@dataclass(frozen=True)
class Qwen3_5Config(Qwen3NextConfig):
    mtp_num_hidden_layers: int

    @classmethod
    def from_pretrained(cls, config: PretrainedConfig) -> Qwen3_5Config:
        _, text = unwrap_text_config(config)
        return cls(
            **cls._from_pretrained_kwargs(config),
            intermediate_size=text.intermediate_size,
            hidden_act=text.hidden_act,
            num_experts=0,
            num_experts_per_tok=0,
            moe_intermediate_size=0,
            shared_expert_intermediate_size=0,
            norm_topk_prob=False,
            mlp_only_layers=[],
            decoder_sparse_step=1,
            mtp_num_hidden_layers=getattr(text, "mtp_num_hidden_layers", 0),
        )


@dataclass(frozen=True)
class Qwen3_5MoeConfig(Qwen3NextConfig):
    mtp_num_hidden_layers: int

    @classmethod
    def from_pretrained(cls, config: PretrainedConfig) -> Qwen3_5MoeConfig:
        _, text = unwrap_text_config(config)
        return cls(
            **cls._from_pretrained_kwargs(config),
            **make_moe_config_kwargs(text),
            intermediate_size=0,
            hidden_act=text.hidden_act,
            shared_expert_intermediate_size=text.shared_expert_intermediate_size,
            mlp_only_layers=list(getattr(text, "mlp_only_layers", None) or []),
            decoder_sparse_step=getattr(text, "decoder_sparse_step", 1),
            mtp_num_hidden_layers=getattr(text, "mtp_num_hidden_layers", 0),
        )


class Qwen3NextDecoderLayer(BaseOP):
    def __init__(
        self,
        config: Qwen3NextConfig,
        layer_id: int,
        *,
        block_type     : str | None = None,
        full_attn_idx  : int = 0,
        linear_attn_idx: int = 0,
    ):
        self.block_type = block_type or config.layer_types[layer_id]
        self._layer_id = layer_id
        if self.block_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_key_heads=config.linear_num_key_heads,
                num_value_heads=config.linear_num_value_heads,
                head_k_dim=config.linear_key_head_dim,
                head_v_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_idx=linear_attn_idx,
            )
        elif self.block_type == "full_attention":
            rotary_config = config.rotary_config
            self.self_attn = GatedRotaryAttention(
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                num_qo_heads=config.num_qo_heads,
                num_kv_heads=config.num_kv_heads,
                layer_id=full_attn_idx,
                rotary_dim=rotary_config.rotary_dim,
                max_position=rotary_config.max_position,
                rope_base=rotary_config.base,
                rope_scaling=(
                    tuple(rotary_config.scaling.items())
                    if rotary_config.scaling else None
                ),
                qk_norm_eps=config.rms_norm_eps,
                zero_centered_norm=True,
            )
        else:
            raise ValueError(f"Invalid layer type {self.block_type}")
        if config.use_moe(layer_id):
            self.mlp = MoEMLP(
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                renormalize=config.norm_topk_prob,
            )
        else:
            self.mlp = GatedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )
        self.input_layernorm = RMSNormFused(
            config.hidden_size, eps=config.rms_norm_eps, zero_centered=True
        )
        self.post_attention_layernorm = RMSNormFused(
            config.hidden_size, eps=config.rms_norm_eps, zero_centered=True
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    @with_speculator(
        HiddenCapturePoint.DECODER_INPUT,
        layer_id_field="_layer_id",
    )
    def forward(
        self,
        x        : torch.Tensor,
        residual : torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        *features, residual = self._prepare_attention(x, residual)
        return self._finish_attention(*self.attend_features(features, positions), residual)

    def attend_features(self, features, positions):
        if self.block_type == "linear_attention":
            mixed_qkv, output_gate, beta, gate = features
            h = self.linear_attn.attend(mixed_qkv, beta, gate)
        else:
            query, output_gate, key, value = features
            h = self.self_attn.attend(query, key, value, positions)
        return h, output_gate

    @graph_piece
    def _prepare_attention(
        self,
        x        : torch.Tensor,
        residual : torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        h, residual = self.input_layernorm.forward(x, residual)
        features = (
            self.linear_attn.proj(h) if self.block_type == "linear_attention"
            else self.self_attn._project(h)
        )
        return (*features, residual)

    @graph_piece
    def _finish_attention(
        self,
        h       : torch.Tensor,
        gate    : torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = (
            self.linear_attn.output(h, gate) if self.block_type == "linear_attention"
            else self.self_attn.output(h, gate)
        )
        h, residual = self.post_attention_layernorm.forward(h, residual)
        return self.mlp.forward(h), residual

    def piecewise_graph_pieces(self):
        weight = self.mlp.router.weight if isinstance(self.mlp, MoEMLP) else self.mlp.gate_up_proj.weight
        def sample(*shape):
            return torch.empty((1,) + shape, device=weight.device, dtype=weight.dtype)
        hidden = sample(weight.shape[1])
        residual = None if self._layer_id == 0 else sample(weight.shape[1])
        pieces = [(self._prepare_attention, (hidden, residual),
                   ("hidden", "residual"),
                   (("qkv", "gate", "beta", "decay", "residual")
                    if self.block_type == "linear_attention"
                    else ("query", "gate", "key", "value", "residual")))]
        if isinstance(self.mlp, GatedMLP):
            if self.block_type == "linear_attention":
                config = self.linear_attn._config
                core = sample(config.num_v_heads, config.head_v_dim)
                gate = sample(config.value_dim)
            else:
                core = sample(self.self_attn.num_qo_heads, self.self_attn.head_dim)
                gate = sample(self.self_attn.num_qo_heads, self.self_attn.head_dim)
            pieces.append((self._finish_attention, (core, gate, sample(weight.shape[1])),
                           ("attention_output", "gate", "residual"),
                           ("hidden", "residual")))
        return pieces


class Qwen3NextModel(BaseOP):
    def __init__(self, config: Qwen3NextConfig):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        layers = []
        full_idx = 0
        linear_idx = 0
        for i, layer_type in enumerate(config.layer_types):
            if layer_type == "full_attention":
                layers.append(
                    Qwen3NextDecoderLayer(config, i, full_attn_idx=full_idx)
                )
                full_idx += 1
            else:
                layers.append(
                    Qwen3NextDecoderLayer(config, i, linear_attn_idx=linear_idx)
                )
                linear_idx += 1
        self.layers = OPList(layers)
        self.norm = RMSNormFused(
            config.hidden_size, eps=config.rms_norm_eps, zero_centered=True
        )

    @nvtx_annotate("Model")
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        positions = get_global_ctx().batch.positions
        residual: torch.Tensor | None = None
        runner = get_global_ctx().piecewise_graph_runner
        if runner is not None and runner.can_run_layers(get_global_ctx().batch):
            x, residual = runner.forward_layers(x, positions)
            return self.norm.forward(x, residual)[0]
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual, positions)
        return self.norm.forward(x, residual)[0]


class Qwen3NextForCausalLM(BaseLLMModel):
    def __init__(self, config: Qwen3NextConfig):
        self.model = Qwen3NextModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits

__all__ = [
    "Qwen3_5Config",
    "Qwen3_5MoeConfig",
    "Qwen3NextConfig",
    "Qwen3NextDecoderLayer",
    "Qwen3NextForCausalLM",
]
