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
    RMSNorm,
    VocabParallelEmbedding,
)
from logssms.speculator.hidden_captor import HiddenCapturePoint, with_speculator
from logssms.utils import nvtx_annotate

from .base import BaseLLMModel
from .config import (
    ModelConfig,
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
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, zero_centered=True
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, zero_centered=True
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    @with_speculator(
        HiddenCapturePoint.DECODER_INPUT,
        layer_id_field="_layer_id",
    )
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.input_layernorm.forward(x)
        if self.block_type == "linear_attention":
            h = self.linear_attn.forward(h)
        else:
            h = self.self_attn.forward(h, positions)
        h = residual + h

        residual = h
        h = self.post_attention_layernorm.forward(h)
        h = self.mlp.forward(h)
        return residual + h


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
        self.norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, zero_centered=True
        )

    @nvtx_annotate("Model")
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        positions = get_global_ctx().batch.positions
        for layer in self.layers.op_list:
            x = layer.forward(x, positions)
        return self.norm.forward(x)


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
