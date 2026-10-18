from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from logssms.core import get_global_ctx
from logssms.layers import BaseOP, GatedMLP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from logssms.layers.attention import RotaryAttention
from logssms.layers.mamba2 import Mamba2
from logssms.layers.piecewise import graph_piece
from logssms.speculator.hidden_captor import HiddenCapturePoint, with_speculator
from logssms.utils import div_even, nvtx_annotate

from .base import BaseLLMModel
from .config import AffineLogCacheSpec, ModelConfig, make_common_config_kwargs


@dataclass(frozen=True)
class FalconH1Config(ModelConfig):
    intermediate_size       : int
    mamba_num_heads         : int
    mamba_num_groups        : int
    mamba_head_dim          : int
    mamba_state_dim         : int
    mamba_conv_kernel_size  : int
    mamba_conv_bias         : bool
    mamba_proj_bias         : bool
    projectors_bias         : bool
    mamba_rms_norm          : bool
    mamba_norm_before_gate  : bool
    embedding_multiplier   : float
    lm_head_multiplier      : float
    attention_in_multiplier : float
    attention_out_multiplier: float
    key_multiplier          : float
    ssm_in_multiplier       : float
    ssm_out_multiplier      : float
    ssm_multipliers         : tuple[float, ...]
    mlp_multipliers         : tuple[float, float]

    @property
    def is_hybrid(self) -> bool:
        return True

    @property
    def num_linear_layers(self) -> int:
        return self.num_layers

    def make_linear_cache_spec(self, tp_size: int) -> AffineLogCacheSpec:
        groups = div_even(self.mamba_num_groups, tp_size, allow_replicate=True)
        heads = div_even(self.mamba_num_heads, tp_size)
        return AffineLogCacheSpec(
            recurrent_type="mamba2", num_layers=self.num_layers,
            state_shape=(heads, self.mamba_state_dim, self.mamba_head_dim),
            conv_shape=(heads * self.mamba_head_dim + 2 * groups * self.mamba_state_dim,
                        self.mamba_conv_kernel_size - 1),
            num_groups=groups,
        )

    def shard_weight(
        self, name: str, tensor: torch.Tensor, rank: int, tp_size: int,
    ) -> torch.Tensor | None:
        if ".mamba." not in name:
            return None
        inner = self.mamba_num_heads * self.mamba_head_dim
        groups = self.mamba_num_groups * self.mamba_state_dim
        if ".in_proj." in name:
            widths = (inner, inner, groups, groups, self.mamba_num_heads)
            replicated = (False, False, self.mamba_num_groups < tp_size,
                          self.mamba_num_groups < tp_size, False)
        elif ".conv1d." in name:
            widths = (inner, groups, groups)
            replicated = (False, self.mamba_num_groups < tp_size,
                          self.mamba_num_groups < tp_size)
        elif name.endswith((".D", ".norm.weight")):
            return tensor.chunk(tp_size, 0)[rank].clone()
        elif name.endswith(".out_proj.bias"):
            return tensor if rank == 0 else torch.zeros_like(tensor)
        else:
            return None
        return torch.cat([
            part if replicate else part.chunk(tp_size, 0)[rank]
            for part, replicate in zip(tensor.split(widths, 0), replicated)
        ], 0).clone()

    @classmethod
    def from_pretrained(cls, config: PretrainedConfig) -> FalconH1Config:
        inner = config.mamba_d_ssm or int(config.mamba_expand * config.hidden_size)
        if inner != config.mamba_n_heads * config.mamba_d_head:
            raise ValueError("Falcon-H1 SSM width must equal mamba_n_heads * mamba_d_head")
        if config.hidden_act != "silu" or config.attention_bias or config.mlp_bias:
            raise ValueError("Falcon-H1 currently requires SiLU and bias-free attention/MLP")
        if getattr(config, "attn_layer_indices", None) is not None:
            raise ValueError("Falcon-H1 currently requires parallel Attention/Mamba2 at every layer")
        return cls(
            **make_common_config_kwargs(config, config),
            intermediate_size=config.intermediate_size,
            mamba_num_heads=config.mamba_n_heads,
            mamba_num_groups=config.mamba_n_groups,
            mamba_head_dim=config.mamba_d_head,
            mamba_state_dim=config.mamba_d_state,
            mamba_conv_kernel_size=config.mamba_d_conv,
            mamba_conv_bias=config.mamba_conv_bias,
            mamba_proj_bias=config.mamba_proj_bias,
            projectors_bias=config.projectors_bias,
            mamba_rms_norm=config.mamba_rms_norm,
            mamba_norm_before_gate=config.mamba_norm_before_gate,
            embedding_multiplier=config.embedding_multiplier,
            lm_head_multiplier=config.lm_head_multiplier,
            attention_in_multiplier=config.attention_in_multiplier,
            attention_out_multiplier=config.attention_out_multiplier,
            key_multiplier=config.key_multiplier,
            ssm_in_multiplier=config.ssm_in_multiplier,
            ssm_out_multiplier=config.ssm_out_multiplier,
            ssm_multipliers=tuple(config.ssm_multipliers),
            mlp_multipliers=tuple(config.mlp_multipliers),
        )


class FalconH1DecoderLayer(BaseOP):
    def __init__(self, config: FalconH1Config, layer_id: int):
        self._layer_id = layer_id
        self._config = config
        rope = config.rotary_config
        self.self_attn = RotaryAttention(
            hidden_size=config.hidden_size, head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads, num_kv_heads=config.num_kv_heads,
            layer_id=layer_id, rotary_dim=rope.rotary_dim,
            max_position=rope.max_position, rope_base=rope.base,
            rope_scaling=tuple(rope.scaling.items()) if rope.scaling else None,
        )
        self.mamba = Mamba2(
            hidden_size=config.hidden_size, num_heads=config.mamba_num_heads,
            num_groups=config.mamba_num_groups, head_dim=config.mamba_head_dim,
            state_dim=config.mamba_state_dim, conv_kernel_size=config.mamba_conv_kernel_size,
            conv_bias=config.mamba_conv_bias, proj_bias=config.mamba_proj_bias,
            output_bias=config.projectors_bias, rms_norm=config.mamba_rms_norm,
            rms_norm_eps=config.rms_norm_eps, norm_before_gate=config.mamba_norm_before_gate,
            in_multiplier=config.ssm_in_multiplier, multipliers=config.ssm_multipliers,
            layer_idx=layer_id,
        )
        self.input_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)
        self.feed_forward = GatedMLP(config.hidden_size, config.intermediate_size, "silu")

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    @with_speculator(HiddenCapturePoint.DECODER_INPUT, layer_id_field="_layer_id")
    def forward(self, x, residual, positions):
        query, key, value, xbc, gate, dt, residual = self._prepare_attention(x, residual)
        return self._finish_attention(
            *self.attend_features((query, key, value, xbc, gate, dt), positions), residual,
        )

    def attend_features(self, features, positions):
        query, key, value, xbc, gate, dt = features
        attn = self.self_attn.attend(query, key, value, positions)
        ssm = self.mamba.attend(xbc, dt)
        return attn, ssm, gate

    @graph_piece
    def _prepare_attention(self, x, residual):
        h, residual = self.input_layernorm.forward(x, residual)
        query, key, value = self.self_attn._project(h * self._config.attention_in_multiplier)
        xbc, gate, dt = self.mamba.proj(h)
        return query, key * self._config.key_multiplier, value, xbc, gate, dt, residual

    @graph_piece
    def _finish_attention(self, attention, ssm, gate, residual):
        config = self._config
        attention = self.self_attn.o_proj.forward(attention.flatten(1))
        ssm = self.mamba.output(ssm, gate)
        h = attention * config.attention_out_multiplier + ssm * config.ssm_out_multiplier
        h, residual = self.pre_ff_layernorm.forward(h, residual)
        gate_up = self.feed_forward.gate_up_proj.forward(h)
        gate, up = gate_up.chunk(2, -1)
        h = torch.nn.functional.silu(gate * config.mlp_multipliers[0]) * up
        h = self.feed_forward.down_proj.forward(h) * config.mlp_multipliers[1]
        return h, residual

    def piecewise_graph_pieces(self):
        weight = self.mamba.in_proj.weight
        def sample(*shape):
            return torch.empty((1,) + shape, device=weight.device, dtype=weight.dtype)
        hidden = sample(self._config.hidden_size)
        residual = None if self._layer_id == 0 else sample(self._config.hidden_size)
        attn = self.self_attn
        ssm = self.mamba._config
        return [
            (self._prepare_attention, (hidden, residual), ("hidden", "residual"),
             ("query", "key", "value", "xbc", "gate", "dt", "residual")),
            (self._finish_attention,
             (sample(attn.num_qo_heads, attn.head_dim), sample(ssm.num_heads, ssm.head_dim),
              sample(ssm.value_dim), sample(self._config.hidden_size)),
             ("attention_output", "ssm_output", "gate", "residual"), ("hidden", "residual")),
        ]


class FalconH1Model(BaseOP):
    def __init__(self, config: FalconH1Config):
        self._embedding_multiplier = config.embedding_multiplier
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList([FalconH1DecoderLayer(config, i) for i in range(config.num_layers)])
        self.final_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids):
        x = self.embed_tokens.forward(input_ids) * self._embedding_multiplier
        residual = None
        positions = get_global_ctx().batch.positions
        runner = get_global_ctx().piecewise_graph_runner
        if runner is not None and runner.can_run_layers(get_global_ctx().batch):
            x, residual = runner.forward_layers(x, positions)
            return self.final_layernorm.forward(x, residual)[0]
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual, positions)
        return self.final_layernorm.forward(x, residual)[0]


class FalconH1ForCausalLM(BaseLLMModel):
    def __init__(self, config: FalconH1Config):
        self._lm_head_multiplier = config.lm_head_multiplier
        self.model = FalconH1Model(config)
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )

    def forward(self):
        h = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(h) * self._lm_head_multiplier
