from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from transformers import PretrainedConfig
from logssms.core import get_global_ctx
from logssms.layers import (
    BaseOP,
    GatedMLP as Qwen2MLP,
    OPList,
    ParallelLMHead,
    RMSNormFused,
    RotaryAttention,
    VocabParallelEmbedding,
)
from logssms.speculator.hidden_captor import HiddenCapturePoint, with_speculator
from logssms.utils import nvtx_annotate

from .base import BaseLLMModel
from .config import ModelConfig, make_common_config_kwargs, unwrap_text_config


@dataclass(frozen=True)
class Qwen2Config(ModelConfig):
    intermediate_size: int
    hidden_act       : str

    @classmethod
    def from_pretrained(cls, config: PretrainedConfig) -> Qwen2Config:
        top, text = unwrap_text_config(config)
        return cls(
            **make_common_config_kwargs(top, text),
            intermediate_size=text.intermediate_size,
            hidden_act=text.hidden_act,
        )


class Qwen2DecoderLayer(BaseOP):
    def __init__(self, config: Qwen2Config, layer_id: int):
        rotary_config = config.rotary_config
        self.self_attn = RotaryAttention(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            layer_id=layer_id,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            rope_base=rotary_config.base,
            rope_scaling=(
                tuple(rotary_config.scaling.items()) if rotary_config.scaling else None
            ),
            has_attn_bias=True,
        )
        self.mlp = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    @with_speculator(
        HiddenCapturePoint.DECODER_INPUT,
        layer_id_field="_layer_id",
    )
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class Qwen2Model(BaseOP):
    def __init__(self, config: Qwen2Config):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen2DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class Qwen2ForCausalLM(BaseLLMModel):
    def __init__(self, config: Qwen2Config):
        self.model = Qwen2Model(config)
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


__all__ = ["Qwen2Config", "Qwen2ForCausalLM"]
