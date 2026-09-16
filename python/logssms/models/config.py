from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    rotary_dim  : int
    max_position: int
    base        : float
    scaling     : dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    num_layers         : int
    num_qo_heads       : int
    num_kv_heads       : int
    head_dim           : int
    hidden_size        : int
    vocab_size         : int
    rms_norm_eps       : float
    rotary_config      : RotaryConfig
    tie_word_embeddings: bool
    architectures      : list[str]

    @property
    def is_moe(self) -> bool:
        return False

    @property
    def is_hybrid(self) -> bool:
        return False

    @property
    def num_attention_layers(self) -> int:
        return self.num_layers

    @property
    def num_linear_layers(self) -> int:
        return 0


def unwrap_text_config(
    config: PretrainedConfig,
) -> tuple[PretrainedConfig, PretrainedConfig]:
    text = getattr(config, "text_config", None)
    return config, text if text is not None else config


def make_common_config_kwargs(
    top : PretrainedConfig,
    text: PretrainedConfig,
) -> dict[str, Any]:
    num_qo_heads = text.num_attention_heads
    head_dim = getattr(text, "head_dim", None) or text.hidden_size // num_qo_heads
    rope_scaling = getattr(text, "rope_scaling", None) or getattr(top, "rope_scaling", None)
    rope_params = getattr(text, "rope_parameters", None) or {}
    rope_theta = (
        getattr(text, "rope_theta", None)
        or getattr(top, "rope_theta", None)
        or rope_params.get("rope_theta")
        or (rope_scaling.get("rope_theta") if rope_scaling else None)
    )
    partial_rotary_factor = float(
        rope_params.get(
            "partial_rotary_factor", getattr(text, "partial_rotary_factor", 1.0)
        )
    )
    architectures = (
        getattr(top, "architectures", None)
        or getattr(text, "architectures", None)
        or ["LlamaForCausalLM"]
    )

    return {
        "num_layers"          : text.num_hidden_layers,
        "num_qo_heads"        : num_qo_heads,
        "num_kv_heads"        : getattr(text, "num_key_value_heads", num_qo_heads),
        "head_dim"            : head_dim,
        "hidden_size"         : text.hidden_size,
        "vocab_size"          : text.vocab_size,
        "rms_norm_eps"        : text.rms_norm_eps,
        "rotary_config"       : RotaryConfig(
            rotary_dim=int(head_dim * partial_rotary_factor),
            max_position=text.max_position_embeddings,
            base=rope_theta,
            scaling=rope_scaling,
        ),
        "tie_word_embeddings" : getattr(text, "tie_word_embeddings", False),
        "architectures"       : list(architectures),
    }


def make_moe_config_kwargs(config: PretrainedConfig) -> dict[str, Any]:
    return {
        "num_experts"          : getattr(
            config, "num_local_experts", getattr(config, "num_experts", 0)
        ),
        "num_experts_per_tok"  : config.num_experts_per_tok,
        "moe_intermediate_size": config.moe_intermediate_size,
        "norm_topk_prob"       : getattr(config, "norm_topk_prob", False),
    }


__all__ = ["ModelConfig", "RotaryConfig", "unwrap_text_config", "make_common_config_kwargs", "make_moe_config_kwargs"]
