from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import torch
from transformers import PretrainedConfig

from logssms.distributed import DistributedInfo
from logssms.utils import load_model_config
from logssms.models import ModelConfig, make_model_config

if TYPE_CHECKING:
    from logssms.cache import BaseCacheConfig
    from logssms.speculator import BaseSpeculatorConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path              : str
    tp_info                 : DistributedInfo
    dtype                   : torch.dtype
    max_running_req         : int              = 128
    scheduling_policy       : str              = "prefill-first"
    mixed_prefill_tokens    : int              = 128
    attention_backend       : str              = "auto"
    moe_backend             : str              = "auto"
    cuda_graph_bs           : list[int] | None = None
    cuda_graph_max_bs       : int | None       = None
    memory_ratio            : float            = 0.9
    distributed_timeout     : float            = 60.0
    use_dummy_weight        : bool             = False
    max_seq_len_override    : int | None       = None
    num_page_override       : int | None       = None
    cache_type              : str              = "radix"
    cache_config            : BaseCacheConfig | None = None

    speculative_algorithm        : str | None                  = None
    speculative_draft_model_path : str | None                  = None
    dt_separation                : bool                        = False
    speculator_config            : BaseSpeculatorConfig | None = None

    def __post_init__(self) -> None:
        if self.cache_config is None:
            from logssms.cache import make_cache_argument_parser

            parser = make_cache_argument_parser(self.cache_type)
            defaults = vars(parser.parser.parse_args([]))
            object.__setattr__(self, "cache_config", parser.make_config(defaults))
        if self.scheduling_policy not in ("prefill-first", "mixed"):
            raise ValueError("scheduling_policy must be prefill-first or mixed")
        alignment = self.cache_config.prefill_alignment
        if self.mixed_prefill_tokens <= 0 or self.mixed_prefill_tokens % alignment:
            raise ValueError(f"mixed_prefill_tokens must be a positive multiple of {alignment}")
        if self.scheduling_policy == "mixed" and self.enable_specualtive_decoding:
            raise ValueError("mixed scheduling does not support speculative decoding")

    @property
    def enable_specualtive_decoding(self) -> bool:
        return self.speculative_algorithm is not None

    @property
    def page_size(self) -> int:
        return self.cache_config.page_size

    @cached_property
    def pretrained_config(self) -> PretrainedConfig:
        return load_model_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        return make_model_config(self.pretrained_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        else:
            return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
