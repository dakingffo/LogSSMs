from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from logssms.distributed import DistributedInfo, try_get_tp_info
from logssms.utils import Registry

from logssms.cache.base import BaseCacheManager, CacheResources

if TYPE_CHECKING:
    from logssms.models import ModelConfig
    from logssms.models.config import LinearCacheSpec
    from logssms.engine import EngineConfig
    from logssms.speculator import SpeculatorReserve


@dataclass(frozen=True)
class CachePlan:
    num_pages          : int
    num_tokens         : int
    max_seq_len        : int
    aligned_max_seq_len: int


class BaseCacheFactory(ABC):
    @abstractmethod
    def plan(
        self,
        config            : EngineConfig,
        available_memory  : int,
        speculator_reserve: SpeculatorReserve | None,
    ) -> CachePlan: ...

    @abstractmethod
    def allocate(
        self,
        config: EngineConfig,
        plan  : CachePlan,
        device: torch.device,
        dtype : torch.dtype,
    ) -> CacheResources: ...

    @abstractmethod
    def make_manager(
        self,
        config   : EngineConfig,
        plan     : CachePlan,
        resources: CacheResources,
    ) -> BaseCacheManager: ...


SUPPORTED_CACHE_FACTORIES = Registry[BaseCacheFactory]("Cache")


def get_cache_factory(cache_type: str) -> BaseCacheFactory:
    return SUPPORTED_CACHE_FACTORIES[cache_type]


_TP_DEFAULT = DistributedInfo(rank=0, size=1)


def make_linear_cache_spec(model_config: ModelConfig) -> LinearCacheSpec:
    tp_size = (try_get_tp_info() or _TP_DEFAULT).size
    spec = model_config.make_linear_cache_spec(tp_size)
    assert spec is not None, "Hybrid model must describe its linear cache storage."
    assert spec.num_layers == model_config.num_linear_layers
    return spec


def linear_state_slot_bytes_for_config(
    model_config: ModelConfig,
    dtype       : torch.dtype,
) -> int:
    if not model_config.is_hybrid:
        return 0
    return make_linear_cache_spec(model_config).slot_bytes(dtype.itemsize)
