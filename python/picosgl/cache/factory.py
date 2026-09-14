from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from picosgl.utils import Registry

from .base import BaseCacheManager, CacheResources

if TYPE_CHECKING:
    from picosgl.engine import EngineConfig
    from picosgl.speculator import SpeculatorReserve


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
