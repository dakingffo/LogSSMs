from __future__ import annotations

from dataclasses import dataclass

from .logstate_pool.layout import LOG_STATE_GRANULARITY

LOG_STATE_PAGE_SIZE = 64
LOG_STATE_GRANULARITIES = (16, 32, 64)


class BaseCacheConfig:
    @property
    def prefill_alignment(self) -> int:
        return 64

    @property
    def page_size(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class StandardCacheConfig(BaseCacheConfig):
    page_size: int = 256

    def __post_init__(self) -> None:
        if self.page_size <= 0:
            raise ValueError("Cache page size must be positive.")


@dataclass(frozen=True)
class LogStateCacheConfig(BaseCacheConfig):
    granularity              : int  = LOG_STATE_GRANULARITY
    enable_eager_state       : bool = False
    enable_memorized_evict   : bool = True
    enable_merge_adjustment  : bool = True
    min_sampling_len         : int  = 1024
    max_merge_intensity      : int  = 16
    default_merge_growth_rate: int = 2

    def __post_init__(self) -> None:
        if self.granularity not in LOG_STATE_GRANULARITIES:
            raise ValueError(
                "LogState granularity must be one of "
                f"{LOG_STATE_GRANULARITIES}, got {self.granularity}."
            )
        if self.min_sampling_len <= 0:
            raise ValueError("Minimum sampling length must be positive.")
        if self.max_merge_intensity < 2:
            raise ValueError("Maximum merge intensity must be at least 2.")
        if self.default_merge_growth_rate < 1:
            raise ValueError("Default merge growth rate must be at least one.")

    @property
    def prefill_alignment(self) -> int:
        return self.granularity

    @property
    def page_size(self) -> int:
        return LOG_STATE_PAGE_SIZE


__all__ = [
    "BaseCacheConfig",
    "LOG_STATE_GRANULARITIES",
    "LOG_STATE_PAGE_SIZE",
    "LogStateCacheConfig",
    "StandardCacheConfig",
]
