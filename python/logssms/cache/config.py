from __future__ import annotations

from dataclasses import dataclass


LOG_STATE_PAGE_SIZE = 64
LOG_STATE_GRANULARITIES = (16, 32, 64)


class BaseCacheConfig:
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
    granularity             : int  = 16
    enable_merge_adjustment : bool = False
    min_sampling_len        : int  = 1024
    max_merge_intensity     : int  = 16
    base_intensity_increment: int  = 2

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
        if self.base_intensity_increment <= 0:
            raise ValueError("Base intensity increment must be positive.")

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
