from __future__ import annotations

from dataclasses import dataclass


LOG_STATE_PAGE_SIZE = 64
LOG_STATE_GRANULARITIES = (1, 2, 4, 8, 16, 32, 64)


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
    granularity            : int  = 16
    enable_merge_adjustment: bool = False

    def __post_init__(self) -> None:
        if self.granularity not in LOG_STATE_GRANULARITIES:
            raise ValueError(
                "LogState granularity must be one of "
                f"{LOG_STATE_GRANULARITIES}, got {self.granularity}."
            )

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
