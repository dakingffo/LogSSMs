from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from picosgl.core import Request

from .kv_pool.base import BaseKVCachePool
from .linear_pool.state_pool import LinearStatePool
from .prefix_cache.base import BaseCacheHandle, MatchResult

if TYPE_CHECKING:
    from picosgl.scheduler.prefill import PendingRequest


@dataclass
class CacheResources:
    kv_pool         : BaseKVCachePool
    page_table      : torch.Tensor
    linear_state    : LinearStatePool | None
    state_table     : torch.Tensor | None
    work_state_table: torch.Tensor | None


class BaseCacheManager(ABC):
    def __init__(
        self,
        page_size        : int,
        resources        : CacheResources,
        prefix_cache_type: str,
    ) -> None:
        from .prefix_cache import make_prefix_cache

        self.device = resources.page_table.device
        self.page_size = page_size
        self.resources = resources
        self.prefix_cache = make_prefix_cache(
            device=self.device, type=prefix_cache_type
        )

    @property
    def page_table(self) -> torch.Tensor:
        return self.resources.page_table

    @property
    def state_table(self) -> torch.Tensor | None:
        return self.resources.state_table

    @property
    def work_state_table(self) -> torch.Tensor | None:
        return self.resources.work_state_table

    @property
    def state_pool(self) -> LinearStatePool | None:
        return self.resources.linear_state

    def match_req(self, req: PendingRequest) -> MatchResult:
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])

    def lock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def prefill_acquire(self, total_len: int, handle: BaseCacheHandle) -> bool:
        if total_len > self.available_size:
            return False
        self.lock(handle)
        if total_len > self.available_size:
            self.unlock(handle)
            return False
        return True

    @property
    @abstractmethod
    def available_size(self) -> int: ...

    @abstractmethod
    def plan_prefill_chunk(self, remain_len: int, token_budget: int) -> int: ...

    @abstractmethod
    def init_with_handle(self, table_idx: int, handle: BaseCacheHandle) -> None: ...

    @abstractmethod
    def allocate(self, reqs: list[Request]) -> None: ...

    @abstractmethod
    def cache_req(self, req: Request, *, finished: bool) -> None: ...

    @abstractmethod
    def free_to_cached(self, req: Request) -> None: ...

    @abstractmethod
    def commit_speculative(self, req: Request, C: int, num_sampled: int) -> None: ...

    @abstractmethod
    def check_integrity(self) -> None: ...

    @abstractmethod
    def lazy_free_region(self): ...
