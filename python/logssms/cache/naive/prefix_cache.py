from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from logssms.cache.prefix_cache import (
    BaseCacheHandle,
    BasePrefixCache,
    InsertResult,
    MatchResult,
    SizeInfo,
)


class NaivePrefixValue(NamedTuple):
    kv_indices: torch.Tensor
    state_indices: torch.Tensor | None = None


@dataclass(frozen=True)
class NaiveCacheHandle(BaseCacheHandle):
    empty_tensor: torch.Tensor
    with_state  : bool

    def get_matched_indices(self) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self.with_state:
            return self.empty_tensor, self.empty_tensor
        else:
            return self.empty_tensor


class NaivePrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device, with_state: bool = False):
        self.device = device
        self.with_state = with_state
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        super().__init__()

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        pass

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        return MatchResult(
            NaiveCacheHandle(0, self.empty_tensor, self.with_state),
            tree_match_len=0,
            recoverable_len=0,
        )

    def insert_prefix(
        self,
        input_ids: torch.Tensor,
        value    : NaivePrefixValue,
    ) -> InsertResult:
        assert (value.state_indices is not None) == self.with_state
        return InsertResult(
            0, NaiveCacheHandle(0, self.empty_tensor, self.with_state)
        )

    def evict(self, size: int) -> NaivePrefixValue:
        if size != 0:
            raise NotImplementedError("NaiveCacheManager does not support eviction.")
        return NaivePrefixValue(self.empty_tensor)

    def reset(self) -> None:
        pass

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=0, protected_size=0)

    def check_integrity(self) -> None:
        pass
