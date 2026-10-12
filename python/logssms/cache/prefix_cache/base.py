from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import torch


@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    cached_len: int

    @abstractmethod
    def get_matched_indices(self) -> tuple[torch.Tensor, ...] | torch.Tensor: ...


class SizeInfo(NamedTuple):
    evictable_size: int
    protected_size: int

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size


class InsertResult(NamedTuple):
    cached_len: int
    handle    : BaseCacheHandle


class MatchResult(NamedTuple):
    cuda_handle: BaseCacheHandle


    tree_match_len : int | None = None
    recoverable_len: int | None = None


class BasePrefixCache(ABC):
    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        pass

    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        pass

    @abstractmethod
    def check_integrity(self) -> None:
        pass
