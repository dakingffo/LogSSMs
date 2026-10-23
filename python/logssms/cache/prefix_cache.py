from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import torch

from logssms.utils import Registry


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

class PrefixCacheCreator(Protocol):
    def __call__(self, device: torch.device) -> BasePrefixCache: ...


SUPPORTED_PREFIX_CACHE = Registry[PrefixCacheCreator]("Prefix Cache")


@SUPPORTED_PREFIX_CACHE.register("naive")
def make_naive_cache(device: torch.device):
    from logssms.cache.naive.prefix_cache import NaivePrefixCache

    return NaivePrefixCache(device=device)


@SUPPORTED_PREFIX_CACHE.register("hybrid_naive")
def make_hybrid_naive_cache(device: torch.device):
    from logssms.cache.naive.prefix_cache import NaivePrefixCache

    return NaivePrefixCache(device=device, with_state=True)


@SUPPORTED_PREFIX_CACHE.register("radix")
def make_radix_cache(device: torch.device):
    from logssms.cache.radix.prefix_cache import RadixTreeNode, RadixPrefixCache

    return RadixPrefixCache(device=device, node_type=RadixTreeNode)


@SUPPORTED_PREFIX_CACHE.register("hybrid_radix")
def make_hybrid_radix_cache(device: torch.device):
    from logssms.cache.radix.prefix_cache import HybridRadixTreeNode, RadixPrefixCache

    return RadixPrefixCache(device=device, node_type=HybridRadixTreeNode)


def make_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    return SUPPORTED_PREFIX_CACHE[type](device)


__all__ = [
    "BasePrefixCache",
    "SUPPORTED_PREFIX_CACHE",
    "make_prefix_cache",
]
