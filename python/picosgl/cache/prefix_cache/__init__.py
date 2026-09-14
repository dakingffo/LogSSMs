from __future__ import annotations

from typing import Protocol

import torch

from picosgl.utils import Registry

from .base import BasePrefixCache


class PrefixCacheCreator(Protocol):
    def __call__(self, device: torch.device) -> BasePrefixCache: ...


SUPPORTED_PREFIX_CACHE = Registry[PrefixCacheCreator]("Prefix Cache")


@SUPPORTED_PREFIX_CACHE.register("naive")
def make_naive_cache(device: torch.device):
    from .naive import NaivePrefixCache

    return NaivePrefixCache(device=device)


@SUPPORTED_PREFIX_CACHE.register("hybrid_naive")
def make_hybrid_naive_cache(device: torch.device):
    from .naive import NaivePrefixCache

    return NaivePrefixCache(device=device, with_state=True)


@SUPPORTED_PREFIX_CACHE.register("radix")
def make_radix_cache(device: torch.device):
    from .radix import RadixTreeNode, RadixPrefixCache

    return RadixPrefixCache(device=device, node_type=RadixTreeNode)


@SUPPORTED_PREFIX_CACHE.register("hybrid_radix")
def make_hybrid_radix_cache(device: torch.device):
    from .radix import HybridRadixTreeNode, RadixPrefixCache

    return RadixPrefixCache(device=device, node_type=HybridRadixTreeNode)


def make_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    return SUPPORTED_PREFIX_CACHE[type](device)


__all__ = [
    "BasePrefixCache",
    "SUPPORTED_PREFIX_CACHE",
    "make_prefix_cache",
]
