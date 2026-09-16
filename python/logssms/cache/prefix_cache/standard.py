from __future__ import annotations

from abc import abstractmethod
from typing import NamedTuple

import torch

from .base import BasePrefixCache, InsertResult


class StandardPrefixValue(NamedTuple):
    kv_indices   : torch.Tensor
    state_indices: torch.Tensor | None = None


class StandardPrefixCacheBase(BasePrefixCache):
    @abstractmethod
    def insert_prefix(
        self,
        input_ids: torch.Tensor,
        value    : StandardPrefixValue,
    ) -> InsertResult: ...

    @abstractmethod
    def evict(self, size: int) -> StandardPrefixValue: ...

