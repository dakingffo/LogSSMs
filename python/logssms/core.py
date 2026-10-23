from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from logssms.cache.metadata import CacheMetadata
    from logssms.engine.piecewise_graph import PiecewiseGraphRunner
    from logssms.utils.batch import BatchIndices
    from logssms.cache import BaseCacheHandle
    from logssms.cache.base import KVCachePool, StateCachePool
    from logssms.layers.attention_backend import BaseAttnBackend, BaseAttnMetadata
    from logssms.layers.linear_attention_backend import (
        BaseLinearAttentionBackend,
        BaseLinearAttentionMetadata,
    )
    from logssms.layers.moe_backend import BaseMoeBackend
    from logssms.speculator.hidden_captor import HiddenCaptorBase


def _make_default_linear_attention_backend():
    from logssms.layers.linear_attention_backend import GatedDeltaBackend

    return GatedDeltaBackend()


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k      : int   = -1
    top_p      : float = 1.0
    ignore_eos : bool  = False
    max_tokens : int   = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Request:
    input_ids      : torch.Tensor
    table_idx      : int
    cached_len     : int
    output_len     : int
    uid            : int
    sampling_params: SamplingParams
    cache_handle   : BaseCacheHandle
    max_device_len : int
    profile_tag    : str | None = None
    return_token_ids: bool = False
    prefix_cached_len: int | None = None
    device_len     : int  = field(init=False)
    aborted        : bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)
        if self.prefix_cached_len is None:
            self.prefix_cached_len = self.cached_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    def __lt__(self, other: Request) -> bool:
        return self.uid < other.uid

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def extend_device_n(self, n: int) -> None:
        self.device_len += min(n, self.max_device_len - self.device_len)

    def complete_to_device_len(self, progress: int = -1) -> None:
        self.cached_len = (
            self.device_len if progress < 0 else
            self.cached_len + min(progress, self.device_len - self.cached_len)
        )

    def complete_n(self, n: int) -> None:
        self.device_len += min(n, self.max_device_len - self.device_len)
        self.cached_len = self.device_len - 1

    def append_host(self, next_token: torch.Tensor) -> None:
        self.input_ids = torch.cat([self.input_ids, next_token])

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0 and not self.aborted

    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )

@dataclass(eq=False)
class ChunkedRequest(Request):
    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedRequest should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False


@dataclass
class Batch:
    reqs     : list[Request]
    phase    : Literal["prefill", "decode", "verify", "mixed"]
    num_prefill_reqs: int = 0
    profile_id: int | None = field(init=False, default=None)

    input_ids  : torch.Tensor  = field(init=False)
    positions  : torch.Tensor  = field(init=False)
    out_loc    : torch.Tensor  = field(init=False)
    padded_reqs: list[Request] = field(init=False)
    prepared_indices: BatchIndices | None = field(init=False, default=None)

    attn_metadata: BaseAttnMetadata = field(init=False)


    draft_tokens: torch.Tensor | None = field(init=False, default=None)
    draft_probs : torch.Tensor | None = field(init=False, default=None)
    hidden_captor: HiddenCaptorBase | None = field(init=False, default=None)
    linear_attn_metadata: BaseLinearAttentionMetadata = field(init=False)
    cache_metadata: CacheMetadata | None = field(init=False, default=None)
    mixed_parts: tuple[Batch, Batch] | None = field(init=False, default=None)
    prefill_token_capacity: int = field(init=False, default=0)
    prefill_partition: tuple[int, ...] = field(init=False, default=())
    sample_indices: torch.Tensor | None = field(init=False, default=None)
    sample_req_indices: torch.Tensor | None = field(init=False, default=None)

    @property
    def is_mixed(self) -> bool:
        return self.phase == "mixed"

    @property
    def prefill_count(self) -> int:
        return self.size if self.is_prefill else self.num_prefill_reqs

    @cached_property
    def prefill_tokens(self) -> int:
        return sum(req.extend_len for req in self.prefill_reqs)

    @property
    def prefill_reqs(self) -> list[Request]:
        return self.reqs[:self.prefill_count]

    @property
    def decode_reqs(self) -> list[Request]:
        return self.reqs[self.prefill_count:] if not self.is_verify else []

    def req_is_prefill(self, index: int) -> bool:
        return index < self.prefill_count

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def is_verify(self) -> bool:
        return self.phase == "verify"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)


@dataclass
class Context:
    page_size: int
    page_table         : torch.Tensor               = field(init=False)
    attn_backend       : BaseAttnBackend            = field(init=False)
    linear_attn_backend: BaseLinearAttentionBackend = field(
        default_factory=_make_default_linear_attention_backend,
        init=False,
    )
    moe_backend        : BaseMoeBackend             = field(init=False)
    kv_cache           : KVCachePool            = field(init=False)
    linear_state       : StateCachePool            = field(init=False)

    state_table        : torch.Tensor | None = field(default=None, init=False)
    work_state_table   : torch.Tensor | None = field(default=None, init=False)
    piecewise_graph_runner: PiecewiseGraphRunner | None = field(default=None, init=False)

    _batch      : Batch | None        = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None | dict[str, Context | None]  = None


def set_global_ctx(ctx: Context, key: str | None = None) -> None:
    global _GLOBAL_CTX
    if key is not None:
        assert isinstance(_GLOBAL_CTX, dict), "Global context is not a dict"
        value = _GLOBAL_CTX.get(key)
        assert value is None, f"Global context for '{key=}' is already set"
        _GLOBAL_CTX[key] = ctx
    else:
        assert _GLOBAL_CTX is None, "Global context is already set"
        _GLOBAL_CTX = ctx


def get_global_ctx(key: str | None = None) -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    if key is not None:
        assert isinstance(_GLOBAL_CTX, dict), "Global context is not a dict"
        value = _GLOBAL_CTX.get(key)
        assert value is not None, f"Global context for '{key=}' is not set"
        return value
    else:
        return _GLOBAL_CTX


__all__ = [
    "SamplingParams",
    "Request",
    "Batch",
    "Context",
    "set_global_ctx",
    "get_global_ctx",
]
