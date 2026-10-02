from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from logssms.engine.prefill_graph import PrefillGraphRunner
    from logssms.utils.batch import BatchIndices
    from logssms.cache import BaseCacheHandle, BaseKVCachePool, LinearCachePool
    from logssms.layers.attention_backend import BaseAttnBackend, BaseAttnMetadata
    from logssms.layers.linear_attention_backend import (
        BaseLinearAttentionBackend,
        BaseLinearAttentionMetadata,
    )
    from logssms.layers.moe_backend import BaseMoeBackend
    from logssms.speculator.hidden_captor import HiddenCaptorBase


def _make_default_linear_attention_backend():
    from logssms.layers.linear_attention_backend import LinearAttentionBackend

    return LinearAttentionBackend()


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
    input_ids      : torch.Tensor  # assert is_cpu
    table_idx      : int
    cached_len     : int
    output_len     : int
    uid            : int
    sampling_params: SamplingParams
    cache_handle   : BaseCacheHandle
    max_device_len : int
    profile_tag    : str | None = None
    return_token_ids: bool = False
    device_len     : int  = field(init=False)
    aborted        : bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)
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
        return False  # avoid being added to ar manager


@dataclass
class Batch:
    reqs     : list[Request]
    phase    : Literal["prefill", "decode", "verify"]
    profile_id: int | None = field(init=False, default=None)
    # these fields should be set by scheduler
    input_ids  : torch.Tensor  = field(init=False)
    positions  : torch.Tensor  = field(init=False)
    out_loc    : torch.Tensor  = field(init=False)
    padded_reqs: list[Request] = field(init=False)
    prepared_indices: BatchIndices | None = field(init=False, default=None)
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)
    # spec-decode: drafts the MTP head produced for this verify batch. draft_tokens is
    # (bs, K) int32; draft_probs is (bs, K, vocab) fp32 = the MTP head's softmax per draft
    # step, used for the residual rejection sampling (needs the full distribution).
    draft_tokens: torch.Tensor | None = field(init=False, default=None)
    draft_probs : torch.Tensor | None = field(init=False, default=None)
    hidden_captor: HiddenCaptorBase | None = field(init=False, default=None)
    linear_attn_metadata: BaseLinearAttentionMetadata = field(init=False)
    logstate_first_offsets: list[int] | None = field(init=False, default=None)
    logstate_replay_counts: list[int] | None = field(init=False, default=None)
    logstate_checkpoint_writes: list[tuple[int, int]] = field(init=False, default_factory=list)
    logstate_checkpoint_ranges: list[tuple[int, int, int]] = field(init=False, default_factory=list)

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
    kv_cache           : BaseKVCachePool            = field(init=False)   # full attention
    linear_state       : LinearCachePool            = field(init=False)   # linear attention

    state_table        : torch.Tensor | None = field(default=None, init=False)
    work_state_table   : torch.Tensor | None = field(default=None, init=False)
    prefill_graph_runner: PrefillGraphRunner | None = field(default=None, init=False)

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


def clear_global_ctx(key: str | None = None) -> None:
    global _GLOBAL_CTX
    if key is not None:
        assert isinstance(_GLOBAL_CTX, dict), "Global context is not a dict"
        value = _GLOBAL_CTX.get(key, -1)
        assert value != -1, f"Global context does not have '{key=}'"
        _GLOBAL_CTX[key] = None
    else:
        _GLOBAL_CTX = None


__all__ = [
    "SamplingParams",
    "Request",
    "Batch",
    "Context",
    "set_global_ctx",
    "get_global_ctx",
    "clear_global_ctx",
]
