from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

import torch
from logssms.cache.prefix_cache import BaseCacheHandle, BasePrefixCache, MatchResult, SizeInfo

_STDLIB_MONOTONIC_NS = time.monotonic_ns


class LogStateKind(Enum):
    LOG = "log"
    CHECKPOINT = "checkpoint"


@dataclass(frozen=True)
class LogStateEntry:
    slot         : int
    kind         : LogStateKind
    begin        : int
    end          : int
    span_granules: int | None = None


@dataclass(frozen=True)
class LogStateRecovery:
    position            : int
    base_checkpoint_slot: int | None
    logs                : tuple[LogStateEntry, ...]
    conv_slot           : int | None
    conv_granule        : int | None

    @property
    def log_slots(self) -> tuple[int, ...]:
        return tuple(entry.slot for entry in self.logs)


class LogRadixTrace(NamedTuple):
    node_id        : int
    matched_len    : int
    is_exact       : bool
    has_children   : bool


_EMPTY_I32 = torch.empty(0, dtype=torch.int32)


@dataclass(frozen=True)
class LogStateCacheHandle(BaseCacheHandle):
    node_id            : int
    recovery           : LogStateRecovery
    recovery_slots_t   : torch.Tensor = field(default=_EMPTY_I32, repr=False, compare=False)
    kv_indices         : torch.Tensor = field(default=_EMPTY_I32, repr=False)
    analysis_owner     : int = -1
    source_owner       : int = -1
    analysis_group     : int = -1
    query_uid          : int | None = None
    request_end        : int = 0
    protected_endpoints: tuple[int, ...] = ()
    match_trace        : LogRadixTrace | None = field(default=None, compare=False)
    trace_revision     : int = field(default=-1, compare=False)
    trace_input_len    : int = field(default=0, compare=False)

    @property
    def recovery_slots(self) -> tuple[int, ...]:
        return tuple(self.recovery_slots_t.tolist()) if self.recovery_slots_t.numel() else ()

    def get_matched_indices(self) -> torch.Tensor:
        return self.kv_indices


class LogStatePrefixProxy(BasePrefixCache):
    """Thin BasePrefixCache proxy delegating all operations to C++ LogStateCore."""

    def __init__(self, core, device: torch.device, sync_clock) -> None:
        self._core = core
        self.device = device
        self._sync_clock = sync_clock
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        self._sync_clock()
        res = self._core["match_prefix"](input_ids.contiguous())
        base_ckpt = int(res["base_ckpt"])
        log_slots = torch.from_dlpack(res["log_slots"]).tolist()
        log_begins = torch.from_dlpack(res["log_begins"]).tolist()
        log_ends = torch.from_dlpack(res["log_ends"]).tolist()
        logs = tuple(
            LogStateEntry(s, LogStateKind.LOG, b, e)
            for s, b, e in zip(log_slots, log_begins, log_ends, strict=True)
        )
        conv_slot = int(res["conv_slot"])
        conv_gran = int(res["conv_granule"])
        cached_len = int(res["cached_len"])
        rec = LogStateRecovery(
            position=cached_len,
            base_checkpoint_slot=None if base_ckpt < 0 else base_ckpt,
            logs=logs,
            conv_slot=None if conv_slot < 0 else conv_slot,
            conv_granule=None if conv_gran < 0 else conv_gran,
        )
        kv_cpu = torch.from_dlpack(res["kv_indices"])
        rec_slots_t = torch.from_dlpack(res["recovery_slots"])
        matched_len = int(res["matched_len"])
        trace = LogRadixTrace(
            node_id=int(res["trace_node"]),
            matched_len=matched_len,
            is_exact=bool(res["trace_is_exact"]),
            has_children=bool(res["trace_has_children"]),
        )
        handle = LogStateCacheHandle(
            cached_len=cached_len,
            node_id=int(res["node_id"]),
            recovery=rec,
            recovery_slots_t=rec_slots_t,
            kv_indices=kv_cpu,
            source_owner=int(res["source_owner"]),
            match_trace=trace,
            trace_revision=int(res["trace_rev"]),
            trace_input_len=int(res["trace_input_len"]),
        )
        return MatchResult(handle, tree_match_len=matched_len, recoverable_len=cached_len)

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, LogStateCacheHandle)
        if unlock:
            self._core["unlock_handle"](
                handle.node_id, handle.cached_len, handle.recovery_slots_t,
            )
        else:
            if handle.recovery_slots_t.numel():
                self._core["pin_recovery"](handle.recovery_slots_t, False)
            if handle.cached_len > 0:
                self._core["lock_node"](handle.node_id, False, handle.cached_len)

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
        return int(self._core["tree_insert_raw"](
            input_ids.contiguous(), indices.cpu().to(torch.int32).contiguous(), [], -1,
        ))

    def evict(self, size: int) -> torch.Tensor:
        return self.empty_tensor

    def reset(self) -> None:
        self._core["tree_reset"]()

    @property
    def size_info(self) -> SizeInfo:
        ev, lk = self._core["size_info"]()
        return SizeInfo(int(ev), int(lk))

    def check_integrity(self) -> None:
        self._core["check_integrity"]([])


__all__ = [
    "LogRadixTrace",
    "LogStateCacheHandle",
    "LogStateEntry",
    "LogStateKind",
    "LogStatePrefixProxy",
    "LogStateRecovery",
]
