from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from time import perf_counter_ns
from typing import TYPE_CHECKING, Sequence

import torch

from logssms.core import Request

from .kv_pool.base import BaseKVCachePool
from .linear_pool.base import LinearCachePool
from .prefix_cache.base import BaseCacheHandle, BasePrefixCache, MatchResult

if TYPE_CHECKING:
    from logssms.core import Batch
    from logssms.message import UserMsg
    from logssms.scheduler.prefill import PendingRequest


@dataclass
class CacheResources:
    kv_pool         : BaseKVCachePool
    page_table      : torch.Tensor
    linear_state    : LinearCachePool | None
    state_table     : torch.Tensor | None
    work_state_table: torch.Tensor | None


class BaseCacheManager(ABC):
    def __init__(
        self,
        page_size        : int,
        resources        : CacheResources,
        prefix_cache_type: str | None       = None,
        prefix_cache     : BasePrefixCache | None = None,
    ) -> None:
        self.device = resources.page_table.device
        self.page_size = page_size
        self.resources = resources
        assert (prefix_cache_type is None) != (prefix_cache is None), (
            "Provide exactly one of prefix_cache_type and prefix_cache."
        )
        if prefix_cache is None:
            from .prefix_cache import make_prefix_cache

            assert prefix_cache_type is not None
            prefix_cache = make_prefix_cache(
                device=self.device, type=prefix_cache_type
            )
        self.prefix_cache = prefix_cache
        self._match_profile_enabled = os.getenv(
            "LOGSSMS_PROFILE_SCHEDULER", "0"
        ).lower() in ("1", "true", "yes")
        self._match_profile_samples: list[dict] = []
        self._profile_session_tokens: dict[str, torch.Tensor] = {}
        self._profile_lifecycle_events: list[dict] = []
        self._profile_arrivals: dict[int, dict] = {}
        self._profile_completed_sessions: dict[str, dict] = {}

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
    def linear_pool(self) -> LinearCachePool | None:
        return self.resources.linear_state

    def match_req(self, req: PendingRequest) -> MatchResult:
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])

    def lock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def prefill_acquire(
        self,
        total_len  : int,
        handle     : BaseCacheHandle,
        request_len: int | None = None,
    ) -> bool:
        if total_len > self.available_size:
            return False
        self.lock(handle)
        if total_len > self.available_size:
            self.unlock(handle)
            return False
        return True

    def prepare_batch(self, batch: Batch) -> None:
        """Prepare cache-specific work after allocation and before model metadata."""
        return None

    def record_prefill_queue(
        self,
        pending_count : int,
        oldest_wait_ms: float,
    ) -> None:
        return None

    def record_prefill_wait(self, wait_ms: float) -> None:
        return None

    def record_arrival_profile(self, req: UserMsg) -> None:
        return None

    def _record_lifecycle_event(self, event: dict) -> None:
        event["seq"] = len(self._profile_lifecycle_events)
        event["time_ns"] = perf_counter_ns()
        self._profile_lifecycle_events.append(event)

    def _record_arrival_snapshot(
        self,
        req            : UserMsg,
        tree_match_len : int,
        recoverable_len: int,
        node_path      : list[int],
    ) -> None:
        tag = req.profile_tag or f"uid:{req.uid}"
        session_key = tag.partition("/turn-")[0] if "/turn-" in tag else None
        previous = self._profile_completed_sessions.get(session_key)
        event = {
            "event": "arrival",
            "tag": tag,
            "uid": req.uid,
            "matchable_len": max(0, len(req.input_ids) - 1),
            "tree_match_len": tree_match_len,
            "recoverable_len": recoverable_len,
            "node_path": node_path,
            "previous_inserted_len": (
                previous["inserted_len"] if previous is not None else None
            ),
        }
        self._record_lifecycle_event(event)
        self._profile_arrivals[req.uid] = event

    def _record_complete_profile(
        self,
        req         : Request,
        inserted_len: int,
        node_path   : list[int],
    ) -> None:
        if not self._match_profile_enabled:
            return
        tag = req.profile_tag or f"uid:{req.uid}"
        event = {
            "event": "complete",
            "tag": tag,
            "uid": req.uid,
            "request_cached_len": req.cached_len,
            "inserted_len": inserted_len,
            "node_path": node_path,
        }
        self._record_lifecycle_event(event)
        if "/turn-" in tag:
            self._profile_completed_sessions[tag.partition("/turn-")[0]] = event

    def record_match_profile(
        self,
        req: PendingRequest,
        match: MatchResult,
        *,
        input_len: int,
    ) -> None:
        if not self._match_profile_enabled:
            return
        handle = match.cuda_handle
        tree_match_len = (
            handle.cached_len
            if match.tree_match_len is None else match.tree_match_len
        )
        recoverable_len = (
            handle.cached_len
            if match.recoverable_len is None else match.recoverable_len
        )
        tag = req.profile_tag or f"uid:{req.uid}"
        session_key = tag.partition("/turn-")[0] if "/turn-" in tag else None
        previous_tokens = (
            self._profile_session_tokens.get(session_key)
            if session_key is not None else None
        )
        session_lcp_len = 0
        if previous_tokens is not None:
            compare_len = min(len(previous_tokens), req.input_len)
            different = torch.nonzero(
                previous_tokens[:compare_len] != req.input_ids[:compare_len]
            )
            session_lcp_len = (
                compare_len if len(different) == 0 else int(different[0, 0])
            )
        session_expected_hit = min(session_lcp_len, max(0, input_len - 1))
        match_granularity = getattr(self, "granularity", self.page_size)
        session_aligned_expected_hit = (
            session_expected_hit // match_granularity * match_granularity
        )
        sample = {
            "tag": tag,
            "input_len": input_len,
            # match_req intentionally excludes the final prompt token.
            "matchable_len": max(0, input_len - 1),
            "tree_match_len": tree_match_len,
            "recoverable_len": recoverable_len,
            "admitted_cached_len": handle.cached_len,
            "state_recovery_gap": max(0, tree_match_len - recoverable_len),
            "unmatched_suffix_len": max(0, input_len - 1 - tree_match_len),
            "session_previous_cached_len": (
                len(previous_tokens) if previous_tokens is not None else 0
            ),
            "session_prefix_lcp_len": session_lcp_len,
            "session_expected_hit_len": session_expected_hit,
            "session_aligned_expected_hit_len": session_aligned_expected_hit,
            "session_cache_miss_len": max(
                0, session_aligned_expected_hit - tree_match_len
            ),
            "admission_wait_ms": (
                perf_counter_ns() - req.queued_at_ns
            ) / 1e6,
        }
        arrival = self._profile_arrivals.pop(req.uid, None)
        sample["arrival_tree_match_len"] = (
            arrival["tree_match_len"] if arrival is not None else None
        )
        sample["arrival_recoverable_len"] = (
            arrival["recoverable_len"] if arrival is not None else None
        )
        sample["arrival_seq"] = arrival["seq"] if arrival is not None else None
        sample["previous_inserted_len"] = (
            arrival["previous_inserted_len"] if arrival is not None else None
        )
        event = {
            "event": "admission",
            "tag": tag,
            "uid": req.uid,
            "tree_match_len": tree_match_len,
            "recoverable_len": recoverable_len,
            "arrival_seq": sample["arrival_seq"],
        }
        self._record_lifecycle_event(event)
        sample["admission_seq"] = event["seq"]
        self._match_profile_samples.append(sample)

    def record_cached_session(self, req: Request, cached_len: int) -> None:
        if not self._match_profile_enabled or not req.profile_tag:
            return
        session_key, separator, _turn = req.profile_tag.partition("/turn-")
        if separator:
            self._profile_session_tokens[session_key] = (
                req.input_ids[:cached_len].clone()
            )

    def record_active_requests(self, reqs: Sequence[Request]) -> None:
        return None

    def poll_profile(self) -> None:
        return None

    def profile_result(self) -> dict | None:
        if not self._match_profile_enabled:
            return None
        return {
            "type": type(self).__name__,
            "match_samples": self._match_profile_samples,
            "lifecycle_events": self._profile_lifecycle_events,
        }

    @property
    @abstractmethod
    def available_size(self) -> int: ...

    @abstractmethod
    def plan_prefill_chunk(self, remain_len: int, token_budget: int) -> int: ...

    @abstractmethod
    def init_with_handle(self, table_idx: int, handle: BaseCacheHandle) -> None: ...

    @abstractmethod
    def allocate(self, reqs: list[Request], *, phase: str | None = None) -> None: ...

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
