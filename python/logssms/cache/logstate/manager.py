from __future__ import annotations

import os
from collections import Counter, OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from time import perf_counter_ns
from typing import TYPE_CHECKING, Sequence

import torch

from logssms.core import Request
from logssms.utils import align_ceil, align_down, div_ceil, init_logger
from logssms.utils.torch_utils import make_device_vectors

from logssms.cache.base import BaseCacheManager, CacheResources
from logssms.cache.logstate.metadata import LogStateCacheMetadata
from logssms.cache.factory import BaseCacheFactory, CachePlan, SUPPORTED_CACHE_FACTORIES
from logssms.cache.logstate.layout import (
    LOG_STATE_GRANULARITY,
    LOG_STATE_REPLAY_WINDOW,
    LogStateSlotLayout,
)
from logssms.cache.logstate.frame_pool import MHALogStatePool
from logssms.cache.prefix_cache import BaseCacheHandle, MatchResult
from logssms.cache.logstate.prefix_cache import (
    LogStateCacheHandle,
    LogStateRecovery,
    LogRadixTrace,
    LogRadixTree,
    LogStateEntry,
    LogStateKind,
    LogStateMergeStats,
)

from .evictor import LogStateEvictor
from .online_analysis_scheduler import (
    BranchProtection,
    CheckpointCoalesce,
    LogStatePlan,
    MaterializeCandidate,
    OnlineAnalysisScheduler,
    _MergePathContext,
    _PendingProtection,
)
from .recycler import (
    LogStateRecycleBin,
    RecycledCheckpoint,
    RecycledMatch,
    RecycledRestorePlan,
    _RecycledRestore,
)

if TYPE_CHECKING:
    from logssms.core import Batch
    from logssms.engine import EngineConfig
    from logssms.message import UserMsg
    from logssms.models.config import AffineLogCacheSpec
    from logssms.scheduler.prefill import PendingRequest
    from logssms.speculator import SpeculatorReserve

logger = init_logger(__name__)


@dataclass(frozen=True)
class RestoreTask:
    output_slot        : int
    base_slot          : int | None
    logs               : tuple[LogStateEntry, ...]
    source_conv_slot   : int
    source_conv_granule: int
    output_conv_granule: int


class LogStateConstruction:
    """Restore execution through the pool API, independent of cache policy."""

    def __init__(self, pool: MHALogStatePool) -> None:
        self.pool = pool

    def restore(
        self,
        restores: Sequence[RestoreTask],
        *,
        profile_enabled: bool,
        pending_cuda_events: list[tuple[dict, torch.cuda.Event, torch.cuda.Event]],
        record_cuda: bool = True,
    ) -> dict:
        """Submit independent restores; append timings to the caller-owned event list.

        The list is supplied per call because its owner may replace it when
        polling completed events. This component never retains or polls it.
        Disable record_cuda when the caller times a larger coalesce/restore step.
        """
        submit_timing_ms: dict[str, float] | None = (
            {} if profile_enabled else None
        )
        step_start = perf_counter_ns() if submit_timing_ms is not None else 0
        events = None
        event_setup_cpu_ms = 0.0
        if profile_enabled and record_cuda and restores:
            event_setup_start = perf_counter_ns()
            events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            events[0].record()
            event_setup_cpu_ms = (
                perf_counter_ns() - event_setup_start
            ) / 1e6
        pool = self.pool
        direct = [task for task in restores if not task.logs]
        if direct:
            assert all(task.base_slot is not None for task in direct)
            pool.copy_states([task.base_slot for task in direct],
                             [task.output_slot for task in direct])
        if submit_timing_ms is not None:
            submit_timing_ms["direct_state_copy_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6
            step_start = perf_counter_ns()
        # Each restore reads immutable cache entries and writes its own work slot.
        chains = [
            (
                task.base_slot,
                [
                    (entry.slot, 0, entry.end - entry.begin)
                    for entry in task.logs
                ],
                task.output_slot,
            )
            for task in restores if task.logs
        ]
        if submit_timing_ms is not None:
            submit_timing_ms["chain_planning_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6

        step_start = perf_counter_ns() if submit_timing_ms is not None else 0
        conv_vectors = make_device_vectors([
            [task.source_conv_slot for task in restores],
            [task.source_conv_granule for task in restores],
            [task.output_slot for task in restores],
            [task.output_conv_granule for task in restores],
        ], pool.device) if restores else ()
        if submit_timing_ms is not None:
            submit_timing_ms["conv_metadata_submit_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6

        construction = {
            "chains": len(chains),
            "waves": int(bool(chains)),
            "segments": sum(len(logs) for _, logs, _ in chains),
            "outputs": len(chains),
            "direct_checkpoint_copies": len(direct),
            "cuda_ms": None if chains else 0.0,
        }
        if chains:
            step_start = perf_counter_ns() if submit_timing_ms is not None else 0
            pool.kernels.restore(pool, chains)
            if submit_timing_ms is not None:
                submit_timing_ms["restore_submit_ms"] = (
                    perf_counter_ns() - step_start
                ) / 1e6
                submit_timing_ms["materialize_submit_ms"] = 0.0

        step_start = perf_counter_ns() if submit_timing_ms is not None else 0
        if restores:
            pool.kernels.copy_conv(pool, *conv_vectors)
        if submit_timing_ms is not None:
            submit_timing_ms["conv_restore_submit_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6
        if events is not None:
            event_end_setup_start = perf_counter_ns()
            events[1].record()
            pending_cuda_events.append(
                (construction, events[0], events[1])
            )
            event_setup_cpu_ms += (
                perf_counter_ns() - event_end_setup_start
            ) / 1e6
        construction["event_setup_cpu_ms"] = event_setup_cpu_ms
        if submit_timing_ms is not None:
            construction["submit_timing_ms"] = submit_timing_ms
        return construction


@dataclass(frozen=True)
class LogStateCachePlan(CachePlan):
    linear_spec       : AffineLogCacheSpec
    num_states        : int
    num_work_states   : int
    state_units       : int


@dataclass(frozen=True)
class _LiveGranule:
    begin           : int
    end             : int
    kv_offset       : int
    state_slot      : int | None
    checkpoint_begin: int | None = None


@dataclass(slots=True)
class _RequestCacheState:
    work_slot         : int
    work_committed_pos: int
    request_end       : int
    live_granules      : list[_LiveGranule] = field(default_factory=list)
    recycle_match     : RecycledMatch | None = None
    live_granule_cursor: int = 0

    def live_granule_at(self, position: int) -> _LiveGranule | None:
        granules = self.live_granules
        if not granules:
            return None
        cursor = min(self.live_granule_cursor, len(granules) - 1)
        granule = granules[cursor]
        if granule.begin <= position < granule.end:
            return granule
        # Decode normally uses the tail, including immediately after prefill.
        if granules[-1].begin <= position < granules[-1].end:
            self.live_granule_cursor = len(granules) - 1
            return granules[-1]
        if position < granule.begin:
            cursor = 0
        while cursor < len(granules) and granules[cursor].end <= position:
            cursor += 1
        if cursor < len(granules) and granules[cursor].begin <= position:
            self.live_granule_cursor = cursor
            return granules[cursor]
        return None


class LogStateCacheManager(BaseCacheManager):

    def __init__(
        self,
        num_pages                : int,
        num_states               : int,
        num_work_states          : int,
        page_size                : int,
        resources                : CacheResources,
        granularity              : int  = LOG_STATE_GRANULARITY,
        enable_memorized_evict   : bool = True,
        enable_merge_adjustment  : bool = True,
        min_sampling_len         : int  = 1024,
        max_merge_intensity      : int  = 16,
        default_merge_growth_rate: int = 2,
    ) -> None:
        pool = resources.linear_state
        state_table = resources.state_table
        work_state_table = resources.work_state_table
        assert isinstance(pool, MHALogStatePool)
        assert state_table is not None
        assert work_state_table is not None
        assert pool.page_size == page_size == granularity
        assert pool.layout.granularity == granularity
        assert num_pages > 0 and num_states > 0
        assert num_work_states == work_state_table.numel()
        assert pool.allocator.num_frames >= num_pages + 1

        tree = LogRadixTree(
            granularity=granularity,
            device=resources.page_table.device,
            on_slot_reclaimable=pool.allocator.set_reclaimable,
            checkpoint_log_count=pool.allocator.checkpoint_log_count,
        )
        super().__init__(
            page_size=page_size,
            resources=resources,
            prefix_cache=tree,
        )
        self._profile_enabled = os.getenv(
            "LOGSSMS_PROFILE_SCHEDULER", "0"
        ).lower() in ("1", "true", "yes")
        self.analyzer = OnlineAnalysisScheduler(
            granularity=granularity,
            checkpoint_log_count=pool.allocator.checkpoint_log_count,
            enable_merge_adjustment=enable_merge_adjustment,
            min_sampling_len=min_sampling_len,
            max_merge_intensity=max_merge_intensity,
            default_merge_growth_rate=default_merge_growth_rate,
            profile_enabled=self._profile_enabled,
        )
        self._construction = LogStateConstruction(pool)
        self._allocator = pool.allocator
        self.recycle_bin = (
            LogStateRecycleBin(64 * 1024 * 1024)
            if enable_memorized_evict else None
        )
        self.num_pages = num_pages
        self.num_states = num_states
        self._request_states: dict[int, _RequestCacheState] = {}
        self._recycled_restores: dict[int, _RecycledRestore] = {}
        self._prefill_protection_key: tuple = ()
        self._prefill_protections: tuple[_PendingProtection, ...] | None = None
        self._prefill_match_epoch: tuple = ()
        self._prefill_matches: OrderedDict[int, tuple] = OrderedDict()
        self._prefix_match_revision = -1
        self._prefix_matches: OrderedDict[int, tuple] = OrderedDict()
        self._protected_prefixes: dict[int, set[int]] = {}
        dummy_kv = pool.reserve(kv=1)
        assert dummy_kv is not None
        self._dummy_kv_frame = dummy_kv[0][0]
        self.page_table[-1].fill_(self._dummy_kv_frame * page_size)
        dummy = pool.reserve(logs=1, works=1)
        assert dummy is not None
        self._dummy_log_slot = dummy[1][0]
        self._dummy_work_slot = dummy[2][0]
        work_state_table[-1, 0] = self._dummy_work_slot
        state_table[-1].fill_(self._dummy_log_slot)
        pool.zero_state(self._dummy_work_slot)
        pool.zero_work_conv(self._dummy_work_slot)
        self.granularity = granularity
        self.enable_merge_adjustment = enable_merge_adjustment

        self.pending_recoveries: dict[int, LogStateRecovery] = {}
        self._pending_matches: dict[int, LogStateCacheHandle] = {}
        self._recovery_pins: dict[int, tuple[LogStateCacheHandle, tuple[int, ...]]] = {}
        self.last_plan: LogStatePlan | None = None
        self._lazy_free_pages: list[int] | None = None
        self._lazy_free_states: list[int] | None = None
        self._profile_stats = {
            "admission_attempts": 0,
            "admission_rejections": 0,
            "queue_observations": 0,
            "max_pending_requests": 0,
            "max_pending_wait_ms": 0.0,
            "current_pending_requests": 0,
            "current_oldest_wait_ms": 0.0,
            "admitted_requests": 0,
            "total_admission_wait_ms": 0.0,
            "max_admission_wait_ms": 0.0,
            "eviction_rounds": 0,
            "eviction_candidate_windows": 0,
            "eviction_full_snapshots": 0,
            "eviction_serialized_nodes": 0,
            "pressure_dropped_checkpoints": 0,
            "pressure_pruned_checkpoints": 0,
            "evicted_kv_tokens": 0,
            "recycle_matches": 0,
            "recycle_matched_tokens": 0,
            "pending_match_reuses": 0,
            "prefix_match_reuses": 0,
            "recycle_committed_checkpoints": 0,
            "recycle_reused_gpu_slots": 0,
            "recycle_committed_tokens": 0,
            "released_state_slots_from_eviction": 0,
            "released_state_slots_from_merge": 0,
            "scheduled_payload_conv_frames_from_merge": 0,
            "merge_batch_outcomes": {},
            "merge_plan_diagnostics": {},
            "merge_commit_skips": {},
        }
        self._profile_samples: list[dict] = []
        self._cache_req_samples: list[dict] = []
        self._profile_allocation_samples: list[dict] = []
        self._profile_active_requests: list[dict] = []
        self._profile_cuda_events: list[
            tuple[dict, torch.cuda.Event, torch.cuda.Event]
        ] = []
        self.evictor = LogStateEvictor(
            tree=tree,
            pool=pool,
            recent_checkpoint_limit=work_state_table.shape[0] - 1,
            profile_enabled=self._profile_enabled,
            profile_stats=self._profile_stats,
            allocation_samples=self._profile_allocation_samples,
            record_lifecycle_event=self._record_lifecycle_event,
        )

    @property
    def tree(self) -> LogRadixTree:
        assert isinstance(self.prefix_cache, LogRadixTree)
        return self.prefix_cache

    @property
    def logstate_pool(self) -> MHALogStatePool:
        pool = self.linear_pool
        assert isinstance(pool, MHALogStatePool)
        return pool

    def match_req(self, req: PendingRequest) -> MatchResult:

        version = None if req.input_ids.is_inference() else req.input_ids._version
        epoch = (
            self.tree.revision, self.tree.lock_revision,
            -1 if self.recycle_bin is None else self.recycle_bin.revision,
        )
        if epoch != self._prefill_match_epoch:
            self._prefill_matches.clear()
            self._prefill_match_epoch = epoch
        cached = self._prefill_matches.get(req.uid)
        if (cached is not None and cached[0] is req and cached[1] is req.input_ids
                and cached[2] == (version, req.output_len)):
            with self.record_schedule_step("match_reuse"):
                result = cached[3]
                handle = result.cuda_handle
                self.tree.touch_handle(handle)
                if self._profile_enabled:
                    self._profile_stats["pending_match_reuses"] += 1
                if handle.recycle_match is not None:
                    self.recycle_bin.touch(handle.recycle_match.branch)
                    if self._profile_enabled:
                        self._profile_stats["recycle_matches"] += 1
                        self._profile_stats["recycle_matched_tokens"] += (
                            handle.recycle_match.matched_tokens(handle.cached_len)
                        )
                return result
        result = self._match_req(req, version)

        epoch = (
            self.tree.revision, self.tree.lock_revision,
            -1 if self.recycle_bin is None else self.recycle_bin.revision,
        )
        if epoch != self._prefill_match_epoch:
            self._prefill_matches.clear()
            self._prefill_match_epoch = epoch
        self._prefill_matches[req.uid] = req, req.input_ids, (version, req.output_len), result
        self._prefill_matches.move_to_end(req.uid)
        limit = self.work_state_table.shape[0]
        while len(self._prefill_matches) > limit:
            self._prefill_matches.popitem(last=False)
        return result

    def _match_req(self, req: PendingRequest, version: int | None) -> MatchResult:
        with self.record_schedule_step("prefix_match"):
            if self._prefix_match_revision != self.tree.revision:
                self._prefix_matches.clear()
                self._prefix_match_revision = self.tree.revision
            cached = self._prefix_matches.get(req.uid)
            if (cached is not None and cached[0] is req and cached[1] is req.input_ids
                    and cached[2] == version):
                result = cached[3]
                self.tree.touch_handle(result.cuda_handle)
                if self._profile_enabled:
                    self._profile_stats["prefix_match_reuses"] += 1
            else:
                result = super().match_req(req)
                if self._prefix_match_revision != self.tree.revision:
                    self._prefix_matches.clear()
                    self._prefix_match_revision = self.tree.revision
                self._prefix_matches[req.uid] = req, req.input_ids, version, result
                self._prefix_matches.move_to_end(req.uid)
                limit = self.work_state_table.shape[0]
                while len(self._prefix_matches) > limit:
                    self._prefix_matches.popitem(last=False)
        handle = result.cuda_handle
        assert isinstance(handle, LogStateCacheHandle)
        if result.tree_match_len > handle.cached_len:
            position = result.tree_match_len
            trace = handle.match_trace
            assert trace is not None
            node = self.tree._node_at_depth(trace.node, position)
            self.tree.protect(node)
            self._protected_prefixes.setdefault(req.uid, set()).add(position)
            result = super().match_req(req)
            handle = result.cuda_handle
            assert isinstance(handle, LogStateCacheHandle)
        protected = tuple(sorted(self._protected_prefixes.get(req.uid, ())))
        values = req.input_ids[: req.input_len - 1]
        with self.record_schedule_step("recycle_match"):
            recycle_match = (
                self.recycle_bin.match(
                    values.cpu(), start=handle.cached_len,
                    granularity=self.granularity,
                )
                if self.recycle_bin is not None else None
            )
        if self._profile_enabled and recycle_match is not None:
            self._profile_stats["recycle_matches"] += 1
            self._profile_stats["recycle_matched_tokens"] += (
                recycle_match.matched_tokens(handle.cached_len)
            )
        if not self.enable_merge_adjustment:
            return MatchResult(
                replace(
                    handle, recycle_match=recycle_match,
                    restore_plan=self._plan_recycled_restore(
                        handle, recycle_match, (),
                    ),
                    request_end=req.input_len + req.output_len,
                    protected_endpoints=protected,
                ),
                tree_match_len=result.tree_match_len,
                recoverable_len=result.recoverable_len,
            )
        with self.record_schedule_step("merge_owner_lookup"):
            trace = handle.match_trace
            assert trace is not None and handle.trace_revision == self.tree.revision
            source_node = trace.node

            if trace.matched_len > 0:
                while source_node.merge_stats is None and len(source_node.children) == 1:
                    source_node = next(iter(source_node.children.values()))
            source = source_node.merge_stats
            if source is not None and not source.origin <= trace.matched_len <= source.tail:
                source = None
            continues_chain = (
                source is not None and trace.is_exact and trace.node.is_leaf()
                and not trace.node.is_locked
                and source.tail == trace.matched_len == handle.cached_len
            )
            match_position = (
                trace.matched_len if recycle_match is None else
                max(trace.matched_len, recycle_match.matched_len)
            )
            stats = source if continues_chain else self.analyzer.make_merge_stats(match_position)
            intensity = source.intensity if source is not None else 2
            if recycle_match is not None and recycle_match.branch.merge_stats:
                origin, tail, intensity, next_update, *hits = recycle_match.branch.merge_stats
                if recycle_match.matched_len == tail == len(recycle_match.branch.tokens):

                    stats = LogStateMergeStats(
                        origin, tail, intensity, deque(hits),
                        None if next_update < 0 else next_update,
                    )
        return MatchResult(
            replace(
                handle,
                merge_stats=stats,
                source_merge_stats=source,
                match_position=match_position,
                recycle_match=recycle_match,
                restore_plan=self._plan_recycled_restore(
                    handle, recycle_match, (),
                    intensity=intensity,
                ),
                request_end=req.input_len + req.output_len,
                protected_endpoints=protected,
            ),
            tree_match_len=result.tree_match_len,
            recoverable_len=result.recoverable_len,
        )

    def _prepare_prefill(self, reqs: Sequence[Request]) -> None:
        key = tuple((req, req.cached_len, req.device_len) for req in reqs)
        if self._prefill_protections is not None and self._prefill_protection_key == key:
            return
        pending = self.analyzer._find_protections(
            self.tree, [req.input_ids for req in reqs], traces=self._prefill_traces(reqs),
        )
        self._apply_protections(self.analyzer._plan_protections(pending))
        self._prefill_protection_key = key
        self._prefill_protections = pending
        points: dict[int, list[int]] = {}
        for value in pending:
            for index in value.request_indices:
                uid = reqs[index].uid
                points.setdefault(uid, []).append(value.position)
                self._protected_prefixes.setdefault(uid, set()).add(value.position)
        restores = [
            (req, self._recycled_restores[req.table_idx]) for req in reqs
            if req.table_idx in self._recycled_restores
            and not self._recycled_restores[req.table_idx].prepared
            and not self._request_states[req.table_idx].live_granules
        ]
        if not restores:
            return
        current_pins = Counter(slot for req in reqs for slot in self._recovery_pins.get(
            id(self._pending_matches.get(req.table_idx)), (None, ()))[1])
        blocked = {slot for slot, refs in self.tree._recovery_slot_refs.items()
                   if refs > current_pins[slot]}
        for req, restore in restores:
            branches = points.get(req.uid, ())
            handle = self._pending_matches[req.table_idx]
            plan = self._plan_recycled_restore(
                handle, handle.recycle_match, branches,
                intensity=restore.plan.intensity, previous=restore.plan,
                blocked_slots=frozenset(blocked),
            )
            assert plan is not None
            restore.plan = plan
            blocked.update(entry.slot for checkpoint in plan.checkpoints for entry in checkpoint.gpu_entries)
            blocked.update(slot for item in plan.gpu_materialize for slot in item.slots)
            blocked.update(slot for item in plan.gpu_coalesce for slot in item.slots)
            restore.slots = {
                checkpoint.end: slot for checkpoint in plan.checkpoints
                if (slot := self._checkpoint_output_slot(checkpoint)) is not None
            }

    def _checkpoint_output_slot(self, checkpoint: RecycledCheckpoint) -> int | None:
        for entry in reversed(checkpoint.gpu_entries):
            if (self.tree.has_entry(entry)
                and len(self._allocator.slot(entry.slot).payload) >= self._allocator.payload_frames):
                return entry.slot
        return None

    def _apply_protections(self, protections: Sequence[BranchProtection]) -> None:
        for protection in protections:
            self.tree.protect_at(protection.node, protection.position)

    def _plan_recycled_restore(
        self,
        handle       : LogStateCacheHandle,
        match        : RecycledMatch | None,
        batch_branches: Sequence[int],
        *,
        intensity    : int = 2,
        previous     : RecycledRestorePlan | None = None,
        blocked_slots: frozenset[int] = frozenset(),
    ) -> RecycledRestorePlan | None:
        if match is None:
            return None
        from logssms.kernel import logstate as planning

        cached_len = handle.cached_len
        gpu = tuple(entry for entry in handle.match_entries if entry.end <= cached_len)
        path = handle.match_path
        protected = frozenset((
            *match.branch.protected_endpoints, *batch_branches,
            *((match.matched_len,) if match.matched_len < len(match.branch.tokens) else ()),
            *(node.depth for node in path if node.is_protected),
        ))
        context = _MergePathContext.build(path, ())
        boundaries = tuple(sorted({*protected, *(node.depth for node in path
                                                if node.is_protected or not node.is_leaf())}))
        eligible_slots = None if previous is None else {
            *(entry.slot for checkpoint in previous.checkpoints for entry in checkpoint.gpu_entries),
            *(slot for item in previous.gpu_materialize for slot in item.slots),
            *(slot for item in previous.gpu_coalesce for slot in item.slots),
        }
        lock_boundaries = context.lock_boundaries if previous is None else previous.lock_boundaries
        pieces = [(entry.begin, entry.end, entry,
                   (not self.tree.entry_owner(entry).is_locked if eligible_slots is None
                    else entry.slot in eligible_slots)
                   and entry.slot not in blocked_slots
                   and not any(entry.begin < position < entry.end for position in boundaries))
                  for entry in gpu]
        for entry in match.branch.logs:
            begin = align_ceil(max(cached_len, entry.begin), self.granularity)
            end = align_down(min(match.matched_len, entry.end), self.granularity)
            pieces.extend((left, left + self.granularity, None, True)
                          for left in range(begin, end, self.granularity))
        pieces.sort(key=lambda item: (item[0], item[1]))
        rows = [value for index, (begin, end, entry, eligible) in enumerate(pieces)
                for value in (index, begin, end,
                              entry.span_granules if entry is not None and entry.kind is LogStateKind.CHECKPOINT else -1,
                              int(eligible))]
        candidates = [(0, index, int(entry is not None)) for index, (_, _, entry, eligible) in enumerate(pieces)
                      if eligible and (entry is None or entry.kind is LogStateKind.LOG)]
        materialize = planning.plan_materialize(
            [rows], [boundaries], candidates, self.granularity, self._allocator.checkpoint_log_count,
        )
        virtual = [(pieces[index][0], pieces[index][1], (index,), False)
                   for index, (_, _, entry, _) in enumerate(pieces)
                   if entry is not None and entry.kind is LogStateKind.CHECKPOINT]
        virtual.extend((pieces[candidates[group[0]][1]][0], pieces[candidates[group[-1]][1]][1],
                        tuple(candidates[index][1] for index in group), True)
                       for group in materialize)
        virtual.sort(key=lambda item: (item[0], item[1]))
        projected = [value for index, (begin, end, indices, _) in enumerate(virtual)
                     for value in (index, begin, end, (end - begin) // self.granularity,
                                   int(all(pieces[i][3] for i in indices)))]
        groups, _ = planning.plan_coalesce(
            [projected], [boundaries], [lock_boundaries],
            [(0, match.matched_len, intensity, int(self.enable_merge_adjustment))],
            self.granularity, False,
        )
        consumed = set()
        checkpoints, gpu_materialize, gpu_coalesce = [], [], []
        for group in groups:
            virtual_indices = group[3::2]
            consumed.update(virtual_indices)
            indices = tuple(index for i in virtual_indices for index in virtual[i][2])
            entries = tuple(pieces[index][2] for index in indices if pieces[index][2] is not None)
            if len(entries) == len(indices):
                gpu_coalesce.append(CheckpointCoalesce(
                    tuple(entry.slot for entry in entries),
                    tuple(entry.span_granules if entry.kind is LogStateKind.CHECKPOINT else 1 for entry in entries),
                    group[0], group[1], group[2],
                ))
            else:
                checkpoints.append(RecycledCheckpoint(group[0], group[1], entries))
        for index, (begin, end, indices, is_materialize) in enumerate(virtual):
            if index in consumed or not is_materialize:
                continue
            entries = tuple(pieces[i][2] for i in indices if pieces[i][2] is not None)
            if len(entries) == len(indices):
                gpu_materialize.append(MaterializeCandidate(tuple(entry.slot for entry in entries), begin, end))
            else:
                checkpoints.append(RecycledCheckpoint(begin, end, entries))
        return RecycledRestorePlan(
            cached_len, match.matched_len, tuple(sorted(checkpoints, key=lambda item: item.end)),
            protected, tuple(gpu_materialize), tuple(gpu_coalesce), intensity, lock_boundaries,
        )

    @property
    def admission_capacity(self) -> int:
        with self.record_schedule_step("reclamation_poll"):
            self.logstate_pool.poll_reclamation()
        with self.record_schedule_step("state_capacity"):
            frames, _ = self._allocator.projected_free_capacity()
        with self.record_schedule_step("kv_capacity"):
            kv_frames = self.tree.evictable_kv_frames()
        with self.record_schedule_step("frame_capacity"):
            return frames + kv_frames

    def reservation_size(
        self,
        token_len: int,
        *,
        handle   : BaseCacheHandle | None = None,
        table_idx: int | None = None,
    ) -> int:
        new_request = table_idx is None
        if new_request:
            assert isinstance(handle, LogStateCacheHandle)
            begin = handle.cached_len
            end = begin + token_len
            plan = handle.restore_plan
            slots = {}
            planned = False
        else:
            state = self._request_states[table_idx]
            end = state.request_end
            granules = state.live_granules
            begin = granules[-1].end if granules else max(0, end - token_len)
            restore = self._recycled_restores.get(table_idx)
            plan = None if restore is None else restore.plan
            slots = {} if restore is None else restore.slots
            planned = restore is None or restore.prepared
        if begin >= end:
            return 0
        kv = div_ceil(end, self.page_size) - begin // self.page_size
        checkpoints = () if plan is None else plan.checkpoints
        covered = sum(
            max(0, min(div_ceil(end, self.page_size), checkpoint.end // self.page_size)
                - max(div_ceil(begin, self.page_size), checkpoint.begin // self.page_size))
            for checkpoint in checkpoints
        )
        if covered and not planned:
            allocator = self._allocator
            log_frames = allocator.log_payload_frames + allocator.num_granules * allocator.conv_frames
            checkpoint_frames = allocator.payload_frames + allocator.conv_frames
            return allocator.required_frames(
                kv, kv, int(new_request), free_decay_records=0,
            ) + covered * max(0, checkpoint_frames - log_frames)
        states = int(new_request) + sum(
            checkpoint.end > begin and checkpoint.begin < end
            and checkpoint.end not in slots
            and self._checkpoint_output_slot(checkpoint) is None
            for checkpoint in checkpoints
        )

        return self._allocator.required_frames(
            kv, kv - covered, states, free_decay_records=0,
        )

    def prefill_acquire(
        self,
        total_len  : int,
        handle     : BaseCacheHandle,
        request_len: int | None = None,
    ) -> bool:
        del request_len
        if self._profile_enabled:
            self._profile_stats["admission_attempts"] += 1
        assert isinstance(handle, LogStateCacheHandle)
        sources = self._recovery_sources(handle)
        if any(not self.tree.has_slot(slot) for slot in sources):
            return False
        if handle.restore_plan is not None and any(
            not self.tree.has_slot(entry.slot)
            or not self.tree.has_entry(entry)
            or self.tree.entry_owner(entry).is_locked
            for checkpoint in handle.restore_plan.checkpoints
            for entry in checkpoint.gpu_entries
        ):
            return False
        if self.enable_merge_adjustment:
            assert isinstance(handle, LogStateCacheHandle)
            stats = handle.merge_stats
            if stats is not None and stats is handle.source_merge_stats and (
                handle.node.merge_stats is not stats
                or not handle.node.is_leaf() or handle.node.is_locked
                or handle.node.depth != stats.tail
            ):

                if self._profile_enabled:
                    self._profile_stats["admission_rejections"] += 1
                return False
        with self.record_schedule_step("capacity_before_lock"):
            available = self.admission_capacity
        if self._allocator.free_frames < total_len <= available:
            with self.record_schedule_step("lock_capacity_projection"):
                kv_frames, slots = self.tree.lock_footprint(handle.lock_node)
                unavailable = set(slots) | {
                    slot for slot in sources if slot in self.tree._evictable_slots
                }
                available -= kv_frames + self._allocator.lock_frame_cost(unavailable)
        if total_len > available:
            if self._profile_enabled:
                self._profile_stats["admission_rejections"] += 1
            return False
        with self.record_schedule_step("lock_handle"):
            self._pin_recovery(handle, sources)
            self.lock(handle)
        with self.record_schedule_step("capacity_after_lock"):
            admitted = total_len <= self.admission_capacity
        if not admitted:
            with self.record_schedule_step("unlock_rejected_handle"):
                self.unlock(handle)
            if self._profile_enabled:
                self._profile_stats["admission_rejections"] += 1
            return False
        if not self.enable_merge_adjustment:
            return True
        assert isinstance(handle, LogStateCacheHandle)
        self.analyzer.record_hit(
            handle.source_merge_stats, handle.match_position
        )
        return True

    @staticmethod
    def _recovery_sources(handle: LogStateCacheHandle) -> tuple[int, ...]:
        recovery = handle.recovery
        slots = set(recovery.log_slots)
        if recovery.base_checkpoint_slot is not None:
            slots.add(recovery.base_checkpoint_slot)
        if recovery.conv_slot is not None:
            slots.add(recovery.conv_slot)
        if handle.restore_plan is not None:
            slots.update(entry.slot for checkpoint in handle.restore_plan.checkpoints
                         for entry in checkpoint.gpu_entries)
            slots.update(slot for item in handle.restore_plan.gpu_materialize for slot in item.slots)
            slots.update(slot for item in handle.restore_plan.gpu_coalesce for slot in item.slots)
        return tuple(sorted(slots))

    def _pin_recovery(self, handle: LogStateCacheHandle, slots: tuple[int, ...]) -> None:
        if slots and id(handle) not in self._recovery_pins:
            self.tree.pin_recovery(slots)
            self._recovery_pins[id(handle)] = handle, slots

    def _release_recovery(self, handle_id: int) -> None:
        pinned = self._recovery_pins.pop(handle_id, None)
        if pinned is not None:
            self.tree.pin_recovery(pinned[1], release=True)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self._release_recovery(id(handle))
        super().unlock(handle)

    def record_prefill_queue(
        self,
        pending_count : int,
        oldest_wait_ms: float,
    ) -> None:
        if not self._profile_enabled:
            return
        stats = self._profile_stats
        stats["queue_observations"] += 1
        stats["current_pending_requests"] = pending_count
        stats["current_oldest_wait_ms"] = oldest_wait_ms
        stats["max_pending_requests"] = max(
            stats["max_pending_requests"], pending_count
        )
        stats["max_pending_wait_ms"] = max(
            stats["max_pending_wait_ms"], oldest_wait_ms
        )

    def record_prefill_wait(self, wait_ms: float) -> None:
        if not self._profile_enabled:
            return
        stats = self._profile_stats
        stats["admitted_requests"] += 1
        stats["total_admission_wait_ms"] += wait_ms
        stats["max_admission_wait_ms"] = max(
            stats["max_admission_wait_ms"], wait_ms
        )

    def record_arrival_profile(self, req: UserMsg) -> None:
        if not self._profile_enabled:
            return
        matchable_len = max(0, len(req.input_ids) - 1)
        usable_len = align_down(matchable_len, self.granularity)
        trace = self.tree.trace(req.input_ids[:usable_len])
        recovery, _ = self.tree._recovery_for_trace(trace)
        self._record_arrival_snapshot(
            req, trace.matched_len, recovery.position,
            [node.uuid for node in self.tree.path(trace.node)],
        )

    def record_active_requests(self, reqs: Sequence[Request]) -> None:
        if self._profile_enabled:
            self._profile_active_requests = self._profile_request_paths(reqs)

    def poll_profile(self) -> None:
        if not self._profile_enabled:
            return
        pending = []
        for construction, start, end in self._profile_cuda_events:
            if end.query():
                construction["cuda_ms"] = start.elapsed_time(end)
            else:
                pending.append((construction, start, end))
        self._profile_cuda_events = pending

    def profile_result(self) -> dict | None:
        if not self._profile_enabled:
            return None
        self.poll_profile()
        stats = dict(self._profile_stats)
        admitted = stats["admitted_requests"]
        stats["avg_admission_wait_ms"] = (
            stats["total_admission_wait_ms"] / admitted
            if admitted else 0.0
        )
        stats["pending_cuda_events"] = len(self._profile_cuda_events)
        completed = [
            sample["construction"]["cuda_ms"]
            for sample in self._profile_samples
            if sample["construction"]["cuda_ms"] is not None
        ]
        stats["construction_cuda_ms"] = sum(completed)
        stats["avg_construction_cuda_ms"] = (
            sum(completed) / len(completed) if completed else 0.0
        )
        stats["max_construction_cuda_ms"] = max(completed, default=0.0)
        resources = [
            sample[key]
            for sample in self._profile_samples
            for key in ("resources_before", "resources_after")
        ]
        constructions = [
            sample["construction"] for sample in self._profile_samples
        ]
        stats["min_free_kv_pages"] = min(
            (value["free_kv_pages"] for value in resources), default=self.num_pages
        )
        stats["min_free_state_slots"] = min(
            (value["free_state_slots"] for value in resources),
            default=self.num_states,
        )
        stats["max_construction_chains"] = max(
            (value["chains"] for value in constructions), default=0
        )
        stats["max_construction_segments"] = max(
            (value["segments"] for value in constructions), default=0
        )
        stats["max_construction_outputs"] = max(
            (value["outputs"] for value in constructions), default=0
        )
        timing_fields = (
            "analysis_and_tasks_cpu_ms",
            "profile_before_cpu_ms",
            "construction_submit_cpu_ms",
            "construction_event_setup_cpu_ms",
            "commit_cpu_ms",
            "profile_after_cpu_ms",
            "manager_total_cpu_ms",
            "unaccounted_cpu_ms",
        )
        timing_totals = {
            key: sum(
                sample.get("timing_ms", {}).get(key, 0.0)
                for sample in self._profile_samples
            )
            for key in timing_fields
        }
        timing_totals["production_cpu_ms"] = (
            timing_totals["analysis_and_tasks_cpu_ms"]
            + timing_totals["construction_submit_cpu_ms"]
            - timing_totals["construction_event_setup_cpu_ms"]
            + timing_totals["commit_cpu_ms"]
        )
        timing_totals["profile_diagnostics_cpu_ms"] = (
            timing_totals["profile_before_cpu_ms"]
            + timing_totals["profile_after_cpu_ms"]
            + timing_totals["construction_event_setup_cpu_ms"]
        )
        timing_totals["avg_manager_total_cpu_ms"] = (
            timing_totals["manager_total_cpu_ms"] / len(self._profile_samples)
            if self._profile_samples else 0.0
        )
        stats["prepare_cpu_timing_ms"] = timing_totals
        allocation_phases: dict[str, dict] = {}
        for sample in self._profile_allocation_samples:
            phase = allocation_phases.setdefault(sample["phase"], {
                "calls": 0, "eviction_calls": 0, "evicted_nodes": 0,
                "timing_ms": {}, "counts": {},
            })
            phase["calls"] += 1
            phase["eviction_calls"] += int(
                sample["evicted_nodes"] > 0
                or sample.get("counts", {}).get("dropped_checkpoints", 0) > 0
            )
            phase["evicted_nodes"] += sample["evicted_nodes"]
            for name, value in sample["timing_ms"].items():
                phase["timing_ms"][name] = (
                    phase["timing_ms"].get(name, 0.0) + value
                )
            for name, value in sample.get("counts", {}).items():
                phase["counts"][name] = phase["counts"].get(name, 0) + value
        return {
            "type": "logstate",
            "capacity": {
                "total_frames": self._allocator.num_frames,
                "recent_checkpoint_limit": self.work_state_table.shape[0] - 1,
                "frame_bytes": self._allocator.frame_bytes,
                "payload_frames_per_slot": self._allocator.payload_frames,
                "log_payload_frames_per_slot": self._allocator.log_payload_frames,
                "conv_frames_per_snapshot": self._allocator.conv_frames,
                "granules_per_log": self._allocator.num_granules,
                "decay_records_per_frame": self._allocator.decay_records_per_frame,
            },
            "aggregate": stats,
            "online_merge": self.analyzer.profile_result(),
            "match_samples": self._match_profile_samples,
            "prefill_batches": self._profile_samples,
            "cache_req_samples": self._cache_req_samples,
            "allocation_profile": {
                "phases": allocation_phases,
                "samples": self._profile_allocation_samples,
            },
            "lifecycle_events": self._profile_lifecycle_events,
        }

    @property
    def available_size(self) -> int:
        if self._schedule_profile_step is not None:
            with self.record_schedule_step("reclamation_poll"):
                self.logstate_pool.poll_reclamation()
            with self.record_schedule_step("kv_capacity"):
                evictable_kv_frames = self.tree.evictable_kv_frames()
            with self.record_schedule_step("state_capacity"):
                free_capacity = self._allocator.projected_free_capacity()
            with self.record_schedule_step("frame_capacity"):
                return self._allocator.admissible_pages(
                    evictable_kv_frames=evictable_kv_frames,
                    free_capacity=free_capacity,
                ) * self.page_size

        self.logstate_pool.poll_reclamation()
        return self._allocator.admissible_pages(
            evictable_kv_frames=self.tree.evictable_kv_frames(),
        ) * self.page_size

    def plan_prefill_chunk(self, remain_len: int, token_budget: int) -> int:
        chunk_size = min(remain_len, token_budget)
        if chunk_size < remain_len:
            chunk_size = align_down(chunk_size, self.granularity)
        return chunk_size

    def init_with_handle(self, table_idx: int, handle: BaseCacheHandle) -> None:
        assert isinstance(handle, LogStateCacheHandle)
        assert table_idx not in self._request_states
        self._pin_recovery(handle, self._recovery_sources(handle))
        with self.record_schedule_step("work_allocate"):
            work_slot = self._allocate_work()
        self._request_states[table_idx] = _RequestCacheState(
            work_slot=work_slot, work_committed_pos=handle.cached_len,
            request_end=handle.request_end, recycle_match=handle.recycle_match,
        )
        work_state_table = self.work_state_table
        assert work_state_table is not None
        with self.record_schedule_step("work_mapping"):
            work_state_table[table_idx, 0] = work_slot
        if handle.cached_len > 0:
            with self.record_schedule_step("cached_page_mapping"):
                self.page_table[table_idx, : handle.cached_len].copy_(
                    handle.kv_indices
                )
        state_table = self.state_table
        assert state_table is not None
        with self.record_schedule_step("state_mapping_clear"):
            state_table[table_idx].fill_(-1)
        self.pending_recoveries[table_idx] = handle.recovery
        self._pending_matches[table_idx] = handle
        if handle.restore_plan is not None:
            restore = _RecycledRestore(handle.restore_plan)
            restore.slots.update({
                checkpoint.end: slot for checkpoint in restore.plan.checkpoints
                if (slot := self._checkpoint_output_slot(checkpoint)) is not None
            })
            self._recycled_restores[table_idx] = restore

    def allocate(self, reqs: list[Request], *, phase: str | None = None) -> None:
        state_table = self.state_table
        work_state_table = self.work_state_table
        assert state_table is not None and work_state_table is not None
        start = perf_counter_ns() if self._profile_enabled else 0
        profile = (
            {"phase": phase or "direct", "batch_size": len(reqs),
             "request_tags": [req.profile_tag for req in reqs],
            "needed_pages": 0, "needed_states": 0, "needed_works": 0,
             "evicted_nodes": 0, "timing_ms": {}}
            if self._profile_enabled else None
        )

        if phase == "prefill":
            with self.record_schedule_step("prefill_restore_plan"):
                self._prepare_prefill(reqs)

        allocation_info: list[tuple[Request, int, int, dict[int, int]]] = []
        needed = 0
        needed_logs = 0
        needed_checkpoints = 0
        fresh_work_slots: list[int] = []
        for req in reqs:
            state = self._request_states[req.table_idx]
            state.request_end = req.max_device_len
            granules = state.live_granules
            begin = granules[-1].end if granules else req.cached_len
            if req.device_len > begin:
                first_span = self.page_size - begin % self.page_size
                count = 1 + div_ceil(
                    max(0, req.device_len - begin - first_span), self.page_size
                )
                restore = self._recycled_restores.get(req.table_idx)
                direct = {
                    checkpoint.end: checkpoint.begin
                    for checkpoint in (() if restore is None else restore.plan.checkpoints)
                    if checkpoint.end > begin and checkpoint.begin < req.device_len
                } if phase == "prefill" else {}
                allocation_end = align_ceil(req.device_len, self.page_size)
                checkpoint_pages = sum(
                    max(0, min(end, allocation_end) - max(start, align_ceil(begin, self.page_size)))
                    // self.page_size
                    for end, start in direct.items()
                )
                allocation_info.append((req, begin, count, direct))
                needed += count
                needed_logs += count - checkpoint_pages
                needed_checkpoints += sum(
                    end <= allocation_end and (restore is None or end not in restore.slots)
                    for end in direct
                )

            recovery = self.pending_recoveries.get(req.table_idx)
            if req.cached_len == 0 and (recovery is None or recovery.position == 0):
                fresh_work_slots.append(state.work_slot)

        zero_start = perf_counter_ns() if profile is not None else 0
        self.logstate_pool.zero_work_slots(fresh_work_slots)
        zero_work_ms = (
            (perf_counter_ns() - zero_start) / 1e6 if profile is not None else 0.0
        )

        if profile is not None:
            now = perf_counter_ns()
            profile["needed_pages"] = needed
            profile["needed_states"] = needed_logs
            profile["needed_works"] = needed_checkpoints
            profile["reused_checkpoint_slots"] = sum(
                checkpoint.end > begin and checkpoint.end <= align_ceil(req.device_len, self.page_size)
                and bool(checkpoint.gpu_entries)
                for req, begin, _count, _direct in allocation_info
                if req.table_idx in self._recycled_restores
                for checkpoint in self._recycled_restores[req.table_idx].plan.checkpoints
            )
            profile["timing_ms"]["scan_and_zero_ms"] = (now - start) / 1e6
            profile["timing_ms"]["zero_work_ms"] = zero_work_ms
            profile["timing_ms"]["scan_ms"] = (
                profile["timing_ms"]["scan_and_zero_ms"] - zero_work_ms
            )
            stage = now
        if needed == 0:
            if profile is not None:
                profile["timing_ms"]["total_ms"] = (
                    perf_counter_ns() - start
                ) / 1e6
                self._profile_allocation_samples.append(profile)
            return

        kv_offsets, state_slots, direct_slots = self._allocate(
            needed, needed_logs, needed_checkpoints, profile=profile
        )
        if profile is not None:
            now = perf_counter_ns()
            profile["timing_ms"]["core_allocate_ms"] = (now - stage) / 1e6
            stage = now
        offset = 0
        log_offset = 0
        checkpoint_offset = 0
        regions = []
        for req, begin, count, direct in allocation_info:
            live_granules = self._request_states[req.table_idx].live_granules
            restore = self._recycled_restores.get(req.table_idx)
            logical_begin = begin
            for index in range(count):
                logical_end = align_ceil(logical_begin + 1, self.page_size)
                kv_offset = int(kv_offsets[offset])
                offset += 1
                checkpoint_begin = direct.get(logical_end)
                inside_checkpoint = any(
                    start <= logical_begin and logical_end <= end
                    for end, start in direct.items()
                )
                if checkpoint_begin is not None:
                    assert restore is not None
                    state_slot = restore.slots.get(logical_end)
                    if state_slot is None:
                        state_slot = int(direct_slots[checkpoint_offset])
                        checkpoint_offset += 1
                        restore.slots[logical_end] = state_slot
                elif inside_checkpoint:
                    state_slot = None
                else:
                    state_slot = int(state_slots[log_offset])
                    log_offset += 1
                live_granules.append(
                    _LiveGranule(
                        begin=logical_begin,
                        end=logical_end,
                        kv_offset=kv_offset,
                        state_slot=state_slot,
                        checkpoint_begin=checkpoint_begin,
                    )
                )
                regions.append((
                    req.table_idx, logical_begin,
                    min(logical_end, req.max_device_len, self.page_table.shape[1]),
                    kv_offset, state_slot if checkpoint_begin is None and not inside_checkpoint else -1,
                ))
                logical_begin = logical_end
            mapped_end = min(
                logical_begin,
                req.max_device_len,
                self.page_table.shape[1],
            )
            assert live_granules[-1].begin < mapped_end <= live_granules[-1].end
        from logssms.kernel.metadata import write_cache_mapping

        write_cache_mapping(regions, self.page_table, state_table, self.page_size)
        assert offset == needed
        assert log_offset == needed_logs
        assert checkpoint_offset == needed_checkpoints
        if profile is not None:
            now = perf_counter_ns()
            profile["timing_ms"]["mapping_write_ms"] = (now - stage) / 1e6
            profile["timing_ms"]["mapping_ms"] = profile["timing_ms"]["mapping_write_ms"]
            profile["timing_ms"]["total_ms"] = (now - start) / 1e6
            self._profile_allocation_samples.append(profile)

    def _profile_resource_snapshot(self) -> dict:
        entries = self.tree.iter_entries()
        checkpoint_spans = Counter(
            entry.span_granules for entry in entries
            if entry.kind is LogStateKind.CHECKPOINT
        )
        log_slots = {
            entry.slot for entry in entries
            if entry.kind is LogStateKind.LOG
        }
        checkpoint_slots = {
            entry.slot for entry in entries
            if entry.kind is LogStateKind.CHECKPOINT
        }
        evictable_slots = self.tree.evictable_slots()
        return {
            "frame_usage": self._allocator.profile_usage(),
            "free_frames": self._allocator.free_frames,
            "pending_merge_reclamation_slots": len(self.logstate_pool.retiring_slots),
            "free_kv_pages": self._allocator.free_frames,
            "free_state_slots": len(self._allocator._free_slot_ids),
            "evictable_kv_tokens": self.tree.size_info.evictable_size,
            "cached_kv_tokens": self.tree.size_info.total_size,
            "evictable_kv_pages": (
                self.tree.size_info.evictable_size / self.page_size
            ),
            "evictable_state_slots": len(evictable_slots),
            "pending_recovery_state_slots": len(self.tree._recovery_slot_refs),
            "evictable_checkpoint_slots": len(checkpoint_slots & set(evictable_slots)),
            "tree_log_entries": sum(
                entry.kind is LogStateKind.LOG for entry in entries
            ),
            "tree_checkpoint_entries": sum(
                entry.kind is LogStateKind.CHECKPOINT for entry in entries
            ),
            "tree_log_slots": len(log_slots),
            "tree_checkpoint_slots": len(checkpoint_slots),
            "checkpoint_span_granules": dict(sorted(checkpoint_spans.items())),
            "checkpoint_covered_pages": sum(
                span * count for span, count in checkpoint_spans.items()
            ),
            "live_log_granules": sum(len(state.live_granules) for state in self._request_states.values()),
        }

    def _profile_request_paths(
        self,
        reqs: Sequence[Request],
    ) -> list[dict]:
        result = []
        for req in reqs:
            recovery = self.pending_recoveries.get(req.table_idx)
            state = self._request_states.get(req.table_idx)
            live = () if state is None else state.live_granules
            restore = self._recycled_restores.get(req.table_idx)
            handle = req.cache_handle
            cached_entries = ()
            merge_stats = None
            if isinstance(handle, LogStateCacheHandle):
                owner = handle.source_merge_stats
                if owner is not None:
                    merge_stats = {
                        "origin": owner.origin,
                        "tail": owner.tail,
                        "intensity": owner.intensity,
                        "queued_hits": len(owner.hit_positions),
                        "next_update_tail": owner.next_update_tail,
                    }
                cached_entries = tuple(
                    entry
                    for node in self.tree.path(handle.node)
                    for entry in node.entries
                    if entry.end <= handle.cached_len
                )
            result.append({
                "uid": req.uid,
                "protected_restore_positions": (
                    list(handle.protected_endpoints) if isinstance(handle, LogStateCacheHandle) else []
                ),
                "cached_len": req.cached_len,
                "device_len": req.device_len,
                "source_merge_stats": merge_stats,
                "recycle_restore_plan": None if restore is None else {
                    "gpu_hit": restore.plan.cached_len,
                    "cpu_hit": restore.plan.matched_len,
                    "checkpoints": [
                        {"begin": checkpoint.begin, "end": checkpoint.end,
                         "reuse_slot": checkpoint.gpu_entries[-1].slot if checkpoint.gpu_entries else None}
                        for checkpoint in restore.plan.checkpoints
                    ],
                },
                "recovery_log_slots": (
                    0 if recovery is None else len(recovery.logs)
                ),
                "recovery_has_checkpoint": bool(
                    recovery is not None
                    and recovery.base_checkpoint_slot is not None
                ),
                "cached_log_slots": len({
                    entry.slot for entry in cached_entries
                    if entry.kind is LogStateKind.LOG
                }),
                "cached_checkpoint_slots": len({
                    entry.slot for entry in cached_entries
                    if entry.kind is LogStateKind.CHECKPOINT
                }),
                "cached_checkpoint_span_granules": dict(sorted(Counter(
                    entry.span_granules for entry in cached_entries
                    if entry.kind is LogStateKind.CHECKPOINT
                ).items())),
                "live_log_granules": len(live),
            })
        return result

    def prepare_batch(self, batch: Batch) -> None:
        batch.cache_metadata = LogStateCacheMetadata()
        if not batch.is_prefill:
            self.last_plan = None
            if batch.is_decode:
                replay_counts = []
                for req in batch.reqs:
                    state = self._request_states[req.table_idx]
                    granule = state.live_granule_at(req.cached_len)
                    assert granule is not None, "Decode token has no LogState page"
                    offset = req.cached_len - granule.begin
                    count = req.cached_len - state.work_committed_pos
                    assert 0 <= count < LOG_STATE_REPLAY_WINDOW
                    assert count <= offset % LOG_STATE_REPLAY_WINDOW
                    replay_counts.append(count)
                    if (offset + 1) % LOG_STATE_REPLAY_WINDOW == 0:
                        state.work_committed_pos = req.cached_len + 1
                batch.cache_metadata.replay_counts = replay_counts + [
                    0 for _ in range(batch.padded_size - batch.size)
                ]
            return
        key = tuple((req, req.cached_len, req.device_len) for req in batch.reqs)
        pending = (
            self._prefill_protections if self._prefill_protection_key == key else None
        )
        self._prefill_protection_key = ()
        self._prefill_protections = None
        profile_start = perf_counter_ns() if self._profile_enabled else 0

        first_offsets = []
        for req in batch.reqs:
            granule = next(
                (granule for granule in self._request_states[req.table_idx].live_granules
                 if granule.begin <= req.cached_len < granule.end),
                None,
            )
            if granule is None:

                break
            first_offsets.append(req.cached_len - granule.begin)
        batch.cache_metadata.first_offsets = (
            first_offsets if len(first_offsets) == len(batch.reqs) else None
        )
        batch.cache_metadata.checkpoint_writes = []
        batch.cache_metadata.checkpoint_ranges = []
        row_offset = 0
        for req_index, req in enumerate(batch.reqs):
            restore = self._recycled_restores.get(req.table_idx)
            if restore is not None:
                batch.cache_metadata.checkpoint_ranges.extend(
                    (req_index, checkpoint.begin, checkpoint.end)
                    for checkpoint in restore.plan.checkpoints
                    if checkpoint.begin < req.device_len and req.cached_len < checkpoint.end
                )
            for granule in self._request_states[req.table_idx].live_granules:
                if (
                    granule.checkpoint_begin is not None
                    and req.cached_len < granule.end <= req.device_len
                ):
                    assert granule.state_slot is not None
                    batch.cache_metadata.checkpoint_writes.append((
                        row_offset + granule.end - req.cached_len - 1,
                        granule.state_slot,
                    ))
            row_offset += req.extend_len
        for req in batch.reqs:
            self._request_states[req.table_idx].work_committed_pos = req.device_len
        first_offsets_cpu_ms = (
            (perf_counter_ns() - profile_start) / 1e6
            if self._profile_enabled else 0.0
        )
        analysis_start = perf_counter_ns() if self._profile_enabled else 0
        analysis_step_timing_ms: dict[str, float] | None = (
            {} if self._profile_enabled else None
        )
        round_start = perf_counter_ns() if self._profile_enabled else 0
        traces = self._prefill_traces(batch.reqs)
        round_info = self.analyzer.begin_round(self.tree, traces)
        if analysis_step_timing_ms is not None:
            analysis_step_timing_ms["eligible_paths_ms"] = (
                perf_counter_ns() - round_start
            ) / 1e6
        recycle_start = perf_counter_ns() if self._profile_enabled else 0
        recycled = [
            self._recycled_restores[req.table_idx] for req in batch.reqs
            if req.table_idx in self._recycled_restores
            and not self._recycled_restores[req.table_idx].prepared
        ]
        blocked = frozenset(
            [entry.slot for restore in recycled
             for checkpoint in restore.plan.checkpoints for entry in checkpoint.gpu_entries]
            + [slot for restore in recycled for item in restore.plan.gpu_materialize for slot in item.slots]
            + [slot for restore in recycled for item in restore.plan.gpu_coalesce for slot in item.slots]
        )
        current_pins = Counter(
            slot for req in batch.reqs
            for slot in self._recovery_pins.get(
                id(self._pending_matches.get(req.table_idx)), (None, ())
            )[1]
        )
        blocked |= frozenset(
            slot for slot, refs in self.tree._recovery_slot_refs.items()
            if refs > current_pins[slot]
        )
        if analysis_step_timing_ms is not None:
            analysis_step_timing_ms["recycle_filter_ms"] = (
                perf_counter_ns() - recycle_start
            ) / 1e6
        if blocked:
            round_info = replace(round_info, eligible_slots=round_info.eligible_slots - blocked)
        plan = self.analyzer.analyze(
            self.tree,
            [req.input_ids for req in batch.reqs],
            round_info,
            collect_diagnostics=self._profile_enabled,
            step_timing_ms=analysis_step_timing_ms,
            traces=traces,
            pending=pending,
        )
        if recycled:
            plan = replace(
                plan,
                materialize=plan.materialize + tuple(
                    item for restore in recycled for item in restore.plan.gpu_materialize
                ),
                coalesce=plan.coalesce + tuple(
                    item for restore in recycled for item in restore.plan.gpu_coalesce
                ),
            )
        self._apply_protections(plan.protections)

        self.last_plan = plan
        task_start = perf_counter_ns() if self._profile_enabled else 0
        restore_works = self._restore_work_slots(batch)
        if analysis_step_timing_ms is not None:
            analysis_step_timing_ms["restore_tasks_ms"] = (
                perf_counter_ns() - task_start
            ) / 1e6
        analysis_and_tasks_cpu_ms = (
            (perf_counter_ns() - analysis_start) / 1e6
            if self._profile_enabled else 0.0
        )
        sample = None
        profile_before_cpu_ms = 0.0
        if self._profile_enabled:
            profile_before_start = perf_counter_ns()
            sample = {
                "batch_size": batch.size,
                "input_tokens": sum(req.extend_len for req in batch.reqs),
                "request_tags": [req.profile_tag for req in batch.reqs],
                "resources_before": self._profile_resource_snapshot(),
                "requests": self._profile_request_paths(batch.reqs),
                "active_requests": self._profile_active_requests,
                "queue": {
                    "pending_requests": self._profile_stats[
                        "current_pending_requests"
                    ],
                    "oldest_wait_ms": self._profile_stats[
                        "current_oldest_wait_ms"
                    ],
                },
                "plan": {
                    "protections": len(plan.protections),
                    "restores": len(restore_works),
                    "materialize": len(plan.materialize),
                    "coalesce": len(plan.coalesce),
                    "planned_merge_releases": sum(
                        len(item.slots) - 1 for item in plan.coalesce
                    ),
                    "merge_diagnostics": dict(plan.merge_diagnostics),
                },
                "analysis_step_timing_ms": analysis_step_timing_ms,
            }
            profile_before_cpu_ms = (
                perf_counter_ns() - profile_before_start
            ) / 1e6
        construction_start = perf_counter_ns() if self._profile_enabled else 0
        construction, merge_commit = self._construct_checkpoints(batch, restore_works, plan)
        handles = tuple(id(self._pending_matches[req.table_idx]) for req in batch.reqs
                        if req.table_idx in self._pending_matches
                        and id(self._pending_matches[req.table_idx]) in self._recovery_pins)
        if handles:
            for handle_id in handles:
                self._release_recovery(handle_id)
        self._detach_recycled_sources(batch)
        construction_submit_cpu_ms = (
            (perf_counter_ns() - construction_start) / 1e6
            if self._profile_enabled else 0.0
        )

        commit_start = perf_counter_ns() if self._profile_enabled else 0
        for req in batch.reqs:
            self.pending_recoveries.pop(req.table_idx, None)
            self._pending_matches.pop(req.table_idx, None)
        released, released_frames, commit_skips = merge_commit
        commit_cpu_ms = (
            (perf_counter_ns() - commit_start) / 1e6
            if self._profile_enabled else 0.0
        )
        if self._profile_enabled:
            self._profile_stats["released_state_slots_from_merge"] += released
            self._profile_stats["scheduled_payload_conv_frames_from_merge"] += (
                released_frames
            )
            for reason, count in commit_skips.items():
                self._profile_stats["merge_commit_skips"][reason] = (
                    self._profile_stats["merge_commit_skips"].get(reason, 0)
                    + count
                )
            for reason, count in plan.merge_diagnostics:
                self._profile_stats["merge_plan_diagnostics"][reason] = (
                    self._profile_stats["merge_plan_diagnostics"].get(reason, 0)
                    + count
                )
            assert sample is not None
            profile_after_start = perf_counter_ns()
            sample["construction"] = construction
            sample["materialized_slots"] = construction.get("materialized_slots", len(plan.materialize))
            sample["released_state_slots"] = released
            sample["scheduled_payload_conv_frames"] = released_frames
            outcome = self._merge_batch_outcome(plan, construction["merged_checkpoint_groups"],
                                               construction["materialized_slots"])
            sample["merge_outcome"] = outcome
            outcomes = self._profile_stats["merge_batch_outcomes"]
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            sample["merge_commit_skips"] = commit_skips
            sample["resources_after"] = self._profile_resource_snapshot()
            profile_after_cpu_ms = (
                perf_counter_ns() - profile_after_start
            ) / 1e6
            timing_ms = {
                "first_offsets_cpu_ms": first_offsets_cpu_ms,
                "analysis_and_tasks_cpu_ms": analysis_and_tasks_cpu_ms,
                "profile_before_cpu_ms": profile_before_cpu_ms,
                "construction_submit_cpu_ms": construction_submit_cpu_ms,
                "construction_event_setup_cpu_ms": construction.get(
                    "event_setup_cpu_ms", 0.0
                ),
                "commit_cpu_ms": commit_cpu_ms,
                "profile_after_cpu_ms": profile_after_cpu_ms,
            }
            timing_ms["manager_total_cpu_ms"] = (
                perf_counter_ns() - profile_start
            ) / 1e6
            accounted_ms = sum(timing_ms.values()) - timing_ms[
                "manager_total_cpu_ms"
            ]
            timing_ms["unaccounted_cpu_ms"] = (
                timing_ms["manager_total_cpu_ms"] - accounted_ms
            )
            sample["timing_ms"] = timing_ms
            self._profile_samples.append(sample)

    def _detach_recycled_sources(self, batch: Batch) -> None:
        retired = []
        for req in batch.reqs:
            restore = self._recycled_restores.get(req.table_idx)
            if restore is None:
                continue
            for checkpoint in restore.plan.checkpoints:
                for entry in checkpoint.gpu_entries:
                    if entry.slot in restore.detached:
                        continue
                    assert self.tree.has_entry(entry)
                    assert not self.tree.entry_is_protected(entry.slot), (
                        "Cannot detach a protected LogState branch anchor"
                    )
                    self.tree.detach_entry(entry.slot)
                    restore.detached.add(entry.slot)
                    if entry.slot != restore.slots.get(checkpoint.end):
                        retired.append(entry.slot)
            restore.prepared = True
        if retired:

            self.logstate_pool.retire_slots(retired)

    def _prefill_traces(self, reqs: Sequence[Request]) -> tuple[LogRadixTrace, ...]:
        traces = []
        for req in reqs:
            handle = self._pending_matches.get(req.table_idx)
            usable_len = align_down(len(req.input_ids), self.granularity)
            if (
                handle is not None and handle.match_trace is not None
                and handle.trace_revision == self.tree.revision
                and usable_len >= handle.trace_input_len
                and (
                    handle.match_trace.matched_len < handle.trace_input_len
                    or usable_len == handle.trace_input_len
                    or (handle.match_trace.is_exact and not handle.match_trace.node.children)
                )
            ):
                traces.append(handle.match_trace)
            else:
                traces.append(self.tree.trace(req.input_ids[:usable_len]))
        return tuple(traces)

    def cache_req(self, req: Request, *, finished: bool) -> None:
        profile_start = perf_counter_ns() if self._profile_enabled else 0
        old_handle = req.cache_handle
        assert isinstance(old_handle, LogStateCacheHandle)
        request_state = self._request_states[req.table_idx]
        granules = request_state.live_granules
        restore = self._recycled_restores.get(req.table_idx)
        protected_endpoints = set(old_handle.protected_endpoints)
        protected_endpoints.update(self._protected_prefixes.get(req.uid, ()))
        if restore is not None:
            protected_endpoints.update(restore.plan.protected_endpoints)
        full_pages = [granule for granule in granules if granule.end <= req.cached_len]
        committed_ends = {granule: granule.end for granule in full_pages}
        if finished:
            for granule in granules:
                if granule.end <= req.cached_len or granule.checkpoint_begin is not None:
                    continue
                position = max((value for value in protected_endpoints
                                if granule.begin < value <= req.cached_len < granule.end), default=0)
                if position:
                    full_pages.append(granule)
                    committed_ends[granule] = position
        insert_len = old_handle.cached_len
        for granule in full_pages:
            if granule.begin != insert_len:
                break
            insert_len = committed_ends[granule]
        if restore is not None:
            for checkpoint in restore.plan.checkpoints:
                if checkpoint.begin < insert_len < checkpoint.end and (
                    insert_len > old_handle.cached_len
                    or any(entry.slot in restore.detached for entry in checkpoint.gpu_entries)
                ):

                    insert_len = checkpoint.begin
                    break
        assert insert_len % self.granularity == 0

        stage_start = perf_counter_ns() if self._profile_enabled else 0
        insert = self.tree.insert(
            req.input_ids[: req.cached_len],
            self.page_table[req.table_idx, : req.cached_len],
            cache_len=insert_len,
            kv_frame_spans=((granule.begin, committed_ends[granule], granule.kv_offset // self.page_size)
                           for granule in full_pages),
        )
        tree_insert_ms = (
            (perf_counter_ns() - stage_start) / 1e6
            if self._profile_enabled else 0.0
        )
        adopted = [
            granule
            for granule in full_pages
            if old_handle.cached_len <= granule.begin
            and committed_ends[granule] <= insert.inserted_len
        ]
        adopted_set = set(adopted)
        stage_start = perf_counter_ns() if self._profile_enabled else 0
        existing = self.tree.recovery_at(insert.node, insert.inserted_len)
        tree_match_ms = (
            (perf_counter_ns() - stage_start) / 1e6
            if self._profile_enabled else 0.0
        )
        covered_len = existing.position
        released_kv_offsets = []
        released_state_slots = []
        retain_kv_ms = 0.0
        for granule in adopted:
            page_end = committed_ends[granule]
            logical_begin = max(granule.begin, covered_len)
            if granule.checkpoint_begin is not None and logical_begin < page_end:
                assert granule.state_slot is not None
                checkpoint_begin = granule.checkpoint_begin
                kind = self._allocator.slot(granule.state_slot).kind
                if kind == "work":
                    self.logstate_pool.commit_checkpoint(granule.state_slot)
                elif kind == "log":
                    self.logstate_pool.materialize(granule.state_slot)
                self.tree.attach_entry(
                    insert.node,
                    LogStateEntry(
                        slot=granule.state_slot,
                        kind=LogStateKind.CHECKPOINT,
                        begin=checkpoint_begin,
                        end=granule.end,
                        span_granules=(granule.end - checkpoint_begin) // self.page_size,
                    ),
                )
                covered_len = granule.end
                if restore is not None:
                    restore.committed.add(granule.end)
                    if self._profile_enabled:
                        self._profile_stats["recycle_committed_checkpoints"] += 1
                        self._profile_stats["recycle_reused_gpu_slots"] += int(kind != "work")
                        self._profile_stats["recycle_committed_tokens"] += (
                            granule.end - max(checkpoint_begin, restore.plan.cached_len)
                        )
            elif granule.state_slot is not None and logical_begin < page_end:
                self.tree.attach_entry(
                    insert.node,
                    LogStateEntry(
                        slot=granule.state_slot,
                        kind=LogStateKind.LOG,
                        begin=logical_begin,
                        end=page_end,
                    ),
                )
                covered_len = page_end
            elif granule.state_slot is not None:
                released_state_slots.append(granule.state_slot)
                if restore is not None and granule.checkpoint_begin is not None:
                    restore.slots.pop(granule.end, None)

            kv_begin = max(granule.begin, insert.previous_len)
            if kv_begin < page_end:
                stage_start = perf_counter_ns() if self._profile_enabled else 0
                self.evictor.cached_page_tokens[granule.kv_offset] = (
                    self.evictor.cached_page_tokens.get(granule.kv_offset, 0)
                    + page_end - kv_begin
                )
                if self._profile_enabled:
                    retain_kv_ms += (perf_counter_ns() - stage_start) / 1e6
            else:
                released_kv_offsets.append(granule.kv_offset)

        duplicates = [
            granule
            for granule in full_pages
            if granule not in adopted_set and committed_ends[granule] <= insert.previous_len
        ]
        self._release_live_granules(req.table_idx, duplicates)
        if released_kv_offsets or released_state_slots:
            self._return_slots(
                released_kv_offsets,
                released_state_slots,
            )

        if protected_endpoints:
            for position in sorted(protected_endpoints):
                if 0 < position <= insert.inserted_len:
                    trace = self.tree.trace(req.input_ids[:position])
                    node = trace.node
                    self.tree.protect_at(node, position)

        merge_stats_start = perf_counter_ns() if self._profile_enabled else 0
        stats = old_handle.merge_stats
        if stats is not None:
            self.analyzer.advance_merge_stats(
                stats, insert.inserted_len, owner_uuid=insert.node.uuid,
            )
            if insert.previous_len < insert.inserted_len:
                if old_handle.node.merge_stats is stats:
                    old_handle.node.merge_stats = None
                assert insert.node.merge_stats in (None, stats)
                insert.node.merge_stats = stats
        merge_stats_ms = (
            (perf_counter_ns() - merge_stats_start) / 1e6
            if self._profile_enabled else 0.0
        )
        new_handle = replace(
            self.tree.make_handle(insert.node, insert.inserted_len),
            merge_stats=stats,
            protected_endpoints=tuple(sorted(protected_endpoints)),
        )
        self.unlock(old_handle)
        self.pending_recoveries.pop(req.table_idx, None)
        self._pending_matches.pop(req.table_idx, None)

        state_table = self.state_table
        assert state_table is not None
        stage_start = perf_counter_ns() if self._profile_enabled else 0
        state_table[req.table_idx, : insert.inserted_len].fill_(-1)
        self.page_table[
            req.table_idx, : insert.inserted_len
        ].copy_(new_handle.kv_indices)
        mapping_update_ms = (
            (perf_counter_ns() - stage_start) / 1e6
            if self._profile_enabled else 0.0
        )

        removed_pages = adopted_set.union(duplicates)
        remaining = [granule for granule in granules if granule not in removed_pages]
        conv_copy_ms = 0.0
        work_slot_lookup_ms = 0.0
        if (
            not finished
            and adopted
            and req.cached_len == adopted[-1].end
        ):
            stage_start = perf_counter_ns() if self._profile_enabled else 0
            work_slot = request_state.work_slot
            if self._profile_enabled:
                work_slot_lookup_ms = (perf_counter_ns() - stage_start) / 1e6
            recovery = new_handle.recovery
            assert recovery.conv_slot is not None
            assert recovery.conv_granule is not None
            stage_start = perf_counter_ns() if self._profile_enabled else 0
            self.logstate_pool.copy_conv_snapshot(
                recovery.conv_slot, recovery.conv_granule, work_slot,
            )
            if self._profile_enabled:
                conv_copy_ms = (perf_counter_ns() - stage_start) / 1e6
        if finished:
            if any(value[0] is req for value in self._prefill_protection_key):
                self._prefill_protection_key = ()
                self._prefill_protections = None
            self._protected_prefixes.pop(req.uid, None)
            recycled = request_state.recycle_match
            request_state.recycle_match = None
            if (
                recycled is not None
                and insert.inserted_len >= recycled.matched_len
                and self.recycle_bin is not None
            ):

                self.recycle_bin.discard(recycled.branch)
            self._release_live_granules(req.table_idx, remaining)
            restore = self._recycled_restores.pop(req.table_idx, None)
            if restore is not None:
                page_slots = {granule.state_slot for granule in remaining}
                uncommitted = [
                    slot for end, slot in restore.slots.items()
                    if end not in restore.committed and slot not in page_slots
                    and slot in restore.detached
                ]
                if uncommitted:
                    self._return_slots(
                        [], uncommitted,
                    )
            state_table[req.table_idx].fill_(-1)
            work_slot = self._request_states.pop(req.table_idx).work_slot
            self._return_slots(
                [], [work_slot],
            )
            work_state_table = self.work_state_table
            assert work_state_table is not None
            work_state_table[req.table_idx, 0] = -1
            self.record_cached_session(req, insert.inserted_len)
            if self._profile_enabled:
                node = self.tree.trace(req.input_ids[:insert.inserted_len]).node
                self._record_complete_profile(
                    req, insert.inserted_len,
                    [value.uuid for value in self.tree.path(node)],
                )
        else:
            req.cache_handle = new_handle
            self.lock(new_handle)
            request_state.live_granules = remaining
            request_state.live_granule_cursor = 0
        if self._profile_enabled:
            self._cache_req_samples.append({
                "tag": req.profile_tag,
                "finished": finished,
                "adopted_pages": len(adopted),
                "tree_insert_ms": tree_insert_ms,
                "tree_match_ms": tree_match_ms,
                "retain_kv_ms": retain_kv_ms,
                "merge_stats_ms": merge_stats_ms,
                "mapping_update_ms": mapping_update_ms,
                "work_slot_lookup_ms": work_slot_lookup_ms,
                "conv_copy_ms": conv_copy_ms,
                "total_ms": (perf_counter_ns() - profile_start) / 1e6,
            })

    def free_to_cached(self, req: Request) -> None:
        state = self._request_states[req.table_idx]
        granules = state.live_granules
        keep = [granule for granule in granules if granule.begin < req.cached_len]
        release = [granule for granule in granules if granule.begin >= req.cached_len]
        self._release_live_granules(req.table_idx, release)
        state.live_granules = keep
        state.live_granule_cursor = 0
        req.device_len = min(req.cached_len + 1, req.max_device_len)

    def commit_speculative(self, req: Request, C: int, num_sampled: int) -> None:
        raise NotImplementedError(
            "LogState speculative commit requires the LogState verify backend."
        )

    def check_integrity(self) -> None:
        self.tree.check_integrity()
        live_granules = sum(len(state.live_granules) for state in self._request_states.values())
        if live_granules != 0:
            raise RuntimeError(
                f"LogState cache has {live_granules} request-owned pages while idle."
            )

        if sum(self.evictor.cached_page_tokens.values()) != self.tree.size_info.total_size:
            raise RuntimeError("LogState cached KV token accounting mismatch.")

        tree_slots = {int(entry.slot) for entry in self.tree.iter_entries()}
        work_state_table = self.work_state_table
        assert work_state_table is not None
        work_slots = {state.work_slot for state in self._request_states.values()} | {self._dummy_work_slot}
        if len(work_slots) != len(self._request_states) + 1:
            raise RuntimeError("LogState work-state table contains duplicate slots.")
        for row, state in self._request_states.items():
            if int(work_state_table[row, 0]) != state.work_slot:
                raise RuntimeError("LogState work-state table disagrees with allocations.")
        kv_pages = {granule // self.page_size for granule in self.evictor.cached_page_tokens}
        kv_pages.update(
            granule.kv_offset // self.page_size
            for state in self._request_states.values() for granule in state.live_granules
        )
        expected_kv = kv_pages | {self._dummy_kv_frame}
        expected_state = tree_slots | {self._dummy_log_slot} | {
            granule.state_slot for state in self._request_states.values() for granule in state.live_granules
            if granule.state_slot is not None
        } | work_slots
        self.logstate_pool.poll_reclamation()

        expected_state.update(self.logstate_pool.retiring_slots)
        self._allocator.check_integrity()
        if self._allocator._reclaimable_slots != (
            set(self.tree.evictable_slots()) | self.logstate_pool.retiring_slots
        ):
            raise RuntimeError("LogState reclaimable slots disagree with tree eligibility and retirement.")
        if (self._allocator.allocated_kv != expected_kv
            or self._allocator.allocated_slots != expected_state):
            raise RuntimeError(
                "LogState allocator ownership accounting mismatch: "
                f"KV extra={self._allocator.allocated_kv - expected_kv}, "
                f"KV missing={expected_kv - self._allocator.allocated_kv}, "
                f"State extra={self._allocator.allocated_slots - expected_state}, "
                f"State missing={expected_state - self._allocator.allocated_slots}."
            )

    @contextmanager
    def lazy_free_region(self):
        assert self._lazy_free_pages is None and self._lazy_free_states is None
        self._lazy_free_pages = []
        self._lazy_free_states = []
        try:
            yield
        finally:
            lazy_pages = self._lazy_free_pages
            lazy_states = self._lazy_free_states
            self._lazy_free_pages = None
            self._lazy_free_states = None
            if lazy_pages or lazy_states:
                self._return_slots(lazy_pages, lazy_states)

    def _allocate(
        self,
        needed_pages: int,
        needed_states: int,
        needed_works: int = 0,
        profile: dict | None = None,
    ) -> tuple[list[int], list[int], list[int]]:
        return self.evictor.allocate(
            needed_pages, needed_states, needed_works, profile,
            recycle_bin=self.recycle_bin,
        )

    def _allocate_work(self) -> int:
        _pages, _logs, works = self._allocate(0, 0, 1)
        return int(works[0])

    def _restore_work_slots(
        self,
        batch: Batch,
    ) -> tuple[int, ...]:
        works = []
        for req in batch.reqs:
            recovery = self.pending_recoveries.get(req.table_idx)
            if recovery is None or recovery.position == 0:
                continue
            assert recovery.conv_slot is not None
            assert recovery.conv_granule is not None
            assert not recovery.logs or recovery.logs[-1].end == recovery.position
            works.append(self._request_states[req.table_idx].work_slot)
        return tuple(dict.fromkeys(works))

    def _construct_checkpoints(
        self, batch: Batch, works: Sequence[int], plan: LogStatePlan,
    ) -> tuple[dict, tuple[int, int, dict[str, int]]]:
        pool = self.logstate_pool
        released = released_frames = merged = segments = materialized = materialized_logs = 0
        skips: Counter[str] = Counter()
        events = None
        if self._profile_enabled and (works or plan.coalesce or plan.materialize):
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()

        for req in batch.reqs:
            self._release_recovery(id(self._pending_matches.get(req.table_idx)))
        merge_groups = {group.slots for group in plan.coalesce}
        consumed = {slot for group in plan.coalesce for slot in group.slots}
        materialize = [CheckpointCoalesce(
            item.slots, (1,) * len(item.slots), item.begin, item.end,
            (item.end - item.begin) // self.granularity,
        ) for item in plan.materialize if not any(slot in consumed for slot in item.slots)]
        groups = iter(sorted((*plan.coalesce, *materialize), key=lambda item: (item.end, item.begin)))
        wave_count = 0
        while True:
            wave, chains, frame_counts, log_counts = [], [], [], []
            for group in groups:
                if any(not self.tree.has_slot(slot)
                       or slot in self.tree._recovery_slot_refs for slot in group.slots):
                    skips["pending_or_missing_source"] += 1
                    continue
                right = self.tree.entry_for_slot(group.right_slot)
                if right.kind is LogStateKind.CHECKPOINT:
                    count, frames, reasons = self._commit_coalesce((group,), allow_logs=True)
                    released += count
                    released_frames += frames
                    merged += bool(count)
                    skips.update(reasons)
                    continue
                if not works:
                    skips["no_restore_work"] += 1
                    continue
                if self._allocator.checkpoint_frame_capacity(group.slots) < self._allocator.payload_frames:
                    skips["insufficient_group_frames"] += 1
                    continue
                if self._allocator.checkpoint_frame_savings(group.slots) < 0:
                    skips["no_memory_gain"] += 1
                    continue
                recovery = self.tree.recovery_at(self.tree.entry_owner(right), right.end)
                assert recovery.position == right.end
                logs = [(entry.slot, 0, entry.end - entry.begin)
                        for entry in recovery.logs]
                wave.append(group)
                chains.append((recovery.base_checkpoint_slot, logs, works[len(wave) - 1]))
                log_counts.append(sum(self.tree.entry_for_slot(slot).kind is LogStateKind.LOG
                                      for slot in group.slots))
                frame_counts.append(sum(len(self._allocator.slot(slot).payload)
                                        + sum(map(len, self._allocator.slot(slot).conv)) for slot in group.slots))
                segments += len(logs)
                if len(wave) == len(works):
                    break
            if not wave:
                break

            pool.kernels.restore(pool, chains)
            committed, committed_works = [], []
            count, _frames, reasons = self._commit_coalesce(tuple(wave), allow_logs=True,
                                                          retire_sources=False)
            released += count
            skips.update(reasons)
            for index, group in enumerate(wave):
                right = self.tree.entry_for_slot(group.right_slot)
                if (right.kind is LogStateKind.CHECKPOINT
                    and right.begin == group.begin and right.end == group.end):
                    committed.append(group.slots)
                    committed_works.append(works[index])
                    materialized_logs += log_counts[index]
                    released_frames += frame_counts[index] - self._allocator.payload_frames - self._allocator.conv_frames
            if committed:
                pool.coalesce_groups(committed, committed_works)
                pool.retire_slots([slot for group in committed for slot in group[:-1]])
            merged += sum(group in merge_groups for group in committed)
            materialized += len(committed)
            wave_count += 1

        refreshed = []
        for req in batch.reqs:
            original = self.pending_recoveries.get(req.table_idx)
            handle = self._pending_matches.get(req.table_idx)
            if original is None or original.position == 0 or handle is None:
                continue
            recovery = self.tree.recovery_at(handle.node, original.position)
            assert recovery.position == original.position
            refreshed.append(RestoreTask(
                self._request_states[req.table_idx].work_slot, recovery.base_checkpoint_slot, recovery.logs,
                recovery.conv_slot, recovery.conv_granule, 0,
            ))
        construction = self._construction.restore(
            refreshed, profile_enabled=self._profile_enabled,
            pending_cuda_events=self._profile_cuda_events, record_cuda=False,
        )
        construction["chains"] += materialized
        construction["segments"] += segments
        construction["outputs"] += materialized
        construction["waves"] += wave_count
        construction["merged_checkpoint_groups"] = merged
        construction["materialized_slots"] = materialized_logs
        if events is not None:
            events[1].record()
            construction["cuda_ms"] = None
            self._profile_cuda_events.append((construction, *events))
        return construction, (released, released_frames, dict(skips))

    @staticmethod
    def _merge_batch_outcome(plan: LogStatePlan, merged: int, materialized: int = 0) -> str:
        if merged:
            return "merged"
        if materialized:
            return "materialized"
        if plan.coalesce:
            return "commit_skipped"
        diagnostics = dict(plan.merge_diagnostics)
        if not diagnostics.get("merge_anchors", 0):
            return "no_match_anchor"
        if not diagnostics.get("checkpoint_candidates", 0):
            return "no_checkpoint_candidate"
        if not diagnostics.get("candidate_groups", 0):
            return "no_adjacent_candidate"
        return "candidates_rejected"

    def _commit_coalesce(
        self,
        plans: Sequence[CheckpointCoalesce],
        *, allow_logs: bool = False, retire_sources: bool = True,
    ) -> tuple[int, int, dict[str, int]]:
        released = []
        released_frames = 0
        skips: dict[str, int] = {}
        for plan in plans:
            if any(slot in self.tree._recovery_slot_refs for slot in plan.slots):
                skips["pending_work_recovery"] = skips.get("pending_work_recovery", 0) + 1
                continue
            if any(
                not self.tree.has_slot(slot)

                for slot in plan.slots
            ):
                skips["missing_slot"] = (
                    skips.get("missing_slot", 0) + 1
                )
                continue
            entries = [
                self.tree.entry_for_slot(slot)
                for slot in plan.slots
            ]
            if any(
                (entry.kind is not LogStateKind.CHECKPOINT and not (
                    allow_logs and entry.kind is LogStateKind.LOG and level == 1
                ))
                or entry.slot != slot
                or (entry.span_granules if entry.kind is LogStateKind.CHECKPOINT else 1) != level
                for entry, slot, level in zip(
                    entries,
                    plan.slots,
                    plan.source_spans,
                    strict=True,
                )
            ):
                skips["checkpoint_metadata_mismatch"] = (
                    skips.get("checkpoint_metadata_mismatch", 0) + 1
                )
                continue
            if entries[0].begin != plan.begin or entries[-1].end != plan.end:
                skips["range_mismatch"] = skips.get("range_mismatch", 0) + 1
                continue
            if any(
                left.end != right.begin
                for left, right in zip(entries, entries[1:])
            ):
                skips["noncontiguous_at_commit"] = (
                    skips.get("noncontiguous_at_commit", 0) + 1
                )
                continue
            owner = self.tree.entry_owner(entries[-1])
            checkpoint_timestamp = (self.tree.checkpoint_timestamp(plan.right_slot)
                                    if entries[-1].kind is LogStateKind.CHECKPOINT else None)
            if self._profile_enabled:
                for slot in plan.slots[:-1]:
                    record = self._allocator.slot(slot)

                    released_frames += len(record.payload) + (
                        sum(map(len, record.conv)) if allow_logs else len(record.conv[-1])
                    )
            for slot in plan.slots:
                self.tree.detach_entry(slot)
            self.tree.attach_entry(
                owner,
                LogStateEntry(
                    slot=plan.right_slot,
                    kind=LogStateKind.CHECKPOINT,
                    begin=plan.begin,
                    end=plan.end,
                    span_granules=plan.next_span,
                ),
                checkpoint_timestamp=checkpoint_timestamp,
            )
            released.extend(plan.slots[:-1])
        if released and retire_sources:

            self.logstate_pool.retire_slots(released)
        return len(released), released_frames, skips

    def _release_live_granules(
        self,
        table_idx: int,
        granules    : Sequence[_LiveGranule],
    ) -> None:
        if not granules:
            return
        state_table = self.state_table
        assert state_table is not None
        restore = self._recycled_restores.get(table_idx)
        released_states = []
        for granule in granules:
            state_table[table_idx, granule.begin:granule.end].fill_(-1)
            if granule.state_slot is None:
                continue
            if restore is not None and granule.checkpoint_begin is not None:
                checkpoint = next(value for value in restore.plan.checkpoints if value.end == granule.end)
                if (
                    granule.state_slot not in restore.detached
                    and any(entry.slot == granule.state_slot for entry in checkpoint.gpu_entries)
                ):
                    continue
                restore.slots.pop(granule.end, None)
            released_states.append(granule.state_slot)
        self._return_slots(
            [granule.kv_offset for granule in granules],
            released_states,
        )

    def _return_slots(
        self,
        granules : Sequence[int],
        states: Sequence[int],
    ) -> None:
        if self._lazy_free_pages is None:
            page_frames = [granule // self.page_size for granule in granules]
            self.logstate_pool.release_kv(page_frames)
            self.logstate_pool.release_slots(states)
        else:
            assert self._lazy_free_states is not None
            # Snapshot the IDs, but leave reclamation and event waits to the
            # pool after the region exits, including exceptional exits.
            self._lazy_free_pages.extend(granules)
            self._lazy_free_states.extend(states)


class LogStateCacheFactory(BaseCacheFactory):

    def plan(
        self,
        config            : EngineConfig,
        available_memory  : int,
        speculator_reserve: SpeculatorReserve | None,
    ) -> LogStateCachePlan:
        from logssms.cache.config import LogStateCacheConfig
        from logssms.distributed import try_get_tp_info
        from logssms.utils import div_even

        assert isinstance(config.cache_config, LogStateCacheConfig)
        assert config.model_config.is_hybrid, (
            "LogState cache is only defined for hybrid models."
        )
        assert speculator_reserve is None, (
            "LogState speculative verify is not implemented yet."
        )
        tp_info = try_get_tp_info()
        assert tp_info is not None

        cache_per_page = (
            2
            * config.model_config.head_dim
            * div_even(
                config.model_config.num_kv_heads,
                tp_info.size,
                allow_replicate=True,
            )
            * config.page_size
            * config.dtype.itemsize
            * config.model_config.num_attention_layers
        )
        from logssms.models.config import AffineLogCacheSpec
        from logssms.cache.factory import make_linear_cache_spec

        spec = make_linear_cache_spec(config.model_config)
        if not isinstance(spec, AffineLogCacheSpec):
            raise NotImplementedError(f"LogState storage for {spec.recurrent_type} is not implemented.")
        num_v_heads, head_k_dim, head_v_dim = spec.state_shape
        conv_dim, state_len = spec.conv_shape
        layout = LogStateSlotLayout(
            page_size=config.page_size,
            granularity=config.cache_config.granularity,
            num_linear_layers=spec.num_layers,
            num_key_heads=spec.num_groups,
            num_value_heads=num_v_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            conv_dim=conv_dim,
            state_len=state_len,
            dtype=config.dtype,
        )
        num_work_states = config.max_running_req + 1
        head_bytes = layout.head_payload_bytes(cache_per_page)
        payload_frames = div_ceil(
            spec.num_layers * num_v_heads * head_bytes,
            cache_per_page,
        )
        conv_frames = div_ceil(
            layout.conv_state_bytes // layout.num_granules, cache_per_page
        )
        assert layout.log_decay_bytes <= cache_per_page
        log_payload_frames = div_ceil(layout.log_payload_bytes, cache_per_page)
        state_units = log_payload_frames + layout.num_granules * conv_frames + 1
        work_units = payload_frames + conv_frames

        num_pages = available_memory // cache_per_page - 1
        if config.num_page_override is not None:
            num_pages = min(num_pages, config.num_page_override)
        num_states = max(0, (
            num_pages - num_work_states * work_units - state_units
        ) // state_units)
        assert num_pages > 1 and num_states > 1 and layout.log_decay_bytes <= cache_per_page, (
            "Not enough memory for LogState cache allocation."
        )

        num_tokens = num_pages * config.page_size
        max_seq_len = min(
            config.max_seq_len,
            num_tokens,
            max(0, num_pages - num_work_states * work_units - state_units)
            // (1 + state_units) * config.page_size,
        )
        return LogStateCachePlan(
            linear_spec=spec,
            num_pages=num_pages,
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            aligned_max_seq_len=align_ceil(max_seq_len, 32),
            num_states=num_states,
            num_work_states=num_work_states,
            state_units=state_units,
        )

    def allocate(
        self,
        config: EngineConfig,
        plan  : CachePlan,
        device: torch.device,
        dtype : torch.dtype,
    ) -> CacheResources:
        from logssms.cache.config import LogStateCacheConfig

        assert isinstance(plan, LogStateCachePlan)
        assert isinstance(config.cache_config, LogStateCacheConfig)
        from logssms.distributed import try_get_tp_info
        from logssms.utils import div_even

        model = config.model_config
        spec = plan.linear_spec
        num_v_heads, head_k_dim, head_v_dim = spec.state_shape
        conv_dim, state_len = spec.conv_shape
        tp_info = try_get_tp_info()
        assert tp_info is not None
        shared_pool = MHALogStatePool(
            num_frames=plan.num_pages + 1,
            page_size=config.page_size,
            num_attention_layers=model.num_attention_layers,
            num_kv_heads=div_even(model.num_kv_heads, tp_info.size, allow_replicate=True),
            head_dim=model.head_dim,
            num_linear_layers=spec.num_layers,
            num_key_heads=spec.num_groups,
            num_value_heads=num_v_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            conv_dim=conv_dim,
            state_len=state_len,
            device=device,
            dtype=dtype,
            granularity=config.cache_config.granularity,
            recurrent_type=spec.recurrent_type,
        )
        assert (
            shared_pool.allocator.log_payload_frames
            + shared_pool.layout.num_granules * shared_pool.allocator.conv_frames
            + 1
        ) == plan.state_units
        page_table = torch.zeros(
            (config.max_running_req + 1, plan.aligned_max_seq_len),
            dtype=torch.int32,
            device=device,
        )
        page_table[-1].fill_(plan.num_tokens)

        linear_state = shared_pool
        state_table = torch.full(
            (config.max_running_req + 1, plan.aligned_max_seq_len),
            -1,
            dtype=torch.int32,
            device=device,
        )
        work_state_table = torch.full(
            (config.max_running_req + 1, 1), -1,
            dtype=torch.int32, device=device,
        )

        logger.info(
            f"Allocating {plan.num_pages + 1} shared cache frames, "
            f"up to {plan.num_work_states} work states allocated on demand = "
            f"{shared_pool.storage.numel() / (1 << 30):.2f} GiB"
        )
        return CacheResources(
            kv_pool=shared_pool,
            page_table=page_table,
            linear_state=linear_state,
            state_table=state_table,
            work_state_table=work_state_table,
        )

    def make_manager(
        self,
        config   : EngineConfig,
        plan     : CachePlan,
        resources: CacheResources,
    ) -> LogStateCacheManager:
        from logssms.cache.config import LogStateCacheConfig

        assert isinstance(plan, LogStateCachePlan)
        assert isinstance(config.cache_config, LogStateCacheConfig)
        return LogStateCacheManager(
            num_pages=plan.num_pages,
            num_states=plan.num_states,
            num_work_states=plan.num_work_states,
            page_size=config.page_size,
            resources=resources,
            granularity=config.cache_config.granularity,
            enable_memorized_evict=(
                config.cache_config.enable_memorized_evict
            ),
            enable_merge_adjustment=(
                config.cache_config.enable_merge_adjustment
            ),
            min_sampling_len=config.cache_config.min_sampling_len,
            max_merge_intensity=config.cache_config.max_merge_intensity,
            default_merge_growth_rate=(
                config.cache_config.default_merge_growth_rate
            ),
        )

SUPPORTED_CACHE_FACTORIES.register("logstate")(LogStateCacheFactory())

__all__ = [
    "LogStateCacheManager",
    "LogStateCacheFactory",
    "LogStateCachePlan",
]
