from __future__ import annotations

import math
import os
from collections import Counter, OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from time import perf_counter_ns
from typing import TYPE_CHECKING, Sequence
from weakref import WeakValueDictionary

import torch

from logssms.core import Request
from logssms.utils import align_ceil, align_down, div_ceil, init_logger

from .base import BaseCacheManager, CacheResources
from .factory import BaseCacheFactory, CachePlan, SUPPORTED_CACHE_FACTORIES
from .logstate_construction import LogStateConstruction, RestoreTask
from .logstate_pool.layout import (
    LOG_STATE_GRANULARITY,
    LOG_STATE_REPLAY_WINDOW,
    LogStateSlotLayout,
)
from .logstate_pool.mha_pool import MHALogStatePool
from .prefix_cache.base import BaseCacheHandle, MatchResult
from .prefix_cache.logstate import (
    LogStateCacheHandle,
    LogStateRecovery,
    LogRadixTrace,
    LogRadixTree,
    LogRadixTreeNode,
    LogStateEntry,
    LogStateKind,
    LogStateMergeStats,
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
class RecycledLog:
    begin: int
    end  : int


@dataclass(frozen=True)
class RecycledBranch:
    tokens            : tuple[int, ...]
    logs              : tuple[RecycledLog, ...]
    protected_endpoints: frozenset[int]
    _serial           : int = field(default=0, compare=False, repr=False)
    merge_stats       : tuple[int, ...] = ()

    @property
    def bytes(self) -> int:


        return (
            256 + 8 * len(self.tokens) + 16 * len(self.logs)
            + 8 * len(self.protected_endpoints)
            + 8 * len(self.merge_stats)
        )


@dataclass(frozen=True)
class RecycledMatch:
    branch     : RecycledBranch
    matched_len: int

    def matched_tokens(self, start: int) -> int:


        return sum(
            max(0, min(log.end, self.matched_len) - max(log.begin, start))
            for log in self.branch.logs
        )


@dataclass(frozen=True)
class RecycledCheckpoint:
    begin      : int
    end        : int
    gpu_entries: tuple[LogStateEntry, ...] = ()


@dataclass(frozen=True)
class RecycledRestorePlan:

    cached_len         : int
    matched_len        : int
    checkpoints        : tuple[RecycledCheckpoint, ...]
    protected_endpoints: frozenset[int]
    gpu_materialize    : tuple[MaterializeCandidate, ...] = ()
    gpu_coalesce       : tuple[CheckpointCoalesce, ...] = ()
    intensity          : int = 2
    lock_boundaries    : frozenset[int] = frozenset()


@dataclass
class _RecycledRestore:
    plan     : RecycledRestorePlan
    slots    : dict[int, int] = field(default_factory=dict)
    detached : set[int] = field(default_factory=set)
    committed: set[int] = field(default_factory=set)
    prepared : bool = False


class LogStateRecycleBin:

    def __init__(self, capacity_bytes: int) -> None:
        from logssms.kernel import logstate as planning

        assert capacity_bytes >= 0
        self.capacity_bytes = capacity_bytes
        self._native = planning.make_recycle_bin(capacity_bytes)
        self._matches: WeakValueDictionary[int, RecycledBranch] = WeakValueDictionary()
        self.revision = 0


    def remember(
        self,
        tree: LogRadixTree,
        node: LogRadixTreeNode,
    ) -> None:
        from logssms.kernel import logstate as planning

        logs = [
            (max(node.begin, entry.begin), min(node.depth, entry.end))
            for entry in node.entries if entry.kind is LogStateKind.LOG
        ]
        if self.capacity_bytes == 0 or not logs:
            return
        self.revision += 1
        path = tree.path(node)
        stats = node.merge_stats
        self._native["remember"](
            [part.key for part in path],
            planning._pack(logs),
            [part.depth for part in path if part.is_protected
             and any(begin <= part.depth <= end for begin, end in logs)],
            [] if stats is None else [
                stats.origin, stats.tail, stats.intensity,
                -1 if stats.next_update_tail is None else stats.next_update_tail,
                *stats.hit_positions,
            ],
        )

    def match(self, tokens: torch.Tensor, *, start: int, granularity: int) -> RecycledMatch | None:
        from logssms.kernel import logstate as planning

        assert tokens.device.type == "cpu"
        result = self._native["match"](tokens, start, granularity)
        if len(result) == 0:
            return None
        serial, matched_len = result
        branch = self._matches.get(serial)
        if branch is None:
            rows = planning._unpack(self._native["get"](serial, tokens))
            branch = RecycledBranch(
                tuple(rows[0]), tuple(RecycledLog(begin, end)
                                      for begin, end in rows[3:]), frozenset(rows[1]), serial,
                tuple(rows[2]),
            )
            self._matches[serial] = branch
        return RecycledMatch(branch, matched_len)

    def discard(self, branch: RecycledBranch) -> None:
        if self._matches.get(branch._serial) is branch:
            self.revision += 1
            self._native["discard"](branch._serial)
            del self._matches[branch._serial]

    def touch(self, branch: RecycledBranch) -> None:
        self._native["touch"](branch._serial)

    def __len__(self) -> int:
        return self._native["usage"]()[1]


class LogStateCapacityError(RuntimeError):
    pass


@dataclass(frozen=True)
class LogStateCachePlan(CachePlan):
    linear_spec       : AffineLogCacheSpec
    num_states        : int
    num_work_states   : int
    state_units       : int


@dataclass(frozen=True)
class LogStateRound:
    eligible_slots: frozenset[int]
    analysis_nodes: tuple[LogRadixTreeNode, ...]
    endpoints     : tuple[_AnalysisEndpoint, ...]


@dataclass(frozen=True)
class _MergePathContext:
    protected_depths : tuple[int, ...]
    protected_entries: tuple[LogStateEntry, ...]
    pending_positions : tuple[int, ...]
    lock_boundaries   : frozenset[int]

    @classmethod
    def build(
        cls,
        tree   : LogRadixTree,
        path   : Sequence[LogRadixTreeNode],
        entries: Sequence[LogStateEntry],
        pending: Sequence[_PendingProtection],
    ) -> _MergePathContext:
        path_ids = {node.uuid for node in path}
        protected_slots: dict[int, bool] = {}
        protected_entries = []
        for entry in entries:
            protected = protected_slots.get(entry.slot)
            if protected is None:
                protected = tree.entry_is_protected(entry.slot)
                protected_slots[entry.slot] = protected
            if protected:
                protected_entries.append(entry)
        return cls(
            protected_depths=tuple(sorted(
                node.depth for node in path if node.is_protected
            )),
            protected_entries=tuple(protected_entries),
            pending_positions=tuple(sorted(
                value.position for value in pending
                if value.trace.node.uuid in path_ids
            )),
            lock_boundaries=frozenset(
                node.depth for node in path
                if node.ref_count > sum(
                    child.ref_count for child in node.children.values()
                )
            ),
        )


@dataclass(frozen=True)
class BranchProtection:
    request_indices: tuple[int, ...]
    position       : int
    log_slot       : int | None
    node           : LogRadixTreeNode | None


@dataclass(frozen=True)
class MaterializeCandidate:
    log_slot: int
    begin   : int
    end     : int

@dataclass(frozen=True)
class CheckpointCoalesce:
    slots        : tuple[int, ...]
    source_spans : tuple[int, ...]
    begin        : int
    end          : int
    next_span    : int

    @property
    def left_slot(self) -> int:
        return self.slots[0]

    @property
    def right_slot(self) -> int:
        return self.slots[-1]


@dataclass(frozen=True)
class LogStatePlan:
    protections      : tuple[BranchProtection, ...]
    materialize      : tuple[MaterializeCandidate, ...]
    coalesce         : tuple[CheckpointCoalesce, ...]
    merge_diagnostics: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class _AnalysisEndpoint:
    node       : LogRadixTreeNode
    matched_len: int


@dataclass(frozen=True)
class _PendingProtection:
    request_indices: tuple[int, ...]
    position       : int
    trace          : LogRadixTrace


@dataclass(frozen=True)
class _LivePage:
    begin     : int
    end       : int
    kv_page   : int
    state_slot: int | None
    checkpoint_begin: int | None = None


@dataclass(slots=True)
class _RequestCacheState:
    work_slot         : int
    work_committed_pos: int
    request_end       : int
    live_pages        : list[_LivePage]     = field(default_factory=list)
    recycle_match     : RecycledMatch | None = None
    live_page_cursor  : int                 = 0

    def live_page_at(self, position: int) -> _LivePage | None:
        pages = self.live_pages
        if not pages:
            return None
        cursor = min(self.live_page_cursor, len(pages) - 1)
        page = pages[cursor]
        if page.begin <= position < page.end:
            return page
        # Decode normally uses the tail, including immediately after prefill.
        if pages[-1].begin <= position < pages[-1].end:
            self.live_page_cursor = len(pages) - 1
            return pages[-1]
        if position < page.begin:
            cursor = 0
        while cursor < len(pages) and pages[cursor].end <= position:
            cursor += 1
        if cursor < len(pages) and pages[cursor].begin <= position:
            self.live_page_cursor = cursor
            return pages[cursor]
        return None


class OnlineAnalysisScheduler:

    def __init__(
        self,
        page_size                : int,
        granularity              : int  = LOG_STATE_GRANULARITY,
        enable_merge_adjustment  : bool = True,
        min_sampling_len         : int  = 1024,
        max_merge_intensity      : int  = 16,
        default_merge_growth_rate: int = 2,
        profile_enabled          : bool = False,
    ) -> None:
        assert page_size > 0
        assert granularity > 0 and page_size % granularity == 0
        assert min_sampling_len > 0
        assert max_merge_intensity >= 2
        assert default_merge_growth_rate >= 1
        self.page_size = page_size
        self.granularity = granularity
        self.enable_merge_adjustment = enable_merge_adjustment
        self.min_sampling_len = min_sampling_len
        self.max_merge_intensity = max_merge_intensity
        self.default_merge_growth_rate = default_merge_growth_rate
        self._merge_profile = (
            {"aggregate": Counter(), "timing_ms": Counter(),
             "updates": deque(maxlen=8192)}
            if profile_enabled else None
        )

    def profile_result(self) -> dict | None:
        profile = self._merge_profile
        if profile is None:
            return None
        return {
            "enabled": self.enable_merge_adjustment,
            "min_sampling_len": self.min_sampling_len,
            "max_merge_intensity": self.max_merge_intensity,
            "default_merge_growth_rate": self.default_merge_growth_rate,
            "aggregate": dict(profile["aggregate"]),
            "timing_ms": dict(profile["timing_ms"]),
            "updates": list(profile["updates"]),
            "dropped_updates": max(
                0, profile["aggregate"]["interval_updates"] - len(profile["updates"])
            ),
        }

    def make_merge_stats(self, origin: int) -> LogStateMergeStats | None:
        if not self.enable_merge_adjustment:
            return None
        if self._merge_profile is not None:
            self._merge_profile["aggregate"]["owners_created"] += 1
        return LogStateMergeStats(
            origin=origin, tail=origin,
            next_update_tail=origin + self.min_sampling_len,
        )

    def record_hit(self, stats: LogStateMergeStats | None, position: int) -> None:
        profile = self._merge_profile
        start = perf_counter_ns() if profile is not None else 0
        if stats is None:
            outcome = "hits_without_owner"
        elif position <= 0:
            outcome = "hits_without_prefix"
        elif position < stats.origin:
            outcome = "hits_before_origin"
        elif position > stats.tail:
            outcome = "hits_after_tail"
        else:
            stats.hit_positions.append(position)
            outcome = "hits_recorded"
        if profile is not None:
            profile["aggregate"]["hit_attempts"] += 1
            profile["aggregate"][outcome] += 1
            profile["timing_ms"]["record_hit_ms"] += (
                perf_counter_ns() - start
            ) / 1e6

    def advance_merge_stats(
        self,
        stats: LogStateMergeStats,
        tail : int,
        *,
        owner_uuid: int | None = None,
    ) -> None:
        if tail <= stats.tail:
            return
        profile = self._merge_profile
        start = perf_counter_ns() if profile is not None else 0
        updates = 0
        boundary = stats.next_update_tail
        if boundary is None:
            boundary = stats.origin + self.min_sampling_len
        while boundary <= tail:
            previous_intensity = stats.intensity
            details = {} if profile is not None else None
            sample_start = perf_counter_ns() if profile is not None else 0
            queued_hits = len(stats.hit_positions) if profile is not None else 0
            bins = {}
            remaining = deque()
            window_begin = boundary - self.min_sampling_len
            for position in stats.hit_positions:
                if position >= boundary:
                    remaining.append(position)
                elif position >= window_begin:
                    distance = max(self.granularity, boundary - position)
                    index = (distance - 1) // self.page_size
                    bins[index] = bins.get(index, 0) + 1
            stats.hit_positions = remaining
            sample_ms = (perf_counter_ns() - sample_start) / 1e6 if profile is not None else 0.0
            fit_start = perf_counter_ns() if profile is not None else 0
            fitted = self._fit_hit_bins(bins, details)
            fit_ms = (perf_counter_ns() - fit_start) / 1e6 if profile is not None else 0.0
            stats.intensity = (
                fitted if fitted is not None else
                min(
                    self.max_merge_intensity,
                    stats.intensity * self.default_merge_growth_rate,
                )
            )
            if profile is not None:
                assert details is not None
                counts = profile["aggregate"]
                counts["interval_updates"] += 1
                counts["fit_updates" if fitted is not None else "fallback_updates"] += 1
                counts[details["reason"]] += 1
                counts["sampled_hits"] += details["sampled_hits"]
                counts["retired_hits"] += queued_hits - len(stats.hit_positions)
                counts["intensity_changes"] += int(stats.intensity != previous_intensity)
                profile["timing_ms"]["fit_ms"] += fit_ms
                profile["timing_ms"]["sample_and_retire_ms"] += sample_ms
                profile["updates"].append({
                    "owner_uuid": owner_uuid,
                    "origin": stats.origin,
                    "tail_before": stats.tail,
                    "tail_after": tail,
                    "window_begin": boundary - self.min_sampling_len,
                    "window_end": boundary,
                    "queued_hits_before": queued_hits,
                    "queued_hits_after": len(stats.hit_positions),
                    "previous_intensity": previous_intensity,
                    "intensity": stats.intensity,
                    "fit_ms": fit_ms,
                    "sample_and_retire_ms": sample_ms,
                    **details,
                })
            updates += 1
            boundary += self.min_sampling_len
        stats.next_update_tail = boundary
        stats.tail = tail
        if profile is not None:
            profile["aggregate"]["tail_advances"] += 1
            profile["aggregate"]["advances_below_update_boundary"] += int(updates == 0)
            profile["timing_ms"]["advance_stats_ms"] += (
                perf_counter_ns() - start
            ) / 1e6

    def _fit_hit_bins(
        self,
        bins: dict[int, int],
        diagnostics: dict | None = None,
    ) -> int | None:
        if len(bins) < 2:
            if diagnostics is not None:
                diagnostics.update(
                    reason="no_hits" if not bins else "insufficient_bins",
                    sampled_hits=sum(bins.values()), occupied_bins=len(bins),
                )
            return None

        xs = [math.log((index + 0.5) * self.page_size) for index in bins]
        ys = [math.log(count / self.page_size) for count in bins.values()]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        denominator = sum((value - x_mean) ** 2 for value in xs)
        if denominator == 0:
            if diagnostics is not None:
                diagnostics.update(
                    reason="degenerate_fit", sampled_hits=sum(bins.values()),
                    occupied_bins=len(bins),
                )
            return None
        slope = sum(
            (x - x_mean) * (y - y_mean)
            for x, y in zip(xs, ys, strict=True)
        ) / denominator
        fitted = math.floor(-slope)
        if diagnostics is not None:
            intercept = y_mean - slope * x_mean
            total_variance = sum((y - y_mean) ** 2 for y in ys)
            residual = sum(
                (y - intercept - slope * x) ** 2
                for x, y in zip(xs, ys, strict=True)
            )
            diagnostics.update(
                reason="fitted", sampled_hits=sum(bins.values()),
                occupied_bins=len(bins), slope=slope, exponent=-slope,
                floored_exponent=fitted,
                clamp=("min" if fitted < 2 else "max" if fitted > self.max_merge_intensity else None),
                r_squared=1 - residual / total_variance if total_variance > 0 else None,
            )
        return min(self.max_merge_intensity, max(2, fitted))

    def begin_round(
        self, tree: LogRadixTree, traces: Sequence[LogRadixTrace],
    ) -> LogStateRound:
        nodes, endpoints = self._analysis_paths(tree, traces)
        return LogStateRound(
            frozenset(entry.slot for node in nodes for entry in node.entries),
            nodes, endpoints,
        )

    def analyze(
        self,
        tree      : LogRadixTree,
        input_ids : Sequence[torch.Tensor],
        round_info: LogStateRound | None = None,
        *,
        collect_diagnostics: bool = False,
        step_timing_ms: dict[str, float] | None = None,
        traces: Sequence[LogRadixTrace] | None = None,
        pending: tuple[_PendingProtection, ...] | None = None,
    ) -> LogStatePlan:
        assert tree.page_size == self.page_size
        assert tree.granularity == self.granularity
        step_start = perf_counter_ns() if step_timing_ms is not None else 0
        if traces is None:
            traces = tuple(
                tree.trace(values[:align_down(len(values), self.granularity)])
                for values in input_ids
            )
        assert len(traces) == len(input_ids)
        if round_info is None:
            round_info = self.begin_round(tree, traces)
        analysis_nodes, endpoints = round_info.analysis_nodes, round_info.endpoints
        if step_timing_ms is not None:
            step_timing_ms["matched_paths_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6
            step_start = perf_counter_ns()
        if pending is None:
            pending = self._find_protections(
                tree, input_ids, traces=traces, step_timing_ms=step_timing_ms,
            )
        else:
            pending = tuple(replace(
                value, trace=(
                    tree.trace_prefix(traces[value.request_indices[0]], value.position)
                    if value.position <= traces[value.request_indices[0]].matched_len
                    else traces[value.request_indices[0]]
                ),
            ) for value in pending)
            if step_timing_ms is not None:
                step_timing_ms["batch_branch_points_ms"] = 0.0
        if step_timing_ms is not None:
            step_timing_ms["protection_total_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6
            step_start = perf_counter_ns()
        protections = self._plan_protections(tree, pending)
        if step_timing_ms is not None:
            step_timing_ms["protection_and_restore_plan_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6
            step_start = perf_counter_ns()
        path_cache = {}
        materialize = self._plan_materialize(
            tree, round_info.eligible_slots, pending, analysis_nodes, path_cache,
        )
        if step_timing_ms is not None:
            step_timing_ms["materialize_plan_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6
            step_start = perf_counter_ns()
        diagnostics = {} if collect_diagnostics else None
        coalesce = self._plan_coalesce(
            tree,
            round_info.eligible_slots,
            pending,
            materialize,
            endpoints,
            diagnostics,
            path_cache,
        )
        if step_timing_ms is not None:
            step_timing_ms["coalesce_plan_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6
        return LogStatePlan(
            protections=tuple(protections),
            materialize=materialize,
            coalesce=coalesce,
            merge_diagnostics=(
                tuple(sorted(diagnostics.items()))
                if diagnostics is not None else ()
            ),
        )

    def _analysis_paths(
        self,
        tree     : LogRadixTree,
        traces   : Sequence[LogRadixTrace],
    ) -> tuple[
        tuple[LogRadixTreeNode, ...],
        tuple[_AnalysisEndpoint, ...],
    ]:
        nodes: dict[int, LogRadixTreeNode] = {}
        endpoints: dict[tuple[int, int], _AnalysisEndpoint] = {}
        for trace in traces:
            if trace.matched_len == 0:
                continue
            endpoint = _AnalysisEndpoint(trace.node, trace.matched_len)
            endpoints[(trace.node.uuid, trace.matched_len)] = endpoint
            for node in tree.path(trace.node):
                if node.uuid in nodes:
                    continue
                nodes[node.uuid] = node
        return (
            tuple(nodes.values()),
            tuple(sorted(
                endpoints.values(),
                key=lambda value: (value.matched_len, value.node.uuid),
            )),
        )

    def _find_protections(
        self,
        tree     : LogRadixTree,
        input_ids: Sequence[torch.Tensor],
        *,
        traces: Sequence[LogRadixTrace] | None = None,
        step_timing_ms: dict[str, float] | None = None,
    ) -> tuple[_PendingProtection, ...]:
        from logssms.kernel import logstate as planning

        branch_start = perf_counter_ns() if step_timing_ms is not None else 0
        if traces is None:
            traces = [tree.trace(values[:align_down(len(values), self.granularity)])
                      for values in input_ids]
        forced = []
        for request_index, values in enumerate(input_ids):
            usable_len = align_down(len(values), self.granularity)
            trace = traces[request_index]
            if trace.matched_len == 0:
                continue
            ends_in_existing_edge = not trace.is_exact
            leaves_existing_route = (
                trace.is_exact
                and trace.matched_len <= usable_len
                and len(trace.node.children) > 0
            )
            if ends_in_existing_edge or leaves_existing_route:
                forced.append((request_index, trace.matched_len))
        candidates = planning.branch_protections(input_ids, forced, self.granularity)
        if step_timing_ms is not None:
            step_timing_ms["batch_branch_points_ms"] = (
                perf_counter_ns() - branch_start
            ) / 1e6
        pending = []
        for position, *request_indices in candidates:
            request_index = request_indices[0]

            trace = traces[request_index]
            if position <= trace.matched_len:
                trace = tree.trace_prefix(trace, position)
            pending.append(
                _PendingProtection(
                    request_indices=tuple(request_indices),
                    position=position,
                    trace=trace,
                )
            )
        return tuple(pending)

    def _plan_protections(
        self,
        tree   : LogRadixTree,
        pending: Sequence[_PendingProtection],
    ) -> tuple[BranchProtection, ...]:
        protections = []
        for value in pending:
            entry = self._covering_log(tree, value)
            node = (
                value.trace.node
                if value.trace.is_exact
                and value.trace.matched_len == value.position
                and not value.trace.node.is_root() else None
            )
            if entry is not None or node is not None:
                protections.append(BranchProtection(
                    request_indices=value.request_indices,
                    position=value.position,
                    log_slot=None if entry is None else entry.slot,
                    node=node,
                ))
        return tuple(protections)

    def _cached_path_entries(
        self,
        tree   : LogRadixTree,
        node   : LogRadixTreeNode,
        cache  : dict,
    ) -> tuple[tuple[LogRadixTreeNode, ...], list[LogStateEntry]]:

        if node.uuid not in cache:
            path = tree.path(node)
            entries = sorted(
                (entry for part in path for entry in part.entries),
                key=lambda entry: (entry.begin, entry.end, entry.slot),
            )
            cache[node.uuid] = (path, entries)
        return cache[node.uuid]

    def _planning_path(
        self,
        tree          : LogRadixTree,
        node          : LogRadixTreeNode,
        eligible_slots: frozenset[int],
        pending       : tuple[_PendingProtection, ...],
        path_cache    : dict,
    ) -> tuple[list[LogStateEntry], list[int], tuple[int, ...], tuple[int, ...]]:

        key = ("native", node.uuid)
        if key not in path_cache:
            path, entries = self._cached_path_entries(tree, node, path_cache)
            context = _MergePathContext.build(tree, path, entries, pending)
            protected_slots = {entry.slot for entry in context.protected_entries}
            rows = [
                value
                for entry in entries
                for value in (
                    entry.slot, entry.begin, entry.end,
                    -1 if entry.span_pages is None else entry.span_pages,
                    entry.physical_begin, int(entry.slot in eligible_slots),
                    int(entry.slot in protected_slots), int(tree.slot_is_fragmented(entry.slot)),
                )
            ]
            path_cache[key] = (
                entries, rows,
                context.protected_depths + context.pending_positions,
                tuple(context.lock_boundaries),
            )
        return path_cache[key]

    def _plan_materialize(
        self,
        tree          : LogRadixTree,
        eligible_slots: frozenset[int],
        pending       : tuple[_PendingProtection, ...],
        analysis_nodes: tuple[LogRadixTreeNode, ...],
        path_cache    : dict,
    ) -> tuple[MaterializeCandidate, ...]:
        from logssms.kernel import logstate as planning

        candidates = []
        paths = []
        path_indices = {}
        entry_indices = []
        source_entries = []
        entries = {entry.slot: entry for node in analysis_nodes for entry in node.entries}


        parent_ids = {node.parent.uuid for node in analysis_nodes}
        owner_paths = {}
        for node in analysis_nodes:
            if node.uuid not in parent_ids:
                for part in tree.path(node):
                    owner_paths.setdefault(part.uuid, node)
        for entry in entries.values():
            if entry.slot not in eligible_slots or entry.kind is not LogStateKind.LOG:
                continue
            owner = owner_paths[tree.entry_owner(entry).uuid]
            if owner.uuid not in path_indices:
                path_indices[owner.uuid] = len(paths)
                path = self._planning_path(tree, owner, eligible_slots, pending, path_cache)
                paths.append(path)
                entry_indices.append({value: i for i, value in enumerate(path[0])})
            path_index = path_indices[owner.uuid]
            candidates.append((path_index, entry_indices[path_index][entry]))
            source_entries.append(entry)
        if not candidates:
            return ()
        rows = planning.plan_materialize(
            [path[1] for path in paths], [path[2] for path in paths],
            candidates, self.page_size,
        )
        result = []
        for row in rows:
            entry = source_entries[row[0]]
            result.append(MaterializeCandidate(entry.slot, entry.begin, entry.end))
        result.sort(key=lambda value: (value.begin, value.end, value.log_slot))
        return tuple(result)

    def _plan_coalesce(
        self,
        tree          : LogRadixTree,
        eligible_slots: frozenset[int],
        pending       : tuple[_PendingProtection, ...],
        materialize   : tuple[MaterializeCandidate, ...],
        endpoints     : tuple[_AnalysisEndpoint, ...],
        diagnostics   : dict[str, int] | None,
        path_cache    : dict,
    ) -> tuple[CheckpointCoalesce, ...]:
        if diagnostics is not None:
            diagnostics.update({
                "matched_endpoints": len(endpoints),
                "internal_endpoints": sum(
                    not endpoint.node.is_leaf() for endpoint in endpoints
                ),
                "leaf_endpoints": sum(
                    endpoint.node.is_leaf() for endpoint in endpoints
                ),
                "merge_anchors": 0,
                "anchors_without_eligible_entries": 0,
                "entries_beyond_match_endpoint": 0,
                "entries_not_eligible_this_round": 0,
                "logs_not_materialized": 0,
                "checkpoint_candidates": 0,
                "protected_entries": 0,
                "protected_boundaries": 0,
                "gaps": 0,
                "lock_boundaries": 0,
                "segments": 0,
                "segments_without_merge_pair": 0,
                "candidate_groups": 0,
                "groups_already_planned": 0,
                "rejected_decreasing_span": 0,
                "rejected_span_mismatch": 0,
                "rejected_discontinuity": 0,
                "rejected_lock_boundary": 0,
                "rejected_protected_interval": 0,
                "planned_groups": 0,
                "planned_slot_releases": 0,
                "same_span_groups": 0,
                "cross_span_groups": 0,
                "planned_covered_tokens": 0,
                "anchors_with_stats": 0,
                "anchors_without_stats": 0,
            })
        from logssms.kernel import logstate as planning

        anchor_limits: dict[int, tuple[LogRadixTreeNode, int]] = {}
        for endpoint in endpoints:
            node = endpoint.node
            limit = min(node.depth, endpoint.matched_len)
            previous = anchor_limits.get(node.uuid)
            if previous is None or limit > previous[1]:
                anchor_limits[node.uuid] = (node, limit)
        if not anchor_limits:
            return ()
        paths = []
        anchors = []
        for node, matched_len in anchor_limits.values():
            stats = node.merge_stats if self.enable_merge_adjustment else None
            if stats is not None and stats.origin > matched_len:
                stats = None
            paths.append(self._planning_path(tree, node, eligible_slots, pending, path_cache))
            anchors.append((
                len(paths) - 1, matched_len,
                stats.intensity if stats is not None else 2,
                int(stats is not None),
            ))
        groups, native_diagnostics = planning.plan_coalesce(
            [path[1] for path in paths], [path[2] for path in paths],
            [path[3] for path in paths], anchors,
            [plan.log_slot for plan in materialize], self.page_size,
            diagnostics is not None,
        )
        if diagnostics is not None:
            diagnostics.update(native_diagnostics)
        result = []
        for row in groups:
            values = tuple(row)
            result.append(CheckpointCoalesce(
                slots=values[3::2], source_spans=values[4::2],
                begin=values[0], end=values[1], next_span=values[2],
            ))
        result.sort(key=lambda value: (value.begin, value.end, value.left_slot))
        return tuple(result)

    def _covering_log(
        self,
        tree   : LogRadixTree,
        pending: _PendingProtection,
    ) -> LogStateEntry | None:
        entries = self._entries_for_node(tree, pending.trace.node)
        candidates = [
            entry
            for entry in entries
            if entry.kind is LogStateKind.LOG
            and entry.physical_begin < pending.position
            < entry.physical_begin + self.page_size
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda entry: entry.end - entry.begin)

    def _entries_for_node(
        self,
        tree: LogRadixTree,
        node: LogRadixTreeNode,
    ) -> list[LogStateEntry]:
        entries = [
            entry
            for path_node in tree.path(node)
            for entry in path_node.entries
        ]
        entries.sort(key=lambda entry: (entry.begin, entry.end, entry.slot))
        return entries


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
        assert pool.page_size == page_size
        assert pool.layout.granularity == granularity
        assert num_pages > 0 and num_states > 0
        assert num_work_states == work_state_table.numel()
        assert pool.allocator.num_frames >= num_pages + 1

        tree = LogRadixTree(
            page_size=page_size,
            device=resources.page_table.device,
            granularity=granularity,
            on_slot_reclaimable=pool.allocator.set_reclaimable,
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
            page_size=page_size,
            granularity=granularity,
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
        self._cached_page_tokens: dict[int, int] = {}
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

    @property
    def tree(self) -> LogRadixTree:
        assert isinstance(self.prefix_cache, LogRadixTree)
        return self.prefix_cache

    @property
    def logstate_pool(self) -> MHALogStatePool:
        pool = self.linear_pool
        assert isinstance(pool, MHALogStatePool)
        return pool

    @property
    def free_pages(self) -> torch.Tensor:
        return torch.tensor(
            self._allocator._free, dtype=torch.int32, device=self.device
        ) * self.page_size

    @property
    def free_states(self) -> torch.Tensor:
        return torch.tensor(
            self._allocator._free_slot_ids, dtype=torch.int32, device=self.device
        )

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
        self._apply_protections(self.analyzer._plan_protections(self.tree, pending))
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
            and not self._request_states[req.table_idx].live_pages
        ]
        if not restores or not pending:
            return
        for req, restore in restores:
            branches = points.get(req.uid, ())
            handle = self._pending_matches[req.table_idx]
            plan = self._plan_recycled_restore(
                handle, handle.recycle_match, branches,
                intensity=restore.plan.intensity, previous=restore.plan,
            )
            assert plan is not None
            restore.plan = plan
            restore.slots = {
                checkpoint.end: checkpoint.gpu_entries[-1].slot
                for checkpoint in plan.checkpoints
                if checkpoint.gpu_entries and not (
                    self.logstate_pool.shared_keys
                    and self._allocator.slot(checkpoint.gpu_entries[-1].slot).kind == "log"
                )
            }

    def _apply_protections(self, protections: Sequence[BranchProtection]) -> None:
        for protection in protections:
            if protection.node is not None:
                self.tree.protect(protection.node)
            if protection.log_slot is not None and self.tree.has_slot(protection.log_slot):
                self.tree.protect_entry(protection.log_slot)

    def _plan_recycled_restore(
        self,
        handle       : LogStateCacheHandle,
        match        : RecycledMatch | None,
        batch_branches: Sequence[int],
        *,
        intensity    : int = 2,
        previous     : RecycledRestorePlan | None = None,
    ) -> RecycledRestorePlan | None:
        if match is None:
            return None
        from logssms.kernel import logstate as planning

        cached_len = handle.cached_len
        gpu = tuple(
            entry for entry in handle.match_entries
            if entry.end <= cached_len
        )
        path = handle.match_path
        protected = frozenset(
            (*match.branch.protected_endpoints, *batch_branches,
             *((match.matched_len,) if match.matched_len < len(match.branch.tokens) else ()),
             *(node.depth for node in path if node.is_protected))
        )
        context = _MergePathContext.build(self.tree, path, gpu, ())
        protected_slots = {entry.slot for entry in context.protected_entries}
        eligible_slots = None if previous is None else {
            *(entry.slot for checkpoint in previous.checkpoints for entry in checkpoint.gpu_entries),
            *(item.log_slot for item in previous.gpu_materialize),
            *(slot for item in previous.gpu_coalesce for slot in item.slots),
        }
        lock_boundaries = context.lock_boundaries if previous is None else previous.lock_boundaries


        protected_pages = {
            position // self.page_size for position in protected
            if position % self.page_size
        }
        pieces: list[tuple[int, int, LogStateEntry | None, bool]] = []
        for entry in gpu:
            eligible = (
                (not self.tree.entry_owner(entry).is_locked if eligible_slots is None
                 else entry.slot in eligible_slots)
                and entry.slot not in protected_slots
                and not self.tree.slot_is_fragmented(entry.slot)
                and entry.begin % self.page_size == 0
                and entry.end % self.page_size == 0
                and not any(entry.begin < position < entry.end for position in protected)
                and not (entry.kind is LogStateKind.LOG and entry.end in protected)
            )
            pieces.append((entry.begin, entry.end, entry, eligible))
        for entry in match.branch.logs:
            begin = max(cached_len, entry.begin)
            end = min(match.matched_len, entry.end)
            begin = align_ceil(begin, self.page_size)
            end = align_down(end, self.page_size)
            if begin >= end:
                continue
            cuts = {begin, end}
            cuts.update(pos for pos in protected if begin < pos < end and pos % self.page_size == 0)
            for page in protected_pages:
                cuts.update(pos for pos in (page * self.page_size, (page + 1) * self.page_size)
                            if begin < pos < end)


            cuts.update(range(begin, end, self.page_size))
            boundaries = sorted(cuts)
            for left, right in zip(boundaries, boundaries[1:]):
                if left // self.page_size in protected_pages:
                    continue
                pieces.append((left, right, None, True))
        pieces.sort(key=lambda item: (item[0], item[1]))
        rows = []
        for index, (begin, end, entry, eligible) in enumerate(pieces):
            rows.extend((
                index, begin, end, (end - begin) // self.page_size,
                begin, int(eligible), int(not eligible), 0,
            ))
        groups, _ = planning.plan_coalesce(
            [rows], [sorted(protected)], [lock_boundaries],
            [(0, match.matched_len, intensity, int(self.enable_merge_adjustment))],
            (), self.page_size, False,
        )
        consumed = set()
        checkpoints = []
        gpu_coalesce = []
        for group in groups:
            indices = group[3::2]
            if all(pieces[index][2] is not None for index in indices):
                gpu_coalesce.append(CheckpointCoalesce(
                    tuple(pieces[index][2].slot for index in indices),
                    tuple(group[4::2]), group[0], group[1], group[2],
                ))
                continue
            consumed.update(indices)
            checkpoints.append(RecycledCheckpoint(
                group[0], group[1], tuple(
                    pieces[index][2] for index in indices if pieces[index][2] is not None
                ),
            ))
        checkpoints.extend(
            RecycledCheckpoint(begin, end)
            for index, (begin, end, entry, eligible) in enumerate(pieces)
            if index not in consumed and entry is None and eligible
        )
        gpu_candidates = []
        for index, (begin, end, entry, eligible) in enumerate(pieces):
            if index in consumed or entry is None or not eligible or entry.kind is not LogStateKind.LOG:
                continue
            gpu_candidates.append(entry)
        if gpu_candidates:
            indices = {entry: i for i, entry in enumerate(gpu)}
            rows = [value for entry in gpu for value in (
                entry.slot, entry.begin, entry.end,
                -1 if entry.span_pages is None else entry.span_pages,
                entry.physical_begin, 1, 0, 0,
            )]
            reachable = planning.plan_materialize(
                [rows], [()], [(0, indices[entry]) for entry in gpu_candidates], self.page_size,
            )
            assert len(reachable) == len(gpu_candidates)
        gpu_materialize = tuple(MaterializeCandidate(entry.slot, entry.begin, entry.end)
                                for entry in gpu_candidates)
        return RecycledRestorePlan(
            cached_len, match.matched_len,
            tuple(sorted(checkpoints, key=lambda entry: entry.end)), protected,
            tuple(gpu_materialize), tuple(gpu_coalesce),
            intensity, lock_boundaries,
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
            pages = state.live_pages
            begin = pages[-1].end if pages else max(0, end - token_len)
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
            and (not checkpoint.gpu_entries or (
                self.logstate_pool.shared_keys
                and self._allocator.slot(checkpoint.gpu_entries[-1].slot).kind == "log"
            )) and checkpoint.end not in slots
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
            or self.tree.entries_for_slot(entry.slot) != (entry,)
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
            slots.update(item.log_slot for item in handle.restore_plan.gpu_materialize)
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
                checkpoint.end: checkpoint.gpu_entries[-1].slot
                for checkpoint in restore.plan.checkpoints
                if checkpoint.gpu_entries and not (
                    self.logstate_pool.shared_keys
                    and self._allocator.slot(checkpoint.gpu_entries[-1].slot).kind == "log"
                )
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
            pages = state.live_pages
            begin = pages[-1].end if pages else req.cached_len
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

        kv_pages, state_slots, direct_slots = self._allocate(
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
            live_pages = self._request_states[req.table_idx].live_pages
            restore = self._recycled_restores.get(req.table_idx)
            logical_begin = begin
            for index in range(count):
                logical_end = align_ceil(logical_begin + 1, self.page_size)
                kv_page = int(kv_pages[offset])
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
                live_pages.append(
                    _LivePage(
                        begin=logical_begin,
                        end=logical_end,
                        kv_page=kv_page,
                        state_slot=state_slot,
                        checkpoint_begin=checkpoint_begin,
                    )
                )
                regions.append((
                    req.table_idx, logical_begin,
                    min(logical_end, req.max_device_len, self.page_table.shape[1]),
                    kv_page, state_slot if checkpoint_begin is None and not inside_checkpoint else -1,
                ))
                logical_begin = logical_end
            mapped_end = min(
                logical_begin,
                req.max_device_len,
                self.page_table.shape[1],
            )
            assert live_pages[-1].begin < mapped_end <= live_pages[-1].end
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
            entry.span_pages for entry in entries
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
            "free_kv_pages": len(self.free_pages),
            "free_state_slots": len(self.free_states),
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
            "checkpoint_span_pages": dict(sorted(checkpoint_spans.items())),
            "checkpoint_covered_pages": sum(
                span * count for span, count in checkpoint_spans.items()
            ),
            "live_log_pages": sum(len(state.live_pages) for state in self._request_states.values()),
        }

    def _profile_request_paths(
        self,
        reqs: Sequence[Request],
    ) -> list[dict]:
        result = []
        for req in reqs:
            recovery = self.pending_recoveries.get(req.table_idx)
            state = self._request_states.get(req.table_idx)
            live = () if state is None else state.live_pages
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
                "cached_checkpoint_span_pages": dict(sorted(Counter(
                    entry.span_pages for entry in cached_entries
                    if entry.kind is LogStateKind.CHECKPOINT
                ).items())),
                "live_log_pages": len(live),
            })
        return result

    def prepare_batch(self, batch: Batch) -> None:
        if not batch.is_prefill:
            self.last_plan = None
            if batch.is_decode:
                replay_counts = []
                for req in batch.reqs:
                    state = self._request_states[req.table_idx]
                    page = state.live_page_at(req.cached_len)
                    assert page is not None, "Decode token has no LogState page"
                    offset = req.cached_len - page.begin
                    count = req.cached_len - state.work_committed_pos
                    assert 0 <= count < LOG_STATE_REPLAY_WINDOW
                    assert count <= offset % LOG_STATE_REPLAY_WINDOW
                    replay_counts.append(count)
                    if (offset + 1) % LOG_STATE_REPLAY_WINDOW == 0:
                        state.work_committed_pos = req.cached_len + 1
                batch.logstate_replay_counts = replay_counts + [
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
            page = next(
                (page for page in self._request_states[req.table_idx].live_pages
                 if page.begin <= req.cached_len < page.end),
                None,
            )
            if page is None:


                break
            first_offsets.append(req.cached_len - page.begin)
        batch.logstate_first_offsets = (
            first_offsets if len(first_offsets) == len(batch.reqs) else None
        )
        batch.logstate_checkpoint_writes = []
        batch.logstate_checkpoint_ranges = []
        row_offset = 0
        for req_index, req in enumerate(batch.reqs):
            restore = self._recycled_restores.get(req.table_idx)
            if restore is not None:
                batch.logstate_checkpoint_ranges.extend(
                    (req_index, checkpoint.begin, checkpoint.end)
                    for checkpoint in restore.plan.checkpoints
                    if checkpoint.begin < req.device_len and req.cached_len < checkpoint.end
                )
            for page in self._request_states[req.table_idx].live_pages:
                if (
                    page.checkpoint_begin is not None
                    and req.cached_len < page.end <= req.device_len
                ):
                    assert page.state_slot is not None
                    batch.logstate_checkpoint_writes.append((
                        row_offset + page.end - req.cached_len - 1,
                        page.state_slot,
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
            + [item.log_slot for restore in recycled for item in restore.plan.gpu_materialize]
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
            outcome = self._merge_batch_outcome(plan, released)
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
                    assert self.tree.entries_for_slot(entry.slot) == (entry,)
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
        pages = request_state.live_pages
        restore = self._recycled_restores.get(req.table_idx)
        protected_endpoints = set(old_handle.protected_endpoints)
        protected_endpoints.update(self._protected_prefixes.get(req.uid, ()))
        if restore is not None:
            protected_endpoints.update(restore.plan.protected_endpoints)
        full_pages = [page for page in pages if page.end <= req.cached_len]
        committed_ends = {page: page.end for page in full_pages}
        if finished:
            for page in pages:
                if page.end <= req.cached_len or page.checkpoint_begin is not None:
                    continue
                position = max((value for value in protected_endpoints
                                if page.begin < value <= req.cached_len < page.end), default=0)
                if position:
                    full_pages.append(page)
                    committed_ends[page] = position
        insert_len = old_handle.cached_len
        for page in full_pages:
            if page.begin != insert_len:
                break
            insert_len = committed_ends[page]
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
            kv_page_spans=((page.begin, committed_ends[page], page.kv_page // self.page_size)
                           for page in full_pages),
        )
        tree_insert_ms = (
            (perf_counter_ns() - stage_start) / 1e6
            if self._profile_enabled else 0.0
        )
        adopted = [
            page
            for page in full_pages
            if old_handle.cached_len <= page.begin
            and committed_ends[page] <= insert.inserted_len
        ]
        adopted_set = set(adopted)
        stage_start = perf_counter_ns() if self._profile_enabled else 0
        existing = self.tree.recovery_at(insert.node, insert.inserted_len)
        tree_match_ms = (
            (perf_counter_ns() - stage_start) / 1e6
            if self._profile_enabled else 0.0
        )
        covered_len = existing.position
        released_kv_pages = []
        released_state_slots = []
        retain_kv_ms = 0.0
        for page in adopted:
            page_end = committed_ends[page]
            logical_begin = max(page.begin, covered_len)
            if page.checkpoint_begin is not None and logical_begin < page_end:
                assert page.state_slot is not None
                checkpoint_begin = max(
                    page.checkpoint_begin, align_down(covered_len, self.page_size),
                )
                kind = self._allocator.slot(page.state_slot).kind
                if kind == "work":
                    self.logstate_pool.commit_checkpoint(page.state_slot)
                elif kind == "log":
                    self.logstate_pool.materialize(page.state_slot)
                self.tree.attach_entry(
                    insert.node,
                    LogStateEntry(
                        slot=page.state_slot,
                        kind=LogStateKind.CHECKPOINT,
                        begin=checkpoint_begin,
                        end=page.end,
                        span_pages=(page.end - checkpoint_begin) // self.page_size,
                    ),
                )
                covered_len = page.end
                if restore is not None:
                    restore.committed.add(page.end)
                    if self._profile_enabled:
                        self._profile_stats["recycle_committed_checkpoints"] += 1
                        self._profile_stats["recycle_reused_gpu_slots"] += int(kind != "work")
                        self._profile_stats["recycle_committed_tokens"] += (
                            page.end - max(checkpoint_begin, restore.plan.cached_len)
                        )
            elif page.state_slot is not None and logical_begin < page_end:
                self.tree.attach_entry(
                    insert.node,
                    LogStateEntry(
                        slot=page.state_slot,
                        kind=LogStateKind.LOG,
                        begin=logical_begin,
                        end=page_end,
                        storage_begin=page.begin,
                    ),
                )
                covered_len = page_end
            elif page.state_slot is not None:
                released_state_slots.append(page.state_slot)
                if restore is not None and page.checkpoint_begin is not None:
                    restore.slots.pop(page.end, None)

            kv_begin = max(page.begin, insert.previous_len)
            if kv_begin < page_end:
                stage_start = perf_counter_ns() if self._profile_enabled else 0
                self._cached_page_tokens[page.kv_page] = (
                    self._cached_page_tokens.get(page.kv_page, 0)
                    + page_end - kv_begin
                )
                if self._profile_enabled:
                    retain_kv_ms += (perf_counter_ns() - stage_start) / 1e6
            else:
                released_kv_pages.append(page.kv_page)

        duplicates = [
            page
            for page in full_pages
            if page not in adopted_set and committed_ends[page] <= insert.previous_len
        ]
        self._release_live_pages(req.table_idx, duplicates)
        if released_kv_pages or released_state_slots:
            self._return_slots(
                released_kv_pages,
                released_state_slots,
            )

        if protected_endpoints:
            for position in sorted(protected_endpoints):
                if 0 < position <= insert.inserted_len:
                    trace = self.tree.trace(req.input_ids[:position])
                    node = trace.node
                    if not trace.is_exact:
                        node = self.tree.split_at(node, trace.matched_in_node)
                    self.tree.protect(node)
                    for entry in self.tree.matched_entries(insert.node):
                        if (entry.kind is LogStateKind.LOG
                            and entry.physical_begin < position < entry.physical_begin + self.page_size):
                            self.tree.protect_entry(entry.slot)

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
        remaining = [page for page in pages if page not in removed_pages]
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
            self._release_live_pages(req.table_idx, remaining)
            restore = self._recycled_restores.pop(req.table_idx, None)
            if restore is not None:
                page_slots = {page.state_slot for page in remaining}
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
            request_state.live_pages = remaining
            request_state.live_page_cursor = 0
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
        pages = state.live_pages
        keep = [page for page in pages if page.begin < req.cached_len]
        release = [page for page in pages if page.begin >= req.cached_len]
        self._release_live_pages(req.table_idx, release)
        state.live_pages = keep
        state.live_page_cursor = 0
        req.device_len = min(req.cached_len + 1, req.max_device_len)

    def commit_speculative(self, req: Request, C: int, num_sampled: int) -> None:
        raise NotImplementedError(
            "LogState speculative commit requires the LogState verify backend."
        )

    def check_integrity(self) -> None:
        self.tree.check_integrity()
        live_pages = sum(len(state.live_pages) for state in self._request_states.values())
        if live_pages != 0:
            raise RuntimeError(
                f"LogState cache has {live_pages} request-owned pages while idle."
            )

        if sum(self._cached_page_tokens.values()) != self.tree.size_info.total_size:
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
        kv_pages = {page // self.page_size for page in self._cached_page_tokens}
        kv_pages.update(
            page.kv_page // self.page_size
            for state in self._request_states.values() for page in state.live_pages
        )
        expected_kv = kv_pages | {self._dummy_kv_frame}
        expected_state = tree_slots | {self._dummy_log_slot} | {
            page.state_slot for state in self._request_states.values() for page in state.live_pages
            if page.state_slot is not None
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
        needed_pages : int,
        needed_states: int,
        needed_works : int = 0,
        profile     : dict | None = None,
    ) -> tuple[list[int], list[int], list[int]]:
        own_profile = profile is None and self._profile_enabled
        if own_profile:
            profile = {
                "phase": "work" if needed_works else "direct",
                "batch_size": 0, "needed_pages": needed_pages,
                "needed_states": needed_states, "needed_works": needed_works,
                "evicted_nodes": 0, "timing_ms": {}, "counts": {},
            }
        start = perf_counter_ns() if profile is not None else 0
        initial_pool_timings = {} if profile is not None else None
        reserved = self.logstate_pool.reserve(
            kv=needed_pages, logs=needed_states, works=needed_works,
            timings=initial_pool_timings,
        )
        if profile is not None:
            now = perf_counter_ns()
            profile["timing_ms"]["initial_reserve_ms"] = (now - start) / 1e6
            self._record_pool_timings(
                profile, "initial", initial_pool_timings
            )
            stage = now
        if reserved is None:
            selected, page_counts, released_slots, checkpoint_pairs = self._plan_eviction(
                needed_pages, needed_states, needed_works
            )
            if profile is not None:
                now = perf_counter_ns()
                profile["timing_ms"]["eviction_plan_ms"] = (now - stage) / 1e6
                profile["evicted_nodes"] = len(selected)
                profile.setdefault("counts", {})["dropped_checkpoints"] = len(checkpoint_pairs)
                stage = now

            dropped_checkpoints = []
            for left_slot, right_slot in checkpoint_pairs:
                entry = self.tree.drop_checkpoint(left_slot, right_slot)
                if self._profile_enabled:
                    dropped_checkpoints.append({
                        "slot": entry.slot, "begin": entry.begin, "end": entry.end,
                        "retained_slot": right_slot,
                    })
            evicted_entry_spans = (
                [
                    {
                        "node_uuid": node.uuid,
                        "kind": entry.kind.value,
                        "begin": entry.begin,
                        "end": entry.end,
                        "slot": entry.slot,
                    }
                    for node in selected for entry in node.entries
                ]
                if self._profile_enabled else None
            )
            evicted_nodes = [] if self._profile_enabled else None
            free_frames_before = self._allocator.free_frames if self._profile_enabled else 0
            if self.recycle_bin is not None:
                for node in selected:
                    self.recycle_bin.remember(self.tree, node)
            eviction = self.tree.evict_nodes(selected, evicted_nodes)
            if profile is not None:
                now = perf_counter_ns()
                profile["timing_ms"]["tree_evict_ms"] = (now - stage) / 1e6
                stage = now
            self._release_cached_kv(page_counts)
            if profile is not None:
                now = perf_counter_ns()
                profile["timing_ms"]["release_kv_ms"] = (now - stage) / 1e6
                stage = now
            for slot in released_slots:
                assert not self.tree.has_slot(slot)


            self.logstate_pool.retire_slots(sorted(released_slots))
            if profile is not None:
                now = perf_counter_ns()
                profile["timing_ms"]["release_states_ms"] = (now - stage) / 1e6
                stage = now
            free_frames_after_evict = (
                self._allocator.free_frames if self._profile_enabled else 0
            )
            retry_pool_timings = {} if profile is not None else None
            reserved = self.logstate_pool.reserve(
                kv=needed_pages, logs=needed_states, works=needed_works,
                timings=retry_pool_timings,
            )
            if profile is not None:
                now = perf_counter_ns()
                profile["timing_ms"]["retry_reserve_ms"] = (now - stage) / 1e6
                self._record_pool_timings(
                    profile, "retry", retry_pool_timings
                )
                stage = now
            assert reserved is not None, "Eviction plan did not free the projected frames"
            if self._profile_enabled:
                self._record_lifecycle_event({
                    "event": "eviction",
                    "phase": profile["phase"] if profile is not None else None,
                    "request_tags": profile.get("request_tags") if profile is not None else None,
                    "nodes": [
                        {"uuid": uuid, "begin": begin, "end": end}
                        for uuid, begin, end in evicted_nodes
                    ],
                    "entries": evicted_entry_spans,
                    "dropped_checkpoints": dropped_checkpoints,
                    "kv_tokens": len(eviction.kv_indices),
                    "state_slots": len(released_slots),
                    "free_frames_before": free_frames_before,
                    "free_frames_after": free_frames_after_evict,
                    "need_kv_pages": needed_pages,
                    "need_log_slots": needed_states,
                    "need_work_slots": needed_works,
                })
                self._profile_stats["eviction_rounds"] += 1
                self._profile_stats["pressure_dropped_checkpoints"] += len(checkpoint_pairs)
                self._profile_stats["pressure_pruned_checkpoints"] += sum(
                    entry["kind"] == LogStateKind.CHECKPOINT.value for entry in evicted_entry_spans
                )
                self._profile_stats["evicted_kv_tokens"] += len(
                    eviction.kv_indices
                )
                self._profile_stats[
                    "released_state_slots_from_eviction"
                ] += len(released_slots)

        ids_start = perf_counter_ns() if profile is not None else 0
        pages = [page * self.page_size for page in reserved[0]]
        states, works = reserved[1:]
        if profile is not None:
            now = perf_counter_ns()
            profile["timing_ms"]["ids_prepare_ms"] = (
                now - ids_start
            ) / 1e6
            profile["timing_ms"]["core_total_ms"] = (now - start) / 1e6
            if own_profile:
                self._profile_allocation_samples.append(profile)
        return pages, states, works

    def _plan_eviction(
        self, needed_pages: int, needed_states: int, needed_works: int,
    ) -> tuple[list[LogRadixTreeNode], dict[int, int], set[int], list[tuple[int, int]]]:
        from logssms.kernel import logstate as planning

        allocator = self._allocator
        for nodes, topology, log_only in self.tree.eviction_candidates():
            if not nodes:
                continue
            if self._profile_enabled:
                self._profile_stats["eviction_candidate_windows"] += int(log_only)
                self._profile_stats["eviction_full_snapshots"] += int(not log_only)
                self._profile_stats["eviction_serialized_nodes"] += len(nodes)
            node_slots = [node.eviction_slots for node in nodes]
            slot_ids = {slot for row in node_slots for slot in row}
            slots = []
            decay_frames = set()
            for slot in slot_ids:
                record = allocator.slot(slot)
                slots.append((
                    slot, self.tree.slot_entry_count(slot),
                    len(record.payload) + sum(map(len, record.conv)),
                    -1 if record.decay is None else record.decay.frame,
                    -1 if record.decay is None else record.decay.index,
                ))
                if record.decay is not None:
                    decay_frames.add(record.decay.frame)
            rows = planning.plan_eviction(
                topology,
                [[value for frame, count in node._kv_page_counts.items() for value in (
                    frame, count, self._cached_page_tokens[frame * self.page_size],
                )] for node in nodes],
                node_slots,
                slots,
                [(frame, allocator._decay_masks[frame].bit_count()) for frame in decay_frames],
                (allocator.free_frames, allocator._free_decay_records,
                 allocator.payload_frames, allocator.conv_frames, allocator.num_granules,
                 allocator.decay_records_per_frame, needed_pages, needed_states, needed_works,
                 allocator.log_payload_frames, self.work_state_table.shape[0] - 1),
                () if log_only else self.tree.checkpoint_eviction_actions(nodes),
            )
            if rows:
                selected, pages, released, compressed = rows
                return ([nodes[index] for index in selected],
                        dict(zip(pages[::2], pages[1::2], strict=True)), set(released),
                        list(zip(compressed[::2], compressed[1::2], strict=True)))
        raise LogStateCapacityError(
            "Insufficient evictable LogState cache capacity: "
            f"need_pages={needed_pages}, need_states={needed_states}, "
            f"free_frames={allocator.free_frames}."
        )

    @staticmethod
    def _record_pool_timings(
        profile: dict, prefix: str, timings: dict | None,
    ) -> None:
        assert timings is not None
        for name, value in timings.items():
            bucket = "counts" if name in (
                "waited_materializations", "descriptor_count",
                "reclamation_count", "failed",
            ) else "timing_ms"
            profile.setdefault(bucket, {})[f"{prefix}_{name}"] = value

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
        released = released_frames = merged = segments = materialized = 0
        skips: Counter[str] = Counter()
        events = None
        if self._profile_enabled and (works or plan.coalesce):
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()


        for req in batch.reqs:
            self._release_recovery(id(self._pending_matches.get(req.table_idx)))
        groups = iter(plan.coalesce)
        wave_count = 0
        while True:
            wave, chains, frame_counts = [], [], []
            for group in groups:
                if any(not self.tree.has_slot(slot) or self.tree.slot_is_fragmented(slot)
                       or slot in self.tree._recovery_slot_refs for slot in group.slots):
                    skips["pending_or_fragmented_source"] += 1
                    continue
                right = self.tree.entries_for_slot(group.right_slot)[0]
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
                if self._allocator.checkpoint_frame_savings(group.slots) <= 0:
                    skips["no_memory_gain"] += 1
                    continue
                recovery = self.tree.recovery_at(self.tree.entry_owner(right), right.end)
                assert recovery.position == right.end
                logs = [(entry.slot, entry.begin - entry.physical_begin, entry.end - entry.physical_begin)
                        for entry in recovery.logs]
                wave.append(group)
                chains.append((recovery.base_checkpoint_slot, logs, works[len(wave) - 1]))
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
                right_entries = self.tree.entries_for_slot(group.right_slot)
                if (len(right_entries) == 1 and right_entries[0].kind is LogStateKind.CHECKPOINT
                    and right_entries[0].begin == group.begin and right_entries[0].end == group.end):
                    committed.append(group.slots)
                    committed_works.append(works[index])
                    released_frames += frame_counts[index] - self._allocator.payload_frames - self._allocator.conv_frames
            if committed:
                pool.coalesce_groups(committed, committed_works)
                pool.retire_slots([slot for group in committed for slot in group[:-1]])
            merged += len(committed)
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
        construction["materialized_slots"] = materialized
        if events is not None:
            events[1].record()
            construction["cuda_ms"] = None
            self._profile_cuda_events.append((construction, *events))
        return construction, (released, released_frames, dict(skips))

    @staticmethod
    def _merge_batch_outcome(plan: LogStatePlan, released: int) -> str:
        if released:
            return "merged"
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
                or self.tree.slot_is_fragmented(slot)
                for slot in plan.slots
            ):
                skips["missing_or_fragmented_slot"] = (
                    skips.get("missing_or_fragmented_slot", 0) + 1
                )
                continue
            entries = [
                self.tree.entries_for_slot(slot)[0]
                for slot in plan.slots
            ]
            if any(
                (entry.kind is not LogStateKind.CHECKPOINT and not (
                    allow_logs and entry.kind is LogStateKind.LOG and level == 1
                ))
                or entry.slot != slot
                or (entry.span_pages if entry.kind is LogStateKind.CHECKPOINT else 1) != level
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
                    span_pages=plan.next_span,
                ),
                checkpoint_timestamp=checkpoint_timestamp,
            )
            released.extend(plan.slots[:-1])
        if released and retire_sources:


            self.logstate_pool.retire_slots(released)
        return len(released), released_frames, skips

    def _release_live_pages(
        self,
        table_idx: int,
        pages    : Sequence[_LivePage],
    ) -> None:
        if not pages:
            return
        state_table = self.state_table
        assert state_table is not None
        restore = self._recycled_restores.get(table_idx)
        released_states = []
        for page in pages:
            state_table[table_idx, page.begin:page.end].fill_(-1)
            if page.state_slot is None:
                continue
            if restore is not None and page.checkpoint_begin is not None:
                checkpoint = next(value for value in restore.plan.checkpoints if value.end == page.end)
                if (
                    page.state_slot not in restore.detached
                    and any(entry.slot == page.state_slot for entry in checkpoint.gpu_entries)
                ):
                    continue
                restore.slots.pop(page.end, None)
            released_states.append(page.state_slot)
        self._return_slots(
            [page.kv_page for page in pages],
            released_states,
        )

    def _return_slots(
        self,
        pages : Sequence[int],
        states: Sequence[int],
    ) -> None:
        if self._lazy_free_pages is None:
            page_frames = [page // self.page_size for page in pages]
            self.logstate_pool.release_kv(page_frames)
            self.logstate_pool.release_slots(states)
        else:
            assert self._lazy_free_states is not None
            # Snapshot the IDs, but leave reclamation and event waits to the
            # pool after the region exits, including exceptional exits.
            self._lazy_free_pages.extend(pages)
            self._lazy_free_states.extend(states)

    def _release_cached_kv(self, page_counts: Counter[int]) -> None:
        released = []
        for frame, count in page_counts.items():
            page = frame * self.page_size
            remaining = self._cached_page_tokens[page] - count
            assert remaining >= 0
            if remaining == 0:
                del self._cached_page_tokens[page]
                released.append(frame)
            else:
                self._cached_page_tokens[page] = remaining
        if released:
            self.logstate_pool.release_kv(released)


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
        assert config.page_size == 64, (
            "LogState currently uses a fixed 64-token physical page."
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
        from .pools import make_linear_cache_spec

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
        log_payload_frames = (payload_frames if layout.num_key_heads == layout.num_value_heads
                              else div_ceil(layout.log_payload_bytes, cache_per_page))
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
    "BranchProtection",
    "CheckpointCoalesce",
    "LogStatePlan",
    "LogStateRound",
    "LogStateCacheManager",
    "LogStateCacheFactory",
    "LogStateCachePlan",
    "OnlineAnalysisScheduler",
    "MaterializeCandidate",
]
