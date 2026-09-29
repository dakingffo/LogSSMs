from __future__ import annotations

import math
import os
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from time import perf_counter_ns
from typing import TYPE_CHECKING, Sequence

import torch

from logssms.core import Request
from logssms.kernel.linear_attention.framed_logstate import (
    construct_framed_log_state_chains,
    restore_framed_log_state_chains,
)
from logssms.utils import align_ceil, align_down, div_ceil, init_logger

from .base import BaseCacheManager, CacheResources
from .factory import BaseCacheFactory, CachePlan, SUPPORTED_CACHE_FACTORIES
from .logstate_pool.layout import (
    LOG_STATE_GRANULARITY,
    LOG_STATE_REPLAY_WINDOW,
    LogStateSlotLayout,
)
from .logstate_pool.frame_pool import FramedMHALogStatePool
from .logstate_recycle import LogStateRecycleBin, RecycledMatch
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
    from logssms.scheduler.prefill import PendingRequest
    from logssms.speculator import SpeculatorReserve


logger = init_logger(__name__)


class LogStateCapacityError(RuntimeError):
    pass


@dataclass(frozen=True)
class LogStateCachePlan(CachePlan):
    num_states        : int
    num_work_states   : int
    cache_per_page    : int
    cache_per_log_slot: int
    state_units       : int


class WorkStateRestoreMode(Enum):
    LOG_REPLAY = "log_replay"
    PREFILL = "prefill"


@dataclass(frozen=True)
class LogStateRound:
    eligible_slots: frozenset[int]


@dataclass(frozen=True)
class BranchProtection:
    request_indices: tuple[int, ...]
    position       : int
    log_slot       : int | None
    node           : LogRadixTreeNode | None


@dataclass(frozen=True)
class WorkStateRestore:
    request_indices     : tuple[int, ...]
    position            : int
    replay_end          : int
    mode                : WorkStateRestoreMode
    base_checkpoint_slot: int | None
    logs                : tuple[LogStateEntry, ...]

    @property
    def log_slots(self) -> tuple[int, ...]:
        return tuple(entry.slot for entry in self.logs)


@dataclass(frozen=True)
class PageMaterialize:
    log_slot            : int
    begin               : int
    end                 : int
    base_checkpoint_slot: int | None
    replay_logs         : tuple[LogStateEntry, ...]

    @property
    def replay_log_slots(self) -> tuple[int, ...]:
        return tuple(entry.slot for entry in self.replay_logs)


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
    tree_revision    : int
    protections      : tuple[BranchProtection, ...]
    restores         : tuple[WorkStateRestore, ...]
    materialize      : tuple[PageMaterialize, ...]
    coalesce         : tuple[CheckpointCoalesce, ...]
    merge_diagnostics: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class _AnalysisEndpoint:
    node       : LogRadixTreeNode
    matched_len: int


@dataclass
class _BatchPrefixNode:
    depth          : int
    children       : dict[tuple[int, ...], _BatchPrefixNode] = field(
        default_factory=dict
    )
    request_indices: list[int] = field(default_factory=list)
    terminal_count : int = 0


@dataclass(frozen=True)
class _PendingProtection:
    request_indices: tuple[int, ...]
    request_index  : int
    position       : int
    trace          : LogRadixTrace


@dataclass(frozen=True)
class _LivePage:
    begin     : int
    end       : int
    kv_page   : int
    state_slot: int | None
    checkpoint_begin: int | None = None


@dataclass(frozen=True)
class _ReconstructionTask:
    output_slot        : int
    base_slot          : int | None
    logs               : tuple[LogStateEntry, ...]
    source_conv_slot   : int
    source_conv_granule: int
    output_conv_granule: int


@dataclass(frozen=True)
class _MaterializeChain:
    base_slot       : int | None
    logs            : tuple[LogStateEntry, ...]
    checkpoint_slots: frozenset[int]


@dataclass(frozen=True)
class _ConstructionChain:
    base_slot  : int | None
    segments   : tuple[LogStateEntry, ...]
    outputs    : tuple[tuple[int, ...], ...]
    materialize: bool


class OnlineAnalysisScheduler:
    """Generate LogState protection, reconstruction, and merge plans.

    With merge adjustment disabled, checkpoint coalescing remains binary. With
    it enabled, each logical chain samples branch-hit endpoints over a moving
    token window, fits their binned density in log-log space, and uses the
    floored power-law exponent as the maximum same-level merge arity. This class
    never allocates slots, launches kernels, or mutates persistent tree entries.
    """

    def __init__(
        self,
        page_size                : int,
        granularity              : int  = LOG_STATE_GRANULARITY,
        enable_merge_adjustment  : bool = False,
        min_sampling_len         : int  = 1024,
        max_merge_intensity      : int  = 16,
        default_merge_growth_rate: int = 2,
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

    def make_merge_stats(self, origin: int) -> LogStateMergeStats | None:
        if not self.enable_merge_adjustment:
            return None
        return LogStateMergeStats(
            origin=origin, tail=origin,
            next_update_tail=origin + self.min_sampling_len,
        )

    def record_hit(self, stats: LogStateMergeStats, position: int) -> None:
        if position < stats.origin or position > stats.tail:
            return
        stats.hit_positions.append(position)

    def advance_merge_stats(
        self,
        stats: LogStateMergeStats,
        tail : int,
    ) -> None:
        if tail <= stats.tail:
            return
        boundary = stats.next_update_tail
        if boundary is None:
            boundary = stats.origin + self.min_sampling_len
        while boundary <= tail:
            fitted = self._fit_merge_intensity(stats, boundary)
            stats.intensity = (
                fitted if fitted is not None else
                min(
                    self.max_merge_intensity,
                    stats.intensity * self.default_merge_growth_rate,
                )
            )
            stats.hit_positions = deque(
                position for position in stats.hit_positions
                if position >= boundary
            )
            boundary += self.min_sampling_len
        stats.next_update_tail = boundary
        stats.tail = tail

    def _fit_merge_intensity(
        self,
        stats: LogStateMergeStats,
        boundary: int,
    ) -> int | None:
        bins: dict[int, int] = {}
        for position in stats.hit_positions:
            if not boundary - self.min_sampling_len <= position < boundary:
                continue
            distance = max(self.granularity, boundary - position)
            index = (distance - 1) // self.page_size
            bins[index] = bins.get(index, 0) + 1
        if len(bins) < 2:
            return None

        xs = [math.log((index + 0.5) * self.page_size) for index in bins]
        ys = [math.log(count / self.page_size) for count in bins.values()]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        denominator = sum((value - x_mean) ** 2 for value in xs)
        if denominator == 0:
            return None
        slope = sum(
            (x - x_mean) * (y - y_mean)
            for x, y in zip(xs, ys, strict=True)
        ) / denominator
        fitted = math.floor(-slope)
        return min(self.max_merge_intensity, max(2, fitted))

    def begin_round(self, tree: LogRadixTree) -> LogStateRound:
        return LogStateRound(
            frozenset(entry.slot for entry in tree.iter_entries())
        )

    def analyze(
        self,
        tree      : LogRadixTree,
        input_ids : Sequence[torch.Tensor],
        round_info: LogStateRound | None = None,
        *,
        collect_diagnostics: bool = False,
        step_timing_ms: dict[str, float] | None = None,
    ) -> LogStatePlan:
        assert tree.page_size == self.page_size
        assert tree.granularity == self.granularity
        if round_info is None:
            round_info = self.begin_round(tree)

        step_start = perf_counter_ns() if step_timing_ms is not None else 0
        analysis_nodes, endpoints = self._analysis_paths(tree, input_ids)
        if step_timing_ms is not None:
            step_timing_ms["matched_paths_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6
            step_start = perf_counter_ns()
        pending = self._find_protections(
            tree, input_ids, step_timing_ms=step_timing_ms,
        )
        if step_timing_ms is not None:
            step_timing_ms["protection_total_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6
            step_start = perf_counter_ns()
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
        restores = tuple(
            restore
            for value in pending
            if (restore := self._plan_work_state_restore(tree, value)) is not None
        )
        if step_timing_ms is not None:
            step_timing_ms["protection_and_restore_plan_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6
            step_start = perf_counter_ns()
        materialize = self._plan_materialize(
            tree, round_info.eligible_slots, pending, analysis_nodes
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
            analysis_nodes,
            endpoints,
            diagnostics,
        )
        if step_timing_ms is not None:
            step_timing_ms["coalesce_plan_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6
        return LogStatePlan(
            tree_revision=tree.revision,
            protections=tuple(protections),
            restores=restores,
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
        input_ids: Sequence[torch.Tensor],
    ) -> tuple[
        tuple[LogRadixTreeNode, ...],
        tuple[_AnalysisEndpoint, ...],
    ]:
        """Return reached path nodes and the exact matched endpoints.

        Only the matched path is part of the current analysis.  Descendant and
        sibling continuations are not maintained until a later prefill request
        actually reaches them.  An entirely new request reaches only the root
        and must not trigger unrelated cache work.  Keep each endpoint's
        matched length separately: a trace can end inside a compressed edge,
        and maintenance must not inspect entries beyond that hit.
        """
        nodes: dict[int, LogRadixTreeNode] = {}
        endpoints: dict[tuple[int, int], _AnalysisEndpoint] = {}
        for values in input_ids:
            usable_len = align_down(len(values), self.granularity)
            trace = tree.trace(values[:usable_len])
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
        step_timing_ms: dict[str, float] | None = None,
    ) -> tuple[_PendingProtection, ...]:
        branch_start = perf_counter_ns() if step_timing_ms is not None else 0
        candidates = self._batch_branch_points(input_ids)
        if step_timing_ms is not None:
            step_timing_ms["batch_branch_points_ms"] = (
                perf_counter_ns() - branch_start
            ) / 1e6

        for request_index, values in enumerate(input_ids):
            usable_len = align_down(len(values), self.granularity)
            trace = tree.trace(values[:usable_len])
            if trace.matched_len == 0:
                continue
            ends_in_existing_edge = not trace.is_exact
            leaves_existing_route = (
                trace.is_exact
                and trace.matched_len <= usable_len
                and len(trace.node.children) > 0
            )
            if ends_in_existing_edge or leaves_existing_route:
                key = tuple(values[:trace.matched_len].tolist())
                request_indices = candidates.setdefault(
                    key, set()
                )
                request_indices.add(request_index)

        pending = []
        for key, request_indices in candidates.items():
            request_index = min(request_indices)
            position = len(key)
            if position == 0 or position % self.granularity != 0:
                continue
            values = input_ids[request_index]
            trace = tree.trace(values[:position])
            pending.append(
                _PendingProtection(
                    request_indices=tuple(sorted(request_indices)),
                    request_index=request_index,
                    position=position,
                    trace=trace,
                )
            )
        pending.sort(key=lambda value: (value.position, value.request_indices))
        return tuple(pending)

    def _batch_branch_points(
        self,
        input_ids: Sequence[torch.Tensor],
    ) -> dict[tuple[int, ...], set[int]]:
        root = _BatchPrefixNode(depth=0)
        for request_index, values in enumerate(input_ids):
            usable_len = align_down(len(values), self.granularity)
            node = root
            node.request_indices.append(request_index)
            for begin in range(0, usable_len, self.granularity):
                key = tuple(values[begin : begin + self.granularity].tolist())
                child = node.children.get(key)
                if child is None:
                    child = _BatchPrefixNode(depth=begin + self.granularity)
                    node.children[key] = child
                node = child
                node.request_indices.append(request_index)
            node.terminal_count += 1

        branches: dict[tuple[int, ...], set[int]] = {}
        stack: list[tuple[_BatchPrefixNode, tuple[int, ...]]] = [(root, ())]
        while stack:
            node, prefix = stack.pop()
            routes = len(node.children) + int(node.terminal_count > 0)
            if node.depth > 0 and (routes >= 2 or node.terminal_count >= 2):
                branches[prefix] = set(node.request_indices)
            for key, child in node.children.items():
                stack.append((child, prefix + key))
        return branches

    def _plan_work_state_restore(
        self,
        tree   : LogRadixTree,
        pending: _PendingProtection,
    ) -> WorkStateRestore | None:
        entries = self._entries_for_trace(tree, pending.trace)
        checkpoints = [
            entry
            for entry in entries
            if entry.kind is LogStateKind.CHECKPOINT
            and entry.end <= pending.position
        ]
        exact = [entry for entry in checkpoints if entry.end == pending.position]
        if exact:
            return None

        base = max(checkpoints, key=lambda entry: entry.end, default=None)
        base_end = 0 if base is None else base.end
        logs, covered_end = self._cover_with_logs(
            entries, base_end, pending.position
        )
        mode = (
            WorkStateRestoreMode.LOG_REPLAY
            if covered_end == pending.position
            else WorkStateRestoreMode.PREFILL
        )
        return WorkStateRestore(
            request_indices=pending.request_indices,
            position=pending.position,
            replay_end=covered_end,
            mode=mode,
            base_checkpoint_slot=None if base is None else base.slot,
            logs=tuple(logs),
        )

    def _plan_materialize(
        self,
        tree          : LogRadixTree,
        eligible_slots: frozenset[int],
        pending       : tuple[_PendingProtection, ...],
        analysis_nodes: tuple[LogRadixTreeNode, ...],
    ) -> tuple[PageMaterialize, ...]:
        result = []
        entries = {
            entry.slot: entry
            for node in analysis_nodes
            for entry in node.entries
        }
        for entry in entries.values():
            if entry.slot not in eligible_slots or entry.kind is not LogStateKind.LOG:
                continue
            if tree.slot_is_fragmented(entry.slot):
                continue
            if (entry.end - entry.begin) % self.page_size != 0:
                # Prefix deduplication can leave a single logical suffix of a
                # physical Log page.  It remains useful for replay, but the
                # persistent Checkpoint representation is page-sized.
                continue
            owner = tree.entry_owner(entry)
            if self._interval_is_protected(tree, owner, entry.begin, entry.end, pending):
                continue
            path_entries = self._entries_for_node(tree, owner)
            checkpoints = [
                value
                for value in path_entries
                if value.kind is LogStateKind.CHECKPOINT
                and value.end <= entry.begin
            ]
            base = max(checkpoints, key=lambda value: value.end, default=None)
            base_end = 0 if base is None else base.end
            logs, covered_end = self._cover_with_logs(
                path_entries, base_end, entry.end
            )
            if covered_end != entry.end:
                continue
            result.append(
                PageMaterialize(
                    log_slot=entry.slot,
                    begin=entry.begin,
                    end=entry.end,
                    base_checkpoint_slot=None if base is None else base.slot,
                    replay_logs=tuple(logs),
                )
            )
        result.sort(key=lambda value: (value.begin, value.end, value.log_slot))
        return tuple(result)

    def _plan_coalesce(
        self,
        tree          : LogRadixTree,
        eligible_slots: frozenset[int],
        pending       : tuple[_PendingProtection, ...],
        materialize   : tuple[PageMaterialize, ...],
        analysis_nodes: tuple[LogRadixTreeNode, ...],
        endpoints     : tuple[_AnalysisEndpoint, ...],
        diagnostics   : dict[str, int] | None,
    ) -> tuple[CheckpointCoalesce, ...]:
        projected = {
            plan.log_slot: LogStateEntry(
                slot=plan.log_slot,
                kind=LogStateKind.CHECKPOINT,
                begin=plan.begin,
                end=plan.end,
                span_pages=1,
            )
            for plan in materialize
        }
        result = []
        used_slots: set[int] = set()
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
                "entries_outside_adaptive_owner": 0,
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
                "rejected_level_mismatch": 0,
                "rejected_span_mismatch": 0,
                "rejected_discontinuity": 0,
                "rejected_lock_boundary": 0,
                "rejected_protected_interval": 0,
                "planned_groups": 0,
                "planned_slot_releases": 0,
            })
        # Every matched endpoint is an anchor, including an internal node
        # without an adaptive owner. Owner nodes on the matched path retain
        # their own statistics; neither kind may inspect an unmatched suffix.
        anchor_limits: dict[int, tuple[LogRadixTreeNode, int]] = {}
        for endpoint in endpoints:
            for node in tree.path(endpoint.node):
                if node is not endpoint.node and (
                    not self.enable_merge_adjustment
                    or node.merge_stats is None
                ):
                    continue
                limit = min(node.depth, endpoint.matched_len)
                previous = anchor_limits.get(node.uuid)
                if previous is None or limit > previous[1]:
                    anchor_limits[node.uuid] = (node, limit)
        anchors = tuple(anchor_limits.values())

        for leaf, matched_len in anchors:
            if diagnostics is not None:
                diagnostics["merge_anchors"] += 1
            stats = leaf.merge_stats
            path_entries = self._entries_for_node(tree, leaf)
            if matched_len is not None:
                in_match = [
                    entry for entry in path_entries
                    if entry.end <= matched_len
                ]
                if diagnostics is not None:
                    diagnostics["entries_beyond_match_endpoint"] += (
                        len(path_entries) - len(in_match)
                    )
                path_entries = in_match
            if diagnostics is not None:
                diagnostics["entries_not_eligible_this_round"] += sum(
                    entry.slot not in eligible_slots for entry in path_entries
                )
            path_entries = [
                entry for entry in path_entries
                if entry.slot in eligible_slots
            ]
            if stats is not None:
                if diagnostics is not None:
                    diagnostics["entries_outside_adaptive_owner"] += sum(
                        not (
                            stats.origin <= entry.begin
                            and entry.end <= stats.tail
                        )
                        for entry in path_entries
                    )
            entries = [
                projected.get(entry.slot, entry)
                for entry in path_entries
                if entry.slot not in used_slots
                and (
                    stats is None
                    or stats.origin <= entry.begin
                    and entry.end <= stats.tail
                )
            ]
            if diagnostics is not None:
                diagnostics["logs_not_materialized"] += sum(
                    entry.kind is LogStateKind.LOG for entry in entries
                )
            entries = [
                entry
                for entry in entries
                if entry.kind is LogStateKind.CHECKPOINT
                and entry.span_pages is not None
            ]
            if diagnostics is not None:
                diagnostics["checkpoint_candidates"] += len(entries)
            if not entries:
                if diagnostics is not None:
                    diagnostics["anchors_without_eligible_entries"] += 1
                continue
            entries.sort(key=lambda entry: (entry.begin, entry.end, entry.slot))
            intensity = (
                stats.intensity
                if self.enable_merge_adjustment and stats is not None else
                2
            )
            segments = self._merge_segments(
                tree, leaf, entries, pending, diagnostics
            )
            if diagnostics is not None:
                diagnostics["segments"] += len(segments)
                diagnostics["segments_without_merge_pair"] += sum(
                    len(segment) < 2 for segment in segments
                )
            groups = tuple(
                group
                for segment in segments
                for group in (
                    self._adaptive_merge_groups(segment, intensity)
                    if self.enable_merge_adjustment else
                    tuple(
                        segment[index : index + 2]
                        for index in range(len(segment) - 1)
                    )
                )
            )
            for group in groups:
                if diagnostics is not None:
                    diagnostics["candidate_groups"] += 1
                if any(entry.slot in used_slots for entry in group):
                    if diagnostics is not None:
                        diagnostics["groups_already_planned"] += 1
                    continue
                reason = self._coalesce_rejection_reason(
                    tree, leaf, group, pending,
                    require_equal_level=True,
                )
                if reason is not None:
                    if diagnostics is not None:
                        diagnostics[f"rejected_{reason}"] += 1
                    continue
                result.append(self._make_coalesce(group))
                used_slots.update(entry.slot for entry in group)
                if diagnostics is not None:
                    diagnostics["planned_groups"] += 1
                    diagnostics["planned_slot_releases"] += len(group) - 1

            if not self.enable_merge_adjustment:
                continue
            remaining = [
                entry for entry in entries if entry.slot not in used_slots
            ]
            for left, right in zip(remaining, remaining[1:]):
                if left.slot in used_slots or right.slot in used_slots:
                    continue
                assert left.span_pages is not None
                assert right.span_pages is not None
                if left.span_pages >= right.span_pages:
                    continue
                group = (left, right)
                if diagnostics is not None:
                    diagnostics["candidate_groups"] += 1
                reason = self._coalesce_rejection_reason(
                    tree, leaf, group, pending,
                    require_equal_level=False,
                )
                if reason is not None:
                    if diagnostics is not None:
                        diagnostics[f"rejected_{reason}"] += 1
                    continue
                result.append(self._make_coalesce(group))
                used_slots.update((left.slot, right.slot))
                if diagnostics is not None:
                    diagnostics["planned_groups"] += 1
                    diagnostics["planned_slot_releases"] += 1
        result.sort(key=lambda value: (value.begin, value.end, value.left_slot))
        return tuple(result)

    def _merge_segments(
        self,
        tree   : LogRadixTree,
        leaf   : LogRadixTreeNode,
        entries: Sequence[LogStateEntry],
        pending: tuple[_PendingProtection, ...],
        diagnostics: dict[str, int] | None,
    ) -> tuple[tuple[LogStateEntry, ...], ...]:
        result = []
        current = []
        for entry in entries:
            entry_is_blocked = self._interval_is_protected(
                tree, leaf, entry.begin, entry.end, pending
            )
            if diagnostics is None:
                boundary_is_blocked = bool(current) and (
                    current[-1].end != entry.begin
                    or tree.is_lock_boundary(leaf, current[-1].end)
                    or self._interval_is_protected(
                        tree, leaf, current[-1].begin, entry.end, pending
                    )
                )
            else:
                gap = bool(current) and current[-1].end != entry.begin
                lock_boundary = bool(current) and tree.is_lock_boundary(
                    leaf, current[-1].end
                )
                protected_boundary = bool(current) and self._interval_is_protected(
                    tree, leaf, current[-1].begin, entry.end, pending
                )
                diagnostics["protected_entries"] += int(entry_is_blocked)
                diagnostics["gaps"] += int(gap)
                diagnostics["lock_boundaries"] += int(lock_boundary)
                diagnostics["protected_boundaries"] += int(protected_boundary)
                boundary_is_blocked = gap or lock_boundary or protected_boundary
            if entry_is_blocked or boundary_is_blocked:
                if current:
                    result.append(tuple(current))
                current = []
            if not entry_is_blocked:
                current.append(entry)
        if current:
            result.append(tuple(current))
        return tuple(result)

    @staticmethod
    def _adaptive_merge_groups(
        entries  : Sequence[LogStateEntry],
        intensity: int,
    ) -> tuple[tuple[LogStateEntry, ...], ...]:
        assert intensity >= 2
        result = []
        index = 0
        while index < len(entries):
            level = entries[index].span_pages
            end = index + 1
            while end < len(entries) and entries[end].span_pages == level:
                end += 1
            run = entries[index:end]
            offset = 0
            while len(run) - offset >= intensity:
                result.append(tuple(run[offset : offset + intensity]))
                offset += intensity
            remaining = len(run) - offset
            if remaining >= 3:
                result.append(tuple(run[offset : len(run) - 1]))
            index = end
        return tuple(result)

    @staticmethod
    def _make_coalesce(
        entries   : Sequence[LogStateEntry],
    ) -> CheckpointCoalesce:
        return CheckpointCoalesce(
            slots=tuple(entry.slot for entry in entries),
            source_spans=tuple(
                entry.span_pages for entry in entries  # type: ignore[arg-type]
            ),
            begin=entries[0].begin,
            end=entries[-1].end,
            next_span=sum(
                entry.span_pages for entry in entries  # type: ignore[arg-type]
            ),
        )

    def _can_coalesce(
        self,
        tree   : LogRadixTree,
        leaf   : LogRadixTreeNode,
        entries: Sequence[LogStateEntry],
        pending: tuple[_PendingProtection, ...],
        *,
        require_equal_level: bool,
    ) -> bool:
        return self._coalesce_rejection_reason(
            tree,
            leaf,
            entries,
            pending,
            require_equal_level=require_equal_level,
        ) is None

    def _coalesce_rejection_reason(
        self,
        tree               : LogRadixTree,
        leaf               : LogRadixTreeNode,
        entries            : Sequence[LogStateEntry],
        pending            : tuple[_PendingProtection, ...],
        *,
        require_equal_level: bool,
    ) -> str | None:
        if len(entries) < 2:
            return "too_few_entries"
        levels = [entry.span_pages for entry in entries]
        assert all(level is not None for level in levels)
        if require_equal_level and len(set(levels)) != 1:
            return "level_mismatch"
        if any(
            entry.end - entry.begin != self.page_size * span
            for entry, span in zip(entries, levels, strict=True)
        ):
            return "span_mismatch"
        for left, right in zip(entries, entries[1:]):
            if left.end != right.begin:
                return "discontinuity"
            if tree.is_lock_boundary(leaf, left.end):
                return "lock_boundary"
        if self._interval_is_protected(
            tree, leaf, entries[0].begin, entries[-1].end, pending
        ):
            return "protected_interval"
        return None

    def _interval_is_protected(
        self,
        tree   : LogRadixTree,
        node   : LogRadixTreeNode,
        begin  : int,
        end    : int,
        pending: tuple[_PendingProtection, ...],
    ) -> bool:
        if any(
            path_node.is_protected and begin < path_node.depth < end
            for path_node in tree.path(node)
        ):
            return True
        entries = [
            entry
            for entry in self._entries_for_node(tree, node)
            if begin <= entry.begin and entry.end <= end
        ]
        if any(tree.entry_is_protected(entry.slot) for entry in entries):
            return True
        path_ids = {value.uuid for value in tree.path(node)}
        for value in pending:
            if not begin < value.position < end:
                continue
            if value.trace.node.uuid not in path_ids:
                continue
            return True
        return False

    def _covering_log(
        self,
        tree   : LogRadixTree,
        pending: _PendingProtection,
    ) -> LogStateEntry | None:
        entries = self._entries_for_trace(tree, pending.trace)
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

    def _entries_for_trace(
        self,
        tree : LogRadixTree,
        trace: LogRadixTrace,
    ) -> list[LogStateEntry]:
        return self._entries_for_node(tree, trace.node)

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

    @staticmethod
    def _cover_with_logs(
        entries: Sequence[LogStateEntry],
        begin  : int,
        end    : int,
    ) -> tuple[list[LogStateEntry], int]:
        cursor = begin
        result = []
        logs = sorted(
            (entry for entry in entries if entry.kind is LogStateKind.LOG),
            key=lambda entry: (entry.begin, entry.end),
        )
        for entry in logs:
            if entry.end <= cursor or entry.begin > cursor:
                continue
            result.append(entry)
            cursor = min(end, entry.end)
            if cursor == end:
                break
        return result, cursor


class LogStateCacheManager(BaseCacheManager):
    """Own LogState cache allocation and request/tree ownership transitions.

    ``CacheResources`` owns the device buffers. This class owns their allocation
    state, the Log-Radix-Tree, and the online maintenance policy. Branch recovery
    writes request work states, while private Log chains are materialized in
    place by the CUDA construction kernel.
    """

    def __init__(
        self,
        num_pages                : int,
        num_states               : int,
        num_work_states          : int,
        page_size                : int,
        resources                : CacheResources,
        granularity              : int  = LOG_STATE_GRANULARITY,
        enable_memorized_evict   : bool = True,
        enable_merge_adjustment  : bool = False,
        min_sampling_len         : int  = 1024,
        max_merge_intensity      : int  = 16,
        default_merge_growth_rate: int = 2,
    ) -> None:
        pool = resources.linear_state
        state_table = resources.state_table
        work_state_table = resources.work_state_table
        assert isinstance(pool, FramedMHALogStatePool)
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
        )
        super().__init__(
            page_size=page_size,
            resources=resources,
            prefix_cache=tree,
        )
        self.analyzer = OnlineAnalysisScheduler(
            page_size=page_size,
            granularity=granularity,
            enable_merge_adjustment=enable_merge_adjustment,
            min_sampling_len=min_sampling_len,
            max_merge_intensity=max_merge_intensity,
            default_merge_growth_rate=default_merge_growth_rate,
        )
        self.state_units = (
            pool.allocator.payload_frames
            + pool.layout.num_granules * pool.allocator.conv_frames
            + 1
        )
        self.kv_units_per_page = 1
        self._allocator = pool.allocator
        self.enable_memorized_evict = enable_memorized_evict
        self.recycle_bin = (
            LogStateRecycleBin(64 * 1024 * 1024)
            if enable_memorized_evict else None
        )
        self.num_pages = num_pages
        self.num_states = num_states
        self.num_work_states = num_work_states
        self._work_slots: dict[int, int] = {}
        self._work_committed_pos: dict[int, int] = {}
        self._recycle_matches: dict[int, RecycledMatch] = {}
        self._recycle_starts: dict[int, int] = {}
        self._recycle_rebuilt_slots: dict[int, set[int]] = {}
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
        self.last_plan: LogStatePlan | None = None
        self._live_pages: dict[int, list[_LivePage]] = {}
        self._cached_page_tokens: dict[int, int] = {}
        self._lazy_free_pages: list[torch.Tensor] | None = None
        self._lazy_free_states: list[torch.Tensor] | None = None
        self._profile_enabled = os.getenv(
            "LOGSSMS_PROFILE_SCHEDULER", "0"
        ).lower() in ("1", "true", "yes")
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
            "evicted_kv_tokens": 0,
            "recycle_matches": 0,
            "recycle_matched_tokens": 0,
            "released_state_slots_from_eviction": 0,
            "released_state_slots_from_merge": 0,
            "scheduled_payload_conv_frames_from_merge": 0,
            "merge_batch_outcomes": {},
            "merge_plan_diagnostics": {},
            "merge_commit_skips": {},
        }
        self._profile_samples: list[dict] = []
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
    def logstate_pool(self) -> FramedMHALogStatePool:
        pool = self.linear_pool
        assert isinstance(pool, FramedMHALogStatePool)
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
        result = super().match_req(req)
        handle = result.cuda_handle
        assert isinstance(handle, LogStateCacheHandle)
        values = req.input_ids[: req.input_len - 1]
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
                recycle_match.matched_len - handle.cached_len
            )
        if not self.enable_merge_adjustment:
            return MatchResult(
                replace(handle, recycle_match=recycle_match),
                tree_match_len=result.tree_match_len,
                recoverable_len=result.recoverable_len,
            )
        usable_len = align_down(len(values), self.granularity)
        trace = self.tree.trace(values[:usable_len])
        return MatchResult(
            replace(
                handle,
                merge_stats=self.analyzer.make_merge_stats(trace.matched_len),
                source_merge_stats=trace.node.merge_stats,
                match_position=trace.matched_len,
                recycle_match=recycle_match,
            ),
            tree_match_len=result.tree_match_len,
            recoverable_len=result.recoverable_len,
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
        admitted = super().prefill_acquire(total_len, handle)
        if not admitted:
            if self._profile_enabled:
                self._profile_stats["admission_rejections"] += 1
            return False
        if not self.enable_merge_adjustment:
            return True
        assert isinstance(handle, LogStateCacheHandle)
        if handle.source_merge_stats is not None and handle.match_position > 0:
            self.analyzer.record_hit(
                handle.source_merge_stats, handle.match_position
            )
        return True

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
        stats["max_construction_workspace_bytes"] = max(
            (value["workspace_bytes"] for value in constructions), default=0
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
            phase["eviction_calls"] += int(sample["evicted_nodes"] > 0)
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
                "frame_bytes": self._allocator.frame_bytes,
                "payload_frames_per_slot": self._allocator.payload_frames,
                "conv_frames_per_snapshot": self._allocator.conv_frames,
                "granules_per_log": self._allocator.num_granules,
                "decay_records_per_frame": self._allocator.decay_records_per_frame,
            },
            "aggregate": stats,
            "match_samples": self._match_profile_samples,
            "prefill_batches": self._profile_samples,
            "allocation_profile": {
                "phases": allocation_phases,
                "samples": self._profile_allocation_samples,
            },
            "lifecycle_events": self._profile_lifecycle_events,
        }

    @property
    def available_size(self) -> int:
        self.logstate_pool.poll_reclamation()
        return self._allocator.admissible_pages(
            evictable_kv_frames=self.tree.evictable_kv_frames(
                self._cached_page_tokens
            ),
            evictable_slots=self.tree.evictable_slots(),
        ) * self.page_size

    def plan_prefill_chunk(self, remain_len: int, token_budget: int) -> int:
        chunk_size = min(remain_len, token_budget)
        if chunk_size < remain_len:
            chunk_size = align_down(chunk_size, self.page_size)
        return chunk_size

    def init_with_handle(self, table_idx: int, handle: BaseCacheHandle) -> None:
        assert isinstance(handle, LogStateCacheHandle)
        assert table_idx not in self._work_slots
        work_slot = self._allocate_work()
        self._work_slots[table_idx] = work_slot
        self._work_committed_pos[table_idx] = handle.cached_len
        work_state_table = self.work_state_table
        assert work_state_table is not None
        work_state_table[table_idx, 0] = work_slot
        if handle.cached_len > 0:
            self.page_table[table_idx, : handle.cached_len].copy_(
                handle.kv_indices
            )
        state_table = self.state_table
        assert state_table is not None
        state_table[table_idx].fill_(-1)
        self.pending_recoveries[table_idx] = handle.recovery
        self._live_pages[table_idx] = []
        if handle.recycle_match is not None:
            self._recycle_matches[table_idx] = handle.recycle_match
            self._recycle_starts[table_idx] = handle.cached_len
            self._recycle_rebuilt_slots[table_idx] = set()

    def _direct_recycled_checkpoints(
        self, req: Request, begin: int, batch_branches: set[tuple[int, ...]],
    ) -> dict[int, int]:
        match = self._recycle_matches.get(req.table_idx)
        if match is None:
            return {}
        result = {}
        for entry in match.branch.entries:
            if entry.kind is not LogStateKind.CHECKPOINT:
                continue
            if not (begin <= entry.begin < entry.end <= min(req.device_len, match.matched_len)):
                continue
            if entry.begin % self.page_size or entry.end % self.page_size:
                continue
            if any(entry.begin < pos < entry.end for pos in match.branch.protected_endpoints):
                continue
            if any(
                entry.begin < len(key) < entry.end
                and tuple(req.input_ids[:len(key)].tolist()) == key
                for key in batch_branches
            ):
                continue
            # A second request may already own a GPU branch within this
            # range.  Use the ordinary Log path rather than overwrite it.
            if self.tree.trace(req.input_ids[:entry.end]).matched_len > begin:
                continue
            result[entry.end] = entry.begin
        return result

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

        allocation_info: list[tuple[Request, int, int, dict[int, int]]] = []
        needed = 0
        needed_logs = 0
        needed_checkpoints = 0
        fresh_work_slots: list[int] = []
        batch_branches = (
            set(self.analyzer._batch_branch_points(
                [req.input_ids for req in reqs]
            )) if phase == "prefill" and any(
                req.table_idx in self._recycle_matches for req in reqs
            )
            else set()
        )
        for req in reqs:
            pages = self._live_pages.setdefault(req.table_idx, [])
            begin = pages[-1].end if pages else req.cached_len
            if req.device_len > begin:
                first_span = self.page_size - begin % self.page_size
                count = 1 + div_ceil(
                    max(0, req.device_len - begin - first_span), self.page_size
                )
                direct = (
                    self._direct_recycled_checkpoints(req, begin, batch_branches)
                    if phase == "prefill" else {}
                )
                checkpoint_pages = sum(
                    (end - start) // self.page_size
                    for end, start in direct.items()
                )
                allocation_info.append((req, begin, count, direct))
                needed += count
                needed_logs += count - checkpoint_pages
                needed_checkpoints += len(direct)

            recovery = self.pending_recoveries.get(req.table_idx)
            if req.cached_len == 0 and (recovery is None or recovery.position == 0):
                fresh_work_slots.append(self._work_slots[req.table_idx])

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

        pages, states, checkpoint_slots = self._allocate(
            needed, needed_logs, needed_checkpoints, profile=profile
        )
        if profile is not None:
            now = perf_counter_ns()
            profile["timing_ms"]["core_allocate_ms"] = (now - stage) / 1e6
            stage = now
        kv_pages = pages.tolist()
        state_slots = states.tolist()
        direct_slots = checkpoint_slots.tolist()
        if profile is not None:
            now = perf_counter_ns()
            profile["timing_ms"]["ids_to_host_ms"] = (now - stage) / 1e6
            stage = now
        offset = 0
        log_offset = 0
        checkpoint_offset = 0
        token_offsets = torch.arange(
            self.page_size, dtype=torch.int32, device=self.device
        )
        for req, begin, count, direct in allocation_info:
            live_pages = self._live_pages[req.table_idx]
            page_state_slots = []
            mapped_pages = []
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
                    state_slot = int(direct_slots[checkpoint_offset])
                    checkpoint_offset += 1
                    page_state_slots.append(-1)
                elif inside_checkpoint:
                    state_slot = None
                    page_state_slots.append(-1)
                else:
                    state_slot = int(state_slots[log_offset])
                    log_offset += 1
                    page_state_slots.append(state_slot)
                live_pages.append(
                    _LivePage(
                        begin=logical_begin,
                        end=logical_end,
                        kv_page=kv_page,
                        state_slot=state_slot,
                        checkpoint_begin=checkpoint_begin,
                    )
                )
                mapped_pages.append((kv_page, logical_end - logical_begin))
                logical_begin = logical_end
            mapped_end = min(
                logical_begin,
                req.max_device_len,
                self.page_table.shape[1],
            )
            mapped_len = mapped_end - begin
            assert live_pages[-1].begin < mapped_end <= live_pages[-1].end
            kv_mapping = torch.cat([
                kv_page + token_offsets[:span]
                for kv_page, span in mapped_pages
            ])[:mapped_len]
            state_mapping = torch.cat([
                torch.full((span,), slot, dtype=torch.int32, device=self.device)
                for slot, (_kv_page, span) in zip(
                    page_state_slots, mapped_pages, strict=True
                )
            ])[:mapped_len]
            self.page_table[req.table_idx, begin:mapped_end].copy_(kv_mapping)
            state_table[req.table_idx, begin:mapped_end].copy_(state_mapping)
        assert offset == needed
        assert log_offset == needed_logs
        assert checkpoint_offset == needed_checkpoints
        if profile is not None:
            now = perf_counter_ns()
            profile["timing_ms"]["mapping_write_ms"] = (now - stage) / 1e6
            profile["timing_ms"]["mapping_ms"] = (
                profile["timing_ms"]["ids_to_host_ms"]
                + profile["timing_ms"]["mapping_write_ms"]
            )
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
            "free_kv_pages": len(self.free_pages),
            "free_state_slots": len(self.free_states),
            "evictable_kv_tokens": self.tree.size_info.evictable_size,
            "cached_kv_tokens": self.tree.size_info.total_size,
            "evictable_kv_pages": (
                self.tree.size_info.evictable_size / self.page_size
            ),
            "evictable_state_slots": len(evictable_slots),
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
            "live_log_pages": sum(map(len, self._live_pages.values())),
        }

    def _profile_request_paths(
        self,
        reqs: Sequence[Request],
    ) -> list[dict]:
        result = []
        for req in reqs:
            recovery = self.pending_recoveries.get(req.table_idx)
            live = self._live_pages.get(req.table_idx, ())
            handle = req.cache_handle
            cached_entries = ()
            if isinstance(handle, LogStateCacheHandle):
                cached_entries = tuple(
                    entry
                    for node in self.tree.path(handle.node)
                    for entry in node.entries
                    if entry.end <= handle.cached_len
                )
            result.append({
                "uid": req.uid,
                "cached_len": req.cached_len,
                "device_len": req.device_len,
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
                    offset = next(
                        (req.cached_len - page.begin
                         for page in self._live_pages[req.table_idx]
                         if page.begin <= req.cached_len < page.end),
                        None,
                    )
                    assert offset is not None, "Decode token has no LogState page"
                    committed_pos = self._work_committed_pos[req.table_idx]
                    count = req.cached_len - committed_pos
                    assert 0 <= count < LOG_STATE_REPLAY_WINDOW
                    assert count <= offset % LOG_STATE_REPLAY_WINDOW
                    replay_counts.append(count)
                    if (offset + 1) % LOG_STATE_REPLAY_WINDOW == 0:
                        self._work_committed_pos[req.table_idx] = req.cached_len + 1
                batch.logstate_replay_counts = replay_counts + [
                    0 for _ in range(batch.padded_size - batch.size)
                ]
            return
        profile_start = perf_counter_ns() if self._profile_enabled else 0
        # The first physical offset is known from allocation. Pass it to the
        # linear backend instead of synchronizing out_loc from GPU per request.
        first_offsets = []
        for req in batch.reqs:
            page = next(
                (page for page in self._live_pages.get(req.table_idx, ())
                 if page.begin <= req.cached_len < page.end),
                None,
            )
            if page is None:
                # Planning-only callers may invoke prepare_batch before allocate.
                # The backend retains its physical-offset fallback for these.
                break
            first_offsets.append(req.cached_len - page.begin)
        batch.logstate_first_offsets = (
            first_offsets if len(first_offsets) == len(batch.reqs) else None
        )
        batch.logstate_checkpoint_writes = []
        batch.logstate_checkpoint_ranges = []
        row_offset = 0
        for req_index, req in enumerate(batch.reqs):
            for page in self._live_pages.get(req.table_idx, ()):
                if (
                    page.checkpoint_begin is not None
                    and req.cached_len < page.end <= req.device_len
                ):
                    assert page.state_slot is not None
                    batch.logstate_checkpoint_writes.append((
                        row_offset + page.end - req.cached_len - 1,
                        page.state_slot,
                    ))
                    batch.logstate_checkpoint_ranges.append((
                        req_index, page.checkpoint_begin, page.end,
                    ))
            row_offset += req.extend_len
        for req in batch.reqs:
            self._work_committed_pos[req.table_idx] = req.device_len
        first_offsets_cpu_ms = (
            (perf_counter_ns() - profile_start) / 1e6
            if self._profile_enabled else 0.0
        )
        analysis_start = perf_counter_ns() if self._profile_enabled else 0
        analysis_step_timing_ms: dict[str, float] | None = (
            {} if self._profile_enabled else None
        )
        round_start = perf_counter_ns() if self._profile_enabled else 0
        round_info = self.analyzer.begin_round(self.tree)
        if analysis_step_timing_ms is not None:
            analysis_step_timing_ms["eligible_tree_scan_ms"] = (
                perf_counter_ns() - round_start
            ) / 1e6
        recycle_start = perf_counter_ns() if self._profile_enabled else 0
        blocked = self._unmaterialized_recycle_slots()
        if analysis_step_timing_ms is not None:
            analysis_step_timing_ms["recycle_filter_ms"] = (
                perf_counter_ns() - recycle_start
            ) / 1e6
        if blocked:
            round_info = LogStateRound(round_info.eligible_slots - blocked)
        plan = self.analyzer.analyze(
            self.tree,
            [req.input_ids for req in batch.reqs],
            round_info,
            collect_diagnostics=self._profile_enabled,
            step_timing_ms=analysis_step_timing_ms,
        )
        for protection in plan.protections:
            if protection.node is not None:
                self.tree.protect(protection.node)
            if (
                protection.log_slot is not None
                and self.tree.has_slot(protection.log_slot)
            ):
                self.tree.protect_entry(protection.log_slot)

        self.last_plan = plan
        task_start = perf_counter_ns() if self._profile_enabled else 0
        tasks = self._make_restore_tasks(batch)
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
                    "restores": len(tasks),
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
        construction = self._construct(tasks, plan.materialize)
        construction_submit_cpu_ms = (
            (perf_counter_ns() - construction_start) / 1e6
            if self._profile_enabled else 0.0
        )

        commit_start = perf_counter_ns() if self._profile_enabled else 0
        for req in batch.reqs:
            self.pending_recoveries.pop(req.table_idx, None)
        self._commit_materialize(plan.materialize)
        released, released_frames, commit_skips = self._commit_coalesce(plan.coalesce)
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
            sample["materialized_slots"] = len(plan.materialize)
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

    def _unmaterialized_recycle_slots(self) -> frozenset[int]:
        blocked = set()
        for row, match in self._recycle_matches.items():
            tokens = torch.tensor(match.branch.tokens, dtype=torch.int32)
            checkpoints = tuple(
                entry for entry in match.branch.entries
                if entry.kind is LogStateKind.CHECKPOINT
                and entry.end <= match.matched_len
                and self.tree.trace(tokens[:entry.end]).matched_len < entry.end
            )
            for slot in self._recycle_rebuilt_slots[row]:
                if not self._slot_on_recycle_route(slot, match):
                    continue
                if any(
                    entry.kind is LogStateKind.LOG
                    and any(
                        checkpoint.begin <= entry.begin
                        and entry.end <= checkpoint.end
                        for checkpoint in checkpoints
                    )
                    for entry in self.tree.entries_for_slot(slot)
                ):
                    blocked.add(slot)
        return frozenset(blocked)

    def _slot_on_recycle_route(
        self, slot: int, match: RecycledMatch,
    ) -> bool:
        if not self.tree.has_slot(slot):
            return False
        for entry in self.tree.entries_for_slot(slot):
            owner = self.tree.entry_owner(entry)
            tokens = tuple(
                token for node in self.tree.path(owner)
                for token in node.key.tolist()
            )
            if tokens != match.branch.tokens[:len(tokens)]:
                return False
        return True

    def cache_req(self, req: Request, *, finished: bool) -> None:
        old_handle = req.cache_handle
        assert isinstance(old_handle, LogStateCacheHandle)
        pages = self._live_pages.get(req.table_idx, [])
        full_pages = [page for page in pages if page.end <= req.cached_len]
        insert_len = old_handle.cached_len
        for page in full_pages:
            if page.begin != insert_len:
                break
            insert_len = page.end
        assert insert_len % self.granularity == 0

        insert = self.tree.insert(
            req.input_ids[: req.cached_len],
            self.page_table[req.table_idx, : req.cached_len],
            cache_len=insert_len,
        )
        adopted = [
            page
            for page in full_pages
            if old_handle.cached_len <= page.begin
            and page.end <= insert.inserted_len
        ]
        existing = self.tree.match_prefix(
            req.input_ids[: insert.inserted_len]
        ).cuda_handle
        assert isinstance(existing, LogStateCacheHandle)
        covered_len = existing.cached_len
        released_kv_pages = []
        released_state_slots = []
        for page in adopted:
            logical_begin = max(page.begin, covered_len)
            if page.checkpoint_begin is not None and logical_begin < page.end:
                assert page.state_slot is not None
                assert page.checkpoint_begin >= covered_len
                self.logstate_pool.commit_checkpoint(page.state_slot)
                self.tree.attach_entry(
                    insert.node,
                    LogStateEntry(
                        slot=page.state_slot,
                        kind=LogStateKind.CHECKPOINT,
                        begin=page.checkpoint_begin,
                        end=page.end,
                        span_pages=(page.end - page.checkpoint_begin) // self.page_size,
                    ),
                )
                covered_len = page.end
            elif page.state_slot is not None and logical_begin < page.end:
                self.tree.attach_entry(
                    insert.node,
                    LogStateEntry(
                        slot=page.state_slot,
                        kind=LogStateKind.LOG,
                        begin=logical_begin,
                        end=page.end,
                        storage_begin=page.begin,
                    ),
                )
                covered_len = page.end
            elif page.state_slot is not None:
                released_state_slots.append(page.state_slot)

            kv_begin = max(page.begin, insert.previous_len)
            if kv_begin < page.end:
                self._retain_cached_kv(
                    self.page_table[req.table_idx, kv_begin:page.end]
                )
            else:
                released_kv_pages.append(page.kv_page)

        duplicates = [
            page
            for page in full_pages
            if page not in adopted and page.end <= insert.previous_len
        ]
        self._release_live_pages(req.table_idx, duplicates)
        if released_kv_pages or released_state_slots:
            self._return_slots(
                torch.tensor(
                    released_kv_pages,
                    dtype=torch.int32,
                    device=self.device,
                ),
                torch.tensor(
                    released_state_slots,
                    dtype=torch.int32,
                    device=self.device,
                ),
            )

        if adopted:
            self._restore_recycled_layout(
                req, insert.node, insert.previous_len, insert.inserted_len,
                old_handle.cached_len, adopted,
            )

        stats = old_handle.merge_stats
        if stats is not None:
            self.analyzer.advance_merge_stats(stats, insert.inserted_len)
            if insert.previous_len < insert.inserted_len:
                if old_handle.node.merge_stats is stats:
                    old_handle.node.merge_stats = None
                assert insert.node.merge_stats in (None, stats)
                insert.node.merge_stats = stats
        new_handle = replace(
            self.tree.make_handle(insert.node, insert.inserted_len),
            merge_stats=stats,
        )
        self.unlock(old_handle)
        self.pending_recoveries.pop(req.table_idx, None)

        state_table = self.state_table
        assert state_table is not None
        state_table[req.table_idx, : insert.inserted_len].fill_(-1)
        self.page_table[
            req.table_idx, : insert.inserted_len
        ].copy_(new_handle.kv_indices)

        remaining = [
            page for page in pages if page not in adopted and page not in duplicates
        ]
        if (
            not finished
            and adopted
            and req.cached_len == adopted[-1].end
        ):
            work_state_table = self.work_state_table
            assert work_state_table is not None
            work_slot = int(work_state_table[req.table_idx, 0])
            assert adopted[-1].state_slot is not None
            source = torch.tensor(
                [adopted[-1].state_slot], dtype=torch.int32, device=self.device
            )
            destination = torch.tensor(
                [work_slot], dtype=torch.int32, device=self.device
            )
            source_granule = torch.tensor(
                [(adopted[-1].end - adopted[-1].begin - 1) // self.granularity],
                dtype=torch.int32, device=self.device,
            )
            zero = torch.zeros(1, dtype=torch.int32, device=self.device)
            for layer in range(self.logstate_pool.layout.num_linear_layers):
                self.logstate_pool.write_conv(
                    destination, zero, layer,
                    self.logstate_pool.read_conv(source, source_granule, layer),
                )
        if finished:
            recycled = self._recycle_matches.pop(req.table_idx, None)
            if (
                recycled is not None
                and insert.inserted_len >= recycled.matched_len
                and self.recycle_bin is not None
            ):
                # A completed GPU route supersedes the remembered CPU shape.
                # Keep it until commit: a pending match can still fail admission.
                self.recycle_bin.discard(recycled.branch)
            self._recycle_starts.pop(req.table_idx, None)
            self._recycle_rebuilt_slots.pop(req.table_idx, None)
            self._release_live_pages(req.table_idx, remaining)
            self._live_pages.pop(req.table_idx, None)
            state_table[req.table_idx].fill_(-1)
            work_slot = self._work_slots.pop(req.table_idx)
            self._work_committed_pos.pop(req.table_idx)
            self._return_slots(
                torch.empty(0, dtype=torch.int32, device=self.device),
                torch.tensor([work_slot], dtype=torch.int32, device=self.device),
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
            self._live_pages[req.table_idx] = remaining

    def _restore_recycled_layout(
        self,
        req         : Request,
        leaf        : LogRadixTreeNode,
        previous_len: int,
        inserted_len: int,
        cached_len  : int,
        adopted     : Sequence[_LivePage],
    ) -> None:
        """Replay an evicted shape after prefill has produced fresh Log values.

        The recycle bin contains no GPU data.  A recovered checkpoint is
        therefore constructed from the just-computed logs; only intervals
        whose complete physical pages were inserted are eligible.  New text
        beyond the ghost hit remains Log until a later prefill round.
        """
        match = self._recycle_matches.get(req.table_idx)
        if match is None or previous_len >= match.matched_len:
            return
        limit = min(inserted_len, match.matched_len)
        if limit <= cached_len:
            return
        # A remembered branch point is a topology constraint, not a cached
        # tensor. Re-create the split on the newly computed route before
        # considering any checkpoint conversion across that boundary.
        for position in sorted(match.branch.protected_endpoints):
            if not previous_len < position <= limit:
                continue
            trace = self.tree.trace(req.input_ids[:position])
            if trace.matched_len != position:
                continue
            node = trace.node
            if not trace.is_exact:
                node = self.tree.split_at(node, trace.matched_in_node)
            if not node.is_root():
                self.tree.protect(node)
        fresh_slots = self._recycle_rebuilt_slots[req.table_idx]
        fresh_slots.intersection_update(
            slot for slot in fresh_slots
            if self._slot_on_recycle_route(slot, match)
        )
        fresh_slots.update({
            page.state_slot for page in adopted
            if page.state_slot is not None
            and page.end <= limit and page.begin >= previous_len
        })
        if not fresh_slots:
            return
        old_slots = {
            entry.slot for entry in self.tree.matched_entries(req.cache_handle.node)
            if entry.end <= cached_len
        }
        targets = sorted(
            (
                entry for entry in match.branch.entries
                if entry.kind is LogStateKind.CHECKPOINT
                and self._recycle_starts[req.table_idx] <= entry.begin
                and entry.end <= limit
            ),
            key=lambda entry: entry.end,
        )
        plans: list[PageMaterialize] = []
        checkpoint_ranges: list[tuple[int, int, tuple[int, ...]]] = []
        for target in targets:
            path_entries = sorted(
                self.tree.matched_entries(leaf),
                key=lambda entry: (entry.begin, entry.end),
            )
            pieces = [
                entry for entry in path_entries
                if target.begin <= entry.begin and entry.end <= target.end
            ]
            if (
                not pieces or pieces[0].begin != target.begin
                or pieces[-1].end != target.end
                or any(left.end != right.begin for left, right in zip(pieces, pieces[1:]))
                or any(
                    entry.kind is not LogStateKind.LOG
                    or entry.slot not in fresh_slots
                    or self.tree.slot_is_fragmented(entry.slot)
                    for entry in pieces
                )
            ):
                continue
            if any(
                self.tree.is_lock_boundary(leaf, position)
                for position in range(target.begin + self.granularity, target.end, self.granularity)
            ):
                continue
            if any(
                node.is_protected and target.begin < node.depth < target.end
                for node in self.tree.path(leaf)
            ):
                continue
            base = max(
                (
                    entry for entry in path_entries
                    if entry.kind is LogStateKind.CHECKPOINT
                    and entry.end <= target.begin
                ),
                key=lambda entry: entry.end,
                default=None,
            )
            base_end = 0 if base is None else base.end
            logs, covered = self.analyzer._cover_with_logs(
                path_entries, base_end, target.end
            )
            if covered != target.end or logs[-1].slot != pieces[-1].slot:
                continue
            plans.append(PageMaterialize(
                log_slot=pieces[-1].slot,
                begin=pieces[-1].begin,
                end=target.end,
                base_checkpoint_slot=None if base is None else base.slot,
                replay_logs=tuple(logs),
            ))
            checkpoint_ranges.append((target.begin, target.end, tuple(
                entry.slot for entry in pieces
            )))

        if plans:
            self._construct((), plans)
            retired = []
            for begin, end, slots in checkpoint_ranges:
                for slot in slots:
                    self.tree.detach_entry(slot)
                self.tree.attach_entry(leaf, LogStateEntry(
                    slot=slots[-1], kind=LogStateKind.CHECKPOINT,
                    begin=begin, end=end,
                    span_pages=(end - begin) // self.page_size,
                ))
                self.logstate_pool.materialize(slots[-1])
                retired.extend(slots[:-1])
            if retired:
                fresh_slots.difference_update(retired)
                self._return_slots(
                    torch.empty(0, dtype=torch.int32, device=self.device),
                    torch.tensor(retired, dtype=torch.int32, device=self.device),
                )

        ghost_slots = {
            entry.slot for entry in self.tree.matched_entries(leaf)
            if entry.slot in fresh_slots and entry.end <= limit
            and (
                entry.kind is LogStateKind.CHECKPOINT
                or any(
                    old.kind is LogStateKind.LOG
                    and old.begin <= entry.begin and entry.end <= old.end
                    for old in match.branch.entries
                )
            )
        }
        if not ghost_slots:
            return
        eligible = frozenset(old_slots | ghost_slots)
        plan = self.analyzer.analyze(
            self.tree, [req.input_ids[:inserted_len]],
            LogStateRound(eligible),
        )
        materialize = tuple(
            item for item in plan.materialize if item.log_slot in ghost_slots
        )
        coalesce = tuple(
            item for item in plan.coalesce
            if any(slot in ghost_slots for slot in item.slots)
            and all(
                slot in ghost_slots or slot in old_slots
                for slot in item.slots
            )
        )
        if materialize:
            self._construct((), materialize)
            self._commit_materialize(materialize)
        if coalesce:
            self._commit_coalesce(coalesce)
            fresh_slots.difference_update(
                slot for item in coalesce for slot in item.slots[:-1]
                if not self.tree.has_slot(slot)
            )

    def free_to_cached(self, req: Request) -> None:
        pages = self._live_pages.get(req.table_idx, [])
        keep = [page for page in pages if page.begin < req.cached_len]
        release = [page for page in pages if page.begin >= req.cached_len]
        self._release_live_pages(req.table_idx, release)
        self._live_pages[req.table_idx] = keep
        req.device_len = min(req.cached_len + 1, req.max_device_len)

    def commit_speculative(self, req: Request, C: int, num_sampled: int) -> None:
        raise NotImplementedError(
            "LogState speculative commit requires the LogState verify backend."
        )

    def check_integrity(self) -> None:
        self.tree.check_integrity()
        live_pages = sum(len(pages) for pages in self._live_pages.values())
        if live_pages != 0:
            raise RuntimeError(
                f"LogState cache has {live_pages} request-owned pages while idle."
            )

        if sum(self._cached_page_tokens.values()) != self.tree.size_info.total_size:
            raise RuntimeError("LogState cached KV token accounting mismatch.")

        tree_slots = {int(entry.slot) for entry in self.tree.iter_entries()}
        work_state_table = self.work_state_table
        assert work_state_table is not None
        work_slots = set(self._work_slots.values()) | {self._dummy_work_slot}
        if self._work_committed_pos.keys() != self._work_slots.keys():
            raise RuntimeError("LogState work cursor ownership disagrees with work slots.")
        if len(work_slots) != len(self._work_slots) + 1:
            raise RuntimeError("LogState work-state table contains duplicate slots.")
        for row, slot in self._work_slots.items():
            if int(work_state_table[row, 0]) != slot:
                raise RuntimeError("LogState work-state table disagrees with allocations.")
        kv_pages = {page // self.page_size for page in self._cached_page_tokens}
        kv_pages.update(
            page.kv_page // self.page_size
            for pages in self._live_pages.values() for page in pages
        )
        expected_kv = kv_pages | {self._dummy_kv_frame}
        expected_state = tree_slots | {self._dummy_log_slot} | {
            page.state_slot for pages in self._live_pages.values() for page in pages
            if page.state_slot is not None
        } | work_slots
        self.logstate_pool.poll_reclamation()
        self._allocator.check_integrity()
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
            if lazy_pages:
                self._return_slots(
                    torch.cat(lazy_pages), torch.empty(0, dtype=torch.int32)
                )
            if lazy_states:
                self._return_slots(
                    torch.empty(0, dtype=torch.int32), torch.cat(lazy_states)
                )

    def _allocate(
        self,
        needed_pages : int,
        needed_states: int,
        needed_works : int = 0,
        profile     : dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            selected: list[LogRadixTreeNode] = []
            page_counts: Counter[int] = Counter()
            slot_counts: Counter[int] = Counter()
            released_slots: set[int] = set()
            releasable_kv = 0
            for node in self.tree.iter_evict_candidates():
                selected.append(node)
                for frame, count in node._kv_page_counts.items():
                    page_counts[frame] += count
                    cached = self._cached_page_tokens[frame * self.page_size]
                    assert page_counts[frame] <= cached
                    if page_counts[frame] == cached:
                        releasable_kv += 1
                for entry in node.entries:
                    slot_counts[entry.slot] += 1
                    if slot_counts[entry.slot] == self.tree.slot_entry_count(entry.slot):
                        released_slots.add(entry.slot)
                projected_frames, projected_decay = (
                    self._allocator.projected_free_capacity(released_slots)
                )
                if self._allocator.required_frames(
                    needed_pages, needed_states, needed_works, projected_decay
                ) <= projected_frames + releasable_kv:
                    break
            else:
                raise LogStateCapacityError(
                    "Insufficient evictable LogState cache capacity: "
                    f"need_pages={needed_pages}, need_states={needed_states}, "
                    f"free_frames={self._allocator.free_frames}."
                )
            if profile is not None:
                now = perf_counter_ns()
                profile["timing_ms"]["eviction_plan_ms"] = (now - stage) / 1e6
                profile["evicted_nodes"] = len(selected)
                stage = now

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
            release_pool_timings = {} if profile is not None else None
            for slot in released_slots:
                assert not self.tree.has_slot(slot)
            self.logstate_pool.release_slots(
                sorted(released_slots), timings=release_pool_timings
            )
            if profile is not None:
                now = perf_counter_ns()
                profile["timing_ms"]["release_states_ms"] = (now - stage) / 1e6
                self._record_pool_timings(
                    profile, "release_pool", release_pool_timings
                )
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
                    "kv_tokens": len(eviction.kv_indices),
                    "state_slots": len(released_slots),
                    "free_frames_before": free_frames_before,
                    "free_frames_after": free_frames_after_evict,
                    "need_kv_pages": needed_pages,
                    "need_log_slots": needed_states,
                    "need_work_slots": needed_works,
                })
                self._profile_stats["eviction_rounds"] += 1
                self._profile_stats["evicted_kv_tokens"] += len(
                    eviction.kv_indices
                )
                self._profile_stats[
                    "released_state_slots_from_eviction"
                ] += len(released_slots)

        tensor_start = perf_counter_ns() if profile is not None else 0
        pages = torch.tensor(
            reserved[0], dtype=torch.int32, device=self.device
        ) * self.page_size
        states = torch.tensor(
            reserved[1], dtype=torch.int32, device=self.device
        )
        works = torch.tensor(
            reserved[2], dtype=torch.int32, device=self.device
        )
        if profile is not None:
            now = perf_counter_ns()
            profile["timing_ms"]["tensor_build_ms"] = (
                now - tensor_start
            ) / 1e6
            profile["timing_ms"]["core_total_ms"] = (now - start) / 1e6
            if own_profile:
                self._profile_allocation_samples.append(profile)
        return pages, states, works

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

    def _make_restore_tasks(
        self,
        batch: Batch,
    ) -> list[_ReconstructionTask]:
        work_state_table = self.work_state_table
        assert work_state_table is not None
        assert work_state_table.shape[1] == 1
        tasks = []
        for req in batch.reqs:
            recovery = self.pending_recoveries.get(req.table_idx)
            if recovery is None or recovery.position == 0:
                continue
            assert recovery.conv_slot is not None
            assert recovery.conv_granule is not None
            assert not recovery.logs or recovery.logs[-1].end == recovery.position
            tasks.append(
                _ReconstructionTask(
                    output_slot=int(work_state_table[req.table_idx, 0]),
                    base_slot=recovery.base_checkpoint_slot,
                    logs=recovery.logs,
                    source_conv_slot=recovery.conv_slot,
                    source_conv_granule=recovery.conv_granule,
                    output_conv_granule=0,
                )
            )
        return tasks

    def _construct(
        self,
        restores   : Sequence[_ReconstructionTask],
        materialize: Sequence[PageMaterialize],
    ) -> dict:
        submit_timing_ms: dict[str, float] | None = (
            {} if self._profile_enabled else None
        )
        step_start = perf_counter_ns() if submit_timing_ms is not None else 0
        events = None
        event_setup_cpu_ms = 0.0
        if self._profile_enabled and (restores or materialize):
            event_setup_start = perf_counter_ns()
            events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            events[0].record()
            event_setup_cpu_ms = (
                perf_counter_ns() - event_setup_start
            ) / 1e6
        conv_restores = list(restores)
        direct = [task for task in restores if not task.logs]
        if direct:
            pool = self.logstate_pool
            assert all(task.base_slot is not None for task in direct)
            for task in direct:
                assert task.base_slot is not None
                pool.copy_state(task.base_slot, task.output_slot)
        if submit_timing_ms is not None:
            submit_timing_ms["direct_state_copy_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6
            step_start = perf_counter_ns()
        restores = [task for task in restores if task.logs]
        chains = self._make_construction_chains(restores, materialize)
        if submit_timing_ms is not None:
            submit_timing_ms["chain_planning_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6
        device = self.device
        pool = self.logstate_pool

        def tensor(values: Sequence[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int32, device=device)

        if chains:
            material_chains = [chain for chain in chains if chain.materialize]
            restore_chains = [chain for chain in chains if not chain.materialize]
            segments = [entry for chain in chains for entry in chain.segments]
            outputs = [
                slot
                for chain in chains
                for values in chain.outputs
                for slot in values
            ]
            construction = {
                "chains": len(chains),
                "segments": len(segments),
                "outputs": len(outputs),
                "direct_checkpoint_copies": len(direct),
                "workspace_bytes": 0,
                "cuda_ms": None,
            }
            if restore_chains:
                step_start = perf_counter_ns() if submit_timing_ms is not None else 0
                restore_framed_log_state_chains(
                    pool,
                    [
                        (
                            chain.base_slot,
                            [
                                (
                                    entry.slot,
                                    entry.begin - entry.physical_begin,
                                    entry.end - entry.physical_begin,
                                )
                                for entry in chain.segments
                            ],
                            chain.outputs[-1][0],
                        )
                        for chain in restore_chains
                    ],
                )
                if submit_timing_ms is not None:
                    submit_timing_ms["restore_submit_ms"] = (
                        perf_counter_ns() - step_start
                    ) / 1e6
            if material_chains:
                step_start = perf_counter_ns() if submit_timing_ms is not None else 0
                construction["workspace_bytes"] = construct_framed_log_state_chains(
                    pool,
                    [
                        (
                            chain.base_slot,
                            [
                                (
                                    entry.slot,
                                    entry.begin - entry.physical_begin,
                                    entry.end - entry.physical_begin,
                                )
                                for entry in chain.segments
                            ],
                            chain.outputs,
                        )
                        for chain in material_chains
                    ],
                )
                if submit_timing_ms is not None:
                    submit_timing_ms["materialize_submit_ms"] = (
                        perf_counter_ns() - step_start
                    ) / 1e6
        else:
            construction = {
                "chains": 0,
                "segments": 0,
                "outputs": 0,
                "direct_checkpoint_copies": len(direct),
                "workspace_bytes": 0,
                "cuda_ms": 0.0,
            }

        step_start = perf_counter_ns() if submit_timing_ms is not None else 0
        if conv_restores:
            from logssms.kernel.linear_attention.conv_copy import (
                copy_framed_conv_snapshots,
            )
            source_slots = tensor([
                task.source_conv_slot for task in conv_restores
            ])
            source_granules = tensor([
                task.source_conv_granule for task in conv_restores
            ])
            output_slots = tensor([
                task.output_slot for task in conv_restores
            ])
            output_granules = tensor([
                task.output_conv_granule for task in conv_restores
            ])
            copy_framed_conv_snapshots(
                pool, source_slots, source_granules,
                output_slots, output_granules,
            )
        if submit_timing_ms is not None:
            submit_timing_ms["conv_restore_submit_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6
        if events is not None:
            event_end_setup_start = perf_counter_ns()
            events[1].record()
            self._profile_cuda_events.append(
                (construction, events[0], events[1])
            )
            event_setup_cpu_ms += (
                perf_counter_ns() - event_end_setup_start
            ) / 1e6
        construction["event_setup_cpu_ms"] = event_setup_cpu_ms
        if submit_timing_ms is not None:
            construction["submit_timing_ms"] = submit_timing_ms
        return construction

    @staticmethod
    def _make_construction_chains(
        restores   : Sequence[_ReconstructionTask],
        materialize: Sequence[PageMaterialize],
    ) -> tuple[_ConstructionChain, ...]:
        material_chains = LogStateCacheManager._make_materialize_chains(materialize)
        unused = set(range(len(restores)))
        result = []

        for chain in material_chains:
            source_logs = chain.logs
            events: dict[int, list[int]] = {}
            for entry in chain.logs:
                if entry.slot in chain.checkpoint_slots:
                    events.setdefault(entry.end, []).append(entry.slot)

            for index, task in enumerate(restores):
                if index not in unused or task.base_slot != chain.base_slot or not task.logs:
                    continue
                task_is_covered = all(any(
                    source.slot == entry.slot
                    and source.begin <= entry.begin
                    and entry.end <= source.end
                    for source in source_logs
                ) for entry in task.logs)
                source_is_covered = all(any(
                    entry.slot == source.slot
                    and entry.begin <= source.begin
                    and source.end <= entry.end
                    for entry in task.logs
                ) for source in source_logs)
                if not task_is_covered and not source_is_covered:
                    continue
                if source_is_covered:
                    source_logs = task.logs
                events.setdefault(task.logs[-1].end, []).append(task.output_slot)
                unused.remove(index)

            segments = []
            outputs = []
            for entry in source_logs:
                boundaries = sorted(
                    position for position in events
                    if entry.begin < position <= entry.end
                )
                begin = entry.begin
                for end in boundaries or [entry.end]:
                    segments.append(entry if begin == entry.begin and end == entry.end
                                    else entry.slice(begin, end))
                    outputs.append(tuple(events.get(end, ())))
                    begin = end
                if begin < entry.end:
                    segments.append(entry.slice(begin, entry.end))
                    outputs.append(())
            result.append(_ConstructionChain(
                chain.base_slot, tuple(segments), tuple(outputs), True
            ))

        for index in sorted(unused):
            task = restores[index]
            result.append(_ConstructionChain(
                task.base_slot,
                task.logs,
                tuple(
                    (task.output_slot,) if i == len(task.logs) - 1 else ()
                    for i in range(len(task.logs))
                ),
                False,
            ))
        return tuple(result)

    @staticmethod
    def _make_materialize_chains(
        plans: Sequence[PageMaterialize],
    ) -> tuple[_MaterializeChain, ...]:
        remaining = {plan.log_slot: plan for plan in plans}
        chains = []
        for plan in sorted(plans, key=lambda value: value.end, reverse=True):
            if plan.log_slot not in remaining:
                continue
            checkpoint_slots = frozenset(
                entry.slot
                for entry in plan.replay_logs
                if entry.slot in remaining
            )
            assert plan.log_slot in checkpoint_slots
            chains.append(
                _MaterializeChain(
                    base_slot=plan.base_checkpoint_slot,
                    logs=plan.replay_logs,
                    checkpoint_slots=checkpoint_slots,
                )
            )
            for slot in checkpoint_slots:
                del remaining[slot]
        assert not remaining
        return tuple(chains)

    def _commit_materialize(
        self,
        plans: Sequence[PageMaterialize],
    ) -> None:
        for plan in plans:
            entries = self.tree.entries_for_slot(plan.log_slot)
            assert len(entries) == 1
            old_entry = entries[0]
            owner = self.tree.entry_owner(old_entry)
            detached = self.tree.detach_entry(plan.log_slot)
            assert detached == old_entry
            self.tree.attach_entry(
                owner,
                LogStateEntry(
                    slot=plan.log_slot,
                    kind=LogStateKind.CHECKPOINT,
                    begin=plan.begin,
                    end=plan.end,
                    span_pages=1,
                ),
            )
            self.logstate_pool.materialize(plan.log_slot)

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
    ) -> tuple[int, int, dict[str, int]]:
        released = []
        released_frames = 0
        skips: dict[str, int] = {}
        for plan in plans:
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
                entry.kind is not LogStateKind.CHECKPOINT
                or entry.slot != slot
                or entry.span_pages != level
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
            if self._profile_enabled:
                for slot in plan.slots[:-1]:
                    record = self._allocator.slot(slot)
                    # Materialization may still be pending on CUDA. Count the
                    # eventual checkpoint layout, not Log-only conv frames.
                    released_frames += len(record.payload) + len(record.conv[-1])
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
            )
            released.extend(plan.slots[:-1])
        if released:
            self._return_slots(
                torch.empty(0, dtype=torch.int32, device=self.device),
                torch.tensor(
                    released, dtype=torch.int32, device=self.device
                ),
            )
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
        for page in pages:
            state_table[table_idx, page.begin:page.end].fill_(-1)
        self._return_slots(
            torch.tensor(
                [page.kv_page for page in pages],
                dtype=torch.int32,
                device=self.device,
            ),
            torch.tensor(
                [page.state_slot for page in pages if page.state_slot is not None],
                dtype=torch.int32,
                device=self.device,
            ),
        )

    def _return_slots(
        self,
        pages : torch.Tensor,
        states: torch.Tensor,
    ) -> None:
        if self._lazy_free_pages is None:
            page_frames = [page // self.page_size for page in pages.tolist()]
            self.logstate_pool.release_kv(page_frames)
            self.logstate_pool.release_slots(states.tolist())
        else:
            assert self._lazy_free_states is not None
            if len(pages) > 0:
                self._lazy_free_pages.append(pages)
            if len(states) > 0:
                self._lazy_free_states.append(states)

    def _retain_cached_kv(self, indices: torch.Tensor) -> None:
        pages = torch.div(
            indices.to(torch.int64), self.page_size,
            rounding_mode="floor",
        ) * self.page_size
        unique, counts = torch.unique(pages, return_counts=True)
        for page, count in zip(unique.tolist(), counts.tolist()):
            self._cached_page_tokens[page] = (
                self._cached_page_tokens.get(page, 0) + count
            )

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
    """Allocate one arena of KV granules shared by KV, log and checkpoints."""

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
        model = config.model_config
        from .pools import local_linear_head_counts
        num_k_heads, num_v_heads = local_linear_head_counts(model)
        conv_dim = (
            num_k_heads * model.linear_key_head_dim * 2
            + num_v_heads * model.linear_value_head_dim
        )
        layout = LogStateSlotLayout(
            page_size=config.page_size,
            granularity=config.cache_config.granularity,
            num_linear_layers=model.num_linear_layers,
            num_key_heads=num_k_heads,
            num_value_heads=num_v_heads,
            head_k_dim=model.linear_key_head_dim,
            head_v_dim=model.linear_value_head_dim,
            conv_dim=conv_dim,
            state_len=model.linear_conv_kernel_dim - 1,
            dtype=config.dtype,
        )
        num_work_states = config.max_running_req + 1
        head_bytes = max(
            layout.head_log_key_bytes + layout.head_log_value_bytes,
            layout.head_recurrent_state_bytes,
        )
        payload_frames = div_ceil(
            model.num_linear_layers * num_v_heads * head_bytes,
            cache_per_page,
        )
        conv_frames = div_ceil(
            layout.conv_state_bytes // layout.num_granules, cache_per_page
        )
        assert layout.log_decay_bytes <= cache_per_page
        state_units = payload_frames + layout.num_granules * conv_frames + 1
        cache_per_log_slot = (
            payload_frames + layout.num_granules * conv_frames
        ) * cache_per_page + layout.log_decay_bytes
        work_units = payload_frames + conv_frames
        # The manager also allocates one dummy KV frame, log slot and work slot.
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
            num_pages=num_pages,
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            aligned_max_seq_len=align_ceil(max_seq_len, 32),
            num_states=num_states,
            num_work_states=num_work_states,
            cache_per_page=cache_per_page,
            cache_per_log_slot=cache_per_log_slot,
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
        from .pools import local_linear_head_counts
        from logssms.distributed import try_get_tp_info
        from logssms.utils import div_even

        model = config.model_config
        num_k_heads, num_v_heads = local_linear_head_counts(model)
        conv_dim = (
            num_k_heads * model.linear_key_head_dim * 2
            + num_v_heads * model.linear_value_head_dim
        )
        tp_info = try_get_tp_info()
        assert tp_info is not None
        shared_pool = FramedMHALogStatePool(
            num_frames=plan.num_pages + 1,
            page_size=config.page_size,
            num_attention_layers=model.num_attention_layers,
            num_kv_heads=div_even(model.num_kv_heads, tp_info.size, allow_replicate=True),
            head_dim=model.head_dim,
            num_linear_layers=model.num_linear_layers,
            num_key_heads=num_k_heads,
            num_value_heads=num_v_heads,
            head_k_dim=model.linear_key_head_dim,
            head_v_dim=model.linear_value_head_dim,
            conv_dim=conv_dim,
            state_len=model.linear_conv_kernel_dim - 1,
            device=device,
            dtype=dtype,
            granularity=config.cache_config.granularity,
        )
        assert (
            shared_pool.allocator.payload_frames
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
    "PageMaterialize",
    "WorkStateRestore",
    "WorkStateRestoreMode",
]
