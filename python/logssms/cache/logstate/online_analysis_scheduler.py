from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, replace
from time import perf_counter_ns
from typing import Sequence

import torch

from logssms.utils import align_down
from .layout import LOG_STATE_GRANULARITY
from .prefix_cache import (
    LogRadixTrace, LogRadixTree, LogRadixTreeNode,
    LogStateEntry, LogStateKind, LogStateMergeStats,
)


@dataclass(frozen=True)
class LogStateRound:
    eligible_slots: frozenset[int]
    analysis_nodes: tuple[LogRadixTreeNode, ...]
    endpoints     : tuple[_AnalysisEndpoint, ...]


@dataclass(frozen=True)
class _MergePathContext:
    pending_positions: tuple[int, ...]
    lock_boundaries  : frozenset[int]

    @classmethod
    def build(
        cls, path: Sequence[LogRadixTreeNode], pending: Sequence[_PendingProtection],
    ) -> _MergePathContext:
        path_ids = {node.uuid for node in path}
        return cls(
            pending_positions=tuple(sorted(
                value.position for value in pending
                if value.trace.node.uuid in path_ids
            )),
            lock_boundaries=frozenset(
                position for node in path for position in node._endpoint_refs
            ),
        )


@dataclass(frozen=True)
class BranchProtection:
    request_indices: tuple[int, ...]
    position       : int
    node           : LogRadixTreeNode


@dataclass(frozen=True)
class MaterializeCandidate:
    slots: tuple[int, ...]
    begin: int
    end  : int

    @property
    def right_slot(self) -> int:
        return self.slots[-1]


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


class OnlineAnalysisScheduler:

    def __init__(
        self,
        granularity              : int = LOG_STATE_GRANULARITY,
        checkpoint_log_count     : int = 1,
        enable_merge_adjustment  : bool = True,
        min_sampling_len         : int  = 1024,
        max_merge_intensity      : int  = 16,
        default_merge_growth_rate: int = 2,
        profile_enabled          : bool = False,
    ) -> None:
        assert granularity > 0 and checkpoint_log_count > 0
        assert min_sampling_len > 0
        assert max_merge_intensity >= 2
        assert default_merge_growth_rate >= 1
        self.checkpoint_log_count = checkpoint_log_count
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
                    index = (distance - 1) // self.granularity
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

        xs = [math.log((index + 0.5) * self.granularity) for index in bins]
        ys = [math.log(count / self.granularity) for count in bins.values()]
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
        limits = self._analysis_limits(nodes, endpoints)
        return LogStateRound(
            frozenset(entry.slot for node in nodes for entry in node.entries
                      if entry.end <= limits[node.uuid]),
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
        protections = self._plan_protections(pending)
        if step_timing_ms is not None:
            step_timing_ms["protection_and_restore_plan_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6
            step_start = perf_counter_ns()
        path_cache = {"limits": self._analysis_limits(analysis_nodes, endpoints)}
        materialize = self._plan_materialize(
            tree, round_info.eligible_slots, pending, analysis_nodes, path_cache, endpoints,
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

    @staticmethod
    def _analysis_limits(
        nodes: Sequence[LogRadixTreeNode],
        endpoints: Sequence[_AnalysisEndpoint],
    ) -> dict[int, int]:
        parents = {node.parent.uuid for node in nodes}
        limits = {node.uuid: node.depth if node.uuid in parents else 0 for node in nodes}
        for endpoint in endpoints:
            limits[endpoint.node.uuid] = max(limits[endpoint.node.uuid], endpoint.matched_len)
        return limits

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
        pending: Sequence[_PendingProtection],
    ) -> tuple[BranchProtection, ...]:
        protections = []
        for value in pending:
            node = (
                value.trace.node
                if value.trace.matched_len == value.position
                and not value.trace.node.is_root() else None
            )
            if node is not None:
                protections.append(BranchProtection(
                    request_indices=value.request_indices,
                    position=value.position,
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
                (entry for part in path for entry in part.entries
                 if entry.end <= cache.get("limits", {}).get(part.uuid, part.depth)),
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
            context = _MergePathContext.build(path, pending)
            rows = [
                value
                for entry in entries
                for value in (
                    entry.slot, entry.begin, entry.end,
                    -1 if entry.span_granules is None else entry.span_granules,
                    int(entry.slot in eligible_slots),
                )
            ]
            path_cache[key] = (
                entries, rows,
                tuple(sorted({*(part.depth for part in path), *context.pending_positions,
                              *context.lock_boundaries})),
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
        endpoints     : tuple[_AnalysisEndpoint, ...],
    ) -> tuple[MaterializeCandidate, ...]:
        from logssms.kernel import logstate as planning

        candidates = []
        paths = []
        path_indices = {}
        entry_indices = []
        source_entries = []
        limits = self._analysis_limits(analysis_nodes, endpoints)
        entries = {entry.slot: entry for node in analysis_nodes for entry in node.entries
                   if entry.end <= limits.get(node.uuid, node.depth)}

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
            candidates.append((path_index, entry_indices[path_index][entry], 1))
            source_entries.append(entry)
        if not candidates:
            return ()
        ordered = sorted(zip(candidates, source_entries), key=lambda item: item[0])
        candidates, source_entries = map(list, zip(*ordered))
        rows = planning.plan_materialize(
            [path[1] for path in paths], [path[2] for path in paths],
            candidates, self.granularity, self.checkpoint_log_count,
        )
        result = []
        for row in rows:
            group = [source_entries[index] for index in row]
            result.append(MaterializeCandidate(
                tuple(entry.slot for entry in group), group[0].begin, group[-1].end,
            ))
        result.sort(key=lambda value: (value.begin, value.end, value.right_slot))
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
        projected = {slot: item for item in materialize for slot in item.slots}
        projected_paths = []
        for path in paths:
            rows = []
            for index, entry in enumerate(path[0]):
                item = projected.get(entry.slot)
                if item is None:
                    rows.extend(path[1][index * 5:(index + 1) * 5])
                elif entry.slot == item.right_slot:
                    rows.extend((item.right_slot, item.begin, item.end,
                                 (item.end - item.begin) // self.granularity,
                                 1))
            projected_paths.append(rows)
        groups, native_diagnostics = planning.plan_coalesce(
            projected_paths, [path[2] for path in paths],
            [path[3] for path in paths], anchors,
            self.granularity,
            diagnostics is not None,
        )
        if diagnostics is not None:
            diagnostics.update(native_diagnostics)
        result = []
        for row in groups:
            values = tuple(row)
            slots, spans = [], []
            for slot, span in zip(values[3::2], values[4::2]):
                item = projected.get(slot)
                slots.extend((slot,) if item is None else item.slots)
                spans.extend((span,) if item is None else (1,) * len(item.slots))
            result.append(CheckpointCoalesce(
                slots=tuple(slots), source_spans=tuple(spans),
                begin=values[0], end=values[1], next_span=values[2],
            ))
        result.sort(key=lambda value: (value.begin, value.end, value.left_slot))
        return tuple(result)
