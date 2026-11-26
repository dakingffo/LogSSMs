from __future__ import annotations

import math
from dataclasses import dataclass, replace
from time import perf_counter_ns
from typing import Sequence

import torch
from logssms.utils import align_down

from .hit_model import (
    PRIOR_EXPOSURE,
    PRIOR_INTERNAL_ALPHA,
    PRIOR_TAIL_ALPHA,
    ReuseModel,
    prior_mass,
)
from .layout import LOG_STATE_GRANULARITY
from .prefix_cache import (
    LogRadixTrace,
    LogRadixTree,
    LogRadixTreeNode,
    LogStateEntry,
    LogStateKind,
    LogStateHitStats,
)


@dataclass(frozen=True)
class LogStatePrefillSnapshot:
    eligible_slots: frozenset[int]
    analysis_nodes: tuple[LogRadixTreeNode, ...]
    endpoints     : tuple[_AnalysisEndpoint, ...]


@dataclass(frozen=True)
class _MaterializationBoundaries:
    pending_positions: tuple[int, ...]
    lock_boundaries  : frozenset[int]

    @classmethod
    def build(
        cls, path: Sequence[LogRadixTreeNode], pending: Sequence[_PendingProtection],
    ) -> _MaterializationBoundaries:
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
class MaterializationPlan:
    slots: tuple[int, ...]
    begin: int
    end  : int

    @property
    def right_slot(self) -> int:
        return self.slots[-1]


@dataclass(frozen=True)
class LogStatePrefillPlan:
    protections      : tuple[BranchProtection, ...]
    materialize      : tuple[MaterializationPlan, ...]


@dataclass(frozen=True)
class _AnalysisEndpoint:
    node           : LogRadixTreeNode
    matched_len: int


@dataclass(frozen=True)
class _PendingProtection:
    request_indices: tuple[int, ...]
    position       : int
    trace          : LogRadixTrace


class OnlineAnalysisScheduler:
    def __init__(
        self,
        granularity: int = LOG_STATE_GRANULARITY,
        checkpoint_log_count: int = 1,
        enable_merge_adjustment: bool = True,
        min_sampling_len: int = 8192,
        max_density_exponent: float = 16.0,
        hit_half_life: int = 1024,
        profile_enabled: bool = False,
        enable_hit_materialization: bool = False,
    ) -> None:
        assert granularity > 0 and checkpoint_log_count > 0
        assert min_sampling_len > 0 and max_density_exponent > 0 and hit_half_life > 0
        self.checkpoint_log_count = checkpoint_log_count
        self.granularity = granularity
        self.enable_merge_adjustment = enable_merge_adjustment
        self.enable_hit_materialization = enable_hit_materialization
        self.min_sampling_len = min_sampling_len
        self.max_density_exponent = max_density_exponent
        self.hit_half_life = hit_half_life
        self.reuse = ReuseModel(granularity, min_sampling_len, hit_half_life,
                                max_density_exponent, profile_enabled)
        self._owners = self.reuse._owners

    @property
    def query_count(self):
        return self.reuse.query

    def profile_result(self) -> dict | None:
        profile = self.reuse.profile
        if profile is None:
            return None
        return dict(
            enabled=self.enable_merge_adjustment,
            min_sampling_len=self.min_sampling_len,
            hit_half_life=self.hit_half_life,
            max_density_exponent=self.max_density_exponent,
            queries=self.query_count,
            aggregate=dict(profile["aggregate"]),
            updates=list(profile["updates"]),
            dropped_updates=max(0, profile["aggregate"]["model_updates"] - len(profile["updates"])),
        )

    def make_hit_stats(self, origin, group=None):
        return self.reuse.new_chain(origin, group) if self.enable_merge_adjustment else None

    def record_hit(self, stats, position):
        self.reuse.record_hit(stats, position)

    def advance_hit_stats(self, stats, tail):
        self.reuse.publish(stats, tail)

    def eviction_models(self, snapshots) -> list[tuple]:
        if self.enable_merge_adjustment:
            self.reuse.refresh(snapshot.node.analysis_owner for snapshot in snapshots)
        result = []
        g = self.granularity
        for snapshot in snapshots:
            stats = snapshot.node.analysis_owner if self.enable_merge_adjustment else None
            if stats is None:
                tail = max(g, snapshot.node.depth)
                pm = prior_mass(g, tail // g)
                result.append((-snapshot.node.uuid - 1, 0, tail, 2.,
                               math.log((PRIOR_INTERNAL_ALPHA / PRIOR_EXPOSURE) / pm),
                               PRIOR_TAIL_ALPHA / PRIOR_EXPOSURE, g,
                               snapshot.recovery_begin, snapshot.end))
                continue
            tail = max(stats.tail, snapshot.node.depth)
            result.append((stats.uuid, stats.origin, tail, stats.exponent,
                           stats.log_amplitude, stats.tail_probability, g,
                           snapshot.recovery_begin, snapshot.end))
        return result

    def active_density(self, owner: LogStateHitStats | None, tail: int):
        from logssms.kernel.logstate import ActiveDensity

        g = self.granularity
        if not self.enable_merge_adjustment or owner is None:
            bins = max(1, tail // g)
            pm = prior_mass(g, bins)
            return ActiveDensity(tail, g, bins, (PRIOR_INTERNAL_ALPHA / PRIOR_EXPOSURE) / pm,
                                 2., PRIOR_TAIL_ALPHA / PRIOR_EXPOSURE)
        self.reuse.refresh((owner,))
        bins = max(1, (tail - owner.origin) // g)
        return ActiveDensity(
            tail, g, bins, math.exp(owner.log_amplitude),
            owner.exponent, owner.tail_probability, owner.origin,
        )

    def snapshot_prefill(
        self, tree: LogRadixTree, traces: Sequence[LogRadixTrace],
    ) -> LogStatePrefillSnapshot:
        nodes, endpoints = self._analysis_paths(tree, traces)
        limits = self._analysis_limits(nodes, endpoints)
        return LogStatePrefillSnapshot(
            frozenset(entry.slot for node in nodes for entry in node.entries
                      if entry.end <= limits[node.uuid]),
            nodes, endpoints,
        )

    def analyze(
        self,
        tree      : LogRadixTree,
        input_ids : Sequence[torch.Tensor],
        snapshot: LogStatePrefillSnapshot | None = None,
        *,
        step_timing_ms: dict[str, float] | None = None,
        traces: Sequence[LogRadixTrace] | None = None,
        pending: tuple[_PendingProtection, ...] | None = None,
    ) -> LogStatePrefillPlan:
        assert tree.granularity == self.granularity
        step_start = perf_counter_ns() if step_timing_ms is not None else 0
        if traces is None:
            traces = tuple(
                tree.trace(values[:align_down(len(values), self.granularity)])
                for values in input_ids
            )
        assert len(traces) == len(input_ids)
        if self.enable_hit_materialization and snapshot is None:
            snapshot = self.snapshot_prefill(tree, traces)
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
            step_timing_ms["protection_plan_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6
            step_start = perf_counter_ns()
        materialize = ()
        if self.enable_hit_materialization:
            assert snapshot is not None
            nodes = snapshot.analysis_nodes
            path_cache = {"limits": self._analysis_limits(nodes, snapshot.endpoints)}
            materialize = self._plan_materialize(
                tree, snapshot.eligible_slots, pending, nodes, path_cache,
            )
        if step_timing_ms is not None:
            step_timing_ms["materialize_plan_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6
        return LogStatePrefillPlan(protections=protections, materialize=materialize)

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
    ) -> tuple[list[LogStateEntry], list[int], tuple[int, ...]]:

        key = ("materialization", node.uuid)
        if key not in path_cache:
            path, entries = self._cached_path_entries(tree, node, path_cache)
            context = _MaterializationBoundaries.build(path, pending)
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
            )
        return path_cache[key]

    def _plan_materialize(
        self,
        tree          : LogRadixTree,
        eligible_slots: frozenset[int],
        pending       : tuple[_PendingProtection, ...],
        analysis_nodes: tuple[LogRadixTreeNode, ...],
        path_cache    : dict,
    ) -> tuple[MaterializationPlan, ...]:
        from logssms.kernel import logstate as planning

        candidates = []
        paths = []
        path_indices = {}
        entry_indices = []
        source_entries = []
        limits = path_cache["limits"]
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
            candidates.append((path_index, entry_indices[path_index][entry]))
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
            result.append(MaterializationPlan(
                tuple(entry.slot for entry in group), group[0].begin, group[-1].end,
            ))
        result.sort(key=lambda value: (value.begin, value.end, value.right_slot))
        return tuple(result)
