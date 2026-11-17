from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, replace
from time import perf_counter_ns
from typing import Sequence
from weakref import WeakSet, WeakValueDictionary

import torch
from logssms.utils import align_down

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
    ) -> None:
        assert granularity > 0 and checkpoint_log_count > 0
        assert min_sampling_len > 0 and max_density_exponent > 0 and hit_half_life > 0
        self.checkpoint_log_count = checkpoint_log_count
        self.granularity = granularity
        self.enable_merge_adjustment = enable_merge_adjustment
        self.min_sampling_len = min_sampling_len
        self.max_density_exponent = max_density_exponent
        self.hit_half_life = hit_half_life
        self.query_count = 0
        self._owners: WeakValueDictionary[int, LogStateHitStats] = WeakValueDictionary()
        self._internal_owners: WeakSet[LogStateHitStats] = WeakSet()
        self._observations = self._owned_hits = self._tail_hits = 0.0
        self._analysis_profile = (
            {"aggregate": Counter(), "updates": deque(maxlen=8192)}
            if profile_enabled else None
        )

    def profile_result(self) -> dict | None:
        if self._analysis_profile is None:
            return None
        return dict(
            enabled=self.enable_merge_adjustment,
            min_sampling_len=self.min_sampling_len,
            hit_half_life=self.hit_half_life,
            max_density_exponent=self.max_density_exponent,
            queries=self.query_count,
            aggregate=dict(self._analysis_profile["aggregate"]),
            updates=list(self._analysis_profile["updates"]),
            dropped_updates=max(
                0, self._analysis_profile["aggregate"]["model_updates"]
                - len(self._analysis_profile["updates"]),
            ),
        )

    @property
    def has_internal_observations(self) -> bool:
        return bool(self._internal_owners)

    def make_hit_stats(self, origin: int) -> LogStateHitStats | None:
        if not self.enable_merge_adjustment:
            return None
        stats = LogStateHitStats(origin=origin, tail=origin,
                                query_begin=self.query_count,
                                exposures=deque([(origin, self.query_count)]))
        return stats

    def record_hit(self, stats: LogStateHitStats | None, position: int) -> None:
        self.query_count += 1
        decay = math.exp2(-1 / self.hit_half_life)
        self._observations = self._observations * decay + 1
        self._owned_hits *= decay
        self._tail_hits *= decay
        if stats is not None and 0 < position and stats.origin <= position <= stats.tail:
            self._owned_hits += 1
            self._tail_hits += position == stats.tail
            stats.hits.append((position, stats.tail, self.query_count))
            if 0 < stats.tail - position <= self.min_sampling_len:
                self._internal_owners.add(stats)
            if len(stats.hits) == stats.hits.maxlen:
                stats.query_begin = max(stats.query_begin, stats.hits[0][2] - 1)

    def _prior(self) -> tuple[float, float]:
        probability = ((self._owned_hits + 1) / (self._observations + 2)
                       / max(1, len(self._owners)))
        tail_fraction = (self._tail_hits + 1) / (self._owned_hits + 2)
        return probability * (1 - tail_fraction), probability * tail_fraction

    def _weight(self, query: int) -> float:
        return math.exp2(-(self.query_count - query) / self.hit_half_life)

    def _exposure(self, begin: int, end: int) -> float:
        if end <= begin:
            return 0.0
        rate = math.log(2) / self.hit_half_life
        return self._weight(end) * (-math.expm1(-rate * (end - begin))) / (-math.expm1(-rate))

    def _bucket_exposures(self, stats: LogStateHitStats) -> list[float]:
        count = min(self.min_sampling_len, stats.tail - stats.origin) // self.granularity
        result = [0.0] * count
        events = list(stats.exposures) or [(stats.tail, stats.query_begin)]
        for i, (tail, begin) in enumerate(events):
            end = events[i + 1][1] if i + 1 < len(events) else self.query_count
            exposure = self._exposure(max(begin, stats.query_begin), end)
            available = min(count, max(0, tail - stats.origin) // self.granularity)
            for j in range(available):
                result[j] += exposure
        return result

    def _fit_hit_bins(
        self, bins: dict[int, float], bucket_exposures: Sequence[float],
    ) -> tuple[float, float]:
        g = self.granularity
        support = [(math.log((i + 1) * g), e * g)
                   for i, e in enumerate(bucket_exposures) if e > 0]
        total = sum(bins.values())
        prior_mass = sum(g * ((i + 1) * g) ** -2 for i in range(max(1, len(bucket_exposures))))
        internal_prior, _ = self._prior()
        alpha, beta = 4.0 * internal_prior, 4.0 * prior_mass
        if not support:
            return 2.0, math.log(alpha / beta)
        observed = sum(c * math.log((i + 1) * g) for i, c in bins.items())

        def terms(k):
            values = [e * math.exp(-k * x) for x, e in support]
            return sum(values), sum(v * x for v, (x, _) in zip(values, support))

        exponent = 2.0
        if total:
            lo, hi = 0.0, self.max_density_exponent
            for _ in range(40):
                mid = (lo + hi) / 2
                mass, moment = terms(mid)
                derivative = (total + alpha) * moment / (mass + beta) - observed - (mid - 2.0)
                if derivative > 0:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 1e-6:
                    break
            exponent = (lo + hi) / 2
        mass, _ = terms(exponent)
        return exponent, math.log((total + alpha) / (mass + beta))

    def refresh_models(self, owners: Sequence[LogStateHitStats | None]) -> None:
        for owner in owners:
            if owner is not None and owner.tail > owner.origin:
                self._owners[owner.uuid] = owner
        for stats in {id(owner): owner for owner in owners if owner is not None}.values():
            if stats.fit_query == self.query_count:
                continue
            bins = Counter()
            tail_hits = 0.0
            latest_internal = -1
            for position, tail, query in stats.hits:
                distance = tail - position
                weight = self._weight(query)
                if distance == 0:
                    tail_hits += weight
                elif distance <= self.min_sampling_len:
                    bins[(distance - 1) // self.granularity] += weight
                    latest_internal = max(latest_internal, query)
            exposures = self._bucket_exposures(stats)
            refit = (
                stats.log_amplitude is None
                or stats.fit_query < 0
                or latest_internal != stats.fitted_hit_query
                or self.query_count - stats.exponent_query >= self.hit_half_life
            )
            if refit:
                stats.exponent, stats.log_amplitude = self._fit_hit_bins(bins, exposures)
                stats.fitted_hit_query = latest_internal
                stats.exponent_query = self.query_count
            else:
                g = self.granularity
                internal_prior, _ = self._prior()
                beta = 4 * sum(g * ((i + 1) * g) ** -2 for i in range(max(1, len(exposures))))
                mass = sum(e * g * ((i + 1) * g) ** -stats.exponent
                           for i, e in enumerate(exposures))
                stats.log_amplitude = math.log((sum(bins.values()) + 4 * internal_prior) / (mass + beta))
            first_tail_query = next(
                (query for tail, query in stats.exposures if tail > stats.origin),
                self.query_count,
            )
            exposure = self._exposure(max(stats.query_begin, first_tail_query), self.query_count)
            _, tail_prior = self._prior()
            stats.tail_probability = min(1.0, (tail_hits + 4 * tail_prior) / (exposure + 4.0))
            stats.fit_query = self.query_count
            if self._analysis_profile is not None:
                self._analysis_profile["aggregate"]["model_updates"] += 1
                self._analysis_profile["aggregate"]["exponent_fits"] += int(refit and bool(bins))
                self._analysis_profile["updates"].append(dict(
                    owner_uuid=stats.uuid, query=self.query_count, exponent=stats.exponent,
                    log_amplitude=stats.log_amplitude, tail_probability=stats.tail_probability,
                    weighted_internal_hits=sum(bins.values()), exposure=exposure,
                ))

    def advance_hit_stats(self, stats: LogStateHitStats, tail: int) -> None:
        if tail > stats.origin:
            self._owners[stats.uuid] = stats
        if tail <= stats.tail:
            return
        stats.exposures.append((tail, self.query_count))
        while len(stats.exposures) > 1 and stats.exposures[1][0] <= tail - self.min_sampling_len:
            stats.exposures.popleft()
            stats.query_begin = max(stats.query_begin, stats.exposures[0][1])
        stats.hits = deque((hit for hit in stats.hits
                            if hit[1] >= tail - self.min_sampling_len and hit[2] > stats.query_begin),
                           maxlen=8192)
        if not any(0 < past_tail - position <= self.min_sampling_len
                   for position, past_tail, _ in stats.hits):
            self._internal_owners.discard(stats)
        stats.tail = tail
        stats.fit_query = -1

    def eviction_models(self, segments) -> list[tuple]:
        self.refresh_models([segment.node.analysis_owner for segment in segments])
        result = []
        for segment in segments:
            stats = segment.node.analysis_owner
            recovery_begin = (segment.begin if segment.recovery_begin is None
                              else segment.recovery_begin)
            if stats is None:
                tail = max(self.granularity, segment.node.depth)
                norm = sum(self.granularity * (j * self.granularity) ** -2
                           for j in range(1, max(1, tail // self.granularity) + 1))
                internal_prior, tail_prior = self._prior()
                result.append((-segment.node.uuid - 1, 0, tail, 2., math.log(internal_prior / norm),
                               tail_prior, self.granularity, recovery_begin, segment.end, 1))
            else:
                result.append((stats.uuid, stats.origin, max(stats.tail, segment.node.depth),
                               stats.exponent, stats.log_amplitude, stats.tail_probability,
                               self.granularity, recovery_begin, segment.end, int(stats.fitted_hit_query < 0)))
        return result

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
        if snapshot is None:
            snapshot = self.snapshot_prefill(tree, traces)
        analysis_nodes, endpoints = snapshot.analysis_nodes, snapshot.endpoints
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
            tree, snapshot.eligible_slots, pending, analysis_nodes, path_cache,
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
            result.append(MaterializationPlan(
                tuple(entry.slot for entry in group), group[0].begin, group[-1].end,
            ))
        result.sort(key=lambda value: (value.begin, value.end, value.right_slot))
        return tuple(result)
