from __future__ import annotations

from collections.abc import Callable, Mapping
from time import perf_counter_ns

from .frame_pool import MHALogStatePool
from .prefix_cache import LogRadixTree, LogRadixEvictionSegment, LogStateKind, _EVICTION_EXPAND
from .recycler import LogStateRecycleBin


class LogStateCapacityError(RuntimeError):
    pass


class LogStateEvictor:
    def __init__(
        self,
        tree: LogRadixTree,
        pool: MHALogStatePool,
        *,
        analyzer,
        profile_enabled: bool,
        profile_stats: dict,
        allocation_samples: list[dict],
        record_lifecycle_event: Callable[[dict], None],
    ) -> None:
        self.tree = tree
        self.pool = pool
        self.page_size = pool.page_size
        self.analyzer = analyzer
        self.cached_page_tokens: dict[int, int] = {}
        self._allocator = pool.allocator
        self._profile_enabled = profile_enabled
        self._profile_stats = profile_stats
        self._profile_allocation_samples = allocation_samples
        self._record_lifecycle_event = record_lifecycle_event
        self._planner = None
        self._planner_epoch = None
        self._planner_model_epoch = None
        self._planner_segments = []
        self._planner_decay_frames = ()
        self._planning_timings = {}
        self._planning_diagnostic = {}
        self._slot_descriptions = {}

    def allocate(
        self,
        needed_pages : int,
        needed_states: int,
        needed_works : int = 0,
        profile     : dict | None = None,
        *,
        recycle_bin: LogStateRecycleBin | None = None,
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
        reserved = self.pool.reserve(
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
            selected, page_counts, released_slots, checkpoint_pairs = self.plan(
                needed_pages, needed_states, needed_works,
            )
            if profile is not None:
                now = perf_counter_ns()
                profile["timing_ms"]["eviction_plan_ms"] = (now - stage) / 1e6
                profile["timing_ms"].update(self._planning_timings)
                profile["planning"] = self._planning_diagnostic
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
                        "node_uuid": segment.node.uuid,
                        "kind": entry.kind.value,
                        "begin": entry.begin,
                        "end": entry.end,
                        "slot": entry.slot,
                    }
                    for segment in selected for entry in segment.entries
                ]
                if self._profile_enabled else None
            )
            evicted_nodes = [] if self._profile_enabled else None
            free_frames_before = self._allocator.free_frames if self._profile_enabled else 0
            if recycle_bin is not None:
                for segment in selected:
                    recycle_bin.remember(self.tree, segment)
            eviction = self.tree.evict(selected, evicted_nodes)
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

            for slot in released_slots:
                self._slot_descriptions.pop(slot, None)
            self.pool.retire_slots(sorted(released_slots))
            if profile is not None:
                now = perf_counter_ns()
                profile["timing_ms"]["release_states_ms"] = (now - stage) / 1e6
                stage = now
            free_frames_after_evict = (
                self._allocator.free_frames if self._profile_enabled else 0
            )
            retry_pool_timings = {} if profile is not None else None
            reserved = self.pool.reserve(
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
            assert reserved is not None, (
                "Eviction plan did not free the projected frames: "
                f"free={self._allocator.free_frames}, "
                f"required={self._allocator.required_frames(needed_pages, needed_states, needed_works)}, "
                f"kv={needed_pages}, logs={needed_states}, works={needed_works}, "
                f"slots={len(released_slots)}, checkpoint_drops={len(checkpoint_pairs)}"
            )
            if self._planner is not None:
                self._planner_epoch = self._eviction_epoch()

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

    def plan(
        self, needed_pages: int, needed_states: int, needed_works: int,
    ) -> tuple[list[LogRadixEvictionSegment], dict[int, int], set[int], list[tuple[int, int]]]:
        if not self._profile_enabled:
            return self._plan(needed_pages, needed_states, needed_works)
        start = perf_counter_ns()
        previous, current = self._planner_epoch, self._eviction_epoch()
        labels = ("structure", "locks", "access", "policy", "observations")
        causes = ([name for name, old, new in zip(labels, previous, current, strict=True)
                   if old != new] if self._planner is not None and previous is not None
                  else ["cold_start"])
        self._planning_diagnostic = {
            "causes": causes, "query": self.analyzer.query_count,
            "mode": "rebuild", "full_snapshots": 0,
        }
        try:
            return self._plan(needed_pages, needed_states, needed_works)
        finally:
            self._planning_diagnostic["total_ms"] = (perf_counter_ns() - start) / 1e6
            self._planning_diagnostic["timing_ms"] = dict(self._planning_timings)

    def _plan(
        self, needed_pages: int, needed_states: int, needed_works: int,
    ) -> tuple[list[LogRadixEvictionSegment], dict[int, int], set[int], list[tuple[int, int]]]:
        from logssms.kernel import logstate as planning

        self._planning_timings = {}
        stage = perf_counter_ns() if self._profile_enabled else 0
        config = self._eviction_capacity(needed_pages, needed_states, needed_works)
        epoch = self._eviction_epoch()
        if self._planner is not None and epoch == self._planner_epoch:
            if self._model_epoch() != self._planner_model_epoch:
                self._refresh_planner_models()
                stage = perf_counter_ns() if self._profile_enabled else 0
            occupied = [(frame, self._allocator.decay_counts.get(frame, 0))
                        for frame in self._planner_decay_frames]
            self._planning_stage("eviction_resource_ms", stage)
            stage = perf_counter_ns() if self._profile_enabled else 0
            rows = self._planner.plan(config, occupied)
            self._planning_stage("eviction_select_ms", stage)
            if rows:
                if self._profile_enabled:
                    self._profile_stats["eviction_planner_reuses"] += 1
                    self._planning_diagnostic["mode"] = "reuse"
                return self._consume_plan(rows)
        if self._profile_enabled and not self._planning_diagnostic["causes"]:
            self._planning_diagnostic["causes"] = ["planner_exhausted"]
        self._planner = None
        self._planner_segments = []
        allocator = self._allocator
        leaf_limit, tail_limit = 1, 16
        use_models = self.analyzer.enable_merge_adjustment and self.analyzer.has_internal_observations
        log_only = not use_models
        while True:
            stage = perf_counter_ns() if self._profile_enabled else 0
            nodes, topology = (
                self.tree.log_eviction_snapshot(leaf_limit, tail_limit=tail_limit)
                if log_only else self.tree.eviction_snapshot()
            )
            if not nodes:
                if log_only:
                    log_only = False
                    continue
                break
            if self._profile_enabled:
                self._profile_stats["eviction_candidate_windows"] += int(log_only)
                self._profile_stats["eviction_full_snapshots"] += int(not log_only)
                self._planning_diagnostic["full_snapshots"] += int(not log_only)
                self._profile_stats["eviction_serialized_nodes"] += len(nodes)
            self._planning_stage("eviction_snapshot_ms", stage)
            stage = perf_counter_ns() if self._profile_enabled else 0
            node_slots = [segment.eviction_slots for segment in nodes]
            slot_ids = {slot for row in node_slots for slot in row}
            slots = []
            decay_frames = set()
            for slot in slot_ids:
                record = allocator.slot(slot)
                entry = self.tree.entry_for_slot(slot)
                cached = self._slot_descriptions.get(slot)
                if cached is None or cached[0] is not record or cached[1] is not entry:
                    row = (slot, len(record.payload) + sum(map(len, record.conv)),
                           -1 if record.decay is None else record.decay.frame,
                           entry.begin, entry.end)
                    cached = self._slot_descriptions[slot] = (record, entry, row)
                slots.append(cached[2])
                if cached[2][2] >= 0:
                    decay_frames.add(cached[2][2])
            kv = [[value for frame, count in segment._kv_frame_counts.items() for value in (
                frame, count, self.cached_page_tokens[frame * self.page_size],
            )] if row[2] != _EVICTION_EXPAND else []
                  for segment, row in zip(nodes, topology, strict=True)]
            occupied = [(frame, allocator.decay_counts.get(frame, 0)) for frame in decay_frames]
            actions = () if log_only else self.tree.checkpoint_eviction_actions(nodes)
            self._planning_stage("eviction_resource_ms", stage)
            stage = perf_counter_ns() if self._profile_enabled else 0
            models = self.analyzer.eviction_models(nodes) if use_models else None
            self._planning_stage("eviction_models_ms", stage)
            stage = perf_counter_ns() if self._profile_enabled else 0
            if use_models:
                self._planner = planning.EvictionPlanner(
                    topology, kv, node_slots, slots, occupied, config, actions, models,
                    [(segment.node.depth, segment.node.uuid, segment.begin) for segment in nodes],
                )
                self._planning_stage("eviction_build_ms", stage)
                self._planner_model_epoch = self._model_epoch()
                self._planner_segments = nodes
                self._planner_decay_frames = tuple(decay_frames)
                stage = perf_counter_ns() if self._profile_enabled else 0
                rows = self._planner.plan(config, occupied)
                self._planning_stage("eviction_select_ms", stage)
                if rows:
                    return self._consume_plan(rows)
                self._planner = None
                self._planner_segments = []
            else:
                rows = planning.plan_eviction(
                    topology, kv, node_slots, slots, occupied, config, actions, models=models,
                )
                self._planning_stage("eviction_build_and_select_ms", stage)
            if rows:
                selected, pages, released, compressed = rows
                return ([nodes[index] for index in selected],
                        dict(zip(pages[::2], pages[1::2], strict=True)), set(released),
                        list(zip(compressed[::2], compressed[1::2], strict=True)))
            if not log_only:
                break
            if rows is None:
                if self._profile_enabled:
                    self._profile_stats["eviction_tail_expansions"] += 1
                tail_limit *= 4
            elif leaf_limit < len(self.tree._log_leaf_heap):
                leaf_limit *= 4
            else:
                log_only = False
        raise LogStateCapacityError(
            "Insufficient evictable LogState cache capacity: "
            f"need_pages={needed_pages}, need_states={needed_states}, "
            f"free_frames={allocator.free_frames}."
        )

    def _planning_stage(self, name: str, start: int) -> None:
        if self._profile_enabled:
            elapsed = (perf_counter_ns() - start) / 1e6
            self._planning_timings[name] = self._planning_timings.get(name, 0.0) + elapsed

    def _eviction_epoch(self):
        return (self.tree.revision, self.tree.lock_revision, self.tree.access_revision,
                self.analyzer.enable_merge_adjustment, self.analyzer.has_internal_observations)

    def _model_epoch(self):
        return self.analyzer.query_count, len(self.analyzer._owners)

    def _refresh_planner_models(self) -> None:
        stage = perf_counter_ns() if self._profile_enabled else 0
        live = [(index, segment) for index, segment in enumerate(self._planner_segments)
                if segment is not None]
        live.sort(key=lambda item: (item[1].node.depth, item[1].node.uuid, item[1].begin))
        models = self.analyzer.eviction_models([segment for _, segment in live])
        self._planning_stage("eviction_models_ms", stage)
        stage = perf_counter_ns() if self._profile_enabled else 0
        self._planner.refresh_models([index for index, _ in live], models)
        self._planner_model_epoch = self._model_epoch()
        self._planning_stage("eviction_rescore_ms", stage)
        if self._profile_enabled:
            self._profile_stats["eviction_model_refreshes"] += 1

    def _eviction_capacity(self, pages, logs, works):
        allocator = self._allocator
        return (allocator.free_frames, allocator._free_decay_records,
                allocator.payload_frames, allocator.conv_frames, allocator.num_granules,
                allocator.decay_records_per_frame, pages, logs, works,
                allocator.log_payload_frames)

    def _consume_plan(self, rows):
        selected, pages, released, compressed = rows
        segments = [self._planner_segments[index] for index in selected]
        for index in selected:
            self._planner_segments[index] = None
        # Ownerless densities depend on the physical node's current endpoint.
        if any(segment.node.analysis_owner is None for segment in segments):
            self._planner = None
            self._planner_segments = []
        self._planner_epoch = None
        return (segments, dict(zip(pages[::2], pages[1::2], strict=True)), set(released),
                list(zip(compressed[::2], compressed[1::2], strict=True)))

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

    def _release_cached_kv(self, page_counts: Mapping[int, int]) -> None:
        released = []
        for frame, count in page_counts.items():
            page = frame * self.page_size
            remaining = self.cached_page_tokens[page] - count
            assert remaining >= 0
            if remaining == 0:
                del self.cached_page_tokens[page]
                released.append(frame)
            else:
                self.cached_page_tokens[page] = remaining
        if released:
            self.pool.release_kv(released)
