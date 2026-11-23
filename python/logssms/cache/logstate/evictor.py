from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import struct
from time import perf_counter_ns

from .frame_pool import MHALogStatePool
from .prefix_cache import (
    LogRadixTree, LogRadixEvictionNode, LogRadixTreeNode, LogStateEntry, LogStateKind,
)


class LogStateCapacityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ReclamationPlan:
    leaves: list[LogRadixEvictionNode]
    kv_references: list[tuple[int, int]]
    retired_slots: list[int]
    checkpoint_merges: list[tuple[int, int]]
    materialization: tuple[int, ...]
    active_action: int | None
    active_frames: int = 0
    active_cost: float = 0.


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
        pressure_work_slot: int,
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
        self._planner_nodes = []
        self._planner_index = {}
        self._planner_topology = {}
        self._planner_frames = {}
        self._planner_needs_rebase = False
        self._planner_decay_frames = set()
        self._planner_decay_version = 0
        self._planner_lock_version = 0
        self._planner_access_version = 0
        self._planner_slot_version = 0
        self._planning_timings = {}
        self._planning_diagnostic = {}
        self._slot_descriptions = {}
        self.pressure_work_slot = pressure_work_slot
        self._pressure_tails: set[int] = set()

    def allocate(
        self,
        needed_pages : int,
        needed_states: int,
        needed_works : int = 0,
        profile     : dict | None = None,
        *,
        active_candidates: Callable[[LogRadixTreeNode | None], list] | None = None,
        apply_active: Callable | None = None,
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
        changed_leaf = None
        while reserved is None:
            active = () if active_candidates is None else active_candidates(changed_leaf)
            plan = self.plan(
                needed_pages, needed_states, needed_works,
                tuple(candidate.native_cost for candidate in active),
            )
            selected = plan.leaves
            checkpoint_pairs = plan.checkpoint_merges
            released_slots = plan.retired_slots
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
                        "node_uuid": snapshot.node.uuid,
                        "kind": entry.kind.value,
                        "begin": entry.begin,
                        "end": entry.end,
                        "slot": entry.slot,
                    }
                    for snapshot in selected for entry in snapshot.entries
                ]
                if self._profile_enabled else None
            )
            evicted_nodes = [] if self._profile_enabled else None
            free_frames_before = self._allocator.free_frames if self._profile_enabled else 0
            self.tree.prune(selected, evicted_nodes)
            if profile is not None:
                now = perf_counter_ns()
                profile["timing_ms"]["tree_evict_ms"] = (now - stage) / 1e6
                stage = now
            self._release_cached_kv(plan.kv_references)
            if profile is not None:
                now = perf_counter_ns()
                profile["timing_ms"]["release_kv_ms"] = (now - stage) / 1e6
                stage = now
            for slot in released_slots:
                assert not self.tree.has_slot(slot)

            for slot in released_slots:
                self._slot_descriptions.pop(slot, None)
            self.pool.retire_slots(released_slots)
            materialized = None
            changed_leaf = None
            if plan.materialization:
                owner, materialized = self._materialize(plan.materialization)
                if owner.is_leaf() and not selected and not checkpoint_pairs:
                    changed_leaf = owner
            active_result = None
            if plan.active_action is not None:
                active_result = apply_active(active[plan.active_action], plan.active_frames, plan.active_cost)
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
            assert (reserved is not None or materialized is not None or active_result is not None
                    or (active and (selected or released_slots))), (
                "Eviction plan did not free the projected frames: "
                f"free={self._allocator.free_frames}, "
                f"required={self._allocator.required_frames(needed_pages, needed_states, needed_works)}, "
                f"kv={needed_pages}, logs={needed_states}, works={needed_works}, "
                f"slots={len(released_slots)}, checkpoint_drops={len(checkpoint_pairs)}"
            )
            if self._planner is not None and not self._planner_needs_rebase:
                self._planner_epoch = self._eviction_epoch()

            if self._profile_enabled:
                evicted_tokens = sum(leaf.end - leaf.begin for leaf in selected)
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
                    "materialization": materialized,
                    "active_reclamation": active_result,
                    "kv_tokens": evicted_tokens,
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
                self._profile_stats["evicted_kv_tokens"] += evicted_tokens
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
        active_candidates=(),
    ) -> ReclamationPlan:
        if not self._profile_enabled:
            return self._plan(needed_pages, needed_states, needed_works, active_candidates)
        start = perf_counter_ns()
        previous, current = self._planner_epoch, self._eviction_epoch()
        labels = ("structure", "locks", "access", "policy")
        causes = ([name for name, old, new in zip(labels, previous, current, strict=True)
                   if old != new] if self._planner is not None and previous is not None
                  else ["cold_start"])
        self._planning_diagnostic = {
            "causes": causes, "query": self.analyzer.query_count,
            "mode": "rebuild", "full_snapshots": 0,
        }
        try:
            return self._plan(needed_pages, needed_states, needed_works, active_candidates)
        finally:
            self._planning_diagnostic["total_ms"] = (perf_counter_ns() - start) / 1e6
            self._planning_diagnostic["timing_ms"] = dict(self._planning_timings)

    def _plan(
        self, needed_pages: int, needed_states: int, needed_works: int,
        active_candidates,
    ) -> ReclamationPlan:
        from logssms.kernel import logstate as planning

        self._planning_timings = {}
        stage = perf_counter_ns() if self._profile_enabled else 0
        config = self._eviction_capacity(needed_pages, needed_states, needed_works)
        epoch = self._eviction_epoch()
        lock_update = None
        if (self._planner is not None and self._planner_epoch is not None
                and epoch[:1] + epoch[2:] == self._planner_epoch[:1] + self._planner_epoch[2:]
                and epoch[1] != self._planner_epoch[1]):
            lock_update = self._planner_lock_updates()
            self._planner_epoch = epoch
            stage = perf_counter_ns() if self._profile_enabled else 0
        if self._planner is not None and epoch == self._planner_epoch:
            model_update = None
            if self._model_epoch() != self._planner_model_epoch:
                model_update = self._planner_model_updates()
                stage = perf_counter_ns() if self._profile_enabled else 0
            decay_changes = self._allocator.decay_changes
            dirty = decay_changes.since(self._planner_decay_version)
            frames = self._planner_decay_frames if dirty is None else dirty & self._planner_decay_frames
            occupied = [(frame, self._allocator.decay_counts.get(frame, 0)) for frame in sorted(frames)]
            self._planner_decay_version = decay_changes.version
            self._planning_stage("eviction_resource_ms", stage)
            stage = perf_counter_ns() if self._profile_enabled else 0
            rows = self._planner.plan(
                config, occupied, active_candidates, active_occupied=self._allocator.decay_counts,
                locks=lock_update, models=model_update,
            )
            self._planning_stage("eviction_select_ms", stage)
            if rows:
                if self._profile_enabled:
                    self._profile_stats["eviction_planner_reuses"] += 1
                    self._planning_diagnostic["mode"] = "reuse"
                return self._consume_plan(rows)
            # The unsuccessful search can have staged hypothetical releases.
            self._planner = None
        if self._profile_enabled and not self._planning_diagnostic["causes"]:
            self._planning_diagnostic["causes"] = ["planner_exhausted"]
        allocator = self._allocator
        if self._planner is not None:
            stage = perf_counter_ns() if self._profile_enabled else 0
            rows = self._rebase_planner(config, active_candidates)
            self._planning_stage("eviction_rebase_ms", stage)
            if rows:
                if self._profile_enabled:
                    self._planning_diagnostic["mode"] = "rebase"
                return self._consume_plan(rows)
            self._planner = None
            self._planner_nodes = []
            raise LogStateCapacityError(
                f"Insufficient evictable LogState cache capacity: need_pages={needed_pages}, "
                f"need_states={needed_states}, free_frames={allocator.free_frames}."
            )

        pressure_groups = self.tree.pressure_materializations(allocator.materialize_log_count)
        self._pressure_tails = {group[1] for group in pressure_groups if group[1] >= 0}
        stage = perf_counter_ns() if self._profile_enabled else 0
        nodes, topology = self.tree.eviction_snapshot()
        if self._profile_enabled:
            self._profile_stats["eviction_full_snapshots"] += 1
            self._planning_diagnostic["full_snapshots"] += 1
            self._profile_stats["eviction_serialized_nodes"] += len(nodes)
        self._planning_stage("eviction_snapshot_ms", stage)
        stage = perf_counter_ns() if self._profile_enabled else 0
        node_slots = [snapshot.eviction_slots for snapshot in nodes]
        slot_ids = {slot for row in node_slots for slot in row}
        slots = []
        decay_frames = set()
        for slot in slot_ids:
            row = self._describe_slot(slot)
            slots.append(row)
            if row[2] >= 0:
                decay_frames.add(row[2])
        kv = [[value for frame, count in snapshot.frame_counts.items() for value in (
            frame, count, self.cached_page_tokens[frame * self.page_size],
        )] for snapshot in nodes]
        occupied = [(frame, allocator.decay_counts.get(frame, 0)) for frame in decay_frames]
        actions = self.tree.checkpoint_eviction_actions(nodes)
        self._planning_stage("eviction_resource_ms", stage)
        stage = perf_counter_ns() if self._profile_enabled else 0
        models, past_tails = self.analyzer.eviction_models(nodes)
        self._planning_stage("eviction_models_ms", stage)
        stage = perf_counter_ns() if self._profile_enabled else 0
        self._planner = planning.EvictionPlanner(
            topology, kv, node_slots, slots, occupied, config, actions, models,
            [(snapshot.node.depth, snapshot.node.uuid, snapshot.begin) for snapshot in nodes],
            pressure_groups, past_tails=past_tails,
        )
        self._planning_stage("eviction_build_ms", stage)
        self._planner_model_epoch = self._model_epoch()
        self._planner_nodes = nodes
        self._planner_index = {snapshot.node.uuid: i for i, snapshot in enumerate(nodes)}
        self._planner_topology = {snapshot.node.uuid: row[1:] for snapshot, row in zip(nodes, topology, strict=True)}
        self._planner_frames = {snapshot.node.uuid: snapshot.frame_counts for snapshot in nodes}
        self._planner_needs_rebase = False
        self._planner_decay_frames = decay_frames
        self._planner_decay_version = allocator.decay_changes.version
        self._planner_lock_version = self.tree.reclamation_lock_changes.version
        self._planner_access_version = self.tree.reclamation_access_changes.version
        self._planner_slot_version = allocator.slot_changes.version
        stage = perf_counter_ns() if self._profile_enabled else 0
        rows = self._planner.plan(config, (), active_candidates, active_occupied=allocator.decay_counts)
        self._planning_stage("eviction_select_ms", stage)
        if rows:
            return self._consume_plan(rows)
        self._planner = None
        self._planner_nodes = []
        raise LogStateCapacityError(
            "Insufficient evictable LogState cache capacity: "
            f"need_pages={needed_pages}, need_states={needed_states}, "
            f"free_frames={allocator.free_frames}."
        )

    def _rebase_planner(self, config, active_candidates):
        tree = self.tree
        nodes = tree.eviction_nodes()
        previous = self._planner_index
        slot_changes = self._allocator.slot_changes
        changed_slots = slot_changes.since(self._planner_slot_version)
        dirty_owners = (None if changed_slots is None else
                        {tree.entry_owner(slot).uuid for slot in changed_slots if tree.has_slot(slot)})
        lock_changes = tree.reclamation_lock_changes
        changed_locks = lock_changes.since(self._planner_lock_version)
        access_changes = tree.reclamation_access_changes
        changed_access = access_changes.since(self._planner_access_version)
        current_ids = {node.uuid for node in nodes}
        expired = previous.keys() - current_ids
        removed = [previous[uuid] for uuid in sorted(expired)]
        used = set(previous.values())
        free_indices = iter(i for i, s in enumerate(self._planner_nodes) if s is None and i not in used)
        new_index = len(self._planner_nodes)
        by_uuid = {}
        for node in nodes:
            uuid = node.uuid
            index = previous.get(uuid)
            if index is None:
                index = next(free_indices, None)
                if index is None:
                    index = new_index
                    new_index += 1
            by_uuid[uuid] = index
        native = [None] * new_index
        frames = {}
        changed_frames = {frame for uuid in expired for frame in self._planner_frames[uuid]}
        next_topology = {}
        dirty_indices = set()
        geometry, actions, groups = [], [], []
        recovery_ends = {}
        current = []
        for node in nodes:
            uuid = node.uuid
            index = by_uuid[uuid]
            parent_index = -1 if node.is_root() else by_uuid[node.parent.uuid]
            recovery = recovery_ends.get(parent_index, 0)
            checkpoint, ends = node.eviction_recovery
            anchor = recovery if checkpoint is None else checkpoint
            recovery_ends[index] = ends.get(anchor, anchor)
            old = self._planner_nodes[index] if uuid in previous else None
            old_frames = self._planner_frames.get(uuid)
            slots = node.eviction_slots
            state_changed = old is None or old.eviction_slots is not slots
            lock_changed = changed_locks is None or uuid in changed_locks
            access_changed = changed_access is None or uuid in changed_access
            row = (tree.eviction_topology(node, parent_index) if state_changed or lock_changed or access_changed
                   else (parent_index, *self._planner_topology[uuid][1:]))
            next_topology[uuid] = row
            frame_counts = frames[uuid] = node.eviction_frame_counts
            if (state_changed or old.begin != node.begin or old.end != node.depth
                    or old.recovery_begin != recovery):
                snapshot = LogRadixEvictionNode(node, node.begin, node.depth, slots, tree, recovery)
            else:
                snapshot = old
            native[index] = snapshot
            current.append(snapshot)
            if (not state_changed and dirty_owners is not None and uuid not in dirty_owners
                    and not lock_changed and not access_changed
                    and self._planner_topology[uuid] == row and old_frames is frame_counts
                    and old.recovery_begin == snapshot.recovery_begin
                    and parent_index not in dirty_indices):
                continue
            dirty_indices.add(index)
            if old_frames is not frame_counts:
                changed_frames.update(frame_counts)
                if old_frames is not None:
                    changed_frames.update(old_frames)
            kv = [value for frame, count in frame_counts.items()
                  for value in (frame, count, self.cached_page_tokens[frame * self.page_size])]
            slots = []
            for slot in snapshot.eviction_slots:
                record = self._describe_slot(slot)
                slots.extend(record)
                if record[2] >= 0:
                    self._planner_decay_frames.add(record[2])
            candidates = tree.pressure_materializations(self._allocator.materialize_log_count, node=node)
            if old is not None:
                self._pressure_tails.difference_update(old.eviction_slots)
            self._pressure_tails.update(group[1] for group in candidates if group[1] >= 0)
            geometry.extend(((index, *row), kv, slots, (len(candidates),)))
            actions.extend((slot, right, tic, index) for slot, right, tic, _
                           in tree.checkpoint_eviction_actions([snapshot]))
            groups.extend(candidates)

        allocator = self._allocator
        occupied = [(frame, allocator.decay_counts.get(frame, 0)) for frame in self._planner_decay_frames]
        kv_totals = [(frame, self.cached_page_tokens.get(frame * self.page_size, 0))
                     for frame in sorted(changed_frames)]
        header = (len(dirty_indices), len(actions), len(groups), len(occupied), len(kv_totals))
        models, past_tails = self.analyzer.eviction_models(current)
        indices = list(by_uuid.values())
        order = [(s.end, s.node.uuid, s.begin) for s in current]
        result = self._planner.rebase(
            new_index, removed, [header, *geometry, *actions, *groups, *occupied, *kv_totals],
            indices, models, order, config, active_candidates,
            past_tails=past_tails, active_occupied=allocator.decay_counts,
        )
        self._planner_nodes = native
        self._planner_index = by_uuid
        self._planner_frames = frames
        self._planner_topology = next_topology
        self._planner_decay_version = allocator.decay_changes.version
        self._planner_lock_version = lock_changes.version
        self._planner_access_version = access_changes.version
        self._planner_slot_version = slot_changes.version
        self._planner_model_epoch = self._model_epoch()
        self._planner_needs_rebase = False
        return result

    def _describe_slot(self, slot: int) -> tuple[int, ...]:
        record = self._allocator.slot(slot)
        entry = self.tree.entry_for_slot(slot)
        cached = self._slot_descriptions.get(slot)
        if cached is None or cached[0] is not record or cached[1] is not entry:
            row = (slot, len(record.payload) + sum(map(len, record.conv)),
                   -1 if record.decay is None else record.decay.frame, entry.begin, entry.end,
                   int(entry.kind is LogStateKind.CHECKPOINT))
            self._slot_descriptions[slot] = (record, entry, row)
            return row
        return cached[2]

    def _update_materialized_node(self, owner: LogRadixTreeNode) -> bool:
        if self._planner is None or self._planner_needs_rebase:
            return False
        index = self._planner_index[owner.uuid]
        old = self._planner_nodes[index]
        assert old is not None
        frontiers = []
        snapshots = []
        groups = []
        stack = [(owner, old.recovery_begin)]
        while stack:
            node, frontier = stack.pop()
            node_index = self._planner_index.get(node.uuid)
            if node_index is None:
                return False
            previous = self._planner_nodes[node_index]
            if previous is None or (node is not owner and previous.eviction_slots is not node.eviction_slots):
                # Updating a descendant's frontier does not publish changes to its slots.
                return False
            candidates = self.tree.pressure_materializations(self._allocator.materialize_log_count, node=node)
            groups.extend(candidates)
            snapshot = LogRadixEvictionNode(
                node, node.begin, node.depth, node.eviction_slots, self.tree, frontier,
            )
            frontiers.append((node_index, frontier, len(candidates)))
            snapshots.append((node_index, snapshot))
            checkpoint, ends = node.eviction_recovery
            if checkpoint is not None:
                frontier = checkpoint
            child_frontier = ends.get(frontier, frontier)
            stack.extend((child, child_frontier) for child in node.children.values())
        actions = [(slot, right, timestamp, index) for slot, right, timestamp, _
                   in self.tree.checkpoint_eviction_actions([snapshots[0][1]])]
        timestamp = min(map(self.tree.checkpoint_timestamp, owner.checkpoint_slots), default=owner.timestamp)
        if not self._planner.replace_materialized_node(
            index, timestamp, map(self._describe_slot, owner.eviction_slots), actions, frontiers, groups,
        ):
            return False
        for node_index, snapshot in snapshots:
            previous = self._planner_nodes[node_index]
            self._pressure_tails.difference_update(previous.eviction_slots)
            self._planner_nodes[node_index] = snapshot
        self._planner_topology[owner.uuid] = (*self._planner_topology[owner.uuid][:2], timestamp)
        self._pressure_tails.update(group[1] for group in groups if group[1] >= 0)
        return True

    def _planning_stage(self, name: str, start: int) -> None:
        if self._profile_enabled:
            elapsed = (perf_counter_ns() - start) / 1e6
            self._planning_timings[name] = self._planning_timings.get(name, 0.0) + elapsed

    def _eviction_epoch(self):
        return (self.tree.revision, self.tree.lock_revision, self.tree.access_revision,
                self.analyzer.enable_merge_adjustment)

    def _planner_lock_updates(self):
        stage = perf_counter_ns() if self._profile_enabled else 0
        changes = self.tree.reclamation_lock_changes
        dirty = changes.since(self._planner_lock_version)
        indices = (range(len(self._planner_nodes)) if dirty is None
                   else sorted(self._planner_index[uuid] for uuid in dirty if uuid in self._planner_index))
        live = [(index, self._planner_nodes[index]) for index in indices
                if self._planner_nodes[index] is not None]
        self._planner_lock_version = changes.version
        blocked = [(index, int(snapshot.node.is_root() or snapshot.node.is_locked
                               or any(slot in self.tree._recovery_slot_refs
                                      for slot in snapshot.eviction_slots if self.tree.has_slot(slot))))
                   for index, snapshot in live]
        actions = self.tree.checkpoint_eviction_actions([snapshot for _, snapshot in live])
        actions = [(slot, right, timestamp, live[index][0])
                   for slot, right, timestamp, index in actions]
        for index, unavailable in blocked:
            node = self._planner_nodes[index].node
            parent, _, timestamp = self._planner_topology[node.uuid]
            self._planner_topology[node.uuid] = parent, unavailable, timestamp
        self._planning_stage("eviction_locks_ms", stage)
        if self._profile_enabled:
            self._profile_stats["eviction_lock_refreshes"] += 1
        return (blocked, actions) if blocked else None

    def _model_epoch(self):
        return self.analyzer.query_count, len(self.analyzer._owners)

    def _planner_model_updates(self):
        stage = perf_counter_ns() if self._profile_enabled else 0
        live = [(index, snapshot) for index, snapshot in enumerate(self._planner_nodes)
                if snapshot is not None]
        live.sort(key=lambda item: (item[1].node.depth, item[1].node.uuid, item[1].begin))
        models, past_tails = self.analyzer.eviction_models([snapshot for _, snapshot in live])
        self._planning_stage("eviction_models_ms", stage)
        self._planner_model_epoch = self._model_epoch()
        if self._profile_enabled:
            self._profile_stats["eviction_model_refreshes"] += 1
        return [index for index, _ in live], models, past_tails

    def _eviction_capacity(self, pages, logs, works):
        allocator = self._allocator
        return (allocator.free_frames, allocator._free_decay_records,
                allocator.payload_frames, allocator.conv_frames, allocator.num_granules,
                allocator.decay_records_per_frame, pages, logs, works,
                allocator.log_payload_frames)

    def _consume_plan(self, rows):
        selected, pages, released, compressed, *materialization = rows
        action = materialization[0] if materialization else ()
        active_action, frames, cost = None, 0, 0.
        if action and action[0] == -2:
            _, active_action, frames, bits = action
            cost, = struct.unpack("d", struct.pack("q", bits))
        snapshots = [self._planner_nodes[index] for index in selected]
        for index in selected:
            self._planner_nodes[index] = None
        if (any(snapshot.node.analysis_owner is None for snapshot in snapshots)
                or not self._pressure_tails.isdisjoint(released)):
            self._planner_needs_rebase = True
        self._planner_epoch = None
        return ReclamationPlan(
            snapshots, list(zip(pages[::2], pages[1::2], strict=True)), released,
            list(zip(compressed[::2], compressed[1::2], strict=True)),
            tuple(action) if active_action is None else (), active_action, frames, cost,
        )

    def _materialize(self, candidate: tuple[int, ...]) -> tuple[LogRadixTreeNode, dict]:
        base, tail, *chain = candidate
        first_slot = next(i for i, slot in enumerate(chain) if slot >= 0)
        prerequisites = [self.tree.entry_for_slot(-slot - 2) for slot in chain[:first_slot]]
        slots = chain[first_slot:]
        entries = [self.tree.entry_for_slot(slot) for slot in slots]
        first, last = entries[0], entries[-1]
        owner = self.tree.entry_owner(last)
        checkpoint = None if base < 0 else self.tree.entry_for_slot(base)
        recovery = [*prerequisites, *entries]
        assert (0 if checkpoint is None else checkpoint.end) == recovery[0].begin
        assert all(left.end == right.begin for left, right in zip(recovery, recovery[1:]))
        assert all(self.tree.entry_owner(entry) is owner and entry.kind is LogStateKind.LOG
                   for entry in entries)
        assert not owner.is_locked and not any(slot in self.tree._recovery_slot_refs for slot in slots)
        absorb = tail < 0
        frame_savings = self._allocator.checkpoint_frame_savings(slots)
        if absorb:
            assert checkpoint is not None and self.tree.entry_owner(checkpoint) is owner
            assert base not in self.tree._recovery_slot_refs and last.end == owner.depth
            frame_savings += self._allocator.payload_frames + self._allocator.conv_frames
            begin, destination = checkpoint.begin, base
        else:
            begin, destination = first.begin, last.slot
        assert frame_savings > 0
        logs = [(entry.slot, 0, entry.end - entry.begin) for entry in recovery]
        for slot in slots:
            self.tree.detach_entry(slot)
            self._slot_descriptions.pop(slot, None)
        if absorb:
            self.tree.detach_entry(base)
            self._slot_descriptions.pop(base, None)
        self.pool.construct_checkpoint(
            None if base < 0 else base, logs, self.pressure_work_slot, tuple(slots), absorb=absorb,
        )
        self.tree.attach_entry(
            owner,
            LogStateEntry(destination, LogStateKind.CHECKPOINT, begin, last.end,
                          span_granules=(last.end - begin) // self.page_size),
            checkpoint_timestamp=owner.timestamp,
        )
        self.pool.retire_slots(slots if absorb else slots[:-1])
        if not self._update_materialized_node(owner):
            self._planner_needs_rebase = True
        return owner, dict(begin=first.begin, end=last.end, log_slots=len(slots), frames=frame_savings,
                           absorbed=absorb, ancestor_logs=len(prerequisites))

    @staticmethod
    def _record_pool_timings(
        profile: dict, prefix: str, timings: dict | None,
    ) -> None:
        assert timings is not None
        for name, value in timings.items():
            bucket = "counts" if name in (
                "waited_merge_slots", "descriptor_count", "merge_reclamation_count", "failed",
            ) else "timing_ms"
            profile.setdefault(bucket, {})[f"{prefix}_{name}"] = value

    def _release_cached_kv(self, page_counts: Iterable[tuple[int, int]]) -> None:
        released = []
        for frame, count in page_counts:
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
