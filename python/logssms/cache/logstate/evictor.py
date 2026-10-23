from __future__ import annotations

from collections.abc import Callable, Mapping
from time import perf_counter_ns

from .frame_pool import MHALogStatePool
from .prefix_cache import LogRadixTree, LogRadixTreeNode, LogStateKind
from .recycler import LogStateRecycleBin


class LogStateCapacityError(RuntimeError):
    pass


class LogStateEvictor:
    def __init__(
        self,
        tree: LogRadixTree,
        pool: MHALogStatePool,
        recent_checkpoint_limit: int,
        *,
        profile_enabled: bool,
        profile_stats: dict,
        allocation_samples: list[dict],
        record_lifecycle_event: Callable[[dict], None],
    ) -> None:
        self.tree = tree
        self.pool = pool
        self.page_size = pool.page_size
        self.recent_checkpoint_limit = recent_checkpoint_limit
        self.cached_page_tokens: dict[int, int] = {}
        self._allocator = pool.allocator
        self._profile_enabled = profile_enabled
        self._profile_stats = profile_stats
        self._profile_allocation_samples = allocation_samples
        self._record_lifecycle_event = record_lifecycle_event

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
            if recycle_bin is not None:
                for node in selected:
                    recycle_bin.remember(self.tree, node)
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

    def plan(
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
                    frame, count, self.cached_page_tokens[frame * self.page_size],
                )] for node in nodes],
                node_slots,
                slots,
                [(frame, allocator._decay_masks[frame].bit_count()) for frame in decay_frames],
                (allocator.free_frames, allocator._free_decay_records,
                 allocator.payload_frames, allocator.conv_frames, allocator.num_granules,
                 allocator.decay_records_per_frame, needed_pages, needed_states, needed_works,
                 allocator.log_payload_frames, self.recent_checkpoint_limit),
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
