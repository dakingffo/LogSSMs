from __future__ import annotations

import time
from bisect import bisect_right
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import cached_property
from heapq import heappop, heappush
from itertools import count
from typing import Any, Callable, Iterable, NamedTuple, TypeAlias, TYPE_CHECKING

if TYPE_CHECKING:
    from logssms.cache.logstate.recycler import (
        RecycledMatch,
        RecycledRestorePlan,
    )

import torch

from logssms.utils import align_down

from logssms.cache.prefix_cache import BaseCacheHandle, BasePrefixCache, MatchResult, SizeInfo

KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]


def _get_key_fn(granularity: int) -> KEY_FN:
    if granularity == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:granularity].tolist())


class LogStateKind(Enum):
    LOG = "log"
    CHECKPOINT = "checkpoint"


@dataclass
class LogStateMergeStats:
    origin          : int
    tail            : int
    intensity       : int = 2
    hit_positions   : deque[int] = field(default_factory=deque)
    next_update_tail: int | None = None


@dataclass(frozen=True)
class LogStateEntry:
    slot         : int
    kind         : LogStateKind
    begin        : int
    end          : int
    span_granules: int | None = None

    def __post_init__(self) -> None:
        assert self.slot >= 0
        assert 0 <= self.begin < self.end
        assert self.span_granules is None or self.span_granules > 0
        if self.kind is LogStateKind.LOG:
            assert self.span_granules is None, "A Log has no checkpoint span."
        else:
            assert self.span_granules is not None, "A Checkpoint needs an exact granule span."


class LogRadixTreeNode:
    counter = count(0)

    def __init__(
        self,
        key_fn     : KEY_FN,
        granularity: int,
        tic        : int | None = None,
    ) -> None:
        self.uuid = next(LogRadixTreeNode.counter)
        self.key_fn = key_fn
        self.granularity = granularity
        self.children: dict[Any, LogRadixTreeNode] = {}
        self._parent: LogRadixTreeNode | None = None
        self._depth = 0
        self.ref_count = 0
        self._endpoint_refs: Counter[int] = Counter()
        self.is_protected = False

        self._timestamp = time.monotonic_ns() if tic is None else tic
        self._log_leaf_index = -1
        self._log_leaf_tree: LogRadixTree | None = None
        self._key = torch.empty(0, dtype=torch.int32)
        self._kv = torch.empty(0, dtype=torch.int32)
        self._kv_frames = torch.empty(0, dtype=torch.int32)
        self._kv_frame_counts: dict[int, int] = {}
        self._entries: list[LogStateEntry] = []
        self._checkpoint_entries: list[LogStateEntry] = []
        self._append_buffers: tuple[torch.Tensor, ...] = ()
        self.merge_stats: LogStateMergeStats | None = None

    def initialize(
        self,
        input_ids : torch.Tensor,
        kv_indices: torch.Tensor,
        kv_frames : torch.Tensor | None,
    ) -> None:
        assert len(input_ids) == len(kv_indices)
        self._key = input_ids
        self._kv = kv_indices.clone()
        self._kv_frames = (
            torch.div(kv_indices.to(device="cpu", dtype=torch.int32),
                      self.granularity, rounding_mode="floor")
            if kv_frames is None else kv_frames.clone()
        )
        self._kv_frame_counts = dict(Counter(self._kv_frames.tolist()))

    def append(self, keys: torch.Tensor, kv: torch.Tensor, frames: torch.Tensor) -> None:
        length, end = self.length, self.length + len(keys)
        if not self._append_buffers or self._append_buffers[0].numel() < end:
            capacity = max(end, max(1, length) * 2)
            self._append_buffers = tuple(
                torch.empty(capacity, dtype=value.dtype, device=value.device)
                for value in (self._key, self._kv, self._kv_frames)
            )
            for buffer, value in zip(self._append_buffers, (self._key, self._kv, self._kv_frames)):
                buffer[:length].copy_(value)
        for buffer, value in zip(self._append_buffers, (keys, kv, frames)):
            buffer[length:end].copy_(value)
        self._key, self._kv, self._kv_frames = (
            buffer[:end] for buffer in self._append_buffers
        )
        self._depth += len(keys)

    def set_parent(self, parent: LogRadixTreeNode) -> None:
        self._parent = parent
        self._depth = parent.depth + self.length
        parent.children[self.key_fn(self._key)] = self

    @property
    def parent(self) -> LogRadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def timestamp(self) -> int:
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value: int) -> None:
        self._timestamp = value
        if self._log_leaf_index >= 0:
            self._log_leaf_tree._update_log_leaf(self)

    @property
    def key(self) -> torch.Tensor:
        return self._key

    @property
    def kv(self) -> torch.Tensor:
        return self._kv

    @property
    def entries(self) -> tuple[LogStateEntry, ...]:
        return tuple(self._entries)

    @cached_property
    def eviction_slots(self) -> tuple[int, ...]:
        return tuple(entry.slot for entry in self._entries)

    @cached_property
    def checkpoint_slots(self) -> tuple[int, ...]:
        return tuple(entry.slot for entry in self._checkpoint_entries)

    @property
    def length(self) -> int:
        return len(self._key)

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def begin(self) -> int:
        return self.depth - self.length

    @property
    def is_locked(self) -> bool:
        return self.ref_count > 0

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        from logssms.kernel import fast_compare_key

        return fast_compare_key(self._key, input_ids)

    def __lt__(self, other: LogRadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


class LogRadixTrace(NamedTuple):
    node            : LogRadixTreeNode
    matched_len     : int
    matched_in_node : int

    @property
    def is_exact(self) -> bool:
        return self.node.is_root() or self.matched_in_node == self.node.length


@dataclass(frozen=True)
class LogStateRecovery:
    position            : int
    base_checkpoint_slot: int | None
    logs                : tuple[LogStateEntry, ...]
    conv_slot           : int | None
    conv_granule        : int | None

    @property
    def log_slots(self) -> tuple[int, ...]:
        return tuple(entry.slot for entry in self.logs)


@dataclass(frozen=True)
class LogStateCacheHandle(BaseCacheHandle):
    node              : LogRadixTreeNode
    lock_node         : LogRadixTreeNode
    recovery          : LogStateRecovery
    merge_stats       : LogStateMergeStats | None = None
    source_merge_stats: LogStateMergeStats | None = None
    match_position    : int = 0
    recycle_match     : RecycledMatch | None = None
    restore_plan      : RecycledRestorePlan | None = None
    request_end       : int = 0
    protected_endpoints: tuple[int, ...] = ()
    match_trace       : LogRadixTrace | None = field(default=None, compare=False)
    trace_revision    : int = field(default=-1, compare=False)
    trace_input_len   : int = field(default=0, compare=False)
    match_path        : tuple[LogRadixTreeNode, ...] = field(default=(), compare=False, repr=False)
    match_entries     : tuple[LogStateEntry, ...] = field(default=(), compare=False, repr=False)
    kv_parts          : tuple[torch.Tensor, ...] = field(default=(), compare=False, repr=False)

    @cached_property
    def kv_indices(self) -> torch.Tensor:

        return torch.cat(self.kv_parts) if self.kv_parts else self.node.kv[:0]

    def get_matched_indices(self) -> torch.Tensor:
        return self.kv_indices


class LogRadixInsert(NamedTuple):
    node        : LogRadixTreeNode
    previous_len: int
    inserted_len: int


class LogRadixEviction(NamedTuple):
    kv_indices: torch.Tensor
    entries   : tuple[LogStateEntry, ...]


@dataclass(frozen=True)
class LogRadixEvictionSegment:
    node: LogRadixTreeNode
    begin: int
    end: int
    uuid: int
    parent_uuid: int
    storage_slots: tuple[int, ...]
    tree: LogRadixTree = field(repr=False, compare=False)

    @property
    def depth(self) -> int:
        return self.end

    @property
    def kv(self) -> torch.Tensor:
        return self.node.kv[self.begin - self.node.begin:self.end - self.node.begin]

    @property
    def entries(self) -> tuple[LogStateEntry, ...]:
        return tuple(self.tree.entry_for_slot(slot) for slot in self.storage_slots
                     if self.tree.has_slot(slot))

    @property
    def eviction_slots(self) -> tuple[int, ...]:
        return self.storage_slots

    @property
    def _kv_frame_counts(self) -> dict[int, int]:
        return dict(Counter(self.node._kv_frames[
            self.begin - self.node.begin:self.end - self.node.begin].tolist()))

    @property
    def timestamp(self) -> int:
        return self.node.timestamp

    @property
    def merge_stats(self):
        return self.node.merge_stats


class LogRadixTree(BasePrefixCache):

    def __init__(
        self,
        granularity         : int,
        device              : torch.device,
        on_slot_reclaimable : Callable[[int, bool], None] | None = None,
        checkpoint_log_count: int = 1,
    ) -> None:
        assert granularity > 0 and checkpoint_log_count > 0
        self.checkpoint_log_count = checkpoint_log_count
        self.granularity = granularity
        self.device = device
        self.key_fn = _get_key_fn(granularity)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.root = LogRadixTreeNode(self.key_fn, granularity)
        self.root.ref_count = 1
        self.evictable_size = 0
        self.locked_size = 0
        self.revision = 0
        self.lock_revision = 0
        self._lock_footprint_epoch = (-1, -1)
        self._lock_footprints: dict[int, tuple[int, tuple[int, ...]]] = {}
        self._entry_owners: dict[int, LogRadixTreeNode] = {}
        self._leaves: dict[int, LogRadixTreeNode] = {}
        self._log_leaf_heap: list[tuple[int, int, LogRadixTreeNode]] = []
        self._entries_by_slot: dict[int, LogStateEntry] = {}
        self._checkpoint_timestamps: dict[int, int] = {}
        self._kv_frame_refs: dict[int, int] = {}
        self._locked_kv_refs: dict[int, int] = {}
        self._evictable_kv_count = 0

        self._unprunable_log_refs: dict[int, int] = {}
        self._recovery_slot_refs: dict[int, int] = {}
        self._evictable_slots: dict[int, None] = {}
        self._on_slot_reclaimable = on_slot_reclaimable

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        usable_len = align_down(len(input_ids), self.granularity)
        trace = self.trace(input_ids[:usable_len])
        path = self.path(trace.node)
        entries = tuple(entry for part in path for entry in part.entries)
        revision = self.revision
        recovery, lock_node = self._recovery_for_trace(trace, entries)
        if recovery.position == 0:
            node = self.root
        else:
            node = self._node_at_depth(trace.node, recovery.position)
            if node is not trace.node or revision != self.revision:
                path = self.path(node)
                entries = tuple(entry for part in path for entry in part.entries)
        if recovery.position == 0:
            path, entries = (), ()

        trace_node = trace.node
        while not trace_node.is_root() and trace.matched_len <= trace_node.begin:
            trace_node = trace_node.parent
        matched_trace = LogRadixTrace(
            trace_node, trace.matched_len,
            trace.matched_len - trace_node.begin,
        )
        handle = LogStateCacheHandle(
            cached_len=recovery.position,
            node=node,
            lock_node=node,
            recovery=recovery,
            match_trace=matched_trace,
            trace_revision=self.revision,
            trace_input_len=usable_len,
            match_path=path,
            match_entries=entries,
            kv_parts=tuple(part.kv for part in path) if path else (self.empty_tensor,),
        )
        self.touch_handle(handle)
        return MatchResult(
            handle,
            tree_match_len=trace.matched_len,
            recoverable_len=recovery.position,
        )

    def touch_handle(self, handle: LogStateCacheHandle) -> None:
        tic = time.monotonic_ns()
        for node in handle.match_path:
            node.timestamp = tic
        slot = handle.recovery.base_checkpoint_slot
        if slot is not None:
            self._checkpoint_timestamps[slot] = tic

    def checkpoint_timestamp(self, slot: int) -> int:
        return self._checkpoint_timestamps[slot]

    def lock_handle(
        self,
        handle: BaseCacheHandle,
        unlock: bool = False,
    ) -> None:
        assert isinstance(handle, LogStateCacheHandle)
        self.lock(handle.lock_node, unlock=unlock, position=handle.cached_len)

    def reset(self) -> None:
        if self._on_slot_reclaimable is not None:
            for slot in self._evictable_slots:
                self._on_slot_reclaimable(slot, False)
        self.root = LogRadixTreeNode(self.key_fn, self.granularity)
        self.root.ref_count = 1
        self.evictable_size = 0
        self.locked_size = 0
        self._entry_owners.clear()
        for node in self._leaves.values():
            node._log_leaf_tree = None
            node._log_leaf_index = -1
        self._leaves.clear()
        self._log_leaf_heap.clear()
        self._entries_by_slot.clear()
        self._checkpoint_timestamps.clear()
        self._kv_frame_refs.clear()
        self._locked_kv_refs.clear()
        self._evictable_kv_count = 0
        self._unprunable_log_refs.clear()
        self._recovery_slot_refs.clear()
        self._evictable_slots.clear()
        self.revision += 1
        self._lock_footprints.clear()

    def trace(self, input_ids: torch.Tensor) -> LogRadixTrace:
        matched_len = 0
        node = self.root
        while matched_len < len(input_ids):
            child = node.children.get(self.key_fn(input_ids[matched_len:]))
            if child is None:
                break
            match_len = align_down(
                child.get_match_len(input_ids[matched_len:]), self.granularity
            )
            matched_len += match_len
            if match_len != child.length:
                return LogRadixTrace(child, matched_len, match_len)
            node = child
        return LogRadixTrace(node, matched_len, node.length)

    def trace_prefix(self, trace: LogRadixTrace, position: int) -> LogRadixTrace:
        assert 0 <= position <= trace.matched_len
        node = trace.node
        while not node.is_root() and position <= node.begin:
            node = node.parent
        return LogRadixTrace(node, position, position - node.begin)

    def insert(
        self,
        input_ids : torch.Tensor,
        kv_indices: torch.Tensor,
        entries   : tuple[LogStateEntry, ...] = (),
        cache_len : int | None = None,
        *,
        kv_frame_spans: Iterable[tuple[int, int, int]] | None = None,
    ) -> LogRadixInsert:
        assert len(input_ids) == len(kv_indices)
        usable_len = align_down(len(input_ids), self.granularity)
        node, previous_len = self._tree_walk(input_ids[:usable_len])
        inserted_len = usable_len if cache_len is None else cache_len
        assert 0 <= inserted_len <= usable_len
        assert inserted_len % self.granularity == 0

        if inserted_len == 0:
            node = self.root
        elif inserted_len < previous_len:
            node = self._node_at_depth(node, inserted_len)
        elif previous_len < inserted_len:
            kv_frames = None
            if kv_frame_spans is not None:
                kv_frames = torch.empty(inserted_len - previous_len, dtype=torch.int32)
                covered = previous_len
                for begin, end, frame in kv_frame_spans:
                    begin, end = max(begin, previous_len), min(end, inserted_len)
                    if begin >= end:
                        continue
                    assert begin == covered
                    kv_frames[begin - previous_len:end - previous_len].fill_(frame)
                    covered = end
                assert covered == inserted_len
            suffix_ids = input_ids[previous_len:inserted_len]
            suffix_kv = kv_indices[previous_len:inserted_len]
            if not node.is_root() and node.is_leaf() and not node.is_protected:
                frame_ids = (torch.div(suffix_kv.to(device="cpu", dtype=torch.int32),
                                      self.granularity, rounding_mode="floor")
                             if kv_frames is None else kv_frames)
                node.append(suffix_ids, suffix_kv, frame_ids)
                for frame, tokens in Counter(frame_ids.tolist()).items():
                    if frame not in node._kv_frame_counts:
                        self._update_kv_frame(frame, 1, int(node.is_locked))
                    node._kv_frame_counts[frame] = node._kv_frame_counts.get(frame, 0) + tokens
                delta = inserted_len - previous_len
                if node.is_locked:
                    self.locked_size += delta
                else:
                    self.evictable_size += delta
                self._update_log_leaf(node)
            else:
                new_node = LogRadixTreeNode(self.key_fn, self.granularity)
                new_node.initialize(suffix_ids, suffix_kv, kv_frames)
                new_node.set_parent(node)
                self._remove_leaf(node)
                self._add_leaf(new_node)
                if not node.is_root() and len(node.children) > 1:
                    self.protect(node)
                node = new_node
                self._update_kv_refs(node, 1)
                self.evictable_size += node.length
            self._refresh_checkpoint_neighbors(node)
            self.revision += 1

        for entry in entries:
            assert entry.end <= inserted_len
            self.attach_entry(node, entry)
        return LogRadixInsert(node, previous_len, inserted_len)

    def attach_entry(
        self,
        leaf : LogRadixTreeNode,
        entry: LogStateEntry,
        *,
        checkpoint_timestamp: int | None = None,
    ) -> None:
        assert entry.end <= leaf.depth
        assert entry.begin % self.granularity == 0
        assert entry.end % self.granularity == 0
        if entry.kind is LogStateKind.LOG:
            assert entry.end - entry.begin == self.granularity
        else:
            assert entry.span_granules is not None
            assert entry.end - entry.begin == entry.span_granules * self.granularity
            assert entry.span_granules % self.checkpoint_log_count == 0
        assert entry.slot not in self._entries_by_slot, (
            f"Linear cache slot {entry.slot} is already owned by node "
            f"{self.entry_owner(entry.slot).uuid}."
        )
        owner = self._find_owner(leaf, entry.end)
        assert entry.kind is LogStateKind.CHECKPOINT or entry.begin >= owner.begin
        self._register_entry(owner, entry, checkpoint_timestamp)
        self.revision += 1

    def detach_entry(self, slot: int) -> LogStateEntry:
        entry = self.entry_for_slot(slot)
        owner = self.entry_owner(entry)
        owner._entries.remove(entry)
        self._unregister_entry(entry)
        self.revision += 1
        return entry

    def split_at(
        self,
        node: LogRadixTreeNode,
        pos : int,
    ) -> LogRadixTreeNode:
        assert not node.is_root()
        assert 0 < pos < node.length
        assert pos % self.granularity == 0
        parent = node.parent
        split_depth = node.begin + pos
        prefix = LogRadixTreeNode(
            self.key_fn, self.granularity, node.timestamp
        )
        prefix._key = node._key[:pos]
        prefix._kv = node._kv[:pos]
        prefix._kv_frames = node._kv_frames[:pos]
        counts = node._kv_frame_counts
        if pos <= node.length - pos:
            prefix._kv_frame_counts = dict(Counter(prefix._kv_frames.tolist()))
            small_counts = prefix._kv_frame_counts
        else:
            prefix._kv_frame_counts = counts
            node._kv_frame_counts = dict(Counter(node._kv_frames[pos:].tolist()))
            small_counts = node._kv_frame_counts
        for frame, tokens in small_counts.items():
            remaining = counts[frame] - tokens
            if remaining:
                counts[frame] = remaining
                self._update_kv_frame(frame, 1, int(node.is_locked))
            else:
                del counts[frame]
        prefix.set_parent(parent)
        prefix.ref_count = node.ref_count
        prefix._endpoint_refs = Counter({
            position: refs for position, refs in node._endpoint_refs.items()
            if position <= split_depth
        })
        suffix_refs = node.ref_count - sum(prefix._endpoint_refs.values())
        node._endpoint_refs = Counter({
            position: refs for position, refs in node._endpoint_refs.items()
            if position > split_depth
        })

        old_entries = node._entries
        node._entries = []
        for entry in old_entries:
            checkpoint_timestamp = self._checkpoint_timestamps.get(entry.slot)
            self._unregister_entry(entry)
            if entry.end <= split_depth:
                self._register_entry(prefix, entry, checkpoint_timestamp)
            else:
                self._register_entry(node, entry, checkpoint_timestamp)

        node._append_buffers = ()
        node._key = node._key[pos:]
        node._kv = node._kv[pos:]
        node._kv_frames = node._kv_frames[pos:]
        node.set_parent(prefix)
        was_locked = node.is_locked
        node.ref_count = suffix_refs
        self._update_size_class(node, was_locked)
        for part in (parent, prefix, node):
            self._refresh_checkpoint_neighbors(part)
        self.revision += 1
        return prefix

    def lock(
        self, node: LogRadixTreeNode, unlock: bool = False,
        *, position: int | None = None,
    ) -> None:
        self.lock_revision += 1
        if position is None:
            position = node.depth
        while not node.is_root() and position <= node.begin:
            node = node.parent
        if not node.is_root():
            refs = node._endpoint_refs[position] + (-1 if unlock else 1)
            assert refs >= 0
            if refs:
                node._endpoint_refs[position] = refs
            else:
                del node._endpoint_refs[position]
        while not node.is_root():
            was_locked = node.is_locked
            if unlock:
                node.ref_count -= 1
                assert node.ref_count >= 0
            else:
                node.ref_count += 1
            self._update_size_class(node, was_locked)
            node = node.parent

    def lock_footprint(self, node: LogRadixTreeNode) -> tuple[int, tuple[int, ...]]:
        epoch = self.revision, self.lock_revision
        if self._lock_footprint_epoch != epoch:
            self._lock_footprints.clear()
            self._lock_footprint_epoch = epoch
        cached = self._lock_footprints.get(node.uuid)
        if cached is not None:
            return cached
        frames, slots = set(), set()
        current = node
        while not current.is_root():
            if not current.is_locked:
                frames.update(frame for frame in current._kv_frame_counts
                              if frame not in self._locked_kv_refs)
                slots.update(entry.slot for entry in current._entries
                             if (entry.kind is LogStateKind.LOG
                                 or self._checkpoint_successor(entry) is None)
                             and entry.slot in self._evictable_slots)
            current = current.parent
        result = len(frames), tuple(slots)
        self._lock_footprints[node.uuid] = result
        return result

    def pin_recovery(self, slots: tuple[int, ...], release: bool = False) -> None:
        self.lock_revision += 1
        for slot in slots:
            refs = self._recovery_slot_refs.get(slot, 0) + (-1 if release else 1)
            assert refs >= 0
            if refs:
                assert self.has_slot(slot)
                self._recovery_slot_refs[slot] = refs
            else:
                self._recovery_slot_refs.pop(slot, None)
            self._update_slot_reclaimability(slot)
            if self.has_slot(slot):
                owner = self.entry_owner(slot)
                self._update_log_leaf(owner)
                self._refresh_checkpoint_neighbors(owner)

    def protect(
        self,
        node     : LogRadixTreeNode,
        protected: bool = True,
    ) -> None:
        assert not node.is_root()
        if node.is_protected != protected:
            node.is_protected = protected
            self.revision += 1
            self._refresh_checkpoint_neighbors(node)

    def protect_at(self, node: LogRadixTreeNode, position: int) -> None:
        self.protect(self._node_at_depth(node, position))

    def _eviction_segments(self, nodes, *, log_only=False, blocked=()):
        selected = {node.uuid for node in nodes}
        terminal = {}
        segments, rows = [], []
        for node in sorted(nodes, key=lambda value: (value.depth, value.uuid)):
            parent = -1 if node.is_root() else terminal[node.parent.uuid]
            by_end: dict[int, LogStateEntry] = {}
            for entry in node._entries:
                by_end.setdefault(entry.end, []).append(entry)
            cuts = [node.begin, *by_end]
            if cuts[-1] != node.depth:
                cuts.append(node.depth)
            pairs = zip(cuts, cuts[1:]) if not node.is_root() else [(0, 0)]
            for begin, end in pairs:
                entries = by_end.get(end, ())
                segment = LogRadixEvictionSegment(node, begin, end, len(segments), parent,
                                                 tuple(entry.slot for entry in entries), self)
                checkpoint = any(entry.kind is LogStateKind.CHECKPOINT for entry in entries)
                unavailable = (node.is_root() or node.is_locked or node.uuid in blocked
                               or any(entry.slot in self._recovery_slot_refs for entry in entries)
                               or (log_only and checkpoint))
                if end == node.depth and any(child.uuid not in selected for child in node.children.values()):
                    unavailable = True
                rows.append((segment.uuid, parent, int(unavailable), int(checkpoint),
                             min((self._checkpoint_timestamps[entry.slot] for entry in entries
                                  if entry.kind is LogStateKind.CHECKPOINT), default=node.timestamp)))
                segments.append(segment)
                parent = segment.uuid
            terminal[node.uuid] = parent
        return segments, rows

    def eviction_snapshot(self):
        return self._eviction_segments(self._iter_nodes())

    def log_eviction_snapshot(
        self, limit: int = 8,
    ) -> tuple[list[LogRadixEvictionSegment], list[tuple[int, ...]]]:
        assert limit > 0
        leaves = []
        heap = self._log_leaf_heap
        pending = [(heap[0][0], heap[0][1], 0)] if heap else []
        while pending and len(leaves) <= limit:
            _timestamp, _uuid, index = heappop(pending)
            leaves.append(heap[index][2])
            for child in (2 * index + 1, 2 * index + 2):
                if child < len(heap):
                    timestamp, uuid, _node = heap[child]
                    heappush(pending, (timestamp, uuid, child))
        if not leaves:
            return [], []
        cutoff = (leaves[-1].timestamp, leaves[-1].uuid) if len(leaves) > limit else None
        selected: dict[int, LogRadixTreeNode] = {}
        children: Counter[int] = Counter()
        for node in leaves[:limit]:
            while node.uuid not in selected:
                selected[node.uuid] = node
                if node.is_root():
                    break
                children[node.parent.uuid] += 1
                node = node.parent
        blocked = {node.uuid for node in selected.values()
                   if cutoff is not None and (node.timestamp, node.uuid) >= cutoff}
        return self._eviction_segments(selected.values(), log_only=True, blocked=blocked)

    def eviction_candidates(
        self,
    ) -> Iterable[tuple[list[LogRadixEvictionSegment], list[tuple[int, ...]], bool]]:
        limit = 8
        while True:
            nodes, rows = self.log_eviction_snapshot(limit)
            if not nodes:
                break
            yield nodes, rows, True
            if limit >= len(self._log_leaf_heap):
                break
            limit *= 4
        nodes, rows = self.eviction_snapshot()
        yield nodes, rows, False

    def checkpoint_eviction_actions(
        self, nodes: list[LogRadixEvictionSegment],
    ) -> list[tuple[int, ...]]:
        result = []
        for index, segment in enumerate(nodes):
            for entry in segment.entries:
                if entry.kind is not LogStateKind.CHECKPOINT:
                    continue
                successor = self._checkpoint_successor(entry)
                tail = () if entry.slot in self._recovery_slot_refs else (index,)
                if not tail:
                    successor = None
                result.append((entry.slot, -1 if successor is None else successor.slot,
                               self._checkpoint_timestamps[entry.slot], *tail))
        return result

    def drop_checkpoint(self, left_slot: int, right_slot: int) -> LogStateEntry:
        left = self.entry_for_slot(left_slot)
        right = self.entry_for_slot(right_slot)
        assert left.kind is right.kind is LogStateKind.CHECKPOINT
        assert left.end == right.begin
        owner = self.entry_owner(right)
        left_owner = self.entry_owner(left)
        assert not (left_owner.is_protected and left.end == left_owner.depth)
        assert left_slot not in self._recovery_slot_refs
        assert right_slot not in self._recovery_slot_refs
        current = owner
        while current is not left_owner:
            current = current.parent
            assert not current.is_protected
            assert len(current.children) == 1
        checkpoint_timestamp = self._checkpoint_timestamps[right_slot]
        self.detach_entry(left_slot)
        self.detach_entry(right_slot)
        self.attach_entry(
            owner, replace(
                right, begin=left.begin, span_granules=left.span_granules + right.span_granules,
            ),
            checkpoint_timestamp=checkpoint_timestamp,
        )
        return left

    def evict(
        self,
        nodes        : Iterable[LogRadixEvictionSegment],
        evicted_nodes: list[tuple[int, int, int]] | None = None,
    ) -> LogRadixEviction:
        evicted_kv: list[torch.Tensor] = []
        evicted_entries: list[LogStateEntry] = []
        for target in nodes:
            node = target.node
            begin = target.begin
            assert node.is_leaf() and not node.is_locked and not node.is_root()
            assert target.depth == node.depth
            if evicted_nodes is not None:
                evicted_nodes.append((node.uuid, begin, node.depth))
            offset = begin - node.begin
            evicted_kv.append(node.kv[offset:])
            self.evictable_size -= node.depth - begin
            self._update_kv_refs(node, -1)
            entries = tuple(entry for entry in node.entries if entry.end > begin)
            evicted_entries.extend(entries)
            for entry in entries:
                node._entries.remove(entry)
                self._unregister_entry(entry)
            if offset:
                node._key = node._key[:offset]
                node._kv = node._kv[:offset]
                node._kv_frames = node._kv_frames[:offset]
                node._append_buffers = ()
                node._depth = begin
                node.is_protected = False
                node._kv_frame_counts = dict(Counter(node._kv_frames.tolist()))
                self._update_kv_refs(node, 1)
                self._update_log_leaf(node)
                self._refresh_checkpoint_neighbors(node)
            else:
                parent = node.parent
                del parent.children[self.key_fn(node.key)]
                self._remove_leaf(node)
                if parent.is_leaf() and not parent.is_root():
                    self._add_leaf(parent)
                self._refresh_checkpoint_neighbors(parent)
        if not evicted_kv:
            return LogRadixEviction(self.empty_tensor, ())
        self.revision += 1
        return LogRadixEviction(
            torch.cat(evicted_kv), tuple(evicted_entries)
        )

    def matched_entries(
        self,
        node: LogRadixTreeNode,
    ) -> tuple[LogStateEntry, ...]:
        return tuple(
            entry
            for path_node in self._path(node)
            for entry in path_node.entries
        )

    def make_handle(
        self,
        node      : LogRadixTreeNode,
        cached_len: int | None = None,
    ) -> LogStateCacheHandle:
        if cached_len is None:
            cached_len = node.depth
        assert 0 <= cached_len <= node.depth
        if cached_len == 0:
            endpoint = self.root
        elif cached_len == node.depth:
            endpoint = node
        else:
            endpoint = self._node_at_depth(node, cached_len)
        trace = LogRadixTrace(endpoint, cached_len, endpoint.length)
        path = self.path(endpoint)
        entries = tuple(entry for part in path for entry in part.entries)
        recovery, lock_node = self._recovery_for_trace(trace, entries)
        assert recovery.position == cached_len, (
            f"Cannot recover LogState handle at {cached_len}: recovery stops at "
            f"{recovery.position}; endpoint depth={endpoint.depth}, "
            f"entries={self.matched_entries(endpoint)}"
        )
        return LogStateCacheHandle(
            cached_len=cached_len,
            node=endpoint,
            lock_node=lock_node,
            recovery=recovery,
            match_path=path,
            match_entries=entries,
            kv_parts=tuple(part.kv for part in path) if path else (self.empty_tensor,),
        )

    def recovery_at(
        self,
        node    : LogRadixTreeNode,
        position: int,
    ) -> LogStateRecovery:
        assert 0 <= position <= node.depth
        endpoint = node
        while not endpoint.is_root() and position <= endpoint.begin:
            endpoint = endpoint.parent
        trace = LogRadixTrace(endpoint, position, position - endpoint.begin)
        return self._recovery_for_trace(trace)[0]

    def has_entry(self, entry: LogStateEntry) -> bool:
        return id(entry) in self._entry_owners

    def entry_owner(
        self,
        value: int | LogStateEntry,
    ) -> LogRadixTreeNode:
        if isinstance(value, LogStateEntry):
            return self._entry_owners[id(value)]
        return self._entry_owners[id(self.entry_for_slot(value))]

    def has_slot(self, slot: int) -> bool:
        return slot in self._entries_by_slot

    def entry_for_slot(self, slot: int) -> LogStateEntry:
        return self._entries_by_slot[slot]

    def evictable_slots(self) -> tuple[int, ...]:
        return tuple(self._evictable_slots)

    def evictable_kv_frames(self) -> int:
        return self._evictable_kv_count

    def iter_entries(self) -> tuple[LogStateEntry, ...]:
        return tuple(
            sorted(
                (
                    entry
                    for node in self._iter_nodes()
                    for entry in node.entries
                ),
                key=lambda entry: (entry.begin, entry.end, entry.slot),
            )
        )

    def path(self, node: LogRadixTreeNode) -> tuple[LogRadixTreeNode, ...]:
        return tuple(self._path(node))

    def entry_is_protected(self, slot: int) -> bool:
        entry = self.entry_for_slot(slot)
        owner = self.entry_owner(entry)
        return owner.is_protected and entry.end == owner.depth

    @property
    def size_info(self) -> SizeInfo:

        return SizeInfo(self.evictable_size, self.locked_size)

    def check_integrity(self) -> None:
        evictable_size = 0
        locked_size = 0
        seen_nodes: set[int] = set()
        seen_entries: dict[int, LogRadixTreeNode] = {}
        seen_slots: dict[int, LogStateEntry] = {}
        seen_leaves: dict[int, LogRadixTreeNode] = {}
        kv_refs: Counter[int] = Counter()
        locked_kv_refs: Counter[int] = Counter()
        unprunable_log_refs: Counter[int] = Counter()
        stack = [self.root]

        while stack:
            node = stack.pop()
            if node.uuid in seen_nodes:
                raise RuntimeError(
                    f"Log-Radix-Tree contains duplicate node {node.uuid}."
                )
            seen_nodes.add(node.uuid)

            if node.is_root():
                if node.ref_count != 1 or node.length != 0:
                    raise RuntimeError("Log-Radix-Tree has an invalid root node.")
            else:
                if node.is_leaf():
                    seen_leaves[node.uuid] = node
                if node.parent.children.get(self.key_fn(node.key)) is not node:
                    raise RuntimeError(
                        f"Node {node.uuid} is not indexed by its parent."
                    )
                if node.length == 0 or node.length % self.granularity != 0:
                    raise RuntimeError(
                        f"Node {node.uuid} has invalid length {node.length}."
                    )
                if len(node.kv) != node.length:
                    raise RuntimeError(
                        f"Node {node.uuid} has mismatched key/KV lengths."
                    )
                if len(node._kv_frames) != node.length:
                    raise RuntimeError(
                        f"Node {node.uuid} has mismatched KV frame metadata."
                    )
                if node.ref_count < 0:
                    raise RuntimeError(
                        f"Node {node.uuid} has a negative reference count."
                    )
                if (node.eviction_slots != tuple(entry.slot for entry in node._entries)
                    or node.checkpoint_slots != tuple(
                        entry.slot for entry in node._entries
                        if entry.kind is LogStateKind.CHECKPOINT
                    )):
                    raise RuntimeError(f"Node {node.uuid} has stale eviction metadata.")
                kv_refs.update(node._kv_frame_counts.keys())
                if node.is_locked:
                    locked_kv_refs.update(node._kv_frame_counts.keys())
                    unprunable_log_refs.update(entry.slot for entry in node._entries
                                               if entry.kind is LogStateKind.LOG)
                for entry in node.entries:
                    if not (0 <= entry.begin < entry.end):
                        raise RuntimeError(
                            f"Node {node.uuid} owns invalid entry {entry}."
                        )
                    if not (node.begin < entry.end <= node.depth):
                        raise RuntimeError(
                            f"Entry {entry} belongs to an ancestor of node {node.uuid}."
                        )
                    if id(entry) in seen_entries:
                        raise RuntimeError(
                            f"Linear cache entry {entry} has multiple owners."
                        )
                    if entry.kind is LogStateKind.LOG and (
                        entry.begin < node.begin or entry.end - entry.begin != self.granularity
                    ):
                        raise RuntimeError(f"Invalid Log granule {entry} in node {node.uuid}.")
                    if entry.slot in seen_slots:
                        raise RuntimeError(f"Linear cache slot {entry.slot} has multiple owners.")
                    seen_entries[id(entry)] = node
                    seen_slots[entry.slot] = entry

                if node.is_locked:
                    locked_size += node.length
                else:
                    evictable_size += node.length

            stack.extend(node.children.values())

        if seen_entries != self._entry_owners:
            raise RuntimeError("Log-Radix-Tree slot ownership index is stale.")
        if seen_leaves != self._leaves:
            raise RuntimeError("Log-Radix-Tree leaf index is stale.")
        eligible = {
            node.uuid for node in seen_leaves.values()
            if not node.is_locked
            and (not node._entries or node._entries[-1].kind is LogStateKind.LOG)
            and (not node._entries or node._entries[-1].slot not in self._recovery_slot_refs)
        }
        if eligible != {uuid for _timestamp, uuid, _node in self._log_leaf_heap}:
            raise RuntimeError("Log-Radix-Tree Log eviction index is stale.")
        for index, (timestamp, uuid, node) in enumerate(self._log_leaf_heap):
            if (node._log_leaf_index != index or node._log_leaf_tree is not self
                or (timestamp, uuid) != (node.timestamp, node.uuid)
                or (index and self._log_leaf_heap[(index - 1) // 2][:2] > (timestamp, uuid))):
                raise RuntimeError("Log-Radix-Tree Log eviction heap is stale.")
        if (
            dict(kv_refs) != self._kv_frame_refs
            or dict(locked_kv_refs) != self._locked_kv_refs
            or dict(unprunable_log_refs) != self._unprunable_log_refs
            or self._evictable_kv_count != sum(frame not in locked_kv_refs for frame in kv_refs)
            or set(self._evictable_slots) != {
                slot for slot in seen_slots if self._slot_reclaimable(slot)
            }
        ):
            raise RuntimeError("Log-Radix-Tree reclaimable capacity index is stale.")
        if set(seen_slots) != set(self._entries_by_slot):
            raise RuntimeError("Log-Radix-Tree slot reference index is stale.")
        if set(self._checkpoint_timestamps) != {
            slot for slot, entry in seen_slots.items()
            if entry.kind is LogStateKind.CHECKPOINT
        }:
            raise RuntimeError("Log-Radix-Tree Checkpoint access index is stale.")
        if any(self._entries_by_slot[slot] is not entry for slot, entry in seen_slots.items()):
            raise RuntimeError("Log-Radix-Tree slot ownership index is stale.")
        if (
            evictable_size != self.evictable_size
            or locked_size != self.locked_size
        ):
            raise RuntimeError(
                "Log-Radix-Tree size accounting mismatch:"
                f" recorded=({self.evictable_size}, {self.locked_size}),"
                f" actual=({evictable_size}, {locked_size})."
            )

    def _tree_walk(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[LogRadixTreeNode, int]:
        matched_len = 0
        node = self.root
        tic = time.monotonic_ns()

        while matched_len < len(input_ids):
            child = node.children.get(self.key_fn(input_ids[matched_len:]))
            if child is None:
                break
            node = child
            match_len = align_down(
                node.get_match_len(input_ids[matched_len:]), self.granularity
            )
            matched_len += match_len
            if match_len != node.length:
                node = self.split_at(node, match_len)
                node.timestamp = tic
                break
            node.timestamp = tic
        return node, matched_len

    def _recovery_for_trace(
        self,
        trace: LogRadixTrace,
        path_entries: tuple[LogStateEntry, ...] | None = None,
    ) -> tuple[LogStateRecovery, LogRadixTreeNode]:
        matched_len = trace.matched_len
        entries = [
            entry
            for entry in (
                self.matched_entries(trace.node) if path_entries is None else path_entries
            )
            if entry.end <= matched_len

        ]
        entries.sort(key=lambda entry: (entry.begin, entry.end, entry.slot))

        checkpoints = [
            entry
            for entry in entries
            if entry.kind is LogStateKind.CHECKPOINT
            and entry.end <= matched_len
        ]
        base = max(checkpoints, key=lambda entry: entry.end, default=None)
        cursor = 0 if base is None else base.end
        logs = []
        last_log_owner = None
        for entry in entries:
            if cursor == matched_len:
                break
            if entry.kind is not LogStateKind.LOG:
                continue
            if entry.end <= cursor or entry.begin > cursor:
                continue
            assert entry.begin == cursor
            logs.append(entry)
            last_log_owner = self.entry_owner(entry)
            cursor = entry.end

        if logs:
            conv_slot = logs[-1].slot
            conv_granule = 0
            assert last_log_owner is not None
            lock_node = last_log_owner
        elif base is not None:
            conv_slot = base.slot
            conv_granule = 0
            lock_node = self.entry_owner(base)
        else:
            conv_slot = None
            conv_granule = None
            lock_node = self.root

        return (
            LogStateRecovery(
                position=cursor,
                base_checkpoint_slot=None if base is None else base.slot,
                logs=tuple(logs),
                conv_slot=conv_slot,
                conv_granule=conv_granule,
            ),
            lock_node,
        )

    def _node_at_depth(
        self,
        node : LogRadixTreeNode,
        depth: int,
    ) -> LogRadixTreeNode:
        while not node.is_root() and depth <= node.begin:
            node = node.parent
        assert node.begin < depth <= node.depth
        if depth < node.depth:
            return self.split_at(node, depth - node.begin)
        return node

    def _find_owner(
        self,
        leaf: LogRadixTreeNode,
        end : int,
    ) -> LogRadixTreeNode:
        node = leaf
        while not node.parent.is_root() and end <= node.parent.depth:
            node = node.parent
        assert node.begin < end <= node.depth
        return node

    def _path(self, node: LogRadixTreeNode) -> list[LogRadixTreeNode]:
        path = []
        while not node.is_root():
            path.append(node)
            node = node.parent
        path.reverse()
        return path

    def _iter_nodes(self) -> list[LogRadixTreeNode]:
        nodes = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children.values())
        return nodes

    def _update_size_class(
        self,
        node      : LogRadixTreeNode,
        was_locked: bool,
    ) -> None:
        if was_locked == node.is_locked:
            return
        self._update_log_leaf(node)
        delta = 1 if node.is_locked else -1
        for frame in node._kv_frame_counts:
            self._update_kv_frame(frame, 0, delta)
        for entry in node._entries:
            if entry.kind is not LogStateKind.LOG:
                self._update_slot_reclaimability(entry.slot)
                continue
            locked = self._unprunable_log_refs.get(entry.slot, 0) + delta
            if locked:
                self._unprunable_log_refs[entry.slot] = locked
            else:
                self._unprunable_log_refs.pop(entry.slot, None)
            self._update_slot_reclaimability(entry.slot)
        if node.is_locked:
            self.evictable_size -= node.length
            self.locked_size += node.length
        else:
            self.evictable_size += node.length
            self.locked_size -= node.length

    def _register_entry(
        self,
        owner: LogRadixTreeNode,
        entry: LogStateEntry,
        checkpoint_timestamp: int | None = None,
    ) -> None:
        if not owner._entries or owner._entries[-1].end <= entry.end:
            owner._entries.append(entry)
        else:
            owner._entries.insert(bisect_right(owner._entries, entry.end, key=lambda value: value.end), entry)
        if entry.kind is LogStateKind.CHECKPOINT:
            owner._checkpoint_entries.insert(bisect_right(owner._checkpoint_entries, entry.end,
                                                          key=lambda value: value.end), entry)
        owner.__dict__.pop("eviction_slots", None)
        owner.__dict__.pop("checkpoint_slots", None)
        self._entry_owners[id(entry)] = owner
        self._entries_by_slot[entry.slot] = entry
        if entry.kind is LogStateKind.CHECKPOINT:
            self._checkpoint_timestamps[entry.slot] = (
                time.monotonic_ns() if checkpoint_timestamp is None else checkpoint_timestamp
            )
        if owner.is_locked and entry.kind is LogStateKind.LOG:
            self._unprunable_log_refs[entry.slot] = (
                self._unprunable_log_refs.get(entry.slot, 0) + 1
            )
        self._update_slot_reclaimability(entry.slot)
        self._update_log_leaf(owner)
        self._refresh_entry_neighbors(owner, entry.end)

    def _unregister_entry(self, entry: LogStateEntry) -> None:
        owner = self._entry_owners[id(entry)]
        owner.__dict__.pop("eviction_slots", None)
        owner.__dict__.pop("checkpoint_slots", None)
        if owner.is_locked and entry.kind is LogStateKind.LOG:
            locked = self._unprunable_log_refs[entry.slot] - 1
            if locked:
                self._unprunable_log_refs[entry.slot] = locked
            else:
                del self._unprunable_log_refs[entry.slot]
        if entry.kind is LogStateKind.CHECKPOINT:
            owner._checkpoint_entries.remove(entry)
        del self._entry_owners[id(entry)]
        del self._entries_by_slot[entry.slot]
        self._checkpoint_timestamps.pop(entry.slot, None)
        self._update_slot_reclaimability(entry.slot)
        self._update_log_leaf(owner)
        self._refresh_entry_neighbors(owner, entry.end)

    def _add_leaf(self, node: LogRadixTreeNode) -> None:
        self._leaves[node.uuid] = node
        node._log_leaf_tree = self
        self._update_log_leaf(node)

    def _remove_leaf(self, node: LogRadixTreeNode) -> None:
        if self._leaves.pop(node.uuid, None) is None:
            return
        node._log_leaf_tree = None
        self._update_log_leaf(node)

    def _update_log_leaf(self, node: LogRadixTreeNode) -> None:
        index = node._log_leaf_index
        eligible = (node._log_leaf_tree is self and not node.is_locked
                    and (not node._entries or node._entries[-1].kind is LogStateKind.LOG)
                    and (not node._entries or node._entries[-1].slot not in self._recovery_slot_refs))
        heap = self._log_leaf_heap
        if eligible:
            item = (node.timestamp, node.uuid, node)
            if index < 0:
                index = len(heap)
                heap.append(item)
            else:
                heap[index] = item
        elif index >= 0:
            node._log_leaf_index = -1
            item = heap.pop()
            if index == len(heap):
                return
            heap[index] = item
        else:
            return
        while index:
            parent = (index - 1) // 2
            if heap[parent] <= item:
                break
            heap[index] = heap[parent]
            heap[index][2]._log_leaf_index = index
            index = parent
        heap[index] = item
        item[2]._log_leaf_index = index
        while 2 * index + 1 < len(heap):
            child = 2 * index + 1
            if child + 1 < len(heap) and heap[child + 1] < heap[child]:
                child += 1
            if item <= heap[child]:
                break
            heap[index] = heap[child]
            heap[index][2]._log_leaf_index = index
            index = child
        heap[index] = item
        item[2]._log_leaf_index = index

    def _checkpoint_successor(self, entry: LogStateEntry) -> LogStateEntry | None:
        if entry.kind is not LogStateKind.CHECKPOINT:
            return None
        node = self.entry_owner(entry)
        if node.is_protected and entry.end == node.depth:
            return None
        index = bisect_right(node._entries, entry.end, key=lambda value: value.end)
        if index < len(node._entries):
            candidate = node._entries[index]
            return candidate if (
                candidate.kind is LogStateKind.CHECKPOINT
                and candidate.begin == entry.end
                and candidate.slot not in self._recovery_slot_refs
            ) else None
        while not node.is_protected and len(node.children) == 1:
            node = next(iter(node.children.values()))
            if node.entries:
                candidate = node.entries[0]
                return candidate if (
                    candidate.kind is LogStateKind.CHECKPOINT
                    and candidate.begin == entry.end
                        and candidate.slot not in self._recovery_slot_refs
                ) else None
        return None

    def _refresh_entry_neighbors(self, node: LogRadixTreeNode, position: int) -> None:
        index = bisect_right(node._checkpoint_entries, position, key=lambda value: value.end)
        for entry in node._checkpoint_entries[max(0, index - 2):index + 1]:
            self._update_slot_reclaimability(entry.slot)
        if index <= 1:
            parent = node.parent
            while not parent.is_root():
                if parent._entries:
                    entry = parent._entries[-1]
                    if entry.kind is LogStateKind.CHECKPOINT:
                        self._update_slot_reclaimability(entry.slot)
                    break
                parent = parent.parent

    def _refresh_checkpoint_neighbors(self, node: LogRadixTreeNode) -> None:
        for entry in node._checkpoint_entries:
            self._update_slot_reclaimability(entry.slot)
        node = node.parent if not node.is_root() else node
        while not node.is_root():
            if node.entries:
                entry = node.entries[-1]
                if entry.kind is LogStateKind.CHECKPOINT:
                    self._update_slot_reclaimability(entry.slot)
                break
            node = node.parent

    def _slot_reclaimable(self, slot: int) -> bool:
        if (slot not in self._entries_by_slot or slot in self._unprunable_log_refs
            or slot in self._recovery_slot_refs):
            return False
        entry = self.entry_for_slot(slot)
        if entry.kind is LogStateKind.CHECKPOINT:
            if self.entry_owner(entry).is_locked:

                return self._checkpoint_successor(entry) is not None
        return True

    def _update_slot_reclaimability(self, slot: int) -> None:
        reclaimable = self._slot_reclaimable(slot)
        if reclaimable == (slot in self._evictable_slots):
            return
        if reclaimable:
            self._evictable_slots[slot] = None
        else:
            del self._evictable_slots[slot]
        if self._on_slot_reclaimable is not None:
            self._on_slot_reclaimable(slot, reclaimable)

    def _update_kv_refs(self, node: LogRadixTreeNode, delta: int) -> None:
        for frame in node._kv_frame_counts:
            self._update_kv_frame(frame, delta, delta if node.is_locked else 0)

    def _update_kv_frame(self, frame: int, refs_delta: int, locked_delta: int) -> None:
        refs = self._kv_frame_refs.get(frame, 0)
        locked = self._locked_kv_refs.get(frame, 0)
        self._evictable_kv_count -= int(refs > 0 and locked == 0)
        refs += refs_delta
        locked += locked_delta
        assert 0 <= locked <= refs
        if refs:
            self._kv_frame_refs[frame] = refs
        else:
            self._kv_frame_refs.pop(frame, None)
        if locked:
            self._locked_kv_refs[frame] = locked
        else:
            self._locked_kv_refs.pop(frame, None)
        self._evictable_kv_count += int(refs > 0 and locked == 0)

__all__ = [
    "LogRadixEviction",
    "LogRadixInsert",
    "LogRadixTrace",
    "LogRadixTree",
    "LogRadixTreeNode",
    "LogStateCacheHandle",
    "LogStateEntry",
    "LogStateKind",
    "LogStateMergeStats",
    "LogStateRecovery",
]
