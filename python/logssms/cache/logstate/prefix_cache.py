from __future__ import annotations

import time
from bisect import bisect_right
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import cached_property
from itertools import count
from typing import Any, Callable, Iterable, NamedTuple, Sequence, TypeAlias

import torch
from logssms.cache.prefix_cache import BaseCacheHandle, BasePrefixCache, MatchResult, SizeInfo
from logssms.utils import align_down

from .metadata import ReclamationChanges

KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]


def _get_key_fn(granularity: int) -> KEY_FN:
    if granularity == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:granularity].tolist())


class LogStateKind(Enum):
    LOG = "log"
    CHECKPOINT = "checkpoint"


@dataclass(eq=False)
class LogStateHitStats:
    origin        : int
    tail          : int
    hits          : deque[tuple[int, int, int]] = field(default_factory=lambda: deque(maxlen=8192))
    exposures     : deque[tuple[int, int]] = field(default_factory=deque)
    exponent      : float = 2.0
    log_amplitude : float | None = None
    tail_probability: float = 0.0
    past_tail_probabilities: tuple[tuple[int, float], ...] = ()
    tail_query_begin: int | None = None
    query_begin   : int = 0
    fit_query     : int = -1
    fitted_hit_query: int = -1
    exponent_query: int = -1
    uuid          : int = field(default_factory=lambda: next(LogStateHitStats.counter))

    counter = count(1)


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
        self._leaf_tree: LogRadixTree | None = None
        self._key = torch.empty(0, dtype=torch.int32)
        self._kv = torch.empty(0, dtype=torch.int32)
        self._kv_frames = torch.empty(0, dtype=torch.int32)
        self._kv_frame_counts: dict[int, int] = {}
        self._full_frame_counts: tuple[torch.Tensor, dict[int, int]] | None = None
        self._entries: list[LogStateEntry] = []
        self._checkpoint_entries: list[LogStateEntry] = []
        self._append_buffers: tuple[torch.Tensor, ...] = ()
        self.analysis_owner: LogStateHitStats | None = None

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

    @property
    def eviction_frame_counts(self) -> dict[int, int]:
        cached = self._full_frame_counts
        if cached is None or cached[0] is not self._kv_frames:
            cached = (self._kv_frames, dict(Counter(self._kv_frames.tolist())))
            self._full_frame_counts = cached
        return cached[1]

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
        if self._leaf_tree is not None:
            self._leaf_tree.access_revision += 1
            self._leaf_tree.reclamation_access_changes.record(self.uuid)

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

    @cached_property
    def log_runs(self) -> tuple[int, dict[int, tuple[int, int]]]:
        entries = self._entries
        run_ends = list(range(len(entries) + 1))
        for index in range(len(entries) - 1, -1, -1):
            entry = entries[index]
            if entry.kind is LogStateKind.LOG:
                end = index + 1
                if end < len(entries) and entry.end == entries[end].begin:
                    run_ends[index] = run_ends[end]
                else:
                    run_ends[index] = end
        after_checkpoint = {}
        for entry in self._checkpoint_entries:
            first = bisect_right(entries, entry.end, key=lambda value: value.end)
            end = run_ends[first]
            if first < end and entry.end == entries[first].begin:
                after_checkpoint[entry.slot] = (first, end)
        return run_ends[0], after_checkpoint

    @cached_property
    def eviction_recovery(self) -> tuple[int | None, dict[int, int]]:
        ends = {}
        for entry in reversed(self._entries):
            if entry.kind is LogStateKind.CHECKPOINT:
                return entry.end, ends
            ends[entry.begin] = ends.get(entry.end, entry.end)
        return None, ends

    def invalidate_entry_metadata(self) -> None:
        self.__dict__.pop("eviction_slots", None)
        self.__dict__.pop("checkpoint_slots", None)
        self.__dict__.pop("log_runs", None)
        self.__dict__.pop("eviction_recovery", None)

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
    node            : LogRadixTreeNode
    lock_node         : LogRadixTreeNode
    recovery          : LogStateRecovery
    analysis_owner: LogStateHitStats | None = None
    source_analysis_owner: LogStateHitStats | None = None
    query_uid: int | None = None
    match_position    : int = 0
    request_end       : int = 0
    protected_endpoints: tuple[int, ...] = ()
    match_trace       : LogRadixTrace | None = field(default=None, compare=False)
    trace_revision    : int = field(default=-1, compare=False)
    trace_input_len   : int = field(default=0, compare=False)
    match_path        : tuple[LogRadixTreeNode, ...] = field(default=(), compare=False, repr=False)
    kv_parts          : tuple[torch.Tensor, ...] = field(default=(), compare=False, repr=False)

    @cached_property
    def kv_indices(self) -> torch.Tensor:

        return torch.cat(self.kv_parts) if self.kv_parts else self.node.kv[:0]

    def get_matched_indices(self) -> torch.Tensor:
        return self.kv_indices


class LogRadixInsert(NamedTuple):
    node            : LogRadixTreeNode
    previous_len: int
    inserted_len: int


class LogRadixEviction(NamedTuple):
    kv_indices: torch.Tensor
    entries   : tuple[LogStateEntry, ...]


@dataclass(frozen=True, slots=True)
class LogRadixEvictionNode:
    node          : LogRadixTreeNode
    begin         : int
    end           : int
    eviction_slots: tuple[int, ...]
    tree          : LogRadixTree = field(repr=False, compare=False)
    recovery_begin: int = 0

    @property
    def entries(self) -> tuple[LogStateEntry, ...]:
        return tuple(self.tree.entry_for_slot(slot) for slot in self.eviction_slots
                     if self.tree.has_slot(slot))

    @property
    def frame_counts(self) -> dict[int, int]:
        return self.node.eviction_frame_counts


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
        self.reclamation_lock_changes = ReclamationChanges()
        self.reclamation_access_changes = ReclamationChanges()
        self.access_revision = 0
        self._lock_footprint_epoch = (-1, -1)
        self._lock_footprints: dict[int, tuple[int, tuple[int, ...]]] = {}
        self._entry_owners: dict[int, LogRadixTreeNode] = {}
        self._leaves: dict[int, LogRadixTreeNode] = {}
        self._entries_by_slot: dict[int, LogStateEntry] = {}
        self._checkpoint_timestamps: dict[int, int] = {}
        self._kv_frame_refs: dict[int, int] = {}
        self._locked_kv_refs: dict[int, int] = {}
        self._evictable_kv_count = 0

        self._unprunable_log_refs: dict[int, int] = {}
        self._recovery_slot_refs: dict[int, int] = {}
        self._evictable_slots: dict[int, None] = {}
        self._on_slot_reclaimable = on_slot_reclaimable
        self.analysis_factory: Callable[[int], LogStateHitStats | None] | None = None
        self._match_undo: list[Callable[[], None]] | None = None
        self._match_touches: list[LogStateCacheHandle] = []

    @contextmanager
    def match_transaction(self):
        assert self._match_undo is None
        undo = self._match_undo = []
        committed = False

        def commit():
            nonlocal committed
            committed = True

        try:
            yield commit
        finally:
            self._match_undo = None
            if committed:
                for handle in self._match_touches:
                    self.touch_handle(handle)
            else:
                for restore in reversed(undo):
                    restore()
            self._match_touches.clear()

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
            kv_parts=tuple(part.kv for part in path) if path else (self.empty_tensor,),
        )
        self.touch_handle(handle)
        return MatchResult(
            handle,
            tree_match_len=trace.matched_len,
            recoverable_len=recovery.position,
        )

    def touch_handle(self, handle: LogStateCacheHandle) -> None:
        if self._match_undo is not None:
            self._match_touches.append(handle)
            return
        self.access_revision += 1
        tic = time.monotonic_ns()
        for node in handle.match_path:
            node.timestamp = tic
            if node._leaf_tree is None:
                self.reclamation_access_changes.record(node.uuid)
        slot = handle.recovery.base_checkpoint_slot
        if slot is not None:
            self._checkpoint_timestamps[slot] = tic
            self.reclamation_access_changes.record(self.entry_owner(slot).uuid)

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
            node._leaf_tree = None
        self._leaves.clear()
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
        analysis_owner: LogStateHitStats | None = None,
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
            owner_boundary = (
                analysis_owner is not None
                and analysis_owner is not node.analysis_owner
                and analysis_owner.origin == previous_len
            )
            if (
                not node.is_root()
                and node.is_leaf()
                and not node.is_protected
                and not owner_boundary
            ):
                if node.analysis_owner is None and self.analysis_factory is not None:
                    node.analysis_owner = self.analysis_factory(node.begin)
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
            else:
                new_node = LogRadixTreeNode(self.key_fn, self.granularity)
                new_node.initialize(suffix_ids, suffix_kv, kv_frames)
                new_node.set_parent(node)
                if analysis_owner is not None and analysis_owner.origin == previous_len:
                    new_node.analysis_owner = analysis_owner
                elif not node.is_root() and len(node.children) == 1:
                    new_node.analysis_owner = node.analysis_owner
                elif self.analysis_factory is not None:
                    new_node.analysis_owner = (
                        analysis_owner
                        if analysis_owner is not None and analysis_owner.origin == previous_len
                        else self.analysis_factory(previous_len)
                    )
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
        assert entry.slot not in self._entries_by_slot, (
            f"Linear cache slot {entry.slot} is already owned by node "
            f"{self.entry_owner(entry.slot).uuid}."
        )
        owner = self._find_owner(leaf, entry.end)
        if entry.kind is LogStateKind.CHECKPOINT and entry.begin < owner.begin:
            entry = replace(entry, begin=owner.begin,
                            span_granules=(entry.end - owner.begin) // self.granularity)
        assert entry.begin >= owner.begin
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
        if self._match_undo is not None:
            saved = node.__dict__.copy()
            saved["_kv_frame_counts"] = node._kv_frame_counts.copy()
            timestamps = {entry.slot: self._checkpoint_timestamps.get(entry.slot)
                          for entry in node._entries}
            self._match_undo.append(
                lambda: self._undo_match_split(parent, prefix, node, saved, timestamps),
            )
        prefix.analysis_owner = node.analysis_owner
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
                if entry.kind is LogStateKind.CHECKPOINT and entry.begin < split_depth:
                    entry = replace(entry, begin=split_depth,
                                    span_granules=(entry.end - split_depth) // self.granularity)
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

    def _undo_match_split(self, parent, prefix, node, saved, timestamps) -> None:
        for part in (prefix, node):
            self._update_kv_refs(part, -1)
            if part.is_locked:
                self.locked_size -= part.length
            else:
                self.evictable_size -= part.length
            entries, part._entries = part._entries, []
            for entry in entries:
                self._unregister_entry(entry)
        entries = saved["_entries"]
        node.__dict__.clear()
        node.__dict__.update(saved)
        node._entries = []
        node._checkpoint_entries = []
        node.set_parent(parent)
        self._update_kv_refs(node, 1)
        if node.is_locked:
            self.locked_size += node.length
        else:
            self.evictable_size += node.length
        for entry in entries:
            self._register_entry(node, entry, timestamps[entry.slot])
        for part in (parent, node):
            self._refresh_checkpoint_neighbors(part)
        self.revision += 1

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
                if refs == int(not release):
                    self.reclamation_lock_changes.record(owner.uuid)
                self._refresh_checkpoint_neighbors(owner)

    def protect(
        self,
        node     : LogRadixTreeNode,
        protected: bool = True,
    ) -> None:
        assert not node.is_root()
        if node.is_protected != protected:
            if self._match_undo is not None:
                self._match_undo.append(lambda: self.protect(node, not protected))
            node.is_protected = protected
            self.revision += 1
            self._refresh_checkpoint_neighbors(node)

    def protect_at(self, node: LogRadixTreeNode, position: int) -> None:
        self.protect(self._node_at_depth(node, position))

    def eviction_topology(self, node: LogRadixTreeNode, parent: int) -> tuple[int, int, int]:
        unavailable = (node.is_root() or node.is_locked
                       or any(slot in self._recovery_slot_refs for slot in node.eviction_slots))
        timestamp = min(map(self.checkpoint_timestamp, node.checkpoint_slots), default=node.timestamp)
        return parent, int(unavailable), timestamp

    def eviction_nodes(self) -> list[LogRadixTreeNode]:
        return sorted(self._iter_nodes(), key=lambda node: (node.depth, node.uuid))

    def eviction_snapshot(self) -> tuple[list[LogRadixEvictionNode], list[tuple[int, ...]]]:
        snapshots = []
        rows = []
        indices = {}
        recoveries = {}
        for node in self.eviction_nodes():
            root = node.is_root()
            parent = -1 if root else indices[node.parent.uuid]
            recovery = 0 if root else recoveries[node.parent.uuid]
            index = len(snapshots)
            slots = node.eviction_slots
            snapshots.append(LogRadixEvictionNode(
                node, node.begin, node.depth, slots, self, recovery,
            ))
            rows.append((index, *self.eviction_topology(node, parent)))
            checkpoint, ends = node.eviction_recovery
            if checkpoint is not None:
                recovery = checkpoint
            recovery = ends.get(recovery, recovery)
            indices[node.uuid] = index
            recoveries[node.uuid] = recovery
        return snapshots, rows

    def _pressure_runs(
        self, node: LogRadixTreeNode, count: int,
    ) -> tuple[tuple[int, ...], dict[int, tuple[int, ...]]]:
        entries = node._entries
        leading, after_checkpoint = node.log_runs
        head = ()
        if leading >= count and entries[0].begin == node.begin:
            recovery = self.recovery_at(node.parent, node.begin)
            if recovery.position == node.begin:
                base = recovery.base_checkpoint_slot
                complete = leading // count * count
                head = (-1 if base is None else base, entries[complete - 1].slot,
                        *(-entry.slot - 2 for entry in recovery.logs),
                        *(entry.slot for entry in entries[:count]))
        groups = {}
        for base, (first, end) in after_checkpoint.items():
            complete = (end - first) // count * count
            if complete:
                groups[base] = (base, entries[first + complete - 1].slot,
                                *(entry.slot for entry in entries[first:first + count]))
            elif end == len(entries) and entries[end - 1].end == node.depth:
                groups[base] = (base, -1, *(entry.slot for entry in entries[first:end]))
        return head, groups

    def pressure_materializations(
        self, count: int, *, node: LogRadixTreeNode | None = None,
    ) -> list[tuple[int, ...]]:
        """Return base, boundary, and Logs. Ancestor Log slots use -slot-2."""
        assert count > 0
        result = []
        checkpoint_candidates = {}
        for part in self._iter_nodes() if node is None else (node,):
            head, groups = self._pressure_runs(part, count)
            if head:
                result.append(head)
            checkpoint_candidates.update(groups)
        if checkpoint_candidates:
            # Candidate IDs break ties in the native planner.
            result.extend(checkpoint_candidates[slot] for slot in self._checkpoint_timestamps
                          if slot in checkpoint_candidates)
        return result

    def checkpoint_eviction_actions(
        self, nodes: list[LogRadixEvictionNode],
    ) -> list[tuple[int, ...]]:
        result = []
        for index, snapshot in enumerate(nodes):
            for slot in snapshot.eviction_slots:
                timestamp = self._checkpoint_timestamps.get(slot)
                if timestamp is None or slot in self._recovery_slot_refs:
                    continue
                successor = self._checkpoint_successor(self.entry_for_slot(slot))
                result.append((slot, -1 if successor is None else successor.slot,
                               timestamp, index))
        return result

    def drop_checkpoint(self, left_slot: int, right_slot: int) -> LogStateEntry:
        left = self.entry_for_slot(left_slot)
        right = self.entry_for_slot(right_slot)
        assert left.kind is right.kind is LogStateKind.CHECKPOINT
        assert left.end == right.begin
        owner = self.entry_owner(right)
        left_owner = self.entry_owner(left)
        assert owner is left_owner
        assert not (left_owner.is_protected and left.end == left_owner.depth)
        assert left_slot not in self._recovery_slot_refs
        assert right_slot not in self._recovery_slot_refs
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
        nodes        : Iterable[LogRadixEvictionNode],
        evicted_nodes: list[tuple[int, int, int]] | None = None,
    ) -> LogRadixEviction:
        selected = list(nodes)
        if not selected:
            return LogRadixEviction(self.empty_tensor, ())
        evicted_kv = [target.node.kv for target in selected]
        evicted_entries = tuple(entry for target in selected for entry in target.node.entries)
        self.prune(selected, evicted_nodes)
        return LogRadixEviction(torch.cat(evicted_kv), evicted_entries)

    def prune(
        self, nodes: Sequence[LogRadixEvictionNode],
        evicted_nodes: list[tuple[int, int, int]] | None = None,
    ) -> None:
        for target in nodes:
            node = target.node
            assert node.is_leaf() and not node.is_locked and not node.is_root()
            assert target.end == node.depth and target.begin == node.begin
            if evicted_nodes is not None:
                evicted_nodes.append((node.uuid, node.begin, node.depth))
            self.evictable_size -= node.length
            self._update_kv_refs(node, -1)
            for entry in node.entries:
                node._entries.remove(entry)
                self._unregister_entry(entry)
            parent = node.parent
            del parent.children[self.key_fn(node.key)]
            self._remove_leaf(node)
            if parent.is_leaf() and not parent.is_root():
                self._add_leaf(parent)
            self._refresh_checkpoint_neighbors(parent)
        if nodes:
            self.revision += 1

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
                    if not (node.begin <= entry.begin < entry.end):
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
            if node._leaf_tree is None:
                self.reclamation_access_changes.record(node.uuid)
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
        self.reclamation_lock_changes.record(node.uuid)
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
        owner.invalidate_entry_metadata()
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
        self._refresh_entry_neighbors(owner, entry.end)

    def _unregister_entry(self, entry: LogStateEntry) -> None:
        owner = self._entry_owners[id(entry)]
        owner.invalidate_entry_metadata()
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
        self._refresh_entry_neighbors(owner, entry.end)

    def _add_leaf(self, node: LogRadixTreeNode) -> None:
        self._leaves[node.uuid] = node
        node._leaf_tree = self

    def _remove_leaf(self, node: LogRadixTreeNode) -> None:
        if self._leaves.pop(node.uuid, None) is None:
            return
        node._leaf_tree = None

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
    "LogStateHitStats",
    "LogStateRecovery",
]
