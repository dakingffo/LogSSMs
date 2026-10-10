from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import cached_property
from itertools import count
from typing import Any, Callable, Iterable, Iterator, NamedTuple, TypeAlias, TYPE_CHECKING

if TYPE_CHECKING:
    from logssms.cache.logstate import RecycledMatch, RecycledRestorePlan

import torch

from logssms.utils import align_down

from ..logstate_pool.layout import LOG_STATE_GRANULARITY
from .base import BaseCacheHandle, BasePrefixCache, MatchResult, SizeInfo


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
    span_pages   : int | None = None
    storage_begin: int | None = None

    def __post_init__(self) -> None:
        assert self.slot >= 0
        assert 0 <= self.begin < self.end
        assert self.span_pages is None or self.span_pages > 0
        if self.kind is LogStateKind.LOG:
            assert self.span_pages is None, "A log page has no checkpoint span."
            storage_begin = self.physical_begin
            assert storage_begin <= self.begin < self.end
        else:
            assert self.span_pages is not None, "A checkpoint needs an exact page span."

    @property
    def physical_begin(self) -> int:
        return self.begin if self.storage_begin is None else self.storage_begin

    def slice(self, begin: int, end: int) -> LogStateEntry:
        assert self.kind is LogStateKind.LOG
        assert self.begin <= begin < end <= self.end
        return replace(
            self,
            begin=begin,
            end=end,
            storage_begin=self.physical_begin,
        )


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
        self.is_protected = False
        self.timestamp = time.monotonic_ns() if tic is None else tic
        self._key = torch.empty(0, dtype=torch.int32)
        self._kv = torch.empty(0, dtype=torch.int32)
        self._kv_pages = torch.empty(0, dtype=torch.int32)
        self._kv_page_counts: dict[int, int] = {}
        self._entries: list[LogStateEntry] = []
        self.merge_stats: LogStateMergeStats | None = None

    def set(
        self,
        begin     : int,
        end       : int,
        input_ids : torch.Tensor,
        kv_indices: torch.Tensor,
        page_size : int,
        kv_pages  : torch.Tensor | None = None,
    ) -> None:
        assert len(input_ids) == len(kv_indices)
        self._key = input_ids[begin:end]
        self._kv = kv_indices[begin:end].clone()
        self._kv_pages = (
            torch.div(
                kv_indices[begin:end].to(device="cpu", dtype=torch.int32),
                page_size,
                rounding_mode="floor",
            )
            if kv_pages is None else kv_pages[begin:end].clone()
        )
        self._kv_page_counts = dict(Counter(self._kv_pages.tolist()))

    def set_parent(self, parent: LogRadixTreeNode) -> None:
        self._parent = parent
        self._depth = parent.depth + self.length
        parent.children[self.key_fn(self._key)] = self

    @property
    def parent(self) -> LogRadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def key(self) -> torch.Tensor:
        return self._key

    @property
    def kv(self) -> torch.Tensor:
        return self._kv

    @property
    def entries(self) -> tuple[LogStateEntry, ...]:
        return tuple(self._entries)

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
    match_trace       : LogRadixTrace | None = field(default=None, compare=False)
    trace_revision    : int = field(default=-1, compare=False)
    trace_input_len   : int = field(default=0, compare=False)
    match_path        : tuple[LogRadixTreeNode, ...] = field(default=(), compare=False, repr=False)
    match_entries     : tuple[LogStateEntry, ...] = field(default=(), compare=False, repr=False)
    kv_parts          : tuple[torch.Tensor, ...] = field(default=(), compare=False, repr=False)

    @cached_property
    def kv_indices(self) -> torch.Tensor:
        # Handles are locked before their indices are consumed. Rejected
        # admission attempts never need to concatenate the device-side history.
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


class LogRadixTree(BasePrefixCache):
    """Radix tree for KV pages and heterogeneous linear-cache representations.

    The tree owns prefix topology and the logical association between token
    intervals and LogState slots. It does not allocate or free physical pages.
    ``LogStateCacheManager`` consumes insertion/eviction results and performs
    the corresponding resource operations.
    """

    def __init__(
        self,
        page_size          : int,
        device             : torch.device,
        granularity        : int = LOG_STATE_GRANULARITY,
        on_slot_reclaimable: Callable[[int, bool], None] | None = None,
    ) -> None:
        assert page_size > 0
        assert granularity > 0 and page_size % granularity == 0
        self.page_size = page_size
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
        self._slot_entries: dict[int, list[LogStateEntry]] = {}
        self._kv_frame_refs: dict[int, int] = {}
        self._locked_kv_refs: dict[int, int] = {}
        self._evictable_kv_count = 0
        self._locked_slot_refs: dict[int, int] = {}
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
            tic = time.monotonic_ns()
            if node is not trace.node or revision != self.revision:
                path = self.path(node)
                entries = tuple(entry for part in path for entry in part.entries)
            for path_node in path:
                path_node.timestamp = tic
        if recovery.position == 0:
            path, entries = (), ()
        # Splitting the recovery endpoint may have shortened the original
        # trace's edge. Retain the raw token-match endpoint in the new topology.
        trace_node = trace.node
        while not trace_node.is_root() and trace.matched_len <= trace_node.begin:
            trace_node = trace_node.parent
        matched_trace = LogRadixTrace(
            trace_node, trace.matched_len,
            trace.matched_len - trace_node.begin,
        )
        return MatchResult(
            LogStateCacheHandle(
                cached_len=recovery.position,
                node=node,
                lock_node=lock_node,
                recovery=recovery,
                match_trace=matched_trace,
                trace_revision=self.revision,
                trace_input_len=usable_len,
                match_path=path,
                match_entries=entries,
                kv_parts=tuple(part.kv for part in path) if path else (self.empty_tensor,),
            ),
            tree_match_len=trace.matched_len,
            recoverable_len=recovery.position,
        )

    def lock_handle(
        self,
        handle: BaseCacheHandle,
        unlock: bool = False,
    ) -> None:
        assert isinstance(handle, LogStateCacheHandle)
        self.lock(handle.lock_node, unlock=unlock)

    def reset(self) -> None:
        if self._on_slot_reclaimable is not None:
            for slot in self._evictable_slots:
                self._on_slot_reclaimable(slot, False)
        self.root = LogRadixTreeNode(self.key_fn, self.granularity)
        self.root.ref_count = 1
        self.evictable_size = 0
        self.locked_size = 0
        self._entry_owners.clear()
        self._slot_entries.clear()
        self._kv_frame_refs.clear()
        self._locked_kv_refs.clear()
        self._evictable_kv_count = 0
        self._locked_slot_refs.clear()
        self._evictable_slots.clear()
        self.revision += 1
        self._lock_footprints.clear()

    def trace(self, input_ids: torch.Tensor) -> LogRadixTrace:
        """Read-only prefix walk at the recovery granularity.

        Unlike :meth:`match`, tracing never splits an existing edge. When the
        match ends inside an edge, ``node`` is that edge's node and
        ``matched_in_node`` identifies the prospective split position.
        """
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
        """Shorten a known token match without comparing the prefix again."""
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
    ) -> LogRadixInsert:
        assert len(input_ids) == len(kv_indices)
        usable_len = align_down(len(input_ids), self.granularity)
        node, previous_len = self._tree_walk(input_ids[:usable_len])
        if cache_len is None:
            inserted_len = previous_len + align_down(
                usable_len - previous_len, self.page_size
            )
        else:
            inserted_len = cache_len
        assert 0 <= inserted_len <= usable_len
        assert inserted_len % self.granularity == 0
        if cache_len is None:
            assert (inserted_len - previous_len) % self.page_size == 0, (
                "New LogState cache admission must consist of complete physical pages."
            )

        if inserted_len == 0:
            node = self.root
        elif inserted_len < previous_len:
            node = self._node_at_depth(node, inserted_len)
        elif previous_len < inserted_len:
            new_node = LogRadixTreeNode(self.key_fn, self.granularity)
            new_node.set(
                previous_len, inserted_len, input_ids, kv_indices,
                self.page_size,
            )
            new_node.set_parent(node)
            self._update_kv_refs(new_node, 1)
            self.evictable_size += new_node.length
            self.revision += 1
            node = new_node

        for entry in entries:
            assert entry.end <= inserted_len
            self.attach_entry(node, entry)
        return LogRadixInsert(node, previous_len, inserted_len)

    def attach_entry(
        self,
        leaf : LogRadixTreeNode,
        entry: LogStateEntry,
    ) -> None:
        assert entry.end <= leaf.depth
        assert entry.begin % self.granularity == 0
        assert entry.end % self.granularity == 0
        if entry.kind is LogStateKind.LOG:
            assert (
                entry.physical_begin <= entry.begin < entry.end
                <= entry.physical_begin + self.page_size
            ), (
                "A logical log interval must stay inside its physical page."
            )
        else:
            assert entry.span_pages is not None
            assert entry.end - entry.begin == entry.span_pages * self.page_size
        assert entry.slot not in self._slot_entries, (
            f"Linear cache slot {entry.slot} is already owned by node "
            f"{self.entry_owner(entry.slot).uuid}."
        )
        owner = self._find_owner(leaf, entry.end)
        if entry.end < owner.depth:
            owner = self.split_at(owner, entry.end - owner.begin)
        self._register_entry(owner, entry)
        self.revision += 1

    def detach_entry(self, slot: int) -> LogStateEntry:
        entries = self._slot_entries.get(slot)
        assert entries is not None and len(entries) == 1, (
            f"Can only detach an unsplit linear cache slot, got {slot=} with "
            f"{0 if entries is None else len(entries)} logical fragments."
        )
        entry = entries[0]
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
        self._update_kv_refs(node, -1)

        prefix = LogRadixTreeNode(
            self.key_fn, self.granularity, node.timestamp
        )
        kv_pages = node._kv_pages
        prefix.set(0, pos, node.key, node.kv, self.page_size, kv_pages)
        prefix.set_parent(parent)
        prefix.ref_count = node.ref_count
        self._update_kv_refs(prefix, 1)

        old_entries = node._entries
        node._entries = []
        for entry in old_entries:
            self._unregister_entry(entry)
            if entry.end <= split_depth:
                self._register_entry(prefix, entry)
            elif (
                entry.kind is LogStateKind.LOG
                and entry.begin < split_depth
            ):
                self._register_entry(
                    prefix, entry.slice(entry.begin, split_depth)
                )
                self._register_entry(
                    node, entry.slice(split_depth, entry.end)
                )
            else:
                self._register_entry(node, entry)

        node.set(pos, node.length, node.key, node.kv, self.page_size, kv_pages)
        node.set_parent(prefix)
        self._update_kv_refs(node, 1)
        self.revision += 1
        return prefix

    def lock(self, node: LogRadixTreeNode, unlock: bool = False) -> None:
        self.lock_revision += 1
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
        """Resources which cease to be reclaimable if this path is locked."""
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
                frames.update(frame for frame in current._kv_page_counts
                              if frame not in self._locked_kv_refs)
                slots.update(entry.slot for entry in current._entries
                             if entry.slot in self._evictable_slots)
            current = current.parent
        result = len(frames), tuple(slots)
        self._lock_footprints[node.uuid] = result
        return result

    def protect(
        self,
        node     : LogRadixTreeNode,
        protected: bool = True,
    ) -> None:
        assert not node.is_root()
        if node.is_protected != protected:
            node.is_protected = protected
            self.revision += 1

    def protect_entry(self, slot: int, protected: bool = True) -> None:
        for entry in self._slot_entries[slot]:
            self.protect(self.entry_owner(entry), protected)

    def evict(
        self,
        size         : int,
        evicted_nodes: list[tuple[int, int, int]] | None = None,
    ) -> LogRadixEviction:
        if size == 0:
            return LogRadixEviction(self.empty_tensor, ())
        assert size <= self.evictable_size, (
            f"Can't evict {size}, only {self.evictable_size}."
        )
        selected = []
        evicted_size = 0
        for node in self.iter_evict_candidates():
            selected.append(node)
            evicted_size += node.length
            if evicted_size >= size:
                break
        assert evicted_size >= size
        return self.evict_nodes(selected, evicted_nodes)

    def iter_evict_candidates(self) -> Iterator[LogRadixTreeNode]:
        """Prefer Log-only leaves; use LRU within each representation class."""
        from logssms.kernel import logstate as planning

        nodes, rows = self.eviction_snapshot()
        for index in planning.eviction_candidates(rows):
            yield nodes[index]

    def eviction_snapshot(self) -> tuple[list[LogRadixTreeNode], list[tuple[int, ...]]]:
        """Borrow topology/priority metadata for one native planning call."""
        nodes = self._iter_nodes()
        rows = [(
            node.uuid, -1 if node.is_root() else node.parent.uuid,
            int(node.is_root() or node.is_locked),
            int(any(entry.kind is LogStateKind.CHECKPOINT for entry in node.entries)),
            node.timestamp,
        ) for node in nodes]
        return nodes, rows

    def slot_entry_count(self, slot: int) -> int:
        return len(self._slot_entries[slot])

    def evict_nodes(
        self,
        nodes        : Iterable[LogRadixTreeNode],
        evicted_nodes: list[tuple[int, int, int]] | None = None,
    ) -> LogRadixEviction:
        evicted_kv: list[torch.Tensor] = []
        evicted_entries: list[LogStateEntry] = []
        for node in nodes:
            assert node.is_leaf() and not node.is_locked and not node.is_root()
            if evicted_nodes is not None:
                evicted_nodes.append((node.uuid, node.begin, node.depth))
            self.evictable_size -= node.length
            self._update_kv_refs(node, -1)
            evicted_kv.append(node.kv)
            evicted_entries.extend(node.entries)
            for entry in node.entries:
                self._unregister_entry(entry)

            parent = node.parent
            del parent.children[self.key_fn(node.key)]
        if not evicted_kv:
            return LogRadixEviction(self.empty_tensor, ())
        self.revision += 1
        return LogRadixEviction(
            torch.cat(evicted_kv), tuple(evicted_entries)
        )

    def matched_kv(self, node: LogRadixTreeNode) -> torch.Tensor:
        values = [path_node.kv for path_node in self._path(node)]
        return torch.cat(values) if values else self.empty_tensor

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
        """Recover an already located prefix without comparing its tokens again."""
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
        entries = self._slot_entries[value]
        return min(
            (self._entry_owners[id(entry)] for entry in entries),
            key=lambda node: node.depth,
        )

    def has_slot(self, slot: int) -> bool:
        return slot in self._slot_entries

    def entries_for_slot(self, slot: int) -> tuple[LogStateEntry, ...]:
        return tuple(self._slot_entries[slot])

    def slot_is_fragmented(self, slot: int) -> bool:
        return len(self._slot_entries[slot]) > 1

    def unique_slots(self) -> tuple[int, ...]:
        return tuple(self._slot_entries)

    def evictable_slots(self) -> tuple[int, ...]:
        return tuple(self._evictable_slots)

    def evictable_kv_frames(self) -> int:
        """Count physical KV frames whose entire cached contents can be evicted."""
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

    def leaves(self) -> tuple[LogRadixTreeNode, ...]:
        leaves = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf():
                if not node.is_root():
                    leaves.append(node)
            else:
                stack.extend(node.children.values())
        return tuple(leaves)

    def path(self, node: LogRadixTreeNode) -> tuple[LogRadixTreeNode, ...]:
        return tuple(self._path(node))

    def entry_is_protected(self, slot: int) -> bool:
        return any(
            self.entry_owner(entry).is_protected
            for entry in self._slot_entries[slot]
        )

    def is_lock_boundary(
        self,
        leaf    : LogRadixTreeNode,
        position: int,
    ) -> bool:
        for node in self._path(leaf):
            if node.depth != position:
                continue
            descendant_refs = sum(
                child.ref_count for child in node.children.values()
            )
            return node.ref_count > descendant_refs
        return False

    @property
    def size_info(self) -> SizeInfo:
        # ``SizeInfo`` keeps the legacy name ``protected_size`` for request-held
        # cache. In Log-Radix-Tree this is strictly the live-lock size and is
        # unrelated to ``node.is_protected`` compression policy.
        return SizeInfo(self.evictable_size, self.locked_size)

    def check_integrity(self) -> None:
        evictable_size = 0
        locked_size = 0
        seen_nodes: set[int] = set()
        seen_entries: dict[int, LogRadixTreeNode] = {}
        seen_slots: dict[int, list[LogStateEntry]] = {}
        kv_refs: Counter[int] = Counter()
        locked_kv_refs: Counter[int] = Counter()
        locked_slot_refs: Counter[int] = Counter()
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
                if len(node._kv_pages) != node.length:
                    raise RuntimeError(
                        f"Node {node.uuid} has mismatched KV frame metadata."
                    )
                if node.ref_count < 0:
                    raise RuntimeError(
                        f"Node {node.uuid} has a negative reference count."
                    )
                kv_refs.update(node._kv_page_counts.keys())
                if node.is_locked:
                    locked_kv_refs.update(node._kv_page_counts.keys())
                    locked_slot_refs.update(entry.slot for entry in node._entries)
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
                    if entry.kind is LogStateKind.LOG:
                        if entry.end > entry.physical_begin + self.page_size:
                            raise RuntimeError(
                                f"Log fragment {entry} exceeds its physical page."
                            )
                    seen_entries[id(entry)] = node
                    seen_slots.setdefault(entry.slot, []).append(entry)

                if node.is_locked:
                    locked_size += node.length
                else:
                    evictable_size += node.length

            stack.extend(node.children.values())

        if seen_entries != self._entry_owners:
            raise RuntimeError("Log-Radix-Tree slot ownership index is stale.")
        if (
            dict(kv_refs) != self._kv_frame_refs
            or dict(locked_kv_refs) != self._locked_kv_refs
            or dict(locked_slot_refs) != self._locked_slot_refs
            or self._evictable_kv_count != sum(frame not in locked_kv_refs for frame in kv_refs)
            or set(self._evictable_slots) != set(seen_slots) - set(locked_slot_refs)
        ):
            raise RuntimeError("Log-Radix-Tree reclaimable capacity index is stale.")
        if set(seen_slots) != set(self._slot_entries):
            raise RuntimeError("Log-Radix-Tree slot reference index is stale.")
        for slot, entries in seen_slots.items():
            indexed = self._slot_entries[slot]
            if {id(entry) for entry in entries} != {
                id(entry) for entry in indexed
            }:
                raise RuntimeError(
                    f"Log-Radix-Tree references for slot {slot} are stale."
                )
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
            or (
                entry.kind is LogStateKind.LOG
                and entry.begin < matched_len < entry.end
            )
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
            end = min(matched_len, entry.end)
            logs.append(entry.slice(cursor, end))
            last_log_owner = self.entry_owner(entry)
            cursor = end
            cursor = align_down(cursor, self.granularity)
            if cursor == matched_len or cursor < entry.end:
                break

        if logs:
            last = logs[-1]
            conv_slot = last.slot
            conv_granule = (
                cursor - last.physical_begin
            ) // self.granularity - 1
            assert last_log_owner is not None
            lock_node = last_log_owner
        elif base is not None:
            conv_slot = base.slot
            conv_granule = self.page_size // self.granularity - 1
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
        delta = 1 if node.is_locked else -1
        for frame in node._kv_page_counts:
            self._update_kv_frame(frame, 0, delta)
        for entry in node._entries:
            locked = self._locked_slot_refs.get(entry.slot, 0) + delta
            if locked:
                self._locked_slot_refs[entry.slot] = locked
            else:
                self._locked_slot_refs.pop(entry.slot, None)
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
    ) -> None:
        owner._entries.append(entry)
        owner._entries.sort(key=lambda value: (value.end, value.begin, value.slot))
        self._entry_owners[id(entry)] = owner
        self._slot_entries.setdefault(entry.slot, []).append(entry)
        if owner.is_locked:
            self._locked_slot_refs[entry.slot] = (
                self._locked_slot_refs.get(entry.slot, 0) + 1
            )
        self._update_slot_reclaimability(entry.slot)

    def _unregister_entry(self, entry: LogStateEntry) -> None:
        if self._entry_owners[id(entry)].is_locked:
            locked = self._locked_slot_refs[entry.slot] - 1
            if locked:
                self._locked_slot_refs[entry.slot] = locked
            else:
                del self._locked_slot_refs[entry.slot]
        del self._entry_owners[id(entry)]
        entries = self._slot_entries[entry.slot]
        entries.remove(entry)
        if not entries:
            del self._slot_entries[entry.slot]
        self._update_slot_reclaimability(entry.slot)

    def _update_slot_reclaimability(self, slot: int) -> None:
        reclaimable = slot in self._slot_entries and slot not in self._locked_slot_refs
        if reclaimable == (slot in self._evictable_slots):
            return
        if reclaimable:
            self._evictable_slots[slot] = None
        else:
            del self._evictable_slots[slot]
        if self._on_slot_reclaimable is not None:
            self._on_slot_reclaimable(slot, reclaimable)

    def _update_kv_refs(self, node: LogRadixTreeNode, delta: int) -> None:
        for frame in node._kv_page_counts:
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
