from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Sequence

import torch

from logssms.core import Request
from logssms.kernel.linear_attention import (
    materialize_log_states_inplace,
    reconstruct_log_states,
)
from logssms.utils import align_ceil, align_down, div_ceil, init_logger

from .base import BaseCacheManager, CacheResources
from .factory import BaseCacheFactory, CachePlan, SUPPORTED_CACHE_FACTORIES
from .linear_pool.logstate_pool import LOG_STATE_GRANULARITY
from .linear_pool.logstate_pool import LogStatePool
from .pools import (
    logstate_slot_bytes_for_config,
    make_kvcache_pool,
    make_logstate_pool,
)
from .prefix_cache.base import BaseCacheHandle
from .prefix_cache.logstate import (
    LogStateCacheHandle,
    LogStateRecovery,
    LogRadixTrace,
    LogRadixTree,
    LogRadixTreeNode,
    LogStateEntry,
    LogStateKind,
)

if TYPE_CHECKING:
    from logssms.core import Batch
    from logssms.engine import EngineConfig
    from logssms.speculator import SpeculatorReserve


logger = init_logger(__name__)


@dataclass(frozen=True)
class LogStateCachePlan(CachePlan):
    num_states        : int
    num_work_states   : int
    cache_per_page    : int
    cache_per_log_slot: int


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
    log_slot       : int
    owner_uuid     : int


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
    left_slot : int
    right_slot: int
    begin     : int
    end       : int
    next_level: int


@dataclass(frozen=True)
class LogStatePlan:
    tree_revision: int
    protections  : tuple[BranchProtection, ...]
    restores     : tuple[WorkStateRestore, ...]
    materialize  : tuple[PageMaterialize, ...]
    coalesce     : tuple[CheckpointCoalesce, ...]

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
    state_slot: int


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


class OnlineAnalysisScheduler:
    """Generate deterministic LogState maintenance plans.

    This first policy has no online cost model. It detects branch endpoints at
    the 16-token recovery granularity, protects the complete log page containing
    each endpoint, and materializes complete unprotected log pages in the next
    round. Newly materialized checkpoints immediately participate in one level
    of adjacent equal-level coalescing in that same round. Branch restoration
    writes directly to ``work_state_table`` and does not create a persistent
    checkpoint. This class never allocates slots, launches kernels or mutates the
    persistent tree.
    """

    def __init__(
        self,
        page_size  : int,
        granularity: int = LOG_STATE_GRANULARITY,
    ) -> None:
        assert page_size > 0
        assert granularity > 0 and page_size % granularity == 0
        self.page_size = page_size
        self.granularity = granularity

    def begin_round(self, tree: LogRadixTree) -> LogStateRound:
        return LogStateRound(
            frozenset(entry.slot for entry in tree.iter_entries())
        )

    def analyze(
        self,
        tree      : LogRadixTree,
        input_ids : Sequence[torch.Tensor],
        round_info: LogStateRound | None = None,
    ) -> LogStatePlan:
        assert tree.page_size == self.page_size
        assert tree.granularity == self.granularity
        if round_info is None:
            round_info = self.begin_round(tree)

        pending = self._find_protections(tree, input_ids)
        protections = tuple(
            BranchProtection(
                request_indices=value.request_indices,
                position=value.position,
                log_slot=entry.slot,
                owner_uuid=tree.entry_owner(entry).uuid,
            )
            for value in pending
            if (entry := self._covering_log(tree, value)) is not None
        )
        restores = tuple(
            restore
            for value in pending
            if (restore := self._plan_work_state_restore(tree, value)) is not None
        )
        materialize = self._plan_materialize(
            tree, round_info.eligible_slots, pending
        )
        coalesce = self._plan_coalesce(
            tree, round_info.eligible_slots, pending, materialize
        )
        return LogStatePlan(
            tree_revision=tree.revision,
            protections=protections,
            restores=restores,
            materialize=materialize,
            coalesce=coalesce,
        )

    def _find_protections(
        self,
        tree     : LogRadixTree,
        input_ids: Sequence[torch.Tensor],
    ) -> tuple[_PendingProtection, ...]:
        candidates = self._batch_branch_points(input_ids)

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
    ) -> tuple[PageMaterialize, ...]:
        result = []
        for entry in tree.iter_entries():
            if entry.slot not in eligible_slots or entry.kind is not LogStateKind.LOG:
                continue
            if tree.slot_is_fragmented(entry.slot):
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
    ) -> tuple[CheckpointCoalesce, ...]:
        projected = {
            plan.log_slot: LogStateEntry(
                slot=plan.log_slot,
                kind=LogStateKind.CHECKPOINT,
                begin=plan.begin,
                end=plan.end,
                merge_level=0,
            )
            for plan in materialize
        }
        result = []
        used_slots: set[int] = set()
        for leaf in tree.leaves():
            entries = [
                projected.get(entry.slot, entry)
                for entry in self._entries_for_node(tree, leaf)
                if entry.slot in eligible_slots
                and entry.slot not in used_slots
            ]
            entries = [
                entry
                for entry in entries
                if entry.kind is LogStateKind.CHECKPOINT
                and entry.merge_level is not None
            ]
            entries.sort(key=lambda entry: (entry.begin, entry.end, entry.slot))
            index = 0
            while index + 1 < len(entries):
                left, right = entries[index : index + 2]
                if not self._can_coalesce(tree, leaf, left, right, pending):
                    index += 1
                    continue
                result.append(
                    CheckpointCoalesce(
                        left_slot=left.slot,
                        right_slot=right.slot,
                        begin=left.begin,
                        end=right.end,
                        next_level=left.merge_level + 1,
                    )
                )
                used_slots.update((left.slot, right.slot))
                index += 2
        result.sort(key=lambda value: (value.begin, value.end, value.left_slot))
        return tuple(result)

    def _can_coalesce(
        self,
        tree   : LogRadixTree,
        leaf   : LogRadixTreeNode,
        left   : LogStateEntry,
        right  : LogStateEntry,
        pending: tuple[_PendingProtection, ...],
    ) -> bool:
        assert left.merge_level is not None and right.merge_level is not None
        expected_span = self.page_size << left.merge_level
        return (
            left.merge_level == right.merge_level
            and left.end == right.begin
            and left.end - left.begin == expected_span
            and right.end - right.begin == expected_span
            and not self._interval_is_protected(
                tree, leaf, left.begin, right.end, pending
            )
        )

    def _interval_is_protected(
        self,
        tree   : LogRadixTree,
        node   : LogRadixTreeNode,
        begin  : int,
        end    : int,
        pending: tuple[_PendingProtection, ...],
    ) -> bool:
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
    place by the CUDA reconstruction kernel.
    """

    def __init__(
        self,
        num_pages      : int,
        num_states     : int,
        num_work_states: int,
        page_size      : int,
        resources      : CacheResources,
        granularity    : int = LOG_STATE_GRANULARITY,
    ) -> None:
        pool = resources.linear_state
        state_table = resources.state_table
        work_state_table = resources.work_state_table
        assert isinstance(pool, LogStatePool)
        assert state_table is not None
        assert work_state_table is not None
        assert pool.page_size == page_size
        assert pool.layout.granularity == granularity
        assert num_pages > 0 and num_states > 0
        assert num_work_states == work_state_table.numel()
        assert pool.num_slots >= num_states + num_work_states

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
        self.analyzer = OnlineAnalysisScheduler(page_size, granularity)
        self.free_pages = torch.arange(
            num_pages, dtype=torch.int32, device=self.device
        ) * page_size
        self.free_states = torch.arange(
            num_states, dtype=torch.int32, device=self.device
        )
        self.num_pages = num_pages
        self.num_states = num_states
        self.num_work_states = num_work_states
        self.granularity = granularity

        self.pending_recoveries: dict[int, LogStateRecovery] = {}
        self.last_plan: LogStatePlan | None = None
        self._live_pages: dict[int, list[_LivePage]] = {}
        self._cached_page_tokens: dict[int, int] = {}
        self._lazy_free_pages: list[torch.Tensor] | None = None
        self._lazy_free_states: list[torch.Tensor] | None = None

    @property
    def tree(self) -> LogRadixTree:
        assert isinstance(self.prefix_cache, LogRadixTree)
        return self.prefix_cache

    @property
    def logstate_pool(self) -> LogStatePool:
        pool = self.linear_pool
        assert isinstance(pool, LogStatePool)
        return pool

    @property
    def available_size(self) -> int:
        evictable_entries = len(self.tree.evictable_slots())
        kv_pages = len(self.free_pages) + self.tree.size_info.evictable_size // self.page_size
        state_pages = len(self.free_states) + evictable_entries
        return min(kv_pages, state_pages) * self.page_size

    def plan_prefill_chunk(self, remain_len: int, token_budget: int) -> int:
        chunk_size = min(remain_len, token_budget)
        if chunk_size < remain_len:
            chunk_size = align_down(chunk_size, self.page_size)
        return chunk_size

    def init_with_handle(self, table_idx: int, handle: BaseCacheHandle) -> None:
        assert isinstance(handle, LogStateCacheHandle)
        if handle.cached_len > 0:
            self.page_table[table_idx, : handle.cached_len].copy_(
                handle.kv_indices
            )
        state_table = self.state_table
        assert state_table is not None
        state_table[table_idx].fill_(-1)
        self.pending_recoveries[table_idx] = handle.recovery
        self._live_pages[table_idx] = []

    def allocate(self, reqs: list[Request]) -> None:
        state_table = self.state_table
        work_state_table = self.work_state_table
        assert state_table is not None and work_state_table is not None

        allocation_info: list[tuple[Request, int, int]] = []
        needed = 0
        for req in reqs:
            pages = self._live_pages.setdefault(req.table_idx, [])
            begin = pages[-1].end if pages else req.cached_len
            if req.device_len > begin:
                count = div_ceil(req.device_len - begin, self.page_size)
                allocation_info.append((req, begin, count))
                needed += count

            if req.cached_len == 0 and req.table_idx not in self.pending_recoveries:
                work_slot = int(work_state_table[req.table_idx, 0])
                self.logstate_pool.recurrent_state[work_slot].zero_()
                self.logstate_pool.conv_state[work_slot].zero_()

        if needed == 0:
            return

        kv_pages, state_slots = self._allocate(needed, needed)
        offset = 0
        token_offsets = torch.arange(
            self.page_size, dtype=torch.int32, device=self.device
        )
        for req, begin, count in allocation_info:
            pages = self._live_pages[req.table_idx]
            for index in range(count):
                logical_begin = begin + index * self.page_size
                logical_end = logical_begin + self.page_size
                kv_page = int(kv_pages[offset])
                state_slot = int(state_slots[offset])
                offset += 1
                pages.append(
                    _LivePage(
                        begin=logical_begin,
                        end=logical_end,
                        kv_page=kv_page,
                        state_slot=state_slot,
                    )
                )
                mapped_end = min(
                    logical_end,
                    req.max_device_len,
                    self.page_table.shape[1],
                )
                mapped_len = mapped_end - logical_begin
                assert 0 < mapped_len <= self.page_size
                self.page_table[
                    req.table_idx, logical_begin:mapped_end
                ].copy_(kv_page + token_offsets[:mapped_len])
                state_table[
                    req.table_idx, logical_begin:mapped_end
                ].fill_(state_slot)
        assert offset == needed

    def prepare_batch(self, batch: Batch) -> None:
        if not batch.is_prefill:
            self.last_plan = None
            return
        round_info = self.analyzer.begin_round(self.tree)
        plan = self.analyzer.analyze(
            self.tree,
            [req.input_ids for req in batch.reqs],
            round_info,
        )
        for protection in plan.protections:
            if self.tree.has_slot(protection.log_slot):
                self.tree.protect_entry(protection.log_slot)

        self.last_plan = plan
        tasks = self._make_restore_tasks(batch)
        self._reconstruct(tasks)
        self._materialize_inplace(plan.materialize)

        for req in batch.reqs:
            self.pending_recoveries.pop(req.table_idx, None)
        self._commit_materialize(plan.materialize)
        self._commit_coalesce(plan.coalesce)

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
        assert (insert_len - old_handle.cached_len) % self.page_size == 0

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
        duplicate_kv_pages = []
        for page in adopted:
            self.tree.attach_entry(
                insert.node,
                LogStateEntry(
                    slot=page.state_slot,
                    kind=LogStateKind.LOG,
                    begin=page.begin,
                    end=page.end,
                ),
            )
            kv_begin = max(page.begin, insert.previous_len)
            if kv_begin < page.end:
                self._retain_cached_kv(
                    self.page_table[req.table_idx, kv_begin:page.end]
                )
            else:
                duplicate_kv_pages.append(page.kv_page)

        duplicates = [
            page
            for page in full_pages
            if page not in adopted and page.end <= insert.previous_len
        ]
        self._release_live_pages(req.table_idx, duplicates)
        if duplicate_kv_pages:
            self._return_slots(
                torch.tensor(
                    duplicate_kv_pages,
                    dtype=torch.int32,
                    device=self.device,
                ),
                torch.empty(0, dtype=torch.int32, device=self.device),
            )

        new_handle = self.tree.make_handle(insert.node, insert.inserted_len)
        self.unlock(old_handle)
        self.pending_recoveries.pop(req.table_idx, None)

        state_table = self.state_table
        assert state_table is not None
        state_table[req.table_idx, : insert.inserted_len].fill_(-1)
        self.page_table[
            req.table_idx, : insert.inserted_len
        ].copy_(new_handle.kv_indices)

        transferred = {page.state_slot for page in adopted + duplicates}
        remaining = [
            page for page in pages if page.state_slot not in transferred
        ]
        if (
            not finished
            and adopted
            and req.cached_len == adopted[-1].end
        ):
            work_state_table = self.work_state_table
            assert work_state_table is not None
            work_slot = int(work_state_table[req.table_idx, 0])
            self.logstate_pool.conv_state[work_slot, :, 0].copy_(
                self.logstate_pool.conv_state[
                    adopted[-1].state_slot, :, -1
                ]
            )
        if finished:
            self._release_live_pages(req.table_idx, remaining)
            self._live_pages.pop(req.table_idx, None)
            state_table[req.table_idx].fill_(-1)
        else:
            req.cache_handle = new_handle
            self.lock(new_handle)
            self._live_pages[req.table_idx] = remaining

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

        if len(torch.unique(self.free_pages)) != len(self.free_pages):
            raise RuntimeError("LogState free KV page list contains duplicates.")
        if torch.any(self.free_pages % self.page_size != 0):
            raise RuntimeError("LogState free KV page list is not page aligned.")
        if len(self.free_pages) + len(self._cached_page_tokens) != self.num_pages:
            raise RuntimeError(
                "LogState KV page accounting mismatch:"
                f" free={len(self.free_pages)},"
                f" cached={len(self._cached_page_tokens)},"
                f" total={self.num_pages}."
            )
        if sum(self._cached_page_tokens.values()) != self.tree.size_info.total_size:
            raise RuntimeError("LogState cached KV token accounting mismatch.")

        tree_slots = {entry.slot for entry in self.tree.iter_entries()}
        free_slots = set(self.free_states.tolist())
        work_state_table = self.work_state_table
        assert work_state_table is not None
        work_slots = set(work_state_table.flatten().tolist())
        if not tree_slots.isdisjoint(free_slots | work_slots):
            raise RuntimeError("LogState linear slots have multiple owners.")
        if len(free_slots) + len(tree_slots) != self.num_states:
            raise RuntimeError(
                "LogState linear slot accounting mismatch:"
                f" free={len(free_slots)}, tree={len(tree_slots)},"
                f" total={self.num_states}."
            )
        if len(work_slots) != self.num_work_states:
            raise RuntimeError("LogState work-state table contains duplicate slots.")

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
                self.free_pages = torch.cat([self.free_pages, *lazy_pages])
            if lazy_states:
                self.free_states = torch.cat([self.free_states, *lazy_states])

    def _allocate(
        self,
        needed_pages : int,
        needed_states: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        while (
            len(self.free_pages) < needed_pages
            or len(self.free_states) < needed_states
        ):
            evictable = self.tree.size_info.evictable_size
            if evictable == 0:
                raise RuntimeError(
                    "Insufficient evictable LogState cache capacity:"
                    f" need_pages={needed_pages}, free_pages={len(self.free_pages)},"
                    f" need_states={needed_states}, free_states={len(self.free_states)}."
                )
            missing_pages = max(0, needed_pages - len(self.free_pages))
            missing_states = max(0, needed_states - len(self.free_states))
            evict_size = max(
                self.granularity,
                missing_pages * self.page_size,
                missing_states * self.granularity,
            )
            eviction = self.tree.evict(min(evict_size, evictable))
            self._release_cached_kv(eviction.kv_indices)
            released_states = sorted({
                entry.slot
                for entry in eviction.entries
                if not self.tree.has_slot(entry.slot)
            })
            self._return_slots(
                torch.empty(0, dtype=torch.int32, device=self.device),
                torch.tensor(
                    released_states,
                    dtype=torch.int32,
                    device=self.device,
                ),
            )

        pages = self.free_pages[:needed_pages]
        states = self.free_states[:needed_states]
        self.free_pages = self.free_pages[needed_pages:]
        self.free_states = self.free_states[needed_states:]
        return pages, states

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

    def _reconstruct(self, tasks: Sequence[_ReconstructionTask]) -> None:
        if not tasks:
            return
        device = self.device
        pool = self.logstate_pool
        logs = [entry for task in tasks for entry in task.logs]
        offsets = [0]
        for task in tasks:
            offsets.append(offsets[-1] + len(task.logs))

        def tensor(values: Sequence[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int32, device=device)

        reconstruct_log_states(
            key_log=pool.key_log,
            residual_value_log=pool.residual_value_log,
            decay_log=pool.decay_log,
            recurrent_state=pool.recurrent_state,
            conv_state=pool.conv_state,
            output_slots=tensor([task.output_slot for task in tasks]),
            base_slots=tensor([
                -1 if task.base_slot is None else task.base_slot
                for task in tasks
            ]),
            task_log_offsets=tensor(offsets),
            log_slots=tensor([entry.slot for entry in logs]),
            log_begins=tensor([
                entry.begin - entry.physical_begin for entry in logs
            ]),
            log_ends=tensor([
                entry.end - entry.physical_begin for entry in logs
            ]),
            source_conv_slots=tensor([
                task.source_conv_slot for task in tasks
            ]),
            source_conv_granules=tensor([
                task.source_conv_granule for task in tasks
            ]),
            output_conv_granules=tensor([
                task.output_conv_granule for task in tasks
            ]),
        )

    def _materialize_inplace(
        self,
        plans: Sequence[PageMaterialize],
    ) -> None:
        chains = self._make_materialize_chains(plans)
        if not chains:
            return

        logs = [entry for chain in chains for entry in chain.logs]
        offsets = [0]
        for chain in chains:
            offsets.append(offsets[-1] + len(chain.logs))

        def tensor(values: Sequence[int]) -> torch.Tensor:
            return torch.tensor(
                values, dtype=torch.int32, device=self.device
            )

        pool = self.logstate_pool
        materialize_log_states_inplace(
            key_log=pool.key_log,
            residual_value_log=pool.residual_value_log,
            decay_log=pool.decay_log,
            recurrent_state=pool.recurrent_state,
            base_slots=tensor([
                -1 if chain.base_slot is None else chain.base_slot
                for chain in chains
            ]),
            chain_log_offsets=tensor(offsets),
            log_slots=tensor([entry.slot for entry in logs]),
            log_begins=tensor([
                entry.begin - entry.physical_begin for entry in logs
            ]),
            log_ends=tensor([
                entry.end - entry.physical_begin for entry in logs
            ]),
            checkpoint_flags=tensor([
                int(entry.slot in chain.checkpoint_slots)
                for chain in chains
                for entry in chain.logs
            ]),
        )

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
                    merge_level=0,
                ),
            )

    def _commit_coalesce(
        self,
        plans: Sequence[CheckpointCoalesce],
    ) -> None:
        released = []
        for plan in plans:
            if (
                not self.tree.has_slot(plan.left_slot)
                or not self.tree.has_slot(plan.right_slot)
                or self.tree.slot_is_fragmented(plan.left_slot)
                or self.tree.slot_is_fragmented(plan.right_slot)
            ):
                continue
            left = self.tree.entries_for_slot(plan.left_slot)[0]
            right = self.tree.entries_for_slot(plan.right_slot)[0]
            if (
                left.kind is not LogStateKind.CHECKPOINT
                or right.kind is not LogStateKind.CHECKPOINT
            ):
                continue
            owner = self.tree.entry_owner(right)
            self.tree.detach_entry(plan.left_slot)
            self.tree.detach_entry(plan.right_slot)
            self.tree.attach_entry(
                owner,
                LogStateEntry(
                    slot=plan.right_slot,
                    kind=LogStateKind.CHECKPOINT,
                    begin=plan.begin,
                    end=plan.end,
                    merge_level=plan.next_level,
                ),
            )
            released.append(plan.left_slot)
        if released:
            self._return_slots(
                torch.empty(0, dtype=torch.int32, device=self.device),
                torch.tensor(
                    released, dtype=torch.int32, device=self.device
                ),
            )

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
                [page.state_slot for page in pages],
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
            if len(pages) > 0:
                self.free_pages = torch.cat([self.free_pages, pages])
            if len(states) > 0:
                self.free_states = torch.cat([self.free_states, states])
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

    def _release_cached_kv(self, indices: torch.Tensor) -> None:
        if len(indices) == 0:
            return
        pages = torch.div(
            indices.to(torch.int64), self.page_size,
            rounding_mode="floor",
        ) * self.page_size
        unique, counts = torch.unique(pages, return_counts=True)
        released = []
        for page, count in zip(unique.tolist(), counts.tolist()):
            remaining = self._cached_page_tokens[page] - count
            assert remaining >= 0
            if remaining == 0:
                del self._cached_page_tokens[page]
                released.append(page)
            else:
                self._cached_page_tokens[page] = remaining
        if released:
            self._return_slots(
                torch.tensor(
                    released, dtype=torch.int32, device=self.device
                ),
                torch.empty(0, dtype=torch.int32, device=self.device),
            )


class LogStateCacheFactory(BaseCacheFactory):
    """Allocate independent KV-page and LogState-slot budgets at a 3:1 ratio."""

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
        cache_per_log_slot = logstate_slot_bytes_for_config(
            config.model_config,
            config.page_size,
            config.cache_config.granularity,
            config.dtype,
        )
        num_work_states = config.max_running_req + 1
        work_state_bytes = num_work_states * cache_per_log_slot
        persistent_memory = available_memory - work_state_bytes
        assert persistent_memory > 0, (
            "Not enough memory for LogState request work states."
        )

        kv_budget = persistent_memory * 3 // 4
        state_budget = persistent_memory - kv_budget
        num_pages = (
            kv_budget // cache_per_page
            if config.num_page_override is None else config.num_page_override
        )
        num_states = state_budget // cache_per_log_slot
        assert num_pages > 1 and num_states > 1, (
            "Not enough memory for LogState cache allocation."
        )

        num_tokens = num_pages * config.page_size
        max_seq_len = min(
            config.max_seq_len,
            num_tokens,
            num_states * config.page_size,
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
        kv_pool = make_kvcache_pool(
            model_config=config.model_config,
            num_pages=plan.num_pages + 1,
            page_size=config.page_size,
            device=device,
            dtype=dtype,
        )
        page_table = torch.zeros(
            (config.max_running_req + 1, plan.aligned_max_seq_len),
            dtype=torch.int32,
            device=device,
        )
        page_table[-1].fill_(plan.num_tokens)

        linear_state = make_logstate_pool(
            model_config=config.model_config,
            num_slots=plan.num_states + plan.num_work_states,
            page_size=config.page_size,
            granularity=config.cache_config.granularity,
            device=device,
            dtype=dtype,
        )
        state_table = torch.full(
            (config.max_running_req + 1, plan.aligned_max_seq_len),
            -1,
            dtype=torch.int32,
            device=device,
        )
        work_state_table = torch.arange(
            plan.num_states,
            plan.num_states + plan.num_work_states,
            dtype=torch.int32,
            device=device,
        ).view(config.max_running_req + 1, 1)
        dummy_work_slot = int(work_state_table[-1, 0])
        state_table[-1].fill_(dummy_work_slot)
        linear_state.recurrent_state[dummy_work_slot].zero_()
        linear_state.conv_state[dummy_work_slot].zero_()

        logger.info(
            f"Allocating {plan.num_pages} KV pages = "
            f"{plan.num_pages * plan.cache_per_page / (1 << 30):.2f} GiB, "
            f"{plan.num_states} persistent LogState slots + "
            f"{plan.num_work_states} work slots = "
            f"{(plan.num_states + plan.num_work_states) * plan.cache_per_log_slot / (1 << 30):.2f} GiB"
        )
        return CacheResources(
            kv_pool=kv_pool,
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
