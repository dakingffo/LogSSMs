from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from picosgl.core import Request
from picosgl.cache import BaseCacheHandle, MatchResult, make_prefix_cache
from picosgl.utils import align_down, div_ceil

if TYPE_CHECKING:
    from .prefill import PendingRequest


class CacheManager:
    def __init__(
        self,
        num_pages       : int,
        page_size       : int,
        page_table      : torch.Tensor,
        type            : str,
        num_states      : int                 = 0,
        num_draft_states: int                 = 0,
        state_table     : torch.Tensor | None = None,
        work_state_table: torch.Tensor | None = None,
        state_pool      : object              = None,
    ):
        device = page_table.device
        # The `_free_pages` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_pages` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        self.free_pages = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        self.free_states = torch.arange(num_states, dtype=torch.int32, device=device)
        self.prefix_cache = make_prefix_cache(device=device, type=type)
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size
        # hybrid linear-state wiring (None for non-hybrid models)
        self.num_states = num_states
        self.num_draft_states = num_draft_states
        self.state_table = state_table
        self.work_state_table = work_state_table
        self.state_pool = state_pool

    def match_req(self, req: PendingRequest) -> MatchResult:
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        return self.prefix_cache.size_info.evictable_size + len(self.free_pages) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, reqs: list[Request]) -> None:
        needed_pages = 0
        allocation_info: list[tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages=needed_pages)[0])
            self._write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    def allocate_state(self, reqs: list[Request]) -> None:
        if self.state_pool is None or self.state_table is None:
            return
        needed_states = 0
        allocation_info: list[tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_states += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_states > 0:
            allocated = self._allocate(needed_states=needed_states)[1]
            assert allocated is not None
            self._write_page_table(
                self.state_table,
                allocated.repeat_interleave(self.page_size),
                allocation_info,
                self.page_size,
            )

    def cache_req(self, req: Request, *, finished: bool) -> None:
        # ==================================== valid cache region ====================================
        # [0, req.cached_len)                       This part is valid for attention kernel read/write.
        # [0, old_handle.cached_len)                This part is in the prefix cache before prefill.
        # [old_handle.cached_len, req.cached_len)   This part is allocated by cache manager for this request.
        # ================================== allocated cache region ==================================
        # [old_handle.cached_len, cached_len)       This part was not in the prefix cache when prefill,
        #                                           but later cached by other requests.
        #                                           We must free them to avoid memory leak.
        # [cached_len, new_handle.cached_len)       This part is newly inserted into the prefix cache.
        # [new_handle.cached_len, req.cached_len)   This part is tailing part that can not inserted into the prefix cache.
        #                                           We should free it if the request has finished.
        insert_ids = req.input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, :req.cached_len]
        old_handle = req.cache_handle
        insert_len = align_down(req.cached_len, self.page_size)
        state_slots = None
        if self.state_table is not None and insert_len > 0:
            state_slots = self.state_table[
                req.table_idx,
                self.page_size - 1 : insert_len : self.page_size,
            ]
        if state_slots is not None:  # hybrid: the tree stores one state slot per page
            cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices, state_slots)
        else:  # dense: NaivePrefixCache / RadixPrefixCache take no state slots
            cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        # unlock until all operations on handle is done
        self.unlock(old_handle)
        # this part is already in the prefix cache, free it
        self._free(page_indices[old_handle.cached_len: cached_len])
        if finished:  # this tail part should be freed
            self._free(page_indices[new_handle.cached_len:])
        else:  # keep the tail part, update the handle
            req.cache_handle = new_handle
            self.lock(new_handle)

        if self.state_table is not None:
            self._cache_req_state(
                req, old_handle, cached_len, insert_len, new_handle, finished
            )

    def _cache_req_state(
        self,
        req       : Request,
        old_handle: BaseCacheHandle,
        prefix_len: int,
        insert_len: int,
        new_handle: BaseCacheHandle,
        finished  : bool,
    ) -> None:
        uid = req.table_idx
        # 1) free this request's own slots absorbed by the tree since prefill (they are
        #    no longer referenced once we re-point below). Free BEFORE the re-point.
        self._free_state_pages(
            uid, old_handle.cached_len // self.page_size, prefix_len // self.page_size
        )
        # 2) free the tail (partially-consumed pages) if finished
        if finished:
            self._free_state_pages(
                uid, insert_len // self.page_size, div_ceil(req.cached_len, self.page_size)
            )
            # 3) wipe this request's live token-to-state mapping. Verify reserves
            #    permanently belong to a separate table and survive table_idx reuse.
            self.state_table[uid] = -1
        else:
            # 3) re-point matched pages to the tree's canonical slots
            _matched_indices, matched_state = new_handle.get_matched_indices()
            self.state_table[uid, :insert_len] = matched_state.repeat_interleave(
                self.page_size
            )

    def _free_state_pages(self, uid: int, lo: int, hi: int) -> None:
        if self.state_pool is None or lo >= hi:
            return
        begin, end = lo * self.page_size, hi * self.page_size
        slots = self.state_table[uid, begin:end:self.page_size]
        live = slots[slots >= 0]
        if len(live) > 0:
            self.free_states = torch.cat([self.free_states, live])
        self.state_table[uid, begin:end] = -1

    def free_page_range(self, req: Request, first_page: int, last_page: int) -> None:
        if first_page >= last_page:
            return
        begin, end = first_page * self.page_size, last_page * self.page_size
        self._free(self.page_table[req.table_idx, begin:end])
        self._free_state_pages(req.table_idx, first_page, last_page)

    def state_commit_verify(self, req: Request, C: int, num_sampled: int) -> None:
        if self.state_pool is None or self.state_table is None:
            return
        uid = req.table_idx
        ps = self.page_size
        reserve = self.work_state_table
        assert reserve is not None and num_sampled <= reserve.shape[1]
        C_end = C + num_sampled
        first_page = C // ps
        last_page = (C_end - 1) // ps
        for page in range(first_page, last_page + 1):
            last_position = min(C_end, (page + 1) * ps) - 1
            candidate = last_position - C
            live_position = page * ps
            live_slot = int(self.state_table[uid, live_position])
            verify_slot = int(reserve[uid, candidate])
            assert live_slot >= 0 and verify_slot >= 0
            self.state_table[
                uid, live_position : live_position + ps
            ] = verify_slot
            reserve[uid, candidate] = live_slot

    def check_integrity(self) -> None:
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_pages) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_pages)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            assert torch.all(self.free_pages % self.page_size == 0)
        if self.state_pool is not None and self.state_table is not None:
            # Idle (the only time this runs): live rows have no request-owned refs.
            # The last row belongs to CUDA Graph padding; verify reserves remain populated.
            free_state = len(self.free_states)
            live_state = int(
                (self.state_table[:-1, ::self.page_size] >= 0).sum().item()
            )
            tree_state = self.prefix_cache.total_state_pages()
            if live_state != 0:
                raise RuntimeError(
                    "CacheManager state integrity check failed:"
                    f" {live_state} slots still referenced in state_table while idle."
                )
            if free_state + tree_state != self.num_states:
                raise RuntimeError(
                    "CacheManager state integrity check failed:"
                    f" free_state({free_state}) + tree_state({tree_state}) !="
                    f" num_states({self.num_states})"
                )
            if self.work_state_table is not None:
                reserve = self.work_state_table.flatten()
                if (
                    len(reserve) != self.num_draft_states
                    or torch.any(reserve < 0)
                    or len(torch.unique(reserve)) != len(reserve)
                ):
                    raise RuntimeError(
                        "CacheManager state integrity check failed: verify reserve "
                        "contains missing or duplicate slot references."
                    )

    @contextmanager
    def lazy_free_region(self):
        lazy_free_list: list[torch.Tensor] = []
        self_free = self._free
        try:
            self._free = lambda indices: lazy_free_list.append(indices[:: self.page_size])
            yield
        finally:
            self._free = self_free
            self.free_pages = torch.cat([self.free_pages] + lazy_free_list)

    def _allocate(self, needed_pages: int = 0, needed_states: int = 0) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        free_pages = len(self.free_pages)
        free_states = len(self.free_states)
        if needed_pages > free_pages or needed_states > free_states:
            # Each evicted radix node frees page_size KV tokens AND one state slot per
            # page, so the shortfall (in pages/slots) is the unit to evict in tokens.
            needed_evict = max(
                0, needed_pages - free_pages, needed_states - free_states
            ) * self.page_size
            evictable = self.prefix_cache.size_info.evictable_size
            if needed_evict > evictable:
                raise RuntimeError(
                    "Insufficient evictable cache capacity:"
                    f" need_pages={needed_pages}, free_pages={free_pages},"
                    f" need_states={needed_states}, free_states={free_states},"
                    f" need_evict_tokens={needed_evict},"
                    f" evictable_tokens={evictable},"
                    f" protected_tokens={self.prefix_cache.size_info.protected_size}."
                )
            evicted_pages, evicted_states = self.prefix_cache.evict(needed_evict)
            self.free_pages = torch.cat([self.free_pages, evicted_pages[:: self.page_size]])
            if self.state_pool is not None and evicted_states is not None:
                self.free_states = torch.cat([self.free_states, evicted_states])
            assert needed_pages == 0 or len(self.free_pages) >= needed_pages, "Eviction did not free enough pages."
            assert needed_states == 0 or len(self.free_states) >= needed_states, "Eviction did not free enough states."

        allocated_pages = None
        if needed_pages != 0:
            allocated_pages = self.free_pages[:needed_pages]
            self.free_pages = self.free_pages[needed_pages:]

        allocated_states = None
        if needed_states != 0:
            allocated_states = self.free_states[:needed_states]
            self.free_states = self.free_states[needed_states:]

        return allocated_pages, allocated_states

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self.free_pages = torch.cat([self.free_pages, indices[:: self.page_size]])

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, dtype=torch.int32, device=self.device)
        return (pages.unsqueeze(1) + offsets).flatten()

    def _write_page_table(
        self,
        page_table: torch.Tensor,
        allocated: torch.Tensor,
        allocation_info: list[tuple[int, int, int]],
        page_size: int,
    ) -> None:
        needed_tokens = len(allocated)
        pin_memory = page_table.is_cuda
        table_idx_host = torch.empty(
            needed_tokens, dtype=torch.int64, pin_memory=pin_memory
        )
        positions_host = torch.empty(
            needed_tokens, dtype=torch.int64, pin_memory=pin_memory
        )
        offset = 0
        for table_idx, first_page, last_page in allocation_info:
            first_pos, last_pos = first_page * page_size, last_page * page_size
            length = last_pos - first_pos
            table_idx_host[offset : offset + length].fill_(table_idx)
            torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
            offset += length
        assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
        table_idx = table_idx_host.to(page_table.device, non_blocking=True)
        positions = positions_host.to(page_table.device, non_blocking=True)
        page_table[table_idx, positions] = allocated
