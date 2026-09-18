from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from logssms.core import Request
from logssms.utils import align_ceil, align_down, div_ceil, init_logger

from .base import BaseCacheManager, CacheResources
from .factory import (
    BaseCacheFactory,
    CachePlan,
    SUPPORTED_CACHE_FACTORIES,
)
from .pools import (
    linear_state_slot_bytes_for_config,
    make_kvcache_pool,
    make_linear_state_pool,
)
from .prefix_cache.base import BaseCacheHandle
from .prefix_cache.standard import StandardPrefixValue

if TYPE_CHECKING:
    from logssms.engine import EngineConfig
    from logssms.speculator import SpeculatorReserve


logger = init_logger(__name__)

_MIN_KV_PAGES = 2


@dataclass(frozen=True)
class StandardCachePlan(CachePlan):
    num_states             : int
    num_draft_states       : int
    draft_slots_per_request: int
    cache_per_page         : int
    cache_per_state        : int


class StandardCacheManager(BaseCacheManager):
    def __init__(
        self,
        num_pages        : int,
        page_size        : int,
        resources        : CacheResources,
        prefix_cache_type: str,
        num_states       : int                 = 0,
        num_draft_states : int                 = 0,
    ):
        super().__init__(page_size, resources, prefix_cache_type)
        # The `_free_pages` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_pages` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        self.free_pages = torch.arange(
            num_pages, dtype=torch.int32, device=self.device
        ) * page_size
        self.free_states = torch.arange(
            num_states, dtype=torch.int32, device=self.device
        )
        self.num_pages = num_pages
        self.num_states = num_states
        self.num_draft_states = num_draft_states

    @property
    def available_size(self) -> int:
        return self.prefix_cache.size_info.evictable_size + len(self.free_pages) * self.page_size

    def plan_prefill_chunk(self, remain_len: int, token_budget: int) -> int:
        chunk_size = min(remain_len, token_budget)
        if chunk_size < remain_len and self.state_pool is not None:
            chunk_size = align_down(chunk_size, 64)
        return chunk_size

    def init_with_handle(self, table_idx: int, handle: BaseCacheHandle) -> None:
        matched = handle.get_matched_indices()
        page_entry = self.page_table[table_idx][:handle.cached_len]
        if (state_table := self.state_table) is not None:
            page_indices, state_indices = matched
            page_entry.copy_(page_indices)
            state_entry = state_table[table_idx][:handle.cached_len]
            state_entry.copy_(state_indices)
        else:
            page_entry.copy_(matched)

    def allocate(self, reqs: list[Request]) -> None:
        needed_pages = 0
        allocation_info: list[tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))

        if needed_pages > 0:
            allocated = self._allocate(
                needed_pages=needed_pages,
                needed_states=needed_pages if self.state_pool is not None else 0,
            )
            if allocated[0] is not None:
                self._write_page_table(
                    self.page_table,
                    self._page_to_token(allocated[0]),
                    allocation_info,
                    self.page_size,
                )
            if allocated[1] is not None:
                assert self.state_table is not None
                self._write_page_table(
                    self.state_table,
                    allocated[1].repeat_interleave(self.page_size),
                    allocation_info,
                    self.page_size,
                )

    def cache_req(self, req: Request, *, finished: bool) -> None:
        insert_ids = req.input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, :req.cached_len]
        state_indices = (
            self.state_table[req.table_idx, :req.cached_len]
            if self.state_table is not None else None
        )
        old_handle = req.cache_handle
        insert_len = align_down(req.cached_len, self.page_size)

        value = StandardPrefixValue(page_indices, state_indices)
        prefix_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, value)
        # unlock until all operations on handle is done
        self.unlock(old_handle)
        self._free(
            page_indices[old_handle.cached_len: prefix_len],
            state_indices=(
                state_indices[old_handle.cached_len: prefix_len]
                if state_indices is not None else None
            ),
        )
        if finished:  # this tail part should be freed
            self._free(
                page_indices[new_handle.cached_len:],
                state_indices=(
                    state_indices[new_handle.cached_len:]
                    if state_indices is not None else None
                ),
            )
            if self.state_table is not None:
                self.state_table[req.table_idx] = -1
        else:  # keep the tail part, update the handle
            req.cache_handle = new_handle
            self.lock(new_handle)
            matched = new_handle.get_matched_indices()
            if self.state_table is not None:
                matched_pages, matched_states = matched
                self.page_table[req.table_idx, :insert_len].copy_(matched_pages)
                self.state_table[req.table_idx, :insert_len].copy_(matched_states)
            else:
                self.page_table[req.table_idx, :insert_len].copy_(matched)

    def _free(
        self,
        page_indices : torch.Tensor,
        state_indices: torch.Tensor | None = None,
    ) -> None:
        if len(page_indices) > 0:
            self.free_pages = torch.cat([self.free_pages, page_indices[:: self.page_size]])
        if state_indices is not None and len(state_indices) > 0:
            self.free_states = torch.cat([self.free_states, state_indices[:: self.page_size]])

    def free_to_cached(self, req: Request) -> None:
        C = align_ceil(req.cached_len, self.page_size)
        D = align_ceil(req.device_len, self.page_size)
        assert C <= D
        if C < D:
            self._free(
                self.page_table[req.table_idx, C:D],
                (
                    self.state_table[req.table_idx, C:D]
                    if self.state_table is not None else None
                ),
            )
            if self.state_table is not None:
                self.state_table[req.table_idx, C:D] = -1
        req.device_len = min(req.cached_len + 1, req.max_device_len)

    def commit_speculative(self, req: Request, C: int, num_sampled: int) -> None:
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
            # The last row belongs to CUDA Graph padding; work slots remain populated.
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
                        "CacheManager state integrity check failed: work state table "
                        "contains missing or duplicate slot references."
                    )

    @contextmanager
    def lazy_free_region(self):
        lazy_free_pages: list[torch.Tensor] = []
        lazy_free_states: list[torch.Tensor] = []
        self_free = self._free

        def lazy_free(
            page_indices : torch.Tensor,
            state_indices: torch.Tensor | None = None,
        ) -> None:
            if len(page_indices) > 0:
                lazy_free_pages.append(
                    page_indices[:: self.page_size].clone()
                )
            if state_indices is not None and len(state_indices) > 0:
                lazy_free_states.append(
                    state_indices[:: self.page_size].clone()
                )

        try:
            self._free = lazy_free
            yield
        finally:
            self._free = self_free
            if lazy_free_pages:
                self.free_pages = torch.cat([self.free_pages] + lazy_free_pages)
            if lazy_free_states:
                self.free_states = torch.cat([self.free_states] + lazy_free_states)

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


class StandardCacheFactory(BaseCacheFactory):
    def __init__(self, prefix_cache_type: str) -> None:
        self.prefix_cache_type = prefix_cache_type

    def plan(
        self,
        config            : EngineConfig,
        available_memory  : int,
        speculator_reserve: SpeculatorReserve | None,
    ) -> StandardCachePlan:
        from logssms.cache.config import StandardCacheConfig
        from logssms.distributed import try_get_tp_info
        from logssms.utils import div_even

        assert isinstance(config.cache_config, StandardCacheConfig)
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
        cache_per_state = linear_state_slot_bytes_for_config(
            config.model_config, config.dtype
        )
        num_draft_states = (
            speculator_reserve.num_draft_slots
            if speculator_reserve is not None else 0
        )
        draft_slots_per_request = (
            speculator_reserve.draft_slots_per_request
            if speculator_reserve is not None else 0
        )
        num_dummy_states = int(config.model_config.is_hybrid)
        extra_state_slots = num_draft_states + num_dummy_states
        extra_state_bytes = extra_state_slots * cache_per_state

        num_pages = config.num_page_override
        if num_pages is None:
            if config.model_config.is_hybrid:
                min_pages_bytes = _MIN_KV_PAGES * (
                    cache_per_page + cache_per_state
                )
                available_states = (
                    available_memory - min_pages_bytes
                ) // cache_per_state
                assert extra_state_slots <= available_states, (
                    f"Not enough memory for {extra_state_slots} reserved linear states: "
                    f"only {available_states} fit alongside {_MIN_KV_PAGES} minimum "
                    "KV pages. Raise --memory-ratio / Reduce --max-running-req, "
                    "or pin --num-pages."
                )
                num_pages = max(
                    _MIN_KV_PAGES,
                    (available_memory - extra_state_bytes)
                    // (cache_per_page + cache_per_state),
                )
            else:
                num_pages = available_memory // cache_per_page

        assert num_pages > 1, (
            "Not enough memory for cache allocation. Raise --memory-ratio or "
            "reduce --num-pages."
        )
        num_tokens = num_pages * config.page_size
        max_seq_len = min(config.max_seq_len, num_tokens)
        return StandardCachePlan(
            num_pages=num_pages,
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            aligned_max_seq_len=align_ceil(max_seq_len, 32),
            num_states=num_pages if config.model_config.is_hybrid else 0,
            num_draft_states=num_draft_states,
            draft_slots_per_request=draft_slots_per_request,
            cache_per_page=cache_per_page,
            cache_per_state=cache_per_state,
        )

    def allocate(
        self,
        config: EngineConfig,
        plan  : CachePlan,
        device: torch.device,
        dtype : torch.dtype,
    ) -> CacheResources:
        assert isinstance(plan, StandardCachePlan)
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

        linear_state = None
        state_table = None
        work_state_table = None
        if config.model_config.is_hybrid:
            linear_state = make_linear_state_pool(
                model_config=config.model_config,
                num_slots=plan.num_states + plan.num_draft_states + 1,
                device=device,
                dtype=dtype,
            )
            state_table = torch.full(
                (config.max_running_req + 1, plan.aligned_max_seq_len),
                -1,
                dtype=torch.int32,
                device=device,
            )
            reserve_cols = plan.draft_slots_per_request
            if reserve_cols != 0:
                work_state_table = torch.arange(
                    plan.num_states,
                    plan.num_states + plan.num_draft_states,
                    dtype=torch.int32,
                    device=device,
                ).view(config.max_running_req, reserve_cols)
            state_table[-1].fill_(plan.num_states + plan.num_draft_states)

        kv_cache_bytes = plan.num_pages * plan.cache_per_page
        logger.info(
            f"Allocating {plan.num_tokens} tokens for KV cache, K + V = "
            f"{kv_cache_bytes / (1 << 30):.2f} GiB"
        )
        if config.model_config.is_hybrid:
            state_slots = plan.num_states + plan.num_draft_states + 1
            state_cache_bytes = state_slots * plan.cache_per_state
            logger.info(
                f"Allocating {state_slots} "
                "linear state slots = "
                f"{state_cache_bytes / (1 << 30):.2f} GiB"
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
    ) -> StandardCacheManager:
        assert isinstance(plan, StandardCachePlan)
        prefix_cache_type = self.prefix_cache_type
        if config.model_config.is_hybrid:
            prefix_cache_type = "hybrid_" + prefix_cache_type
        return StandardCacheManager(
            num_pages=plan.num_pages,
            page_size=config.page_size,
            resources=resources,
            prefix_cache_type=prefix_cache_type,
            num_states=plan.num_states,
            num_draft_states=plan.num_draft_states,
        )


SUPPORTED_CACHE_FACTORIES.register("naive")(StandardCacheFactory("naive"))
SUPPORTED_CACHE_FACTORIES.register("radix")(StandardCacheFactory("radix"))
