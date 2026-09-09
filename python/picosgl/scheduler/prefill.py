from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from picosgl.core import (
    SamplingParams, 
    Request, 
    ChunkedRequest, 
    Batch, 
    Context 
)
from picosgl.utils import init_logger, align_ceil, align_down

from .ar import ForwardInput, ForwardOutput

if TYPE_CHECKING:
    from picosgl.cache import BaseCacheHandle
    from picosgl.message import UserMsg

    from .cache import CacheManager
    from .table import TableManager

logger = init_logger(__name__)


@dataclass
class PendingRequest:
    uid            : int
    input_ids      : torch.Tensor
    sampling_params: SamplingParams
    chunked_req    : ChunkedRequest | None = None

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens

    
@dataclass
class PrefillAdder:
    token_budget : int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager

    def _try_allocate_one(self, req: PendingRequest) -> tuple[BaseCacheHandle, int] | None:
        if self.table_manager.available_size == 0:
            return None

        # TODO: consider host cache match case
        handle = self.cache_manager.match_req(req).cuda_handle
        cached_len = handle.cached_len
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len
        estimated_len = align_ceil(extend_len + req.output_len, self.cache_manager.page_size)

        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        self.cache_manager.lock(handle)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)

        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            matched = handle.get_matched_indices()
            # borrow the matched pages' linear-state slots (per page) instead of allocating
            if (state_table := self.cache_manager.state_table) is not None:
                matched_indices, matched_state = matched
                ps = self.cache_manager.page_size
                state_table[table_idx, :cached_len] = matched_state.repeat_interleave(ps)
            else:
                matched_indices = matched
            page_entry.copy_(matched_indices)

        return handle, table_idx

    def _add_one_req(
        self,
        pending_req : PendingRequest,
        cache_handle: BaseCacheHandle,
        table_idx   : int,
        cached_len  : int,
    ) -> Request | ChunkedRequest:
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len
        if is_chunked and getattr(self.cache_manager, "state_table", None) is not None:
            # GDN exposes recurrent checkpoints at 64-token chunk boundaries.
            # Keep every intermediate ChunkedRequest aligned so its state can
            # become the exact initial state of the next prefill batch.
            chunk_size = align_down(chunk_size, 64)
            assert chunk_size > 0
        CLS = ChunkedRequest if is_chunked else Request
        self.token_budget -= chunk_size
        self.reserved_size += align_ceil(remain_len + pending_req.output_len, self.cache_manager.page_size)
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        device_ids = self.table_manager.token_pool[table_idx, cached_len: cached_len + chunk_size]
        device_ids.copy_(
            pending_req.input_ids[cached_len: cached_len + chunk_size].pin_memory(),
            non_blocking=True
        )
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
            max_device_len=len(pending_req.input_ids) + pending_req.output_len,
        )

    def try_add_one(self, pending_req: PendingRequest) -> Request | ChunkedRequest | None:
        if self.token_budget <= 0:
            return None

        if chunked_req := pending_req.chunked_req:
            remain_len = pending_req.input_len - chunked_req.cached_len
            if (
                getattr(self.cache_manager, "state_table", None) is not None
                and remain_len > self.token_budget
                and self.token_budget < 64
            ):
                return None
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            remain_len = pending_req.input_len - cache_handle.cached_len
            if (
                getattr(self.cache_manager, "state_table", None) is not None
                and remain_len > self.token_budget
                and self.token_budget < 64
            ):
                self.table_manager.free(table_idx)
                self.cache_manager.unlock(cache_handle)
                return None
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None


class PrefillManager:
    def __init__(
        self,
        token_pool : torch.Tensor,
        page_size  : int
    ):
        self.token_pool = token_pool
        self.pending_list: list[PendingRequest] = []
        self.inflight_reqs: dict[int, Request] = dict()
        self.page_size = page_size

    def add_one_req(self, req: UserMsg) -> None:
        self.pending_list.append(PendingRequest(req.uid, req.input_ids, req.sampling_params))

    @property
    def need_tokens(self) -> int:
        return sum(
            align_ceil(req.remain_len, self.page_size)
            for req in self.inflight_reqs.values()
        )
        
    def schedule_next_batch(self, adder: PrefillAdder) -> Batch | None:
        self.inflight_reqs.clear()
        if len(self.pending_list) == 0:
            return None

        reqs: list[Request] = []
        chunked_list: list[PendingRequest] = []
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                pending_req.chunked_req = None
                if isinstance(req, ChunkedRequest):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
                self.inflight_reqs[req.uid] = req
            else:
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> tuple[Request | None, bool]:
        # check ChunkedRequest
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                inflight_req = self.inflight_reqs.pop(uid, None)
                if inflight_req is not None:
                    inflight_req.aborted = True
                return req.chunked_req, inflight_req is not None
            
        # check Request
        inflight_req = self.inflight_reqs.pop(uid, None)
        if inflight_req is not None:
            inflight_req.aborted = True
        return inflight_req, inflight_req is not None

    def advance_for_next_schedule(
        self,
        ctx          : Context,
        forward_input: ForwardInput,
        output       : ForwardOutput,
    ) -> None:
        batch, _, _, output_mapping = forward_input
        if not batch.is_verify:
            self.token_pool[output_mapping] = output.next_tokens_gpu
        if batch.is_prefill:
            for req in batch.reqs:
                req.complete_to_device_len()

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
