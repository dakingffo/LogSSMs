from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import TYPE_CHECKING

import torch

from logssms.core import (
    SamplingParams, 
    Request, 
    ChunkedRequest, 
    Batch, 
    Context 
)
from logssms.utils import init_logger, align_ceil

from .ar import ForwardInput, ForwardOutput

if TYPE_CHECKING:
    from logssms.cache import BaseCacheHandle, BaseCacheManager
    from logssms.message import UserMsg

    from .table import TableManager

logger = init_logger(__name__)


@dataclass
class PendingRequest:
    uid            : int
    input_ids      : torch.Tensor
    sampling_params: SamplingParams
    profile_tag    : str | None = None
    return_token_ids: bool = False
    chunked_req    : ChunkedRequest | None = None
    queued_at_ns   : int = field(default_factory=perf_counter_ns)
    admitted       : bool = False

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
    cache_manager: BaseCacheManager
    table_manager: TableManager
    rejection_reason: str | None = field(default=None, init=False)

    def _try_allocate_one(
        self,
        req: PendingRequest,
    ) -> tuple[BaseCacheHandle, int, int] | None:
        if self.table_manager.available_size == 0:
            self.rejection_reason = "no free request table slot"
            return None

        # TODO: consider host cache match case
        match = self.cache_manager.match_req(req)
        handle = match.cuda_handle
        cached_len = handle.cached_len
        remain_len = req.input_len - cached_len
        chunk_size = self.cache_manager.plan_prefill_chunk(
            remain_len, self.token_budget
        )
        if chunk_size == 0:
            self.rejection_reason = (
                f"zero prefill chunk: remain={remain_len}, "
                f"token_budget={self.token_budget}"
            )
            return None

        # TODO: better estimate policy
        estimated_len = align_ceil(remain_len + req.output_len, self.cache_manager.page_size)
        total_len = estimated_len + self.reserved_size
        if not self.cache_manager.prefill_acquire(
            total_len, handle, request_len=estimated_len
        ):
            self.rejection_reason = (
                f"cache admission rejected: request={estimated_len}, "
                f"reserved={self.reserved_size}, total={total_len}, "
                f"available={self.cache_manager.available_size}, "
                f"matched={cached_len}"
            )
            return None

        self.cache_manager.record_match_profile(
            req, match, input_len=req.input_len
        )

        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
        self.cache_manager.init_with_handle(table_idx, handle)

        return handle, table_idx, chunk_size

    def _add_one_req(
        self,
        pending_req : PendingRequest,
        cache_handle: BaseCacheHandle,
        table_idx   : int,
        cached_len  : int,
        chunk_size  : int,
    ) -> Request | ChunkedRequest:
        if not pending_req.admitted:
            self.cache_manager.record_prefill_wait(
                (perf_counter_ns() - pending_req.queued_at_ns) / 1e6
            )
            pending_req.admitted = True
        remain_len = pending_req.input_len - cached_len
        is_chunked = chunk_size < remain_len
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
            profile_tag=pending_req.profile_tag,
            return_token_ids=pending_req.return_token_ids,
            max_device_len=len(pending_req.input_ids) + pending_req.output_len,
        )

    def try_add_one(self, pending_req: PendingRequest) -> Request | ChunkedRequest | None:
        if self.token_budget <= 0:
            self.rejection_reason = "prefill token budget exhausted"
            return None

        if chunked_req := pending_req.chunked_req:
            remain_len = pending_req.input_len - chunked_req.cached_len
            chunk_size = self.cache_manager.plan_prefill_chunk(
                remain_len, self.token_budget
            )
            if chunk_size == 0:
                return None
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
                chunk_size=chunk_size,
            )

        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx, chunk_size = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
                chunk_size=chunk_size,
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
        self.pending_list.append(PendingRequest(
            req.uid,
            req.input_ids,
            req.sampling_params,
            profile_tag=req.profile_tag,
            return_token_ids=req.return_token_ids,
        ))

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
