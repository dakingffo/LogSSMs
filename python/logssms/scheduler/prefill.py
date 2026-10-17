from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import nullcontext
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
    from .profile import SchedulerProfiler

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
    prefix_cached_len: int = 0

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
    profiler     : SchedulerProfiler | None = None
    rejection_reason: str | None = field(default=None, init=False)

    def _profile_step(self, name: str):
        if self.profiler is None:
            return nullcontext()
        return self.profiler.record_schedule_step(name)

    def _try_allocate_one(
        self,
        req: PendingRequest,
    ) -> tuple[BaseCacheHandle, int, int] | None:
        if self.table_manager.available_size == 0:
            self.rejection_reason = "no free request table slot"
            return None


        with self._profile_step("match_req"):
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


        estimated_len = self.cache_manager.reservation_size(
            remain_len + req.output_len, handle=handle,
        )
        total_len = estimated_len + self.reserved_size
        with self._profile_step("prefill_acquire"):
            admitted = self.cache_manager.prefill_acquire(
                total_len, handle, request_len=estimated_len
            )
        if not admitted:
            with self._profile_step("rejection_diagnostics"):
                self.rejection_reason = (
                    f"cache admission rejected: request={estimated_len}, "
                    f"reserved={self.reserved_size}, total={total_len}, "
                    f"available={self.cache_manager.admission_capacity}, "
                    f"matched={cached_len}"
                )
            return None

        with self._profile_step("record_match"):
            self.cache_manager.record_match(
                req, match, input_len=req.input_len
            )

        with self._profile_step("table_allocate"):
            table_idx = self.table_manager.allocate()
        if cached_len > 0:
            with self._profile_step("cached_token_copy"):
                device_ids = self.table_manager.token_pool[table_idx][:cached_len]
                device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
        with self._profile_step("init_with_handle"):
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
            with self._profile_step("record_prefill_wait"):
                self.cache_manager.record_prefill_wait(
                    (perf_counter_ns() - pending_req.queued_at_ns) / 1e6
                )
            pending_req.admitted = True
        remain_len = pending_req.input_len - cached_len
        is_chunked = chunk_size < remain_len
        CLS = ChunkedRequest if is_chunked else Request
        self.token_budget -= chunk_size
        self.reserved_size += self.cache_manager.reservation_size(
            remain_len + pending_req.output_len, table_idx=table_idx,
        )

        with self._profile_step("input_token_copy"):
            device_ids = self.table_manager.token_pool[table_idx, cached_len: cached_len + chunk_size]
            device_ids.copy_(
                pending_req.input_ids[cached_len: cached_len + chunk_size].pin_memory(),
                non_blocking=True
            )
        with self._profile_step("request_build"):
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
                prefix_cached_len=pending_req.prefix_cached_len,
                max_device_len=len(pending_req.input_ids) + pending_req.output_len,
            )

    def try_add_one(self, pending_req: PendingRequest) -> Request | ChunkedRequest | None:
        if self.profiler is None:
            return self._try_add_one(pending_req)
        with self.profiler.record_schedule_request(pending_req) as sample:
            sample["token_budget_before"] = self.token_budget
            sample["reserved_size_before"] = self.reserved_size
            self.rejection_reason = None
            req = self._try_add_one(pending_req)
            sample["admitted"] = req is not None
            sample["rejection_reason"] = self.rejection_reason
            sample["cached_len"] = None if req is None else req.cached_len
            sample["chunk_size"] = None if req is None else req.extend_len
            sample["token_budget_after"] = self.token_budget
            sample["reserved_size_after"] = self.reserved_size
            return req

    def _try_add_one(self, pending_req: PendingRequest) -> Request | ChunkedRequest | None:
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
        page_size  : int,
        cache_manager: BaseCacheManager | None = None,
    ):
        self.token_pool = token_pool
        self.pending_list: list[PendingRequest] = []
        self.inflight_reqs: dict[int, Request] = dict()
        self.page_size = page_size
        self.cache_manager = cache_manager

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

    @property
    def need_capacity(self) -> int:
        if self.cache_manager is None:
            return self.need_tokens
        return sum(
            self.cache_manager.reservation_size(req.remain_len, table_idx=req.table_idx)
            for req in self.inflight_reqs.values()
        )
        
    def schedule_next_batch(
        self, adder: PrefillAdder, max_requests: int | None = None,
    ) -> Batch | None:
        self.inflight_reqs.clear()
        if len(self.pending_list) == 0:
            return None

        reqs: list[Request] = []
        chunked_list: list[PendingRequest] = []
        for pending_req in self.pending_list:
            if max_requests is not None and len(reqs) >= max_requests:
                break
            if req := adder.try_add_one(pending_req):
                pending_req.chunked_req = None
                if isinstance(req, ChunkedRequest):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
                self.inflight_reqs[req.uid] = req
            else:
                break
        if len(reqs) == 0:
            return None
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> tuple[Request | None, bool]:

        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                inflight_req = self.inflight_reqs.pop(uid, None)
                if inflight_req is not None:
                    inflight_req.aborted = True
                return req.chunked_req, inflight_req is not None
            

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
            tokens = output.next_tokens_gpu
            # The engine's output is aligned to real requests; write_mapping
            # excludes intermediate chunks, which have no sampled token.
            if any(isinstance(req, ChunkedRequest) for req in batch.reqs):
                sampled = [
                    i for i, req in enumerate(batch.reqs)
                    if not isinstance(req, ChunkedRequest)
                ]
                tokens = tokens[sampled]
            self.token_pool[output_mapping] = tokens
        for req in batch.prefill_reqs:
            req.complete_to_device_len()

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
