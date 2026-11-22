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
from logssms.utils import init_logger

from .ar import ForwardInput, ForwardOutput

if TYPE_CHECKING:
    from logssms.cache import BaseCacheHandle, BaseCacheManager
    from logssms.cache.base import PrefillAdmission
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
    cache_manager: BaseCacheManager
    table_manager: TableManager
    admission    : PrefillAdmission | None = None
    profiler     : SchedulerProfiler | None = None
    mixed_token_budget: int | None = None
    mixed_partitions: tuple[tuple[int, ...], ...] = ()
    _mixed_lengths: list[int] = field(default_factory=list, init=False)
    use_mixed: bool = field(default=False, init=False)
    _budget_selected: bool = field(default=False, init=False)
    rejection_reason: str | None = field(default=None, init=False)

    def _select_budget(self, remain_len: int) -> None:
        if self._budget_selected:
            return
        self._budget_selected = True
        self.use_mixed = (
            self.mixed_token_budget is not None
            and remain_len <= self.mixed_token_budget
        )
        if self.use_mixed:
            self.token_budget = min(self.token_budget, self.mixed_token_budget)

    def _profile_step(self, name: str):
        if self.profiler is None:
            return nullcontext()
        return self.profiler.record_schedule_step(name)

    def _chunk_budget(self, remain_len: int) -> int:
        if not self.use_mixed or not self.mixed_partitions:
            return self.token_budget
        index = len(self._mixed_lengths)
        candidates = [p[index] for p in self.mixed_partitions if len(p) > index
                      and all(n <= cap for n, cap in zip(self._mixed_lengths, p))]
        if not candidates:
            return 0
        desired = min(remain_len, self.token_budget)
        complete = [cap for cap in candidates if cap >= desired]
        return min(self.token_budget, min(complete) if complete else max(candidates))

    def _remaining_prefill_budget(self, remain_len: int) -> int:
        self._select_budget(remain_len)
        return self._chunk_budget(remain_len)

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
        if self.use_mixed:
            self._mixed_lengths.append(chunk_size)

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
            self.rejection_reason = None
            req = self._try_add_one(pending_req, profile=sample)
            sample["admitted"] = req is not None
            sample["rejection_reason"] = self.rejection_reason
            sample["cached_len"] = None if req is None else req.cached_len
            sample["chunk_size"] = None if req is None else req.extend_len
            sample["token_budget_after"] = self.token_budget
            return req

    def _try_add_one(
        self, pending_req: PendingRequest, *, profile: dict | None = None,
    ) -> Request | ChunkedRequest | None:
        if self.token_budget <= 0:
            self.rejection_reason = "prefill token budget exhausted"
            return None
        if self._chunk_budget(1) == 0:
            self.rejection_reason = "mixed partition capacity exhausted"
            return None

        if pending_req.chunked_req is None and self.table_manager.available_size == 0:
            self.rejection_reason = "no free request table slot"
            return None

        assert self.admission is not None
        result = self.cache_manager.admit_prefill(
            pending_req, self.admission, self._remaining_prefill_budget,
            lambda: self.table_manager.allocate(), profile=profile,
        )
        if isinstance(result, str):
            self.rejection_reason = result
            return None
        if pending_req.chunked_req is None and result.cached_len:
            with self._profile_step("cached_token_copy"):
                device_ids = self.table_manager.token_pool[result.table_idx][:result.cached_len]
                device_ids.copy_(pending_req.input_ids[:result.cached_len].pin_memory(), non_blocking=True)
        return self._add_one_req(pending_req, *result)


class PrefillManager:
    def __init__(self, token_pool: torch.Tensor):
        self.token_pool = token_pool
        self.pending_list: list[PendingRequest] = []
        self.inflight_reqs: dict[int, Request] = {}

    def add_one_req(self, req: UserMsg) -> None:
        self.pending_list.append(PendingRequest(
            req.uid,
            req.input_ids,
            req.sampling_params,
            profile_tag=req.profile_tag,
            return_token_ids=req.return_token_ids,
        ))

    def schedule_next_batch(
        self, adder: PrefillAdder, max_requests: int | None = None,
        *, mixed_max_requests: int | None = None,
    ) -> Batch | None:
        self.inflight_reqs.clear()
        if len(self.pending_list) == 0:
            return None

        reqs: list[Request] = []
        chunked_list: list[PendingRequest] = []
        for pending_req in self.pending_list:
            request_limit = (
                mixed_max_requests
                if mixed_max_requests is not None and adder.use_mixed
                else max_requests
            )
            if request_limit is not None and len(reqs) >= request_limit:
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
