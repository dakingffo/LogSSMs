from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from logssms.core import Request, ChunkedRequest, Batch, Context
from logssms.engine import ForwardInput, ForwardOutput
from logssms.message import DetokenizeMsg
from logssms.utils import align_ceil

if TYPE_CHECKING:
    from logssms.speculator import HiddenCaptorBase

    from logssms.cache import BaseCacheManager
    from .config import SchedulerConfig
    from .table import TableManager


class ARManagerBase:
    def __init__(
        self,
        config       : SchedulerConfig,
        device       : torch.device,
        cache_manager: BaseCacheManager,
        table_manager: TableManager,
        eos_token_id : int,
    ) -> None:
        self.config = config
        self.cache_manager = cache_manager
        self.token_pool = table_manager.token_pool
        self.eos_token_id = eos_token_id
        self.device = device
        self.token_budget = config.decode_batch_budget
        self.running_reqs: dict[int, Request] = {}
        self.inflight_uids: list[set[int], set[int]] = [set(), set()] # (last, next)
        self.finished_reqs: set[Request] = set()

    @property
    def page_size(self) -> int:
        return self.config.page_size

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0

    @property
    def need_tokens(self) -> int:
        return sum(
            align_ceil(self._reserve_remain_len(req), self.page_size)
            for req in self.running_reqs.values()
        )

    def _reserve_remain_len(self, req: Request) -> int:
        return req.remain_len

    def abort_req(self, uid: int) -> tuple[Request | None, bool]:
        inflight: bool = uid in self.inflight_uids[1]
        self.inflight_uids[1].discard(uid)
        req = self.running_reqs.pop(uid, None)
        if req is None:
            return None, False
        else:
            req.aborted = True
            return req, inflight

    def on_prefill_done(
        self,
        req       : Request,
        req_captor: HiddenCaptorBase,
    ) -> None:
        """Prefill -> AR handoff hook. Decode does not need speculator features."""
        return None

    def advance_for_overlap(self, batch: Batch) -> None:
        """Decode -> decode overlap hook. Verify can not advance for overlap."""
        return None

    def on_prefill_scheduled(self) -> None:
        """Close the AR overlap window when the next submitted batch is prefill."""
        self.inflight_uids[0] = self.inflight_uids[1]
        self.inflight_uids[1] = set()

    def process(
        self,
        ctx          : Context,
        forward_input: ForwardInput,
        output       : ForwardOutput
    ) -> tuple[list[DetokenizeMsg], list[Request]]:
        batch = forward_input.batch
        next_tokens_cpu = output.next_tokens_cpu
        reply: list[DetokenizeMsg] = []
        new_finished: set[Request] = set()

        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedRequest) or req.aborted:
                    continue
                if not batch.is_prefill and req.uid not in self.running_reqs:
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                if batch.is_prefill:
                    req.complete_n(1)

                finished = req.max_device_len - (req.device_len - (req.uid in self.inflight_uids[1])) <= 0
                                                # real device len = devel_len - 1 if inflight
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                if finished and req not in self.finished_reqs:
                    if req not in self.inflight_uids[1]:
                        self._finish_req(req)
                    else:
                        self.abort_req(req.uid)
                    new_finished.add(req)
                elif batch.is_prefill:
                    if req.can_decode:
                        self.running_reqs[req.uid] = req
                        if batch.hidden_captor is not None:
                            mapping = forward_input.input_tuple[0]
                            self.on_prefill_done(
                                req,
                                batch.hidden_captor.select(mapping == req.table_idx)
                            )
                    self.cache_manager.cache_req(req, finished=False)

        self.inflight_uids[0] = set()
        self.finished_reqs = new_finished
        return reply, new_finished

    def _finish_req(self, req: Request) -> None:
        self.running_reqs.pop(req.uid, None)
