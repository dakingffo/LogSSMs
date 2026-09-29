from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from logssms.engine import ForwardOutput
from logssms.core import Batch, Request, Context
from logssms.message import (
    DetokenizeMsg,
    SpeculatorStepMsg,
    SpeculatorStepReq,
    make_init_message,
)
from logssms.speculator import DraftState, HiddenCaptorBase
from logssms.utils import div_ceil

from .ar import ARManagerBase, ForwardInput

if TYPE_CHECKING:
    from logssms.speculator import SpeculatorClientBase

    from logssms.cache import BaseCacheManager
    from .config import SchedulerConfig
    from .table import TableManager


@dataclass
class VerifyState(DraftState):
    carry_tokens   : list[int] = field(default_factory=list, init=False)
    carry_positions: list[int] = field(default_factory=list, init=False)
    carry_hidden   : torch.Tensor | None = None

    def set_carry(self, tokens, positions, hidden):
        self.carry_tokens = tokens
        self.carry_positions = positions
        self.carry_hidden = hidden

    def get_carry(self):
        return (
            self.carry_tokens,
            self.carry_positions,
            self.carry_hidden
        )


class VerifyManager(ARManagerBase):
    def __init__(
        self,
        config        : SchedulerConfig,
        device        : torch.device,
        cache_manager : BaseCacheManager,
        table_manager : TableManager,
        eos_token_id  : int,
        client        : SpeculatorClientBase,
        vocab_size    : int,
    ) -> None:
        super().__init__(config, device, cache_manager, table_manager, eos_token_id)
        speculator_config = config.speculator_config
        assert speculator_config is not None
        speculative_algorithm = config.speculative_algorithm
        assert speculative_algorithm is not None

        self.speculative_algorithm = speculative_algorithm
        self.speculator_config = speculator_config
        self.num_spec_tokens = speculator_config.num_draft_tokens
        self.client = client
        self.vocab_size = vocab_size
        self._state_table: dict[int, VerifyState] = {}

    def _reserve_remain_len(self, req: Request) -> int:
        if req.uid in self.inflight_uids[1]:
            return req.max_device_len - (req.cached_len + 1)
        else:
            return req.remain_len

    def abort_req(self, uid: int) -> tuple[Request | None, bool]:
        inflight: bool = uid in self.inflight_uids[1]
        self.inflight_uids[1].discard(uid)
        req = self.running_reqs.pop(uid, None)
        if req is None:
            return None, False
        else:
            req.aborted = True
            self._state_table.pop(req.table_idx, None)
            self.client.remove(uid)
            return req, inflight

    def on_prefill_done(
        self,
        req       : Request,
        req_captor: HiddenCaptorBase,
    ) -> None:
        msg, hidden = make_init_message(
            self.speculative_algorithm,
            self.speculator_config,
            req.uid,
            req.table_idx,
            req.cached_len,
            self.token_pool[req.table_idx],
            req_captor,
            req.sampling_params,
        )
        self.client.init(msg, hidden)
        self.running_reqs[req.uid] = req
        self._state_table[req.table_idx] = VerifyState()

    def schedule_next_batch(self) -> Batch | None:
        self.inflight_uids[0] = self.inflight_uids[1]
        self.inflight_uids[1] = set()

        K = self.num_spec_tokens

        if (0 < len(self.inflight_uids[0]) * K < self.token_budget 
            and len(self.running_reqs) > len(self.inflight_uids[0])):
            # skip one iteration to try achieving a larger batch size
            return None

        scheduled_token = 0
        reqs = []
        step_reqs = []
        hidden_rows = []
        for uid, req in self.running_reqs.items():
            if scheduled_token >= self.token_budget:
                break
            elif uid not in self.inflight_uids[0]:
                self.inflight_uids[1].add(uid)
                reqs.append(req)
                C = req.cached_len
                remain = req.remain_len
                n_drafts = min(K, remain - 1) if remain > 0 else 0
                scheduled_token += n_drafts + 1
                req.extend_device_n(n_drafts)
                toks, pos, hids = self._state_table[req.table_idx].get_carry()
                step_reqs.append(
                    SpeculatorStepReq(
                        uid=uid,
                        n_drafts=n_drafts,
                        append_positions=pos,
                        append_tokens=toks,
                        sampling=not req.sampling_params.is_greedy,
                    )
                )
                if hids is not None:
                    hidden_rows.append(hids)

        if not reqs:
            return None

        appended_hidden = torch.cat(hidden_rows, dim=0) if hidden_rows else None
        input_rows = sum(len(sr.append_positions) for sr in step_reqs)
        output_rows = sum(sr.n_drafts for sr in step_reqs if sr.sampling)
        step_msg = SpeculatorStepMsg(
            reqs=step_reqs,
            input_rows=input_rows,
            output_rows=output_rows,
        )
        reply, probs = self.client.step(step_msg, appended_hidden)

        reply_by_uid = {r.uid: r for r in reply.reqs}
        probs_by_uid: dict[int, torch.Tensor] = {}
        if probs is not None:
            off = 0
            for sr in step_reqs:
                if sr.sampling:
                    probs_by_uid[sr.uid] = probs[off: off + sr.n_drafts]
                    off += sr.n_drafts

        batch = Batch(reqs=reqs, phase="verify")
        bs = len(reqs)
        draft_tokens = torch.full((bs, K), -1, dtype=torch.int32, device=self.device)
        draft_probs: torch.Tensor | None = None
        for i, req in enumerate(reqs):
            st = self._state_table[req.table_idx]
            st.draft_tokens = reply_by_uid[req.uid].draft_tokens
            if req.uid in probs_by_uid:
                st.draft_probs = probs_by_uid[req.uid]
                if draft_probs is None:
                    draft_probs = torch.zeros(
                        bs, K, self.vocab_size, dtype=torch.float32, device=self.device
                    )
                n = st.draft_probs.shape[0]
                draft_probs[i, :n] = st.draft_probs
            if st.draft_tokens:
                C = req.cached_len
                self.token_pool[req.table_idx, C + 1: C + 1 + len(st.draft_tokens)].copy_(
                    torch.tensor(st.draft_tokens, dtype=torch.int32, device=self.device)
                )
                draft_tokens[i, : len(st.draft_tokens)].copy_(
                    torch.tensor(st.draft_tokens, dtype=torch.int32, device=self.device)
                )
        batch.draft_tokens = draft_tokens
        batch.draft_probs = draft_probs

        return batch

    def process(
        self,
        ctx          : Context,
        forward_input: ForwardInput,
        output       : ForwardOutput,
    ) -> tuple[list[DetokenizeMsg], list[Request]]:
        batch = forward_input.batch
        if not batch.is_verify:
            return super().process(ctx, forward_input, output)  # prefill commit

        extend_tokens = output.next_tokens_gpu  # (bs, K+1) int32
        hidden_captor = batch.hidden_captor
        assert hidden_captor is not None
        reply: list[DetokenizeMsg] = []
        new_finished: set[Request] = set()
        offset = 0

        with self.cache_manager.lazy_free_region():
            for req, target_tokens in zip(batch.reqs, extend_tokens):
                verify_len = req.extend_len
                row_start = offset
                offset += verify_len
                if req.uid not in self.running_reqs:
                    continue

                st = self._state_table[req.table_idx]
                n_drafts = verify_len - 1
                C = req.cached_len

                sampled_token = []
                for j in range(n_drafts):
                    tok = int(target_tokens[j].item())
                    sampled_token.append(tok)
                    if tok != st.draft_tokens[j]:
                        break
                else:
                    sampled_token.append(
                        int(target_tokens[n_drafts].item())  # full-accept bonus
                    )
                num_sampled = len(sampled_token)

                req.complete_to_device_len(progress=num_sampled)
                # Promote the accepted snapshots by swapping reserve and live page slot ids.
                self.cache_manager.commit_speculative(req, C, num_sampled)
                self.token_pool[req.table_idx, req.cached_len] = sampled_token[-1]
                st.set_carry(
                    sampled_token,
                    list(range(C + 1, C + 1 + num_sampled)),
                    hidden_captor.get_carry_hidden(
                        slice(row_start, row_start + num_sampled)
                    ),
                )
                self.cache_manager.free_to_cached(req)

                stop, finish = num_sampled, not req.can_decode
                for idx, tok in enumerate(sampled_token):
                    if not req.sampling_params.ignore_eos and tok == self.eos_token_id:
                        stop, finish = idx + 1, True
                        break

                reply.append(
                    DetokenizeMsg(
                        uid=req.uid, next_token=sampled_token[:stop], finished=finish,
                        return_token_ids=req.return_token_ids,
                    )
                )
                req.append_host(torch.tensor(sampled_token, dtype=torch.int32))

                if finish and req not in self.finished_reqs:
                    self._finish_req(req)
                    new_finished.add(req)

        self.inflight_uids[0] = set()
        self.finished_reqs = new_finished
        return reply, new_finished

    def _finish_req(self, req: Request) -> None:
        self.running_reqs.pop(req.uid, None)
        self._state_table.pop(req.table_idx, None)
        self.client.remove(req.uid)
