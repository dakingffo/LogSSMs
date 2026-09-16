from __future__ import annotations

import torch

from logssms.message import (
    SpeculatorHandshakeAckMsg,
    SpeculatorHandshakeMsg,
    SpeculatorInitMsg,
    SpeculatorReply,
    SpeculatorReplyMsg,
    SpeculatorStepMsg,
)
from logssms.message.speculator.mtp import (
    MTPHandshakeAckMsg,
    MTPHandshakeMsg,
    MTPInitMsg,
)

from ...base import DraftManagerBase
from .engine import MTPEngine
from .state import MTPState


class MTPDraftManager(DraftManagerBase):
    def __init__(self, engine: MTPEngine) -> None:
        super().__init__(engine)
        self.states: dict[int, MTPState] = {}
        self.window_size = 0

    def handshake(
        self, msg: SpeculatorHandshakeMsg
    ) -> SpeculatorHandshakeAckMsg:
        assert isinstance(msg, MTPHandshakeMsg)
        assert msg.window_size == self.engine.pool.window_size
        self.window_size = msg.window_size
        return MTPHandshakeAckMsg()

    def init(self, msg: SpeculatorInitMsg, tensor: torch.Tensor) -> None:
        assert isinstance(msg, MTPInitMsg)
        assert tensor.shape[0] == len(msg.carry_positions)
        self.states[msg.uid] = MTPState(
            table_idx=msg.table_idx,
            sampling_params=msg.sampling_params,
            window_positions=msg.carry_positions,
            window_tokens=msg.carry_tokens,
            window_hidden=tensor,
            window_size=self.window_size,
        )

    def step(
        self,
        msg   : SpeculatorStepMsg,
        tensor: torch.Tensor | None,
    ) -> tuple[SpeculatorReplyMsg, torch.Tensor | None]:
        total_rows = sum(len(r.append_positions) for r in msg.reqs)
        assert total_rows == msg.input_rows
        assert (tensor is not None) == (total_rows > 0)
        off = 0
        for req in msg.reqs:
            state = self.states[req.uid]
            state.n_drafts = req.n_drafts
            if n := len(req.append_positions):
                assert tensor is not None
                state.update_window(
                    req.append_positions,
                    req.append_tokens,
                    tensor[off: off + n],
                )
                off += n

        self.engine.draft([self.states[req.uid] for req in msg.reqs])

        probs = None
        sampling_reqs = [
            req for req in msg.reqs if req.sampling and req.n_drafts > 0
        ]
        if sampling_reqs:
            probs = torch.cat(
                [
                    self.states[req.uid].draft_probs[: req.n_drafts]
                    for req in sampling_reqs
                ],
                dim=0,
            )
        reply = SpeculatorReplyMsg(
            reqs=[
                SpeculatorReply(
                    uid=req.uid,
                    draft_tokens=self.states[req.uid].draft_tokens,
                )
                for req in msg.reqs
            ]
        )
        return reply, probs

    def remove(self, uid: int) -> None:
        self.states.pop(uid, None)
