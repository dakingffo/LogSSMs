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
from logssms.message.speculator.eagle3 import (
    Eagle3HandshakeAckMsg,
    Eagle3HandshakeMsg,
    Eagle3InitMsg,
)

from ...base import DraftManagerBase
from .engine import Eagle3Engine
from .state import Eagle3State


class Eagle3DraftManager(DraftManagerBase):
    def __init__(self, engine: Eagle3Engine) -> None:
        super().__init__(engine)
        self.states: dict[int, Eagle3State] = {}
        self.window_size = 0

    def handshake(
        self, msg: SpeculatorHandshakeMsg
    ) -> SpeculatorHandshakeAckMsg:
        assert isinstance(msg, Eagle3HandshakeMsg)
        assert msg.window_size == self.engine.pool.window_size
        assert msg.hidden_size == self.engine.drafter.hidden_size * 3
        self.window_size = msg.window_size
        return Eagle3HandshakeAckMsg()

    def init(self, msg: SpeculatorInitMsg, tensor: torch.Tensor) -> None:
        assert isinstance(msg, Eagle3InitMsg)
        assert tensor.shape[0] == len(msg.carry_positions)
        self.states[msg.uid] = Eagle3State(
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
        total_rows = sum(len(req.append_positions) for req in msg.reqs)
        assert total_rows == msg.input_rows
        assert (tensor is not None) == (total_rows > 0)
        offset = 0
        for req in msg.reqs:
            state = self.states[req.uid]
            state.n_drafts = req.n_drafts
            if num_rows := len(req.append_positions):
                assert tensor is not None
                state.update_window(
                    req.append_positions,
                    req.append_tokens,
                    tensor[offset: offset + num_rows],
                )
                offset += num_rows

        self.engine.draft([self.states[req.uid] for req in msg.reqs])
        sampling_reqs = [
            req for req in msg.reqs if req.sampling and req.n_drafts > 0
        ]
        probs = (
            torch.cat(
                [
                    self.states[req.uid].draft_probs[: req.n_drafts]
                    for req in sampling_reqs
                ],
                dim=0,
            )
            if sampling_reqs else None
        )
        return (
            SpeculatorReplyMsg(
                reqs=[
                    SpeculatorReply(
                        uid=req.uid,
                        draft_tokens=self.states[req.uid].draft_tokens,
                    )
                    for req in msg.reqs
                ]
            ),
            probs,
        )

    def remove(self, uid: int) -> None:
        self.states.pop(uid, None)


__all__ = ["Eagle3DraftManager"]
