from __future__ import annotations

import torch

from picosgl.message import (
    SpeculatorHandshakeAckMsg,
    SpeculatorHandshakeMsg,
    SpeculatorInitMsg,
    SpeculatorReply,
    SpeculatorReplyMsg,
    SpeculatorStepMsg,
)
from picosgl.message.speculator.dflash import (
    DFlashHandshakeAckMsg,
    DFlashHandshakeMsg,
    DFlashInitMsg,
)

from ...base import DraftManagerBase
from .engine import DFlashEngine
from .state import DFlashState


class DFlashDraftManager(DraftManagerBase):
    def __init__(self, engine: DFlashEngine) -> None:
        super().__init__(engine)
        self.states: dict[int, DFlashState] = {}

    def handshake(
        self, msg: SpeculatorHandshakeMsg
    ) -> SpeculatorHandshakeAckMsg:
        assert isinstance(msg, DFlashHandshakeMsg)
        assert msg.window_size == self.engine.window_size
        assert msg.block_size == self.engine.block_size
        assert msg.hidden_size == (
            self.engine.drafter.hidden_size * len(self.engine.drafter.target_layer_ids)
        )
        return DFlashHandshakeAckMsg()

    def init(self, msg: SpeculatorInitMsg, tensor: torch.Tensor) -> None:
        assert isinstance(msg, DFlashInitMsg)
        assert tensor.shape[0] == len(msg.context_positions)
        self.states[msg.uid] = DFlashState(
            table_idx=msg.table_idx,
            sampling_params=msg.sampling_params,
            context_positions=msg.context_positions,
            context_hidden=tensor,
            anchor_position=msg.anchor_position,
            anchor_token=msg.anchor_token,
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
                state.update(
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


__all__ = ["DFlashDraftManager"]
