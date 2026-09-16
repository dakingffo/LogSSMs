from types import SimpleNamespace

import torch

from logssms.core import SamplingParams
from logssms.message import (
    BaseSpeculatorMsg,
    SpeculatorReplyMsg,
    SpeculatorStepMsg,
    SpeculatorStepReq,
    make_handshake_message,
    make_init_message,
)
from logssms.message.speculator.mtp import (
    MTPHandshakeAckMsg,
    MTPHandshakeMsg,
    MTPInitMsg,
)
from logssms.speculator.drafters.mtp import (
    MTPDraftManager,
    MTPHiddenCaptor,
    MTPSpeculatorConfig,
)
from logssms.speculator.hidden_captor import HiddenCapturePoint


class FakeEngine:
    def __init__(self, window_size: int):
        self.pool = SimpleNamespace(window_size=window_size)

    def draft(self, states) -> None:
        for state in states:
            state.draft_tokens = list(range(state.n_drafts))


def test_mtp_message_factories_and_roundtrip():
    config = MTPSpeculatorConfig(num_draft_tokens=4, window_size=3)
    handshake = make_handshake_message("MTP", config, b"id", 8, 4, 16, 32)
    assert handshake == MTPHandshakeMsg(b"id", 8, 4, 16, 32, 3)

    full_hidden = torch.arange(20).reshape(5, 4)
    hidden_captor = MTPHiddenCaptor(config)
    hidden_captor.capture(HiddenCapturePoint.LM_HEAD_INPUT, None, full_hidden)
    msg, hidden = make_init_message(
        "MTP",
        config,
        7,
        2,
        4,
        torch.tensor([10, 11, 12, 13, 14]),
        hidden_captor,
        SamplingParams(),
    )
    assert isinstance(msg, MTPInitMsg)
    assert msg.carry_positions == [2, 3, 4]
    assert msg.carry_tokens == [12, 13, 14]
    torch.testing.assert_close(hidden, full_hidden[-3:])

    assert BaseSpeculatorMsg.decoder(handshake.encoder()) == handshake
    assert BaseSpeculatorMsg.decoder(msg.encoder()) == msg


def test_mtp_types_are_not_exported_from_generic_packages():
    import logssms.message as message
    import logssms.message.speculator as speculator_message
    import logssms.speculator as speculator

    message_types = ("MTPHandshakeMsg", "MTPHandshakeAckMsg", "MTPInitMsg")
    speculator_types = (
        "MTPDraftManager",
        "MTPEngine",
        "MTPHiddenCaptor",
        "MTPSpeculatorConfig",
        "MTPState",
    )
    assert not any(hasattr(message, name) for name in message_types)
    assert not any(hasattr(speculator_message, name) for name in message_types)
    assert not any(hasattr(speculator, name) for name in speculator_types)


def test_mtp_draft_manager_owns_protocol_details():
    init_hidden = torch.arange(12).reshape(3, 4)
    append_hidden = torch.arange(8).reshape(2, 4)
    manager = MTPDraftManager(FakeEngine(3))

    ack = manager.handshake(MTPHandshakeMsg(b"id", 8, 4, 16, 32, 3))
    assert isinstance(ack, MTPHandshakeAckMsg)
    manager.init(
        MTPInitMsg(3, 7, 2, [0, 1, 2], [10, 11, 12], SamplingParams()),
        init_hidden,
    )
    reply, probs = manager.step(
        SpeculatorStepMsg(
            [SpeculatorStepReq(7, 2, [3, 4], [13, 14], False)],
            input_rows=2,
            output_rows=0,
        ),
        append_hidden,
    )

    assert isinstance(reply, SpeculatorReplyMsg)
    assert reply.reqs[0].uid == 7
    assert reply.reqs[0].draft_tokens == [0, 1]
    assert probs is None
    assert manager.states[7].window_positions == [2, 3, 4]
    manager.remove(7)
    assert 7 not in manager.states
