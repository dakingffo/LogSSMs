from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from picosgl.core import SamplingParams
from picosgl.utils import Registry

from ..utils import deserialize_type

from .base import (
    BaseSpeculatorMsg,
    SpeculatorHandshakeAckMsg,
    SpeculatorHandshakeMsg,
    SpeculatorInitMsg,
    SpeculatorRemoveMsg,
    SpeculatorReply,
    SpeculatorReplyMsg,
    SpeculatorStepMsg,
    SpeculatorStepReq,
)
from . import mtp as _mtp
from . import eagle3 as _eagle3
from . import dflash as _dflash

if TYPE_CHECKING:
    from picosgl.speculator import BaseSpeculatorConfig, HiddenCaptorBase


HandshakeMessageCreator = Callable[
    ["BaseSpeculatorConfig", bytes, int, int, int, int],
    SpeculatorHandshakeMsg,
]
InitMessageCreator = Callable[
    [
        "BaseSpeculatorConfig",
        int,
        int,
        int,
        torch.Tensor,
        "HiddenCaptorBase",
        SamplingParams,
    ],
    tuple[SpeculatorInitMsg, torch.Tensor],
]

SUPPORTED_HANDSHAKE_MESSAGE = Registry[HandshakeMessageCreator](
    "Speculator Handshake Message"
)
SUPPORTED_INIT_MESSAGE = Registry[InitMessageCreator]("Speculator Init Message")
SUPPORTED_HANDSHAKE_MESSAGE.register("MTP")(_mtp.make_mtp_handshake_message)
SUPPORTED_INIT_MESSAGE.register("MTP")(_mtp.make_mtp_init_message)
SUPPORTED_HANDSHAKE_MESSAGE.register("EAGLE3")(_eagle3.make_eagle3_handshake_message)
SUPPORTED_INIT_MESSAGE.register("EAGLE3")(_eagle3.make_eagle3_init_message)
SUPPORTED_HANDSHAKE_MESSAGE.register("DFLASH")(_dflash.make_dflash_handshake_message)
SUPPORTED_INIT_MESSAGE.register("DFLASH")(_dflash.make_dflash_init_message)


def make_handshake_message(
    algorithm        : str,
    speculator_config: BaseSpeculatorConfig,
    connection_id    : bytes,
    max_hidden_rows  : int,
    hidden_size      : int,
    max_prob_rows    : int,
    vocab_size       : int,
) -> SpeculatorHandshakeMsg:
    return SUPPORTED_HANDSHAKE_MESSAGE[algorithm](
        speculator_config,
        connection_id,
        max_hidden_rows,
        hidden_size,
        max_prob_rows,
        vocab_size,
    )


def make_init_message(
    algorithm        : str,
    speculator_config: BaseSpeculatorConfig,
    uid              : int,
    table_idx        : int,
    end_position     : int,
    token_ids        : torch.Tensor,
    hidden_captor    : HiddenCaptorBase,
    sampling_params  : SamplingParams,
) -> tuple[SpeculatorInitMsg, torch.Tensor]:
    return SUPPORTED_INIT_MESSAGE[algorithm](
        speculator_config,
        uid,
        table_idx,
        end_position,
        token_ids,
        hidden_captor,
        sampling_params,
    )


_MESSAGE_TYPES = {
    cls.__name__: cls
    for cls in (
        SpeculatorHandshakeAckMsg,
        SpeculatorHandshakeMsg,
        SpeculatorInitMsg,
        SpeculatorRemoveMsg,
        SpeculatorReply,
        SpeculatorReplyMsg,
        SpeculatorStepMsg,
        SpeculatorStepReq,
        _mtp.MTPHandshakeAckMsg,
        _mtp.MTPHandshakeMsg,
        _mtp.MTPInitMsg,
        _eagle3.Eagle3HandshakeAckMsg,
        _eagle3.Eagle3HandshakeMsg,
        _eagle3.Eagle3InitMsg,
        _dflash.DFlashHandshakeAckMsg,
        _dflash.DFlashHandshakeMsg,
        _dflash.DFlashInitMsg,
        SamplingParams,
    )
}


def decode_speculator_message(json: dict) -> BaseSpeculatorMsg:
    return deserialize_type(_MESSAGE_TYPES, json)


__all__ = [
    "BaseSpeculatorMsg",
    "SpeculatorHandshakeMsg",
    "SpeculatorHandshakeAckMsg",
    "SpeculatorInitMsg",
    "SpeculatorStepMsg",
    "SpeculatorStepReq",
    "SpeculatorReply",
    "SpeculatorReplyMsg",
    "SpeculatorRemoveMsg",
    "make_handshake_message",
    "make_init_message",
]
