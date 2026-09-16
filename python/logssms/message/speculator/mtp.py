from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from logssms.core import SamplingParams

from .base import (
    SpeculatorHandshakeMsg,
    SpeculatorHandshakeAckMsg,
    SpeculatorInitMsg,
)

if TYPE_CHECKING:
    import torch

    from logssms.speculator import BaseSpeculatorConfig, HiddenCaptorBase


@dataclass
class MTPHandshakeMsg(SpeculatorHandshakeMsg):
    window_size    : int


@dataclass
class MTPHandshakeAckMsg(SpeculatorHandshakeAckMsg):
    pass


@dataclass
class MTPInitMsg(SpeculatorInitMsg):
    uid            : int
    table_idx      : int
    carry_positions: list[int]
    carry_tokens   : list[int]
    sampling_params: SamplingParams


def make_mtp_handshake_message(
    speculator_config: BaseSpeculatorConfig,
    connection_id    : bytes,
    max_hidden_rows  : int,
    hidden_size      : int,
    max_prob_rows    : int,
    vocab_size       : int,
) -> MTPHandshakeMsg:
    from logssms.speculator.drafters.mtp import MTPSpeculatorConfig

    assert isinstance(speculator_config, MTPSpeculatorConfig)
    return MTPHandshakeMsg(
        connection_id=connection_id,
        max_hidden_rows=max_hidden_rows,
        hidden_size=hidden_size,
        max_prob_rows=max_prob_rows,
        vocab_size=vocab_size,
        window_size=speculator_config.window_size,
    )


def make_mtp_init_message(
    speculator_config: BaseSpeculatorConfig,
    uid              : int,
    table_idx        : int,
    end_position     : int,
    token_ids        : torch.Tensor,
    hidden_captor    : HiddenCaptorBase,
    sampling_params  : SamplingParams,
) -> tuple[MTPInitMsg, torch.Tensor]:
    from logssms.speculator.drafters.mtp import (
        MTPHiddenCaptor,
        MTPSpeculatorConfig,
    )

    assert isinstance(speculator_config, MTPSpeculatorConfig)
    assert isinstance(hidden_captor, MTPHiddenCaptor)
    full_hidden = hidden_captor.full_hidden
    window_len = min(speculator_config.window_size, full_hidden.shape[0])
    positions = list(range(end_position + 1 - window_len, end_position + 1))
    return (
        MTPInitMsg(
            input_rows=window_len,
            uid=uid,
            table_idx=table_idx,
            carry_positions=positions,
            carry_tokens=token_ids[positions].tolist(),
            sampling_params=sampling_params,
        ),
        full_hidden[-window_len:].contiguous(),
    )
