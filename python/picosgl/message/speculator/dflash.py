from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from picosgl.core import SamplingParams

from .base import (
    SpeculatorHandshakeAckMsg,
    SpeculatorHandshakeMsg,
    SpeculatorInitMsg,
)

if TYPE_CHECKING:
    import torch

    from picosgl.speculator import BaseSpeculatorConfig, HiddenCaptorBase


@dataclass
class DFlashHandshakeMsg(SpeculatorHandshakeMsg):
    window_size: int
    block_size : int


@dataclass
class DFlashHandshakeAckMsg(SpeculatorHandshakeAckMsg):
    pass


@dataclass
class DFlashInitMsg(SpeculatorInitMsg):
    uid              : int
    table_idx        : int
    context_positions: list[int]
    anchor_position  : int
    anchor_token     : int
    sampling_params  : SamplingParams


def make_dflash_handshake_message(
    speculator_config: BaseSpeculatorConfig,
    connection_id    : bytes,
    max_hidden_rows  : int,
    hidden_size      : int,
    max_prob_rows    : int,
    vocab_size       : int,
) -> DFlashHandshakeMsg:
    from picosgl.speculator.drafters.dflash import DFlashSpeculatorConfig

    assert isinstance(speculator_config, DFlashSpeculatorConfig)
    return DFlashHandshakeMsg(
        connection_id=connection_id,
        max_hidden_rows=max_hidden_rows,
        hidden_size=hidden_size,
        max_prob_rows=max_prob_rows,
        vocab_size=vocab_size,
        window_size=speculator_config.window_size,
        block_size=speculator_config.block_size,
    )


def make_dflash_init_message(
    speculator_config: BaseSpeculatorConfig,
    uid              : int,
    table_idx        : int,
    end_position     : int,
    token_ids        : torch.Tensor,
    hidden_captor    : HiddenCaptorBase,
    sampling_params  : SamplingParams,
) -> tuple[DFlashInitMsg, torch.Tensor]:
    from picosgl.speculator.drafters.dflash import (
        DFlashHiddenCaptor,
        DFlashSpeculatorConfig,
    )

    assert isinstance(speculator_config, DFlashSpeculatorConfig)
    assert isinstance(hidden_captor, DFlashHiddenCaptor)
    full_hidden = hidden_captor.full_hidden
    context_len = min(speculator_config.window_size, full_hidden.shape[0])
    context_end = end_position
    context_positions = list(range(context_end - context_len, context_end))
    return (
        DFlashInitMsg(
            input_rows=context_len,
            uid=uid,
            table_idx=table_idx,
            context_positions=context_positions,
            anchor_position=end_position,
            anchor_token=int(token_ids[end_position].item()),
            sampling_params=sampling_params,
        ),
        full_hidden[-context_len:].contiguous(),
    )


__all__ = [
    "DFlashHandshakeAckMsg",
    "DFlashHandshakeMsg",
    "DFlashInitMsg",
    "make_dflash_handshake_message",
    "make_dflash_init_message",
]
