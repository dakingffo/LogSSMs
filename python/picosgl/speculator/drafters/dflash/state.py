from __future__ import annotations

import torch

from picosgl.core import SamplingParams

from ...base import DraftState


class DFlashState(DraftState):
    def __init__(
        self,
        table_idx        : int,
        sampling_params  : SamplingParams,
        context_positions: list[int],
        context_hidden   : torch.Tensor,
        anchor_position  : int,
        anchor_token     : int,
    ) -> None:
        super().__init__()
        self.table_idx = table_idx
        self.sampling_params = sampling_params
        self.pending_positions = list(context_positions)
        self.pending_hidden = context_hidden
        self.anchor_position = anchor_position
        self.anchor_token = anchor_token
        self.cache_initialized = False
        self.n_drafts = 0

    def update(
        self,
        output_positions: list[int],
        output_tokens   : list[int],
        hidden          : torch.Tensor,
    ) -> None:
        assert len(output_positions) == len(output_tokens) == hidden.shape[0]
        assert output_positions
        # Target logits at position p predict the output token at p+1. Therefore the
        # captured verify rows become DFlash context at output_position - 1, while the
        # final output token becomes the next noisy block's anchor.
        self.pending_positions.extend(position - 1 for position in output_positions)
        self.pending_hidden = torch.cat([self.pending_hidden, hidden], dim=0)
        self.anchor_position = output_positions[-1]
        self.anchor_token = output_tokens[-1]

    def clear_pending(self) -> None:
        self.pending_positions = []
        self.pending_hidden = self.pending_hidden[:0]


__all__ = ["DFlashState"]
