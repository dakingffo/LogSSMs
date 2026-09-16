from __future__ import annotations

import torch

from logssms.core import SamplingParams

from ...base import DraftState


class MTPState(DraftState):
    def __init__(
        self,
        table_idx       : int,
        sampling_params : SamplingParams,
        window_positions: list[int],
        window_tokens   : list[int],
        window_hidden   : torch.Tensor,
        window_size     : int,
    ) -> None:
        super().__init__()
        self.table_idx        = table_idx
        self.sampling_params  = sampling_params
        self.window_positions = list(window_positions)
        self.window_tokens    = list(window_tokens)
        self.window_size      = window_size
        self.pending_positions = list(window_positions)
        self.pending_tokens    = list(window_tokens)
        self.pending_hidden    = window_hidden
        self.cache_initialized = False
        self.n_drafts         = 0

    def update_window(self, positions: list[int], tokens: list[int], hidden: torch.Tensor) -> None:
        assert len(positions) == len(tokens) == hidden.shape[0]
        self.window_positions.extend(positions)
        self.window_tokens.extend(tokens)
        self.pending_positions.extend(positions)
        self.pending_tokens.extend(tokens)
        self.pending_hidden = (
            torch.cat([self.pending_hidden, hidden], dim=0)
            if self.pending_hidden is not None else hidden
        )
        if reserved_len := max(0, len(self.window_positions) - self.window_size):
            self.window_positions = self.window_positions[reserved_len:]
            self.window_tokens    = self.window_tokens[reserved_len:]

    def clear_pending(self) -> None:
        self.pending_positions = []
        self.pending_tokens    = []
        self.pending_hidden    = self.pending_hidden[:0]
