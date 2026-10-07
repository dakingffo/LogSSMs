from __future__ import annotations

from dataclasses import dataclass

import torch


LOG_STATE_GRANULARITY = 32
LOG_STATE_REPLAY_WINDOW = 16


@dataclass(frozen=True)
class LogStateSlotLayout:
    page_size        : int
    num_linear_layers: int
    num_key_heads    : int
    num_value_heads  : int
    head_k_dim       : int
    head_v_dim       : int
    conv_dim         : int
    state_len        : int
    dtype            : torch.dtype
    granularity      : int = LOG_STATE_GRANULARITY

    def __post_init__(self) -> None:
        assert self.page_size > 0
        assert self.granularity > 0
        assert self.page_size % self.granularity == 0, (
            f"LogState page size {self.page_size} must be divisible by "
            f"the {self.granularity}-token storage granularity."
        )
        assert self.num_linear_layers > 0
        assert self.num_key_heads > 0
        assert self.num_value_heads > 0
        assert self.num_value_heads % self.num_key_heads == 0, (
            "LogState requires an integer number of value heads per key head."
        )
        assert self.head_k_dim > 0
        assert self.head_v_dim > 0
        assert self.conv_dim > 0
        assert self.state_len > 0

    @property
    def num_granules(self) -> int:
        return self.page_size // self.granularity

    @property
    def head_log_key_bytes(self) -> int:
        return self.page_size * self.head_k_dim * self.dtype.itemsize

    @property
    def head_log_value_bytes(self) -> int:
        return self.page_size * self.head_v_dim * self.dtype.itemsize

    @property
    def head_recurrent_state_bytes(self) -> int:
        return self.head_k_dim * self.head_v_dim * self.dtype.itemsize

    def head_payload_bytes(self, frame_bytes: int) -> int:
        """Keep each head overlay inside one frame for direct kernel addressing."""
        size = max(self.head_log_key_bytes + self.head_log_value_bytes,
                   self.head_recurrent_state_bytes)
        if frame_bytes % size == 0:
            return size
        padded = 1 << (size - 1).bit_length()
        if frame_bytes % padded != 0:
            raise ValueError("Linear head payload cannot be aligned within a KV frame")
        return padded

    @property
    def log_decay_bytes(self) -> int:
        return (
            self.num_linear_layers * self.page_size * self.num_value_heads
            * torch.float32.itemsize
        )

    @property
    def conv_state_bytes(self) -> int:
        return (
            self.num_linear_layers * self.num_granules * self.conv_dim
            * self.state_len * self.dtype.itemsize
        )
