from __future__ import annotations

import torch


class LinearStatePool:
    def __init__(
        self,
        num_slots       : int,
        num_linear_layers: int,
        conv_dim        : int,
        kernel_size     : int,
        num_v_heads     : int,
        head_k_dim      : int,
        head_v_dim      : int,
        device          : torch.device,
        dtype           : torch.dtype,
    ) -> None:
        self.conv_state = torch.zeros(
            num_slots, num_linear_layers, conv_dim, kernel_size - 1,
            device=device, dtype=dtype,
        )
        self.recurrent_state = torch.zeros(
            num_slots, num_linear_layers, num_v_heads, head_k_dim, head_v_dim,
            device=device, dtype=dtype,
        )
        self._device = device
        self._dtype = dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype