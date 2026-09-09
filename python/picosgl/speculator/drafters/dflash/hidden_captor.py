from __future__ import annotations

from typing import Any

import torch

from ...base import BaseSpeculatorConfig
from ...hidden_captor import HiddenCaptorBase, HiddenCapturePoint
from .config import DFlashSpeculatorConfig


class DFlashHiddenCaptor(HiddenCaptorBase):
    """Capture target layer outputs through the following decoder's input hook."""

    def __init__(
        self,
        config     : BaseSpeculatorConfig,
        full_hidden: torch.Tensor | None = None,
    ) -> None:
        assert isinstance(config, DFlashSpeculatorConfig)
        self.config = config
        self._layers: dict[int, torch.Tensor] = {}
        self._full_hidden = full_hidden

    def capture(
        self,
        point   : HiddenCapturePoint,
        layer_id: int | None,
        *args   : Any,
        **kwargs: Any,
    ) -> None:
        if point is not HiddenCapturePoint.DECODER_INPUT or layer_id is None:
            return
        target_layer_id = layer_id - 1
        if target_layer_id not in self.config.target_layer_ids:
            return

        hidden = args[0] if args else kwargs["x"]
        assert isinstance(hidden, torch.Tensor)
        self._layers[target_layer_id] = hidden

    @property
    def full_hidden(self) -> torch.Tensor:
        if self._full_hidden is None:
            missing = set(self.config.target_layer_ids) - self._layers.keys()
            assert not missing, f"DFLASH did not capture target layers {sorted(missing)}"
            self._full_hidden = torch.cat(
                [self._layers[layer_id] for layer_id in self.config.target_layer_ids],
                dim=-1,
            )
            self._layers.clear()
        return self._full_hidden

    def select(self, index) -> DFlashHiddenCaptor:
        return DFlashHiddenCaptor(
            self.config,
            self.full_hidden[index].contiguous(),
        )

    def get_carry_hidden(self, index) -> torch.Tensor:
        return self.full_hidden[index].contiguous()


__all__ = ["DFlashHiddenCaptor"]
