from __future__ import annotations

from typing import Any

import torch

from ...base import BaseSpeculatorConfig
from ...hidden_captor import HiddenCaptorBase, HiddenCapturePoint
from .config import MTPSpeculatorConfig


class MTPHiddenCaptor(HiddenCaptorBase):
    def __init__(
        self,
        config     : BaseSpeculatorConfig,
        full_hidden: torch.Tensor | None = None,
    ) -> None:
        assert isinstance(config, MTPSpeculatorConfig)
        self.config = config
        self._full_hidden = full_hidden

    def capture(
        self,
        point   : HiddenCapturePoint,
        layer_id: int | None,
        *args   : Any,
        **kwargs: Any,
    ) -> None:
        if point is HiddenCapturePoint.LM_HEAD_INPUT:
            self._full_hidden = args[0] if args else kwargs["x"]

    @property
    def full_hidden(self) -> torch.Tensor:
        assert self._full_hidden is not None, "LM head input was not captured"
        return self._full_hidden

    def select(self, index) -> MTPHiddenCaptor:
        return MTPHiddenCaptor(
            self.config,
            self.full_hidden[index].contiguous(),
        )

    def get_carry_hidden(self, index) -> torch.Tensor:
        return self.full_hidden[index].contiguous()


__all__ = ["MTPHiddenCaptor"]
