from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from logssms.layers import BaseOP

if TYPE_CHECKING:
    import torch


class BaseDrafterModel(ABC, BaseOP):

    @abstractmethod
    def load_weights(self, model_path: str, device: torch.device) -> None:
        pass
