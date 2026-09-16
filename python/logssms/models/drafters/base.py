from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from logssms.layers import BaseOP

if TYPE_CHECKING:
    import torch


class BaseDrafterModel(ABC, BaseOP):
    """Common lifecycle interface for standalone speculative drafter models."""

    @abstractmethod
    def load_weights(self, model_path: str, device: torch.device) -> None:
        """Load drafter weights onto ``device``."""
