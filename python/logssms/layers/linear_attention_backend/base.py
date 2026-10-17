from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from logssms.core import Batch
    from logssms.layers.gated_delta import GatedDeltaConfig
    from logssms.layers.mamba2 import Mamba2Config


@dataclass
class BaseLinearAttentionMetadata(ABC):
    phase: Literal["prefill", "decode", "verify"]


class BaseLinearAttentionBackend(ABC):
    def forward_mixed(self, batch: Batch, token_inputs: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        assert batch.mixed_parts is not None
        split = batch.prefill_token_capacity
        output = None
        for part, begin, end in (
            (batch.mixed_parts[0], 0, split),
            (batch.mixed_parts[1], split, next(iter(token_inputs.values())).shape[0]),
        ):
            result = self.forward(
                **{name: tensor[begin:end] for name, tensor in token_inputs.items()},
                **kwargs, batch=part,
            )
            if output is None:
                output = result.new_empty((next(iter(token_inputs.values())).shape[0],) + result.shape[1:])
            output[begin:end].copy_(result)
        assert output is not None
        return output

    def profile_result(self) -> dict | None:
        return None

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None: ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...


class BaseGatedDeltaBackend(BaseLinearAttentionBackend):

    @abstractmethod
    def forward(
        self,
        mixed_qkv  : torch.Tensor,
        gate       : torch.Tensor,
        beta       : torch.Tensor,
        conv_weight: torch.Tensor,
        A_log      : torch.Tensor,
        dt_bias    : torch.Tensor,
        config     : GatedDeltaConfig,
        batch      : Batch,
    ) -> torch.Tensor: ...


class BaseMamba2Backend(BaseLinearAttentionBackend):
    @abstractmethod
    def forward(
        self,
        xbc        : torch.Tensor,
        dt         : torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias  : torch.Tensor | None,
        A_log      : torch.Tensor,
        dt_bias    : torch.Tensor,
        D          : torch.Tensor,
        config     : Mamba2Config,
        batch      : Batch,
    ) -> torch.Tensor: ...


__all__ = [
    "BaseLinearAttentionBackend", "BaseLinearAttentionMetadata",
    "BaseGatedDeltaBackend", "BaseMamba2Backend",
]
