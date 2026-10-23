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
class BaseLinearAttnMetadata(ABC):
    phase: Literal["prefill", "decode", "verify"]


class BaseLinearAttnBackend(ABC):
    def forward_mixed(self, batch: Batch, token_inputs: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        from logssms.layers.piecewise import attention_output_buffer

        assert batch.mixed_parts is not None
        split = batch.prefill_token_capacity or batch.prefill_tokens
        example = next(iter(token_inputs.values()))
        config = kwargs["config"]
        heads, dim = ((config.num_v_heads, config.head_v_dim) if isinstance(self, BaseGatedDeltaBackend)
                      else (config.num_heads, config.head_dim))
        output = attention_output_buffer(
            example, (len(example), heads, dim),
            "attention_output" if isinstance(self, BaseGatedDeltaBackend) else "ssm_output",
        )
        for part, begin, end in (
            (batch.mixed_parts[0], 0, split),
            (batch.mixed_parts[1], split, next(iter(token_inputs.values())).shape[0]),
        ):
            self.forward(
                **{name: tensor[begin:end] for name, tensor in token_inputs.items()},
                **kwargs, batch=part, out=output[begin:end],
            )
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


class BaseGatedDeltaBackend(BaseLinearAttnBackend):

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
        out        : torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class BaseMamba2Backend(BaseLinearAttnBackend):
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
        out        : torch.Tensor | None = None,
    ) -> torch.Tensor: ...


__all__ = [
    "BaseLinearAttnBackend", "BaseLinearAttnMetadata",
    "BaseGatedDeltaBackend", "BaseMamba2Backend",
]
