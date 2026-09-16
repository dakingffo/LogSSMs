from __future__ import annotations

import torch

from logssms.utils import nvtx_annotate

from .activation import gelu_and_mul, silu_and_mul
from .base import BaseOP
from .linear import (
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
)
from .moe import MoELayer


class GatedMLP(BaseOP):
    def __init__(
        self,
        hidden_size      : int,
        intermediate_size: int,
        hidden_act       : str,
    ):
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size,
            [intermediate_size, intermediate_size],
            has_bias=False,
        )

        fn_map = {"silu": silu_and_mul, "gelu": gelu_and_mul}
        if act_fn := fn_map.get(hidden_act, None):
            self.act_fn = act_fn
        else:
            raise ValueError(f"Unsupported activation function: {hidden_act}")

        self.down_proj = LinearRowParallel(
            intermediate_size,
            hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        del x
        y = self.act_fn(gate_up)
        del gate_up
        return self.down_proj.forward(y)


class MoEMLP(BaseOP):
    def __init__(
        self,
        num_experts      : int,
        top_k            : int,
        hidden_size      : int,
        intermediate_size: int,
        renormalize      : bool,
    ):
        self.router = LinearReplicated(
            hidden_size,
            num_experts,
            has_bias=False,
        )
        self.experts = MoELayer(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            renormalize=renormalize,
        )


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.router.forward(hidden_states)
        final_hidden_states = self.experts.forward(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        return final_hidden_states.view(num_tokens, hidden_dim)