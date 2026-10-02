from __future__ import annotations

from dataclasses import dataclass

import torch

from logssms.distributed import get_tp_info
from logssms.utils import div_even, nvtx_annotate

from .base import BaseOP
from .linear import LinearColParallelMerged, LinearRowParallel
from .norm import RMSNormGated


@dataclass(frozen=True)
class GatedDeltaConfig:
    num_k_heads: int
    num_v_heads: int
    head_k_dim : int
    head_v_dim : int
    conv_dim   : int
    state_len  : int
    layer_idx  : int

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim


class _Conv1d(BaseOP):
    def __init__(self, in_channels: int, kernel_size: int):
        self.weight = torch.empty(in_channels, 1, kernel_size)
        self.kernel_size = kernel_size

    @property
    def in_channels(self) -> int:
        return self.weight.shape[0]


class GatedDeltaNet(BaseOP):
    def __init__(
        self,
        hidden_size     : int,
        num_key_heads   : int,
        num_value_heads : int,
        head_k_dim      : int,
        head_v_dim      : int,
        conv_kernel_size: int,
        rms_norm_eps    : float,
        layer_idx       : int,
    ):
        tp_size = get_tp_info().size
        local_key_heads = div_even(num_key_heads, tp_size)
        local_value_heads = div_even(num_value_heads, tp_size)
        key_dim = local_key_heads * head_k_dim
        value_dim = local_value_heads * head_v_dim
        conv_dim = key_dim * 2 + value_dim

        self._config = GatedDeltaConfig(
            num_k_heads=local_key_heads,
            num_v_heads=local_value_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            conv_dim=conv_dim,
            state_len=conv_kernel_size - 1,
            layer_idx=layer_idx,
        )
        self._linear_layer_idx = layer_idx

        self.conv1d = _Conv1d(conv_dim, conv_kernel_size)
        self.dt_bias = torch.ones(local_value_heads)
        self.A_log = torch.zeros(local_value_heads, dtype=torch.float32)
        self.norm = RMSNormGated(head_v_dim, eps=rms_norm_eps)
        self.out_proj = LinearRowParallel(
            num_value_heads * head_v_dim, hidden_size, has_bias=False
        )
        self.in_proj_qkvz = LinearColParallelMerged(
            hidden_size,
            [
                num_key_heads * head_k_dim * 2 + num_value_heads * head_v_dim,
                num_value_heads * head_v_dim,
            ],
            has_bias=False,
        )
        self.in_proj_ba = LinearColParallelMerged(
            hidden_size, [num_value_heads, num_value_heads], has_bias=False
        )

    @nvtx_annotate("GatedDeltaNet", layer_id_field="_linear_layer_idx")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed_qkv, z, beta, gate = self.proj(x)
        core_output = self.attend(mixed_qkv, beta, gate)
        return self.output(core_output, z)

    def proj(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        config = self._config
        mixed_qkv, z = self.in_proj_qkvz.forward(x).split(
            [config.conv_dim, config.value_dim], dim=-1
        )
        beta, gate = self.in_proj_ba.forward(x).chunk(2, dim=-1)
        return mixed_qkv, z, beta, gate

    def attend(
        self,
        mixed_qkv: torch.Tensor,
        beta     : torch.Tensor,
        gate     : torch.Tensor,
    ) -> torch.Tensor:
        from logssms.core import get_global_ctx

        ctx = get_global_ctx()
        return ctx.linear_attn_backend.forward(
            mixed_qkv=mixed_qkv,
            gate=gate,
            beta=beta,
            conv_weight=self.conv1d.weight,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            config=self._config,
            batch=ctx.batch,
        )

    def output(self, core_output: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        config = self._config
        shape = (core_output.shape[0], config.num_v_heads, config.head_v_dim)
        output = self.norm.forward(
            core_output.reshape(shape), z.view(shape)
        ).reshape(core_output.shape[0], -1)
        return self.out_proj.forward(output)
