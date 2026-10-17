from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from logssms.core import get_global_ctx
from logssms.distributed import DistributedCommunicator, get_tp_info
from logssms.utils import div_even, nvtx_annotate

from .base import BaseOP
from .linear import LinearTPImpl, LinearRowParallel


@dataclass(frozen=True)
class Mamba2Config:
    num_heads : int
    num_groups: int
    head_dim  : int
    state_dim : int
    state_len : int
    layer_idx : int

    @property
    def value_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def conv_dim(self) -> int:
        return self.value_dim + 2 * self.num_groups * self.state_dim


class Mamba2Conv(BaseOP):
    def __init__(self, channels: int, kernel_size: int, has_bias: bool):
        self.weight = torch.empty(channels, 1, kernel_size)
        self.bias = torch.empty(channels) if has_bias else None


class Mamba2GatedNorm(BaseOP):

    def __init__(self, size: int, groups: int, eps: float, before_gate: bool):
        tp = get_tp_info().size
        self._groups = div_even(groups, tp, allow_replicate=True)
        self._group_ranks = max(1, tp // groups)
        self._comm = DistributedCommunicator()
        self._eps = eps
        self._before_gate = before_gate
        self.weight = torch.empty(div_even(size, tp))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        dtype = x.dtype
        x = x.float()
        gate = F.silu(gate.float())
        if not self._before_gate:
            x = x * gate
        grouped = x.reshape(shape[0], self._groups, -1)
        variance = grouped.square().mean(-1, keepdim=True)
        if self._group_ranks > 1:


            variance = self._comm.all_reduce(variance) / self._group_ranks
        x = (grouped * torch.rsqrt(variance + self._eps)).reshape(shape)
        x = x * self.weight
        if self._before_gate:
            x = x * gate
        return x.to(dtype)


class Mamba2(BaseOP):
    def __init__(
        self,
        *,
        hidden_size     : int,
        num_heads       : int,
        num_groups      : int,
        head_dim        : int,
        state_dim       : int,
        conv_kernel_size: int,
        conv_bias       : bool,
        proj_bias       : bool,
        output_bias     : bool,
        rms_norm        : bool,
        rms_norm_eps    : float,
        norm_before_gate: bool,
        in_multiplier   : float,
        multipliers     : tuple[float, ...],
        layer_idx       : int,
    ):
        tp = get_tp_info().size
        if num_groups < tp and num_groups != 1:
            raise ValueError("Mamba2 TP currently supports sharded groups or one replicated group")
        self._config = Mamba2Config(
            num_heads=div_even(num_heads, tp),
            num_groups=div_even(num_groups, tp, allow_replicate=True),
            head_dim=head_dim, state_dim=state_dim,
            state_len=conv_kernel_size - 1, layer_idx=layer_idx,
        )
        config = self._config
        self._in_multiplier = in_multiplier
        self._multipliers = multipliers
        self.in_proj = LinearTPImpl(
            hidden_size, config.value_dim + config.conv_dim + config.num_heads,
            has_bias=proj_bias,
        )
        self.conv1d = Mamba2Conv(config.conv_dim, conv_kernel_size, conv_bias)
        self.A_log = torch.empty(config.num_heads, dtype=torch.float32)
        self.dt_bias = torch.empty(config.num_heads)
        self.D = torch.empty(config.num_heads)
        self.norm = (
            Mamba2GatedNorm(num_heads * head_dim, num_groups, rms_norm_eps, norm_before_gate)
            if rms_norm else None
        )
        self.out_proj = LinearRowParallel(
            num_heads * head_dim, hidden_size, has_bias=output_bias,
        )

    def proj(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        config = self._config
        widths = (config.value_dim, config.value_dim,
                  config.num_groups * config.state_dim,
                  config.num_groups * config.state_dim, config.num_heads)
        parts = self.in_proj.forward(x * self._in_multiplier).split(widths, -1)
        z, value, key, query, dt = (
            part * scale for part, scale in zip(parts, self._multipliers)
        )
        return torch.cat((value, key, query), -1), z, dt

    @nvtx_annotate("Mamba2", layer_id_field="_layer_idx")
    def attend(self, xbc: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        if ctx.batch.is_mixed:
            return ctx.linear_attn_backend.forward_mixed(
                ctx.batch, {"xbc": xbc, "dt": dt},
                conv_weight=self.conv1d.weight, conv_bias=self.conv1d.bias,
                A_log=self.A_log, dt_bias=self.dt_bias, D=self.D,
                config=self._config,
            )
        return ctx.linear_attn_backend.forward(
            xbc=xbc, dt=dt, conv_weight=self.conv1d.weight,
            conv_bias=self.conv1d.bias, A_log=self.A_log,
            dt_bias=self.dt_bias, D=self.D, config=self._config, batch=ctx.batch,
        )

    @property
    def _layer_idx(self) -> int:
        return self._config.layer_idx

    def output(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        x = self.norm.forward(x, gate) if self.norm is not None else x * F.silu(gate)
        return self.out_proj.forward(x)
