from __future__ import annotations

from typing import Any

import torch

from logssms.core import get_global_ctx
from logssms.distributed import get_tp_info
from logssms.kernel import sigmoid_and_mul
from logssms.utils import div_even, nvtx_annotate

from .base import BaseOP
from .linear import (
    LinearColParallelPartitioned,
    LinearRowParallel,
)
from .norm import RMSNorm
from .rotary import get_rope


class MHAAttentionImpl(BaseOP):
    def __init__(
        self,
        layer_id: int,
    ):
        self._layer_id = layer_id

    @nvtx_annotate("MHA")
    def forward(
        self,
        query: torch.Tensor,
        key  : torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        ctx = get_global_ctx()
        output = ctx.attn_backend.forward(
            query,
            key.flatten(-2),
            value.flatten(-2),
            self._layer_id,
            ctx.batch,
        )
        return output.reshape(query.shape)


class RotaryAttention(BaseOP):
    def __init__(
        self,
        hidden_size : int,
        head_dim    : int,
        num_qo_heads: int,
        num_kv_heads: int,
        layer_id    : int,
        rotary_dim  : int,
        max_position: int,
        rope_base   : float,
        rope_scaling: tuple[tuple[str, Any], ...] | None,
        *,
        has_attn_bias: bool = False,
        qk_norm_eps  : float | None = None,
    ):
        tp_size = get_tp_info().size
        self.head_dim = head_dim
        self.num_qo_heads = div_even(num_qo_heads, tp_size)
        self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
        self._layer_id = layer_id

        self.qkv_proj = LinearColParallelPartitioned(
            input_size=hidden_size,
            partition_size=head_dim,
            partitions=[(num_qo_heads, False)] + [(num_kv_heads, True)] * 2,
            has_bias=has_attn_bias,
        )
        if qk_norm_eps is not None:
            self.q_norm = RMSNorm(head_dim, eps=qk_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=qk_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=rope_base,
            rope_scaling=rope_scaling,
        )
        self.attn = MHAAttentionImpl(layer_id)
        self.o_proj = LinearRowParallel(
            head_dim * num_qo_heads,
            hidden_size,
            has_bias=False,
        )

    def proj(
        self,
        x        : torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = x.shape[:-1]
        qkv = self.qkv_proj.forward(x)
        query, key, value = qkv.split(
            [
                self.num_qo_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        query = query.view(shape + (self.num_qo_heads, self.head_dim))
        key = key.view(shape + (self.num_kv_heads, self.head_dim))
        value = value.view(shape + (self.num_kv_heads, self.head_dim))
        if self.q_norm is not None:
            self.q_norm.forward_inplace(query)
        if self.k_norm is not None:
            self.k_norm.forward_inplace(key)
        query, key = self.rotary.forward(positions, query, key)
        return query, key, value

    @nvtx_annotate("RotaryAttention", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query, key, value = self.proj(x, get_global_ctx().batch.positions)
        output = self.attn.forward(query, key, value)
        return self.o_proj.forward(output.flatten(-2))


class GatedRotaryAttention(BaseOP):
    def __init__(
        self,
        hidden_size       : int,
        head_dim          : int,
        num_qo_heads      : int,
        num_kv_heads      : int,
        layer_id          : int,
        rotary_dim        : int,
        max_position      : int,
        rope_base         : float,
        rope_scaling      : tuple[tuple[str, Any], ...] | None,
        qk_norm_eps       : float,
        zero_centered_norm: bool,
    ):
        tp_size = get_tp_info().size
        self.head_dim = head_dim
        self.num_qo_heads = div_even(num_qo_heads, tp_size)
        self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
        self._layer_id = layer_id

        self.qkv_proj = LinearColParallelPartitioned(
            input_size=hidden_size,
            partition_size=head_dim,
            partitions=[(num_qo_heads * 2, False)] + [(num_kv_heads, True)] * 2,
            has_bias=False,
        )
        self.o_proj = LinearRowParallel(
            num_qo_heads * head_dim, hidden_size, has_bias=False
        )
        self.q_norm = RMSNorm(
            head_dim, eps=qk_norm_eps, zero_centered=zero_centered_norm
        )
        self.k_norm = RMSNorm(
            head_dim, eps=qk_norm_eps, zero_centered=zero_centered_norm
        )
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=rope_base,
            rope_scaling=rope_scaling,
        )
        self.attn = MHAAttentionImpl(layer_id)

    def proj(
        self,
        x        : torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = x.shape[:-1]
        qkv = self.qkv_proj.forward(x)
        q_gate, key, value = qkv.split(
            [
                self.num_qo_heads * self.head_dim * 2,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        q_gate = q_gate.view(
            shape + (self.num_qo_heads, self.head_dim * 2)
        )
        query, gate = q_gate.chunk(2, dim=-1)
        key = key.view(
            shape + (self.num_kv_heads, self.head_dim)
        )
        value = value.view(
            shape + (self.num_kv_heads, self.head_dim)
        )
        query = self.q_norm.forward(query)
        key = self.k_norm.forward(key)
        query, key = self.rotary.forward(positions, query, key)
        return query, gate, key, value

    @nvtx_annotate("GatedRotaryAttention", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        query, gate, key, value = self.proj(x, positions)
        output = self.attn.forward(query, key, value)
        sigmoid_and_mul(output, gate, out=output)
        return self.o_proj.forward(output.flatten(-2))
