from __future__ import annotations

from typing import Any

import torch

from logssms.core import get_global_ctx
from logssms.distributed import get_tp_info
from logssms.kernel import qk_norm_rope_store, sigmoid_and_mul
from logssms.utils import div_even, nvtx_annotate

from .base import BaseOP
from .linear import (
    LinearColParallelPartitioned,
    LinearRowParallel,
)
from .norm import RMSNorm
from .rotary import RotaryEmbedding, get_rope


@nvtx_annotate("QKV::prepare")
def _prepare_cached_query(
    query    : torch.Tensor,
    key      : torch.Tensor,
    value    : torch.Tensor,
    positions: torch.Tensor,
    rotary   : RotaryEmbedding,
    q_norm   : RMSNorm | None,
    k_norm   : RMSNorm | None,
    layer_id : int,
) -> torch.Tensor:
    ctx = get_global_ctx()
    return qk_norm_rope_store(
        query, key, value, positions, rotary._cos_sin_cache,
        ctx.batch.out_loc, ctx.kv_cache.k_cache(layer_id), ctx.kv_cache.v_cache(layer_id),
        q_weight=q_norm.weight if q_norm is not None else None,
        k_weight=k_norm.weight if k_norm is not None else None,
        q_eps=q_norm.eps if q_norm is not None else 0.0,
        k_eps=k_norm.eps if k_norm is not None else 0.0,
        q_zero_centered=q_norm.zero_centered if q_norm is not None else False,
        k_zero_centered=k_norm.zero_centered if k_norm is not None else False,
    )


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
    ) -> torch.Tensor:
        ctx = get_global_ctx()
        output = ctx.attn_backend.forward(query, self._layer_id, ctx.batch)
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

    def _project(
        self,
        x        : torch.Tensor,
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
        return query, key, value

    def proj(
        self,
        x        : torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = self._project(x)
        if self.q_norm is not None:
            self.q_norm.forward_inplace(query)
        if self.k_norm is not None:
            self.k_norm.forward_inplace(key)
        query, key = self.rotary.forward(positions, query, key)
        return query, key, value

    def attend(
        self,
        query    : torch.Tensor,
        key      : torch.Tensor,
        value    : torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        query = _prepare_cached_query(
            query, key, value, positions,
            self.rotary, self.q_norm, self.k_norm, self._layer_id,
        )
        return self.attn.forward(query)

    @nvtx_annotate("RotaryAttention", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query, key, value = self._project(x)
        output = self.attend(query, key, value, get_global_ctx().batch.positions)
        return self.o_proj.forward(output.flatten(-2))


class GatedRotaryAttention(RotaryAttention):
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

    def _project(
        self,
        x        : torch.Tensor,
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
        return query, gate, key, value

    def proj(
        self,
        x        : torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query, gate, key, value = self._project(x)
        query = self.q_norm.forward(query)
        key = self.k_norm.forward(key)
        query, key = self.rotary.forward(positions, query, key)
        return query, gate, key, value

    @nvtx_annotate("GatedRotaryAttention", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        query, gate, key, value = self._project(x)
        output = self.attend(query, key, value, positions)
        return self.output(output, gate)

    def output(self, output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        sigmoid_and_mul(output, gate, out=output)
        return self.o_proj.forward(output.flatten(-2))
