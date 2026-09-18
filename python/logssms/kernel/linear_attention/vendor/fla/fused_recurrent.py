# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in this directory.

"""Inference-only fused recurrent Gated DeltaNet kernel derived from FLA 0.5.2."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .op import exp


@triton.jit
def _softplus(x):
    return tl.where(x < 20.0, tl.log(1.0 + tl.exp(x)), x)


@triton.jit(do_not_specialize=["SEQ_LEN"])
def _fused_recurrent_gated_delta_kernel(
    query,
    key,
    value,
    gate,
    beta,
    A_log,
    dt_bias,
    initial_state,
    output,
    states,
    prepared_key,
    residual_value,
    decay_log,
    scale,
    SEQ_LEN,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_K_DIM : tl.constexpr,
    HEAD_V_DIM : tl.constexpr,
    BLOCK_K    : tl.constexpr,
    BLOCK_V    : tl.constexpr,
    STORE_LOG  : tl.constexpr,
):
    value_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = batch_head // NUM_V_HEADS
    value_head = batch_head % NUM_V_HEADS
    key_head = value_head // (NUM_V_HEADS // NUM_K_HEADS)

    k_offset = tl.arange(0, BLOCK_K)
    v_offset = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    k_mask = k_offset < HEAD_K_DIM
    v_mask = v_offset < HEAD_V_DIM
    state_mask = k_mask[:, None] & v_mask[None, :]

    state_ptr = (
        initial_state
        + batch_head * HEAD_K_DIM * HEAD_V_DIM
        + k_offset[:, None] * HEAD_V_DIM
        + v_offset[None, :]
    )
    state = tl.load(state_ptr, mask=state_mask, other=0.0).to(tl.float32)
    A = tl.load(A_log + value_head).to(tl.float32)
    bias = tl.load(dt_bias + value_head).to(tl.float32)

    query_ptr = (
        query
        + (batch_idx * SEQ_LEN * NUM_K_HEADS + key_head) * HEAD_K_DIM
        + k_offset
    )
    key_ptr = (
        key
        + (batch_idx * SEQ_LEN * NUM_K_HEADS + key_head) * HEAD_K_DIM
        + k_offset
    )
    value_ptr = (
        value
        + (batch_idx * SEQ_LEN * NUM_V_HEADS + value_head) * HEAD_V_DIM
        + v_offset
    )
    gate_ptr = gate + batch_idx * SEQ_LEN * NUM_V_HEADS + value_head
    beta_ptr = beta + batch_idx * SEQ_LEN * NUM_V_HEADS + value_head
    output_ptr = (
        output
        + (batch_idx * SEQ_LEN * NUM_V_HEADS + value_head) * HEAD_V_DIM
        + v_offset
    )

    for seq_idx in tl.range(0, SEQ_LEN):
        q = tl.load(query_ptr, mask=k_mask, other=0.0).to(tl.float32)
        k = tl.load(key_ptr, mask=k_mask, other=0.0).to(tl.float32)
        v = tl.load(value_ptr, mask=v_mask, other=0.0).to(tl.float32)
        q = q * tl.rsqrt(tl.sum(q * q) + 1e-6) * scale
        k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
        decay = exp(-exp(A) * _softplus(tl.load(gate_ptr).to(tl.float32) + bias))
        beta_value = tl.sigmoid(tl.load(beta_ptr).to(tl.float32))

        state *= decay
        delta = (v - tl.sum(state * k[:, None], axis=0)) * beta_value
        state += k[:, None] * delta[None, :]
        result = tl.sum(state * q[:, None], axis=0)
        tl.store(output_ptr, result, mask=v_mask)

        if STORE_LOG:
            prepared_key_ptr = (
                prepared_key
                + (batch_idx * SEQ_LEN * NUM_V_HEADS + value_head)
                * HEAD_K_DIM
                + k_offset
            )
            residual_value_ptr = (
                residual_value
                + (batch_idx * SEQ_LEN * NUM_V_HEADS + value_head)
                * HEAD_V_DIM
                + v_offset
            )
            decay_ptr = (
                decay_log
                + (batch_idx * SEQ_LEN + seq_idx) * NUM_V_HEADS
                + value_head
            )
            tl.store(residual_value_ptr, delta, mask=v_mask)
            if value_block == 0:
                tl.store(prepared_key_ptr, k, mask=k_mask)
                tl.store(decay_ptr, decay)

        snapshot_ptr = (
            states
            + (
                (batch_idx * SEQ_LEN + seq_idx) * NUM_V_HEADS
                + value_head
            ) * HEAD_K_DIM * HEAD_V_DIM
            + k_offset[:, None] * HEAD_V_DIM
            + v_offset[None, :]
        )
        tl.store(snapshot_ptr, state, mask=state_mask)

        query_ptr += NUM_K_HEADS * HEAD_K_DIM
        key_ptr += NUM_K_HEADS * HEAD_K_DIM
        value_ptr += NUM_V_HEADS * HEAD_V_DIM
        gate_ptr += NUM_V_HEADS
        beta_ptr += NUM_V_HEADS
        output_ptr += NUM_V_HEADS * HEAD_V_DIM


@torch.compiler.disable
def fused_recurrent_gated_delta_rule(
    query        : torch.Tensor,
    key          : torch.Tensor,
    value        : torch.Tensor,
    gate         : torch.Tensor,
    beta         : torch.Tensor,
    A_log        : torch.Tensor,
    dt_bias      : torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, states, _, _, _ = _fused_recurrent_gated_delta_rule(
        query,
        key,
        value,
        gate,
        beta,
        A_log,
        dt_bias,
        initial_state,
        store_log=False,
    )
    return output, states


@torch.compiler.disable
def fused_recurrent_gated_delta_rule_with_log(
    query        : torch.Tensor,
    key          : torch.Tensor,
    value        : torch.Tensor,
    gate         : torch.Tensor,
    beta         : torch.Tensor,
    A_log        : torch.Tensor,
    dt_bias      : torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run short GDN recurrence and expose its lossless update log."""
    output, states, prepared_key, residual_value, decay = (
        _fused_recurrent_gated_delta_rule(
            query,
            key,
            value,
            gate,
            beta,
            A_log,
            dt_bias,
            initial_state,
            store_log=True,
        )
    )
    assert prepared_key is not None
    assert residual_value is not None
    assert decay is not None
    return output, states, prepared_key, residual_value, decay


def _fused_recurrent_gated_delta_rule(
    query        : torch.Tensor,
    key          : torch.Tensor,
    value        : torch.Tensor,
    gate         : torch.Tensor,
    beta         : torch.Tensor,
    A_log        : torch.Tensor,
    dt_bias      : torch.Tensor,
    initial_state: torch.Tensor,
    store_log    : bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Run a short GDN recurrence and return every post-token state."""
    assert query.is_cuda and key.is_cuda and value.is_cuda
    assert query.ndim == key.ndim == value.ndim == 4
    batch_size, seq_len, num_k_heads, head_k_dim = query.shape
    num_v_heads = value.shape[2]
    head_v_dim = value.shape[3]
    assert key.shape == query.shape
    assert value.shape[:2] == (batch_size, seq_len)
    assert num_v_heads % num_k_heads == 0
    assert gate.shape == beta.shape == (batch_size, seq_len, num_v_heads)
    assert initial_state.shape == (
        batch_size, num_v_heads, head_k_dim, head_v_dim
    )

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    gate = gate.contiguous()
    beta = beta.contiguous()
    initial_state = initial_state.contiguous()
    output = torch.empty_like(value)
    states = torch.empty(
        batch_size, seq_len, num_v_heads, head_k_dim, head_v_dim,
        dtype=initial_state.dtype,
        device=initial_state.device,
    )
    prepared_key = torch.empty_like(key) if store_log else None
    residual_value = torch.empty_like(value) if store_log else None
    decay = torch.empty_like(gate, dtype=torch.float32) if store_log else None

    block_k = triton.next_power_of_2(head_k_dim)
    block_v = min(8, triton.next_power_of_2(head_v_dim))
    grid = (triton.cdiv(head_v_dim, block_v), batch_size * num_v_heads)
    _fused_recurrent_gated_delta_kernel[grid](
        query,
        key,
        value,
        gate,
        beta,
        A_log,
        dt_bias,
        initial_state,
        output,
        states,
        prepared_key,
        residual_value,
        decay,
        head_k_dim ** -0.5,
        seq_len,
        NUM_K_HEADS=num_k_heads,
        NUM_V_HEADS=num_v_heads,
        HEAD_K_DIM=head_k_dim,
        HEAD_V_DIM=head_v_dim,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        STORE_LOG=store_log,
        num_warps=1,
        num_stages=3,
    )
    return output, states, prepared_key, residual_value, decay


__all__ = [
    "fused_recurrent_gated_delta_rule",
    "fused_recurrent_gated_delta_rule_with_log",
]
