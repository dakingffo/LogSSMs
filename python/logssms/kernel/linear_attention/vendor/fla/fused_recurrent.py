# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in this directory.


from __future__ import annotations

import torch
import triton
import triton.language as tl

from logssms.cache.logstate_pool.layout import LOG_STATE_REPLAY_WINDOW
from ...layout import log_pointer, payload_pointer

from .op import exp


@triton.jit
def _softplus(x):
    return tl.where(x < 20.0, tl.log(1.0 + tl.exp(x)), x)


@triton.jit
def _frame_byte_offset(frame, frame_bytes: tl.constexpr):

    return frame.to(tl.int64) * frame_bytes


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
    log_slots,
    token_offsets,
    work_slots,
    read_slots,
    write_slots,
    key_log_pool,
    value_log_pool,
    replay_counts,
    scale,
    SEQ_LEN,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_K_DIM : tl.constexpr,
    HEAD_V_DIM : tl.constexpr,
    BLOCK_K    : tl.constexpr,
    BLOCK_V    : tl.constexpr,
    DIRECT_STORE: tl.constexpr,
    SLOTTED_STORE: tl.constexpr,
    LAYER      : tl.constexpr,
    STATE_SLOT_STRIDE: tl.constexpr,
    STATE_LAYER_STRIDE: tl.constexpr,
    STATE_HEAD_STRIDE: tl.constexpr,
    READ_SLOT_STRIDE: tl.constexpr,
    WRITE_SLOT_STRIDE: tl.constexpr,
    GATE_BATCH_STRIDE: tl.constexpr,
    GATE_SEQ_STRIDE: tl.constexpr,
    GATE_HEAD_STRIDE: tl.constexpr,
    BETA_BATCH_STRIDE: tl.constexpr,
    BETA_SEQ_STRIDE: tl.constexpr,
    BETA_HEAD_STRIDE: tl.constexpr,
    FRAME_BYTES: tl.constexpr,
    FRAMES_PER_SLOT: tl.constexpr,
    HEAD_PAYLOAD_BYTES: tl.constexpr,
    DECAY_RECORD_BYTES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    REPLAY_WINDOW: tl.constexpr,
    REPLAY_WORK: tl.constexpr,
    LAYERS: tl.constexpr = 1,
    GRANULARITY: tl.constexpr = 64,
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

    if SLOTTED_STORE:
        read_slot = tl.load(read_slots + batch_idx * READ_SLOT_STRIDE)
        write_slot = tl.load(write_slots + batch_idx * WRITE_SLOT_STRIDE)
        state_ptr = (
            initial_state
            + read_slot * STATE_SLOT_STRIDE
            + LAYER * STATE_LAYER_STRIDE
            + value_head * STATE_HEAD_STRIDE
            + k_offset[:, None] * HEAD_V_DIM
            + v_offset[None, :]
        )
        write_state_ptr = (
            initial_state
            + write_slot * STATE_SLOT_STRIDE
            + LAYER * STATE_LAYER_STRIDE
            + value_head * STATE_HEAD_STRIDE
            + k_offset[:, None] * HEAD_V_DIM
            + v_offset[None, :]
        )
    elif DIRECT_STORE:
        work_slot = tl.load(work_slots + batch_idx)
        state_ptr = payload_pointer(
            initial_state.to(tl.pointer_type(query.dtype.element_ty)), key_log_pool,
            work_slot, LAYER, value_head, k_offset[:, None] * HEAD_V_DIM + v_offset[None, :],
            LAYERS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, GRANULARITY, PAGE_SIZE,
            HEAD_PAYLOAD_BYTES, FRAME_BYTES, FRAMES_PER_SLOT,
        )
    else:
        state_ptr = (
            initial_state
            + batch_head * HEAD_K_DIM * HEAD_V_DIM
            + k_offset[:, None] * HEAD_V_DIM
            + v_offset[None, :]
        )
    state = tl.load(state_ptr, mask=state_mask, other=0.0).to(tl.float32)
    if REPLAY_WORK:
        count = tl.load(replay_counts + batch_idx)
        log_slot = tl.load(log_slots + batch_idx)
        current_offset = tl.load(token_offsets + batch_idx)
        decay_frame = tl.load(value_log_pool + log_slot * 2)
        decay_index = tl.load(value_log_pool + log_slot * 2 + 1)
        decay_head = (initial_state
            + _frame_byte_offset(decay_frame, FRAME_BYTES)
            + decay_index * DECAY_RECORD_BYTES
        ).to(tl.pointer_type(tl.float32))
        replay_index = tl.arange(0, REPLAY_WINDOW)
        prior_offsets = current_offset - count + replay_index
        valid = replay_index < count
        log_decays = tl.load(
            decay_head + (LAYER * PAGE_SIZE + prior_offsets) * NUM_V_HEADS
            + value_head,
            mask=valid, other=0.0,
        )
        total_decay = tl.sum(log_decays, axis=0)
        suffix_decay = tl.exp(total_decay - tl.cumsum(log_decays))
        logged_keys = tl.load(
            log_pointer(
                initial_state.to(tl.pointer_type(query.dtype.element_ty)), key_log_pool,
                log_slot, LAYER, value_head, prior_offsets[None, :], k_offset[:, None],
                LAYERS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, GRANULARITY, PAGE_SIZE,
                HEAD_PAYLOAD_BYTES, FRAME_BYTES, FRAMES_PER_SLOT, False, NUM_K_HEADS,
            ),
            mask=k_mask[:, None] & valid[None, :], other=0.0,
        )
        logged_values = tl.load(
            log_pointer(
                initial_state.to(tl.pointer_type(query.dtype.element_ty)), key_log_pool,
                log_slot, LAYER, value_head, prior_offsets[:, None], v_offset[None, :],
                LAYERS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, GRANULARITY, PAGE_SIZE,
                HEAD_PAYLOAD_BYTES, FRAME_BYTES, FRAMES_PER_SLOT, True, NUM_K_HEADS,
            ),
            mask=valid[:, None] & v_mask[None, :], other=0.0,
        )
        scaled_values = (
            logged_values.to(tl.float32) * suffix_decay[:, None]
        ).to(query.dtype.element_ty)
        state = state * tl.exp(total_decay) + tl.dot(
            logged_keys, scaled_values,
        )
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
    gate_ptr = gate + batch_idx * GATE_BATCH_STRIDE + value_head * GATE_HEAD_STRIDE
    beta_ptr = beta + batch_idx * BETA_BATCH_STRIDE + value_head * BETA_HEAD_STRIDE
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
        log_decay = -exp(A) * _softplus(tl.load(gate_ptr).to(tl.float32) + bias)
        decay = exp(log_decay)
        beta_value = tl.sigmoid(tl.load(beta_ptr).to(tl.float32))

        if REPLAY_WORK:
            if value_block == 0:
                tl.store(
                    decay_head + (LAYER * PAGE_SIZE + current_offset) * NUM_V_HEADS
                    + value_head, log_decay,
                )

            state_k = tl.sum(state * k[:, None], axis=0) * decay
            state_q = tl.sum(state * q[:, None], axis=0) * decay
            delta = (v - state_k) * beta_value
            result = state_q + delta * tl.sum(k * q)
        else:
            state *= decay
            delta = (v - tl.sum(state * k[:, None], axis=0)) * beta_value
            state += k[:, None] * delta[None, :]
            result = tl.sum(state * q[:, None], axis=0)
        tl.store(output_ptr, result, mask=v_mask)

        if DIRECT_STORE:
            token_row = batch_idx * SEQ_LEN + seq_idx
            log_slot = tl.load(log_slots + token_row)
            token_offset = tl.load(token_offsets + token_row)
            prepared_key_ptr = log_pointer(
                initial_state.to(tl.pointer_type(query.dtype.element_ty)), key_log_pool,
                log_slot, LAYER, value_head, token_offset, k_offset,
                LAYERS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, GRANULARITY, PAGE_SIZE,
                HEAD_PAYLOAD_BYTES, FRAME_BYTES, FRAMES_PER_SLOT, False, NUM_K_HEADS,
            )
            residual_value_ptr = log_pointer(
                initial_state.to(tl.pointer_type(query.dtype.element_ty)), key_log_pool,
                log_slot, LAYER, value_head, token_offset, v_offset,
                LAYERS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, GRANULARITY, PAGE_SIZE,
                HEAD_PAYLOAD_BYTES, FRAME_BYTES, FRAMES_PER_SLOT, True, NUM_K_HEADS,
            )
            decay_frame = tl.load(value_log_pool + log_slot * 2)
            decay_index = tl.load(value_log_pool + log_slot * 2 + 1)
            decay_ptr = (initial_state
                + _frame_byte_offset(decay_frame, FRAME_BYTES)
                + decay_index * DECAY_RECORD_BYTES
            ).to(tl.pointer_type(tl.float32)) + (
                (LAYER * PAGE_SIZE + token_offset) * NUM_V_HEADS
                + value_head
            )
            tl.store(residual_value_ptr, delta, mask=v_mask)
            if value_block == 0:
                if value_head % (NUM_V_HEADS // NUM_K_HEADS) == 0:
                    tl.store(prepared_key_ptr, k, mask=k_mask)
                if not REPLAY_WORK:
                    tl.store(decay_ptr, log_decay)

        if SLOTTED_STORE:
            snapshot_ptr = write_state_ptr
        elif DIRECT_STORE:
            snapshot_ptr = state_ptr
        else:
            snapshot_ptr = (
                states
                + (
                    (batch_idx * SEQ_LEN + seq_idx) * NUM_V_HEADS
                    + value_head
                ) * HEAD_K_DIM * HEAD_V_DIM
                + k_offset[:, None] * HEAD_V_DIM
                + v_offset[None, :]
            )
        if REPLAY_WORK:
            if (current_offset + 1) % REPLAY_WINDOW == 0:
                updated_state = state * decay + k[:, None] * delta[None, :]
                tl.store(snapshot_ptr, updated_state, mask=state_mask)
        else:
            tl.store(snapshot_ptr, state, mask=state_mask)

        query_ptr += NUM_K_HEADS * HEAD_K_DIM
        key_ptr += NUM_K_HEADS * HEAD_K_DIM
        value_ptr += NUM_V_HEADS * HEAD_V_DIM
        gate_ptr += GATE_SEQ_STRIDE
        beta_ptr += BETA_SEQ_STRIDE
        output_ptr += NUM_V_HEADS * HEAD_V_DIM


@torch.compiler.disable
def fused_recurrent_gated_delta_rule_to_logstate(
    query        : torch.Tensor,
    key          : torch.Tensor,
    value        : torch.Tensor,
    gate         : torch.Tensor,
    beta         : torch.Tensor,
    A_log        : torch.Tensor,
    dt_bias      : torch.Tensor,
    pool,
    work_slots   : torch.Tensor,
    log_slots    : torch.Tensor,
    token_offsets: torch.Tensor,
    layer        : int,
    replay_counts: torch.Tensor,
    enable_eager_state: bool = False,
) -> torch.Tensor:
    batch_size, seq_len, num_k_heads, head_k_dim = query.shape
    num_v_heads = value.shape[2]
    head_v_dim = value.shape[3]
    assert seq_len == 1
    assert head_k_dim == pool.layout.head_k_dim
    assert head_v_dim == pool.layout.head_v_dim
    assert num_v_heads == pool.layout.num_value_heads
    assert replay_counts.shape == (batch_size,)
    assert pool.frame_bytes % pool.head_payload_bytes == 0
    output = torch.empty_like(value)
    full_k = triton.next_power_of_2(head_k_dim)
    tiled = not enable_eager_state and head_k_dim == head_v_dim == 128 and batch_size >= 32
    block_k = 32 if tiled else None
    block_v = 64 if tiled else (8 if head_v_dim < 128 else (16 if batch_size == 1 else 32))
    num_stages = 1 if tiled and batch_size < 64 else 3
    block_v = min(block_v, triton.next_power_of_2(head_v_dim))
    if not enable_eager_state:
        block_v = max(block_v, 16)
    grid = (triton.cdiv(head_v_dim, block_v), batch_size * num_v_heads)
    if block_k is not None:
        from .gdn_replay import _gdn_replay_kernel

        _gdn_replay_kernel[grid](
            query.contiguous(), key.contiguous(), value.contiguous(),
            gate, beta, A_log, dt_bias, pool.storage, output,
            work_slots, log_slots, token_offsets, replay_counts,
            pool.payload_frame_table, pool.decay_frame_table,
            GATE_BATCH_STRIDE=gate.stride(0), GATE_HEAD_STRIDE=gate.stride(2),
            BETA_BATCH_STRIDE=beta.stride(0), BETA_HEAD_STRIDE=beta.stride(2),
            NUM_K_HEADS=num_k_heads, NUM_V_HEADS=num_v_heads,
            HEAD_K_DIM=head_k_dim, HEAD_V_DIM=head_v_dim,
            FULL_K=full_k, BLOCK_K=block_k, BLOCK_V=block_v,
            FRAME_BYTES=pool.frame_bytes,
            FRAMES_PER_SLOT=pool.allocator.payload_frames,
            HEAD_PAYLOAD_BYTES=pool.head_payload_bytes,
            DECAY_RECORD_BYTES=pool.layout.log_decay_bytes,
            LAYER=layer, PAGE_SIZE=pool.page_size,
            LAYERS=pool.layout.num_linear_layers, GRANULARITY=pool.layout.granularity,
            REPLAY_WINDOW=LOG_STATE_REPLAY_WINDOW,
            num_warps=4,
            num_stages=num_stages,
        )
        return output
    _fused_recurrent_gated_delta_kernel[grid](
        query.contiguous(), key.contiguous(), value.contiguous(),
        gate, beta, A_log, dt_bias,
        pool.storage, output, None, log_slots, token_offsets, work_slots, None, None,
        pool.payload_frame_table, pool.decay_frame_table, replay_counts,
        head_k_dim ** -0.5, 1,
        NUM_K_HEADS=num_k_heads,
        NUM_V_HEADS=num_v_heads,
        HEAD_K_DIM=head_k_dim,
        HEAD_V_DIM=head_v_dim,
        BLOCK_K=full_k,
        BLOCK_V=block_v,
        DIRECT_STORE=True,
        SLOTTED_STORE=False,
        LAYER=layer,
        STATE_SLOT_STRIDE=0,
        STATE_LAYER_STRIDE=0,
        STATE_HEAD_STRIDE=0,
        READ_SLOT_STRIDE=0,
        WRITE_SLOT_STRIDE=0,
        GATE_BATCH_STRIDE=gate.stride(0),
        GATE_SEQ_STRIDE=gate.stride(1),
        GATE_HEAD_STRIDE=gate.stride(2),
        BETA_BATCH_STRIDE=beta.stride(0),
        BETA_SEQ_STRIDE=beta.stride(1),
        BETA_HEAD_STRIDE=beta.stride(2),
        FRAME_BYTES=pool.frame_bytes,
        FRAMES_PER_SLOT=pool.allocator.payload_frames,
        HEAD_PAYLOAD_BYTES=pool.head_payload_bytes,
        DECAY_RECORD_BYTES=pool.layout.log_decay_bytes,
        PAGE_SIZE=pool.page_size,
        LAYERS=pool.layout.num_linear_layers, GRANULARITY=pool.layout.granularity,
        REPLAY_WINDOW=LOG_STATE_REPLAY_WINDOW,
        REPLAY_WORK=not enable_eager_state,
        num_warps=4 if not enable_eager_state else 1,
        num_stages=num_stages,
    )
    return output


@torch.compiler.disable
def fused_recurrent_gated_delta_rule_to_slots(
    query       : torch.Tensor,
    key         : torch.Tensor,
    value       : torch.Tensor,
    gate        : torch.Tensor,
    beta        : torch.Tensor,
    A_log       : torch.Tensor,
    dt_bias     : torch.Tensor,
    state_pool  : torch.Tensor,
    read_slots  : torch.Tensor,
    write_slots : torch.Tensor,
    layer       : int,
) -> torch.Tensor:
    batch_size, seq_len, num_k_heads, head_k_dim = query.shape
    num_v_heads, head_v_dim = value.shape[2:]
    assert seq_len == 1 and state_pool.ndim == 5
    assert read_slots.shape == write_slots.shape == (batch_size,)
    assert state_pool.shape[2:] == (num_v_heads, head_k_dim, head_v_dim)
    assert num_v_heads % num_k_heads == 0
    output = torch.empty_like(value)
    block_k = triton.next_power_of_2(head_k_dim)
    block_v = min(32, triton.next_power_of_2(head_v_dim))
    grid = (triton.cdiv(head_v_dim, block_v), batch_size * num_v_heads)
    _fused_recurrent_gated_delta_kernel[grid](
        query.contiguous(), key.contiguous(), value.contiguous(),
        gate, beta, A_log, dt_bias,
        state_pool, output, None, None, None, None, read_slots, write_slots,
        None, None, None,
        head_k_dim ** -0.5, 1,
        NUM_K_HEADS=num_k_heads,
        NUM_V_HEADS=num_v_heads,
        HEAD_K_DIM=head_k_dim,
        HEAD_V_DIM=head_v_dim,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        DIRECT_STORE=False,
        SLOTTED_STORE=True,
        LAYER=layer,
        STATE_SLOT_STRIDE=state_pool.stride(0),
        STATE_LAYER_STRIDE=state_pool.stride(1),
        STATE_HEAD_STRIDE=state_pool.stride(2),
        READ_SLOT_STRIDE=read_slots.stride(0),
        WRITE_SLOT_STRIDE=write_slots.stride(0),
        GATE_BATCH_STRIDE=gate.stride(0),
        GATE_SEQ_STRIDE=gate.stride(1),
        GATE_HEAD_STRIDE=gate.stride(2),
        BETA_BATCH_STRIDE=beta.stride(0),
        BETA_SEQ_STRIDE=beta.stride(1),
        BETA_HEAD_STRIDE=beta.stride(2),
        FRAME_BYTES=0,
        FRAMES_PER_SLOT=0,
        HEAD_PAYLOAD_BYTES=0,
        DECAY_RECORD_BYTES=0,
        PAGE_SIZE=0,
        REPLAY_WINDOW=0,
        REPLAY_WORK=False,
        num_warps=1,
        num_stages=3,
    )
    return output


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
    initial_state = initial_state.contiguous()
    output = torch.empty_like(value)
    states = torch.empty(
        batch_size, seq_len, num_v_heads, head_k_dim, head_v_dim,
        dtype=initial_state.dtype,
        device=initial_state.device,
    )

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
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        head_k_dim ** -0.5,
        seq_len,
        NUM_K_HEADS=num_k_heads,
        NUM_V_HEADS=num_v_heads,
        HEAD_K_DIM=head_k_dim,
        HEAD_V_DIM=head_v_dim,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        DIRECT_STORE=False,
        SLOTTED_STORE=False,
        LAYER=0,
        STATE_SLOT_STRIDE=0,
        STATE_LAYER_STRIDE=0,
        STATE_HEAD_STRIDE=0,
        READ_SLOT_STRIDE=0,
        WRITE_SLOT_STRIDE=0,
        GATE_BATCH_STRIDE=gate.stride(0),
        GATE_SEQ_STRIDE=gate.stride(1),
        GATE_HEAD_STRIDE=gate.stride(2),
        BETA_BATCH_STRIDE=beta.stride(0),
        BETA_SEQ_STRIDE=beta.stride(1),
        BETA_HEAD_STRIDE=beta.stride(2),
        FRAME_BYTES=0,
        FRAMES_PER_SLOT=0,
        HEAD_PAYLOAD_BYTES=0,
        DECAY_RECORD_BYTES=0,
        PAGE_SIZE=0,
        REPLAY_WINDOW=0,
        REPLAY_WORK=False,
        num_warps=1,
        num_stages=3,
    )
    return output, states


__all__ = [
    "fused_recurrent_gated_delta_rule",
    "fused_recurrent_gated_delta_rule_to_logstate",
    "fused_recurrent_gated_delta_rule_to_slots",
]
