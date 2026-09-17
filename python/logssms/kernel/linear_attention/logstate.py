from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _reconstruct_log_states_kernel(
    key_log,
    value_log,
    decay_log,
    recurrent_state,
    output_slots,
    base_slots,
    task_log_offsets,
    log_slots,
    log_begins,
    log_ends,
    key_slot_stride       : tl.constexpr,
    key_layer_stride      : tl.constexpr,
    key_token_stride      : tl.constexpr,
    key_head_stride       : tl.constexpr,
    value_slot_stride     : tl.constexpr,
    value_layer_stride    : tl.constexpr,
    value_token_stride    : tl.constexpr,
    value_head_stride     : tl.constexpr,
    decay_slot_stride     : tl.constexpr,
    decay_layer_stride    : tl.constexpr,
    decay_token_stride    : tl.constexpr,
    state_slot_stride     : tl.constexpr,
    state_layer_stride    : tl.constexpr,
    state_head_stride     : tl.constexpr,
    num_layers            : tl.constexpr,
    num_key_heads         : tl.constexpr,
    num_value_heads       : tl.constexpr,
    head_k_dim            : tl.constexpr,
    head_v_dim            : tl.constexpr,
    BLOCK_K               : tl.constexpr,
    BLOCK_V               : tl.constexpr,
):
    tile = tl.program_id(0)
    group = tl.program_id(1)

    num_k_tiles = tl.cdiv(head_k_dim, BLOCK_K)
    k_tile = tile % num_k_tiles
    v_tile = tile // num_k_tiles
    task = group // (num_layers * num_value_heads)
    layer_head = group % (num_layers * num_value_heads)
    layer = layer_head // num_value_heads
    value_head = layer_head % num_value_heads
    key_head = value_head // (num_value_heads // num_key_heads)

    k_offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
    v_offsets = v_tile * BLOCK_V + tl.arange(0, BLOCK_V)
    state_mask = (
        (k_offsets[:, None] < head_k_dim)
        & (v_offsets[None, :] < head_v_dim)
    )

    base_slot = tl.load(base_slots + task)
    state_ptr = (
        recurrent_state
        + base_slot * state_slot_stride
        + layer * state_layer_stride
        + value_head * state_head_stride
        + k_offsets[:, None] * head_v_dim
        + v_offsets[None, :]
    )
    state = tl.load(
        state_ptr,
        mask=state_mask & (base_slot >= 0),
        other=0.0,
    ).to(tl.float32)

    log_index = tl.load(task_log_offsets + task)
    log_stop = tl.load(task_log_offsets + task + 1)
    while log_index < log_stop:
        slot = tl.load(log_slots + log_index)
        token = tl.load(log_begins + log_index)
        token_stop = tl.load(log_ends + log_index)
        while token < token_stop:
            decay = tl.load(
                decay_log
                + slot * decay_slot_stride
                + layer * decay_layer_stride
                + token * decay_token_stride
                + value_head
            ).to(tl.float32)
            key = tl.load(
                key_log
                + slot * key_slot_stride
                + layer * key_layer_stride
                + token * key_token_stride
                + key_head * key_head_stride
                + k_offsets,
                mask=k_offsets < head_k_dim,
                other=0.0,
            ).to(tl.float32)
            value = tl.load(
                value_log
                + slot * value_slot_stride
                + layer * value_layer_stride
                + token * value_token_stride
                + value_head * value_head_stride
                + v_offsets,
                mask=v_offsets < head_v_dim,
                other=0.0,
            ).to(tl.float32)
            state = state * decay + key[:, None] * value[None, :]
            token += 1
        log_index += 1

    output_slot = tl.load(output_slots + task)
    output_ptr = (
        recurrent_state
        + output_slot * state_slot_stride
        + layer * state_layer_stride
        + value_head * state_head_stride
        + k_offsets[:, None] * head_v_dim
        + v_offsets[None, :]
    )
    tl.store(output_ptr, state, mask=state_mask)


def reconstruct_log_states(
    key_log            : torch.Tensor,
    residual_value_log : torch.Tensor,
    decay_log          : torch.Tensor,
    recurrent_state    : torch.Tensor,
    conv_state         : torch.Tensor,
    output_slots       : torch.Tensor,
    base_slots         : torch.Tensor,
    task_log_offsets   : torch.Tensor,
    log_slots          : torch.Tensor,
    log_begins         : torch.Tensor,
    log_ends           : torch.Tensor,
    source_conv_slots  : torch.Tensor,
    source_conv_granules: torch.Tensor,
    output_conv_granules: torch.Tensor,
) -> None:
    """Reconstruct independent recurrent states from LogState intervals.

    ``log_begins`` and ``log_ends`` are offsets within the physical log slot.
    Every task writes a distinct output slot, allowing work-state restoration
    and persistent checkpoint materialization to share one launch.
    """
    num_tasks = len(output_slots)
    if num_tasks == 0:
        return

    assert key_log.is_cuda
    assert residual_value_log.is_cuda and decay_log.is_cuda
    assert recurrent_state.is_cuda and conv_state.is_cuda
    tensors = (
        output_slots,
        base_slots,
        task_log_offsets,
        log_slots,
        log_begins,
        log_ends,
        source_conv_slots,
        source_conv_granules,
        output_conv_granules,
    )
    assert all(tensor.is_cuda for tensor in tensors)
    assert all(tensor.dtype in (torch.int32, torch.int64) for tensor in tensors)
    assert len(base_slots) == num_tasks
    assert len(task_log_offsets) == num_tasks + 1
    assert len(source_conv_slots) == num_tasks
    assert len(source_conv_granules) == num_tasks
    assert len(output_conv_granules) == num_tasks
    assert len(log_slots) == len(log_begins) == len(log_ends)

    num_slots, num_layers, page_size, num_key_heads, head_k_dim = key_log.shape
    value_shape = residual_value_log.shape
    assert value_shape[:3] == (num_slots, num_layers, page_size)
    num_value_heads, head_v_dim = value_shape[3:]
    assert num_value_heads % num_key_heads == 0
    assert decay_log.shape == (
        num_slots, num_layers, page_size, num_value_heads
    )
    assert recurrent_state.shape == (
        num_slots, num_layers, num_value_heads, head_k_dim, head_v_dim
    )
    assert conv_state.shape[:2] == (num_slots, num_layers)

    block_k = min(16, triton.next_power_of_2(head_k_dim))
    block_v = min(16, triton.next_power_of_2(head_v_dim))
    grid = (
        triton.cdiv(head_k_dim, block_k) * triton.cdiv(head_v_dim, block_v),
        num_tasks * num_layers * num_value_heads,
    )
    _reconstruct_log_states_kernel[grid](
        key_log,
        residual_value_log,
        decay_log,
        recurrent_state,
        output_slots,
        base_slots,
        task_log_offsets,
        log_slots,
        log_begins,
        log_ends,
        key_slot_stride=key_log.stride(0),
        key_layer_stride=key_log.stride(1),
        key_token_stride=key_log.stride(2),
        key_head_stride=key_log.stride(3),
        value_slot_stride=residual_value_log.stride(0),
        value_layer_stride=residual_value_log.stride(1),
        value_token_stride=residual_value_log.stride(2),
        value_head_stride=residual_value_log.stride(3),
        decay_slot_stride=decay_log.stride(0),
        decay_layer_stride=decay_log.stride(1),
        decay_token_stride=decay_log.stride(2),
        state_slot_stride=recurrent_state.stride(0),
        state_layer_stride=recurrent_state.stride(1),
        state_head_stride=recurrent_state.stride(2),
        num_layers=num_layers,
        num_key_heads=num_key_heads,
        num_value_heads=num_value_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        num_warps=4,
    )

    layer_indices = torch.arange(
        num_layers, dtype=torch.int64, device=conv_state.device
    )
    source = conv_state[
        source_conv_slots[:, None],
        layer_indices[None, :],
        source_conv_granules[:, None],
    ]
    conv_state[
        output_slots[:, None],
        layer_indices[None, :],
        output_conv_granules[:, None],
    ] = source


__all__ = ["reconstruct_log_states"]
