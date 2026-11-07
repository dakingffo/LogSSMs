# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, Tri Dao.
# Adapted from vLLM causal_conv1d.py at 2cc2e68d1c5652a5bc7ae10422697308e2193359.
# LogSSMs changes: indexed frame reads, separate state destinations, and Q/K/V stores.
# License: third_party/licenses/vllm-Apache-2.0.txt

import torch
import triton
import triton.language as tl

from .packed_conv import _scale_conv_input, _conv_input_scale_args
from .layout import conv_pointer


@triton.jit
def _frame_element_offset(frame, frame_elems: tl.constexpr):
    return frame.to(tl.int64) * frame_elems


@triton.jit(
    do_not_specialize=['LAYER'],
    do_not_specialize_on_alignment=['FrameTable', 'ReadSlots', 'ReadGranules', 'WriteSlots', 'WriteGranules', 'WorkSlots'],
)
def _conv_decode_kernel(
    X, W, Bias, Q, K, V, StandardState, Storage, FrameTable,
    ReadSlots, ReadGranules, WriteSlots, WriteGranules, WorkSlots,
    C: tl.constexpr, KEY_DIM: tl.constexpr, STATE_LEN: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr, X_COL_STRIDE: tl.constexpr,
    STATE_SLOT_STRIDE: tl.constexpr, STATE_LAYER_STRIDE: tl.constexpr, LAYER,
    READ_STRIDE: tl.constexpr, WRITE_STRIDE: tl.constexpr,
    FRAME_ELEMS: tl.constexpr, NUM_GRANULES: tl.constexpr, CONV_FRAMES: tl.constexpr,
    HAS_BIAS: tl.constexpr, SPLIT_QKV: tl.constexpr, ROUND_CONV: tl.constexpr,
    LOGSTATE: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr,
    END0: tl.constexpr, END1: tl.constexpr,
    SCALE0: tl.constexpr, SCALE1: tl.constexpr, SCALE2: tl.constexpr,
):
    b = tl.program_id(0)
    ch = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)

    read_slot = tl.load(ReadSlots + b * READ_STRIDE)
    write_slot = tl.load(WriteSlots + b * WRITE_STRIDE)
    if LOGSTATE:
        read_granule = tl.load(ReadGranules + b)
        write_granule = tl.load(WriteGranules + b)
        work_slot = tl.load(WorkSlots + b)
        state_ptr = tl.cast(Storage, tl.pointer_type(X.dtype.element_ty))
        if CONV_FRAMES == 1:
            read_frame = tl.load(FrameTable + read_slot * NUM_GRANULES + read_granule)
            write_frame = tl.load(FrameTable + write_slot * NUM_GRANULES + write_granule)
            work_frame = tl.load(FrameTable + work_slot * NUM_GRANULES)
            old_base = _frame_element_offset(read_frame, FRAME_ELEMS) + LAYER * C * STATE_LEN
            new_base = _frame_element_offset(write_frame, FRAME_ELEMS) + LAYER * C * STATE_LEN
            work_base = _frame_element_offset(work_frame, FRAME_ELEMS) + LAYER * C * STATE_LEN
        else:
            old_base = 0
            new_base = 0
            work_base = 0
    else:
        state_ptr = StandardState
        old_base = read_slot * STATE_SLOT_STRIDE + LAYER * STATE_LAYER_STRIDE
        new_base = write_slot * STATE_SLOT_STRIDE + LAYER * STATE_LAYER_STRIDE
        work_base = 0

    KERNEL_WIDTH: tl.constexpr = STATE_LEN + 1
    # STEP 1: READ init_state data
    conv_states_base = state_ptr + old_base + ch * STATE_LEN
    mask_w = ch < C

    prior_tokens = conv_states_base
    if KERNEL_WIDTH >= 2:
        if LOGSTATE and CONV_FRAMES > 1:
            conv_states_ptrs = conv_pointer(state_ptr, FrameTable, read_slot, read_granule,
                                           LAYER * C * STATE_LEN + ch * STATE_LEN + 0,
                                           NUM_GRANULES, CONV_FRAMES, FRAME_ELEMS, ch < C)
        else:
            conv_states_ptrs = prior_tokens  # [BLOCK_C]
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        if LOGSTATE and CONV_FRAMES > 1:
            conv_states_ptrs = conv_pointer(state_ptr, FrameTable, read_slot, read_granule,
                                           LAYER * C * STATE_LEN + ch * STATE_LEN + 1,
                                           NUM_GRANULES, CONV_FRAMES, FRAME_ELEMS, ch < C)
        else:
            conv_states_ptrs = prior_tokens + 1  # [BLOCK_C]
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        if LOGSTATE and CONV_FRAMES > 1:
            conv_states_ptrs = conv_pointer(state_ptr, FrameTable, read_slot, read_granule,
                                           LAYER * C * STATE_LEN + ch * STATE_LEN + 2,
                                           NUM_GRANULES, CONV_FRAMES, FRAME_ELEMS, ch < C)
        else:
            conv_states_ptrs = prior_tokens + 2  # [BLOCK_C]
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 5:
        if LOGSTATE and CONV_FRAMES > 1:
            conv_states_ptrs = conv_pointer(state_ptr, FrameTable, read_slot, read_granule,
                                           LAYER * C * STATE_LEN + ch * STATE_LEN + 3,
                                           NUM_GRANULES, CONV_FRAMES, FRAME_ELEMS, ch < C)
        else:
            conv_states_ptrs = prior_tokens + 3  # [BLOCK_C]
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 6:
        if LOGSTATE and CONV_FRAMES > 1:
            conv_states_ptrs = conv_pointer(state_ptr, FrameTable, read_slot, read_granule,
                                           LAYER * C * STATE_LEN + ch * STATE_LEN + 4,
                                           NUM_GRANULES, CONV_FRAMES, FRAME_ELEMS, ch < C)
        else:
            conv_states_ptrs = prior_tokens + 4  # [BLOCK_C]
        col4 = tl.load(conv_states_ptrs, mask_w, 0.0)

    # Shift the convolution history by one token.
    idx_tokens = tl.arange(0, BLOCK_S)  # [BLOCK_M]

    conv_state_ptrs_source = conv_states_base[None, :] + (idx_tokens + 1)[:, None]
    mask = (
        ((idx_tokens + 1) < STATE_LEN)[:, None]
        & (ch < C)[None, :]
    )
    if LOGSTATE and CONV_FRAMES > 1:
        conv_state_ptrs_source = conv_pointer(state_ptr, FrameTable, read_slot, read_granule,
                                             LAYER * C * STATE_LEN + ch[None, :] * STATE_LEN + (idx_tokens + 1)[:, None],
                                             NUM_GRANULES, CONV_FRAMES, FRAME_ELEMS, mask)
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = STATE_LEN - 1
    x_base = X + (b * X_ROW_STRIDE) + (ch * X_COL_STRIDE)  # [BLOCK_C]

    x_ptrs = (
        tl.broadcast_to(x_base[None, :], (BLOCK_S, BLOCK_C))
    )  # [BLOCK_M, BLOCK_C]

    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < 1)[:, None]
        & (ch < C)[None, :]
    )  # token-index  # token-index  # feature-index
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    loaded_x = _scale_conv_input(loaded_x, ch[None, :], END0, END1, SCALE0, SCALE1, SCALE2)
    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    conv_state_ptrs_target = (
        state_ptr + new_base + ch[None, :] * STATE_LEN + idx_tokens[:, None]
    )
    mask = (idx_tokens < STATE_LEN)[:, None] & (ch < C)[None, :]
    if LOGSTATE and CONV_FRAMES > 1:
        conv_state_ptrs_target = conv_pointer(state_ptr, FrameTable, write_slot, write_granule,
                                             LAYER * C * STATE_LEN + ch[None, :] * STATE_LEN + idx_tokens[:, None],
                                             NUM_GRANULES, CONV_FRAMES, FRAME_ELEMS, mask)
    tl.store(conv_state_ptrs_target, new_conv_state, mask)
    if LOGSTATE:
        work_ptrs = state_ptr + work_base + ch[None, :] * STATE_LEN + idx_tokens[:, None]
        if CONV_FRAMES > 1:
            work_ptrs = conv_pointer(state_ptr, FrameTable, work_slot, 0,
                                     LAYER * C * STATE_LEN + ch[None, :] * STATE_LEN + idx_tokens[:, None],
                                     NUM_GRANULES, CONV_FRAMES, FRAME_ELEMS, mask)
        tl.store(work_ptrs, new_conv_state, mask)

    # STEP 3: init accumulator
    acc_preload = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # STEP 4:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_C features in range
    w_base = W + (ch * KERNEL_WIDTH)  # [BLOCK_C,]
    mask_w = ch < C
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + 0  # [BLOCK_C] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + 1  # [BLOCK_C] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + 2  # [BLOCK_C] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + 3  # [BLOCK_C] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 5:
        w_ptrs = w_base + 4  # [BLOCK_C] tensor
        w_col4 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 6:
        w_ptrs = w_base + 5  # [BLOCK_C] tensor
        w_col5 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base  # starting of chunk [BLOCK_C]
    mask_x_1d = ch < C
    current_x = tl.load(x_base_1d, mask_x_1d, other=0.)
    current_x = _scale_conv_input(current_x, ch, END0, END1, SCALE0, SCALE1, SCALE2)

    # STEP 5: compute each token

    for idx_token in tl.range(1):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    matrix_x = current_x
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = current_x
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = current_x
            elif KERNEL_WIDTH == 5:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    matrix_x = current_x
            elif KERNEL_WIDTH == 6:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    matrix_x = col4
                elif j == 5:
                    matrix_w = w_col5
                    matrix_x = current_x

            acc += matrix_x * matrix_w  # [BLOCK_C]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x
        elif KERNEL_WIDTH == 5:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = matrix_x
        elif KERNEL_WIDTH == 6:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = col4
            col4 = matrix_x

        if HAS_BIAS:
            acc += tl.load(Bias + ch, ch < C, other=0).to(tl.float32)
        if ROUND_CONV:
            acc = acc.to(X.dtype.element_ty).to(tl.float32)
        acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < 1) & (
            ch < C
        )  # token-index  # feature-index
        if SPLIT_QKV:
            tl.store(Q + b * KEY_DIM + ch, acc, mask_1d & (ch < KEY_DIM))
            tl.store(K + b * KEY_DIM + ch - KEY_DIM, acc,
                     mask_1d & (ch >= KEY_DIM) & (ch < 2 * KEY_DIM))
            tl.store(V + b * (C - 2 * KEY_DIM) + ch - 2 * KEY_DIM, acc,
                     mask_1d & (ch >= 2 * KEY_DIM))
        else:
            tl.store(Q + b * C + ch, acc, mask_1d)



def conv_decode(
    mixed_qkv: torch.Tensor,
    weight: torch.Tensor,
    config,
    *,
    bias: torch.Tensor | None = None,
    split_qkv: bool = False,
    round_conv: bool = False,
    read_slots: torch.Tensor,
    write_slots: torch.Tensor,
    standard_state: torch.Tensor | None = None,
    pool=None,
    read_granules: torch.Tensor | None = None,
    write_granules: torch.Tensor | None = None,
    work_slots: torch.Tensor | None = None,
    input_boundaries: tuple[int, int] | None = None,
    input_scales: tuple[float, float, float] = (1., 1., 1.),
):
    assert mixed_qkv.is_cuda and mixed_qkv.ndim == 2
    assert 1 <= config.state_len <= 5
    bs, channels = mixed_qkv.shape
    assert channels == config.conv_dim
    if split_qkv:
        q = torch.empty((bs, 1, config.num_k_heads, config.head_k_dim),
                        dtype=mixed_qkv.dtype, device=mixed_qkv.device)
        k = torch.empty_like(q)
        v = torch.empty((bs, 1, config.num_v_heads, config.head_v_dim),
                        dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    else:
        q = torch.empty((bs, channels), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
        k = v = q
    logstate = pool is not None
    if logstate:
        assert read_granules is not None and write_granules is not None
        assert work_slots is not None
        state = pool.storage
        table = pool.conv_frame_table
        frame_elems = pool.frame_bytes // mixed_qkv.element_size()
        num_granules = pool.layout.num_granules
        slot_stride = layer_stride = 0
    else:
        assert standard_state is not None
        state = standard_state
        table = standard_state
        read_granules = write_granules = work_slots = read_slots
        frame_elems = num_granules = 1
        slot_stride, layer_stride = standard_state.stride()[:2]
    _conv_decode_kernel[(bs, triton.cdiv(channels, 256))](
        mixed_qkv, weight, bias, q, k, v, state, state, table,
        read_slots, read_granules, write_slots, write_granules, work_slots,
        channels, config.key_dim if split_qkv else 0, config.state_len,
        mixed_qkv.stride(0), mixed_qkv.stride(1),
        slot_stride, layer_stride, config.layer_idx,
        read_slots.stride(0), write_slots.stride(0), frame_elems, num_granules,
        pool.allocator.conv_frames if logstate else 1,
        bias is not None, split_qkv, round_conv, logstate, 256, triton.next_power_of_2(config.state_len),
        *_conv_input_scale_args(channels, input_boundaries, input_scales),
        num_warps=4,
    )
    return (q, k, v) if split_qkv else q
