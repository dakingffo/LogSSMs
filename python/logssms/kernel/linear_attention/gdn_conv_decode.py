# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, Tri Dao.
# Adapted from vLLM causal_conv1d.py at 2cc2e68d1c5652a5bc7ae10422697308e2193359.
# LogSSMs changes: indexed frame reads, separate state destinations, and Q/K/V stores.
# License: third_party/licenses/vllm-Apache-2.0.txt

import torch
import triton
import triton.language as tl


@triton.jit
def _frame_element_offset(frame, frame_elems: tl.constexpr):
    return frame.to(tl.int64) * frame_elems


@triton.jit
def _gdn_conv_decode(
    X, W, Q, K, V, StandardState, Storage, FrameTable,
    ReadSlots, ReadGranules, WriteSlots, WriteGranules, WorkSlots,
    C: tl.constexpr, KEY_DIM: tl.constexpr, STATE_LEN: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr, X_COL_STRIDE: tl.constexpr,
    NUM_LAYERS: tl.constexpr, LAYER: tl.constexpr,
    FRAME_ELEMS: tl.constexpr, NUM_GRANULES: tl.constexpr,
    LOGSTATE: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    ch = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)

    read_slot = tl.load(ReadSlots + b)
    write_slot = tl.load(WriteSlots + b)
    if LOGSTATE:
        read_granule = tl.load(ReadGranules + b)
        write_granule = tl.load(WriteGranules + b)
        work_slot = tl.load(WorkSlots + b)
        read_frame = tl.load(
            FrameTable + read_slot * NUM_GRANULES + read_granule
        )
        write_frame = tl.load(
            FrameTable + write_slot * NUM_GRANULES + write_granule
        )
        work_frame = tl.load(FrameTable + work_slot * NUM_GRANULES)
        state_ptr = tl.cast(Storage, tl.pointer_type(X.dtype.element_ty))
        old_base = _frame_element_offset(read_frame, FRAME_ELEMS) + LAYER * C * STATE_LEN
        new_base = _frame_element_offset(write_frame, FRAME_ELEMS) + LAYER * C * STATE_LEN
        work_base = _frame_element_offset(work_frame, FRAME_ELEMS) + LAYER * C * STATE_LEN
    else:
        state_ptr = StandardState
        old_base = (read_slot * NUM_LAYERS + LAYER) * C * STATE_LEN
        new_base = (write_slot * NUM_LAYERS + LAYER) * C * STATE_LEN
        work_base = 0

    KERNEL_WIDTH: tl.constexpr = STATE_LEN + 1
    # STEP 1: READ init_state data
    conv_states_base = state_ptr + old_base + ch * STATE_LEN
    mask_w = ch < C

    prior_tokens = conv_states_base
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens  # [BLOCK_C]
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1  # [BLOCK_C]
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2  # [BLOCK_C]
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 5:
        conv_states_ptrs = prior_tokens + 3  # [BLOCK_C]
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 6:
        conv_states_ptrs = prior_tokens + 4  # [BLOCK_C]
        col4 = tl.load(conv_states_ptrs, mask_w, 0.0)

    # Shift the convolution history by one token.
    idx_tokens = tl.arange(0, BLOCK_S)  # [BLOCK_M]

    conv_state_ptrs_source = conv_states_base[None, :] + (idx_tokens + 1)[:, None]
    mask = (
        ((idx_tokens + 1) < STATE_LEN)[:, None]
        & (ch < C)[None, :]
    )
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
    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    conv_state_ptrs_target = (
        state_ptr + new_base + ch[None, :] * STATE_LEN + idx_tokens[:, None]
    )
    mask = (idx_tokens < STATE_LEN)[:, None] & (ch < C)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)
    if LOGSTATE:
        work_ptrs = state_ptr + work_base + ch[None, :] * STATE_LEN + idx_tokens[:, None]
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

    # STEP 5: compute each token

    for idx_token in tl.range(1):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d  # [BLOCK_C]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d  # [BLOCK_C]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d  # [BLOCK_C]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
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
                    x_ptrs_1d = x_base_1d  # [BLOCK_C]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
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
                    x_ptrs_1d = x_base_1d  # [BLOCK_C]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

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

        acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < 1) & (
            ch < C
        )  # token-index  # feature-index
        tl.store(Q + b * KEY_DIM + ch, acc, mask_1d & (ch < KEY_DIM))
        tl.store(K + b * KEY_DIM + ch - KEY_DIM, acc,
                 mask_1d & (ch >= KEY_DIM) & (ch < 2 * KEY_DIM))
        tl.store(V + b * (C - 2 * KEY_DIM) + ch - 2 * KEY_DIM, acc,
                 mask_1d & (ch >= 2 * KEY_DIM))


def conv_decode_qkv(
    mixed_qkv: torch.Tensor,
    weight: torch.Tensor,
    config,
    *,
    read_slots: torch.Tensor,
    write_slots: torch.Tensor,
    standard_state: torch.Tensor | None = None,
    pool=None,
    read_granules: torch.Tensor | None = None,
    write_granules: torch.Tensor | None = None,
    work_slots: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mixed_qkv.is_cuda and mixed_qkv.ndim == 2
    assert 1 <= config.state_len <= 5
    bs, channels = mixed_qkv.shape
    assert channels == config.conv_dim
    q = torch.empty((bs, 1, config.num_k_heads, config.head_k_dim),
                    dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    k = torch.empty_like(q)
    v = torch.empty((bs, 1, config.num_v_heads, config.head_v_dim),
                    dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    logstate = pool is not None
    if logstate:
        assert read_granules is not None and write_granules is not None
        assert work_slots is not None and pool.allocator.conv_frames == 1
        state = pool.storage
        table = pool.conv_frame_table
        frame_elems = pool.frame_bytes // mixed_qkv.element_size()
        num_granules = pool.layout.num_granules
        num_layers = pool.layout.num_linear_layers
    else:
        assert standard_state is not None
        state = standard_state
        table = standard_state
        read_granules = write_granules = work_slots = read_slots
        frame_elems = num_granules = 1
        num_layers = standard_state.shape[1]
    _gdn_conv_decode[(bs, triton.cdiv(channels, 256))](
        mixed_qkv, weight, q, k, v, state, state, table,
        read_slots, read_granules, write_slots, write_granules, work_slots,
        channels, config.key_dim, config.state_len,
        mixed_qkv.stride(0), mixed_qkv.stride(1),
        num_layers, config.layer_idx, frame_elems, num_granules,
        logstate, 256, triton.next_power_of_2(config.state_len),
        num_warps=4,
    )
    return q, k, v
