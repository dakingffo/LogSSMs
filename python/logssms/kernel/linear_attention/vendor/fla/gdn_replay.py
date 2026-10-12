import triton
import triton.language as tl

from .fused_recurrent import _frame_byte_offset, _softplus
from .op import exp
from ...layout import log_pointer, payload_pointer


@triton.jit
def _reconstruct_tile(
    storage, payload_frames, work_slot, log_slot, value_head, scaled_values, prior_offsets, valid,
    k_offset, v_offset, total_decay,
    HEAD_K_DIM: tl.constexpr, HEAD_V_DIM: tl.constexpr,
    LAYER: tl.constexpr, LAYERS: tl.constexpr, NUM_V_HEADS: tl.constexpr,
    GRANULARITY: tl.constexpr, PAGE_SIZE: tl.constexpr, HEAD_PAYLOAD_BYTES: tl.constexpr,
    FRAME_BYTES: tl.constexpr, FRAMES_PER_SLOT: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
):
    state = tl.load(
        payload_pointer(
            storage, payload_frames, work_slot, LAYER, value_head,
            k_offset[:, None] * HEAD_V_DIM + v_offset[None, :],
            LAYERS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, GRANULARITY, PAGE_SIZE,
            HEAD_PAYLOAD_BYTES, FRAME_BYTES, FRAMES_PER_SLOT,
        ),
        mask=(k_offset[:, None] < HEAD_K_DIM) & (v_offset[None, :] < HEAD_V_DIM),
        other=0.0,
    ).to(tl.float32)
    keys = tl.load(
        log_pointer(
            storage, payload_frames, log_slot, LAYER, value_head,
            prior_offsets[None, :], k_offset[:, None],
            LAYERS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, GRANULARITY, PAGE_SIZE,
            HEAD_PAYLOAD_BYTES, FRAME_BYTES, FRAMES_PER_SLOT, False, NUM_K_HEADS,
        ),
        mask=(k_offset[:, None] < HEAD_K_DIM) & valid[None, :], other=0.0,
    )
    return state * total_decay + tl.dot(keys, scaled_values)


@triton.jit
def _gdn_replay_kernel(
    query, key, value, gate, beta, A_log, dt_bias, storage, output,
    work_slots, log_slots, token_offsets, replay_counts, payload_frames, decay_frames,
    GATE_BATCH_STRIDE: tl.constexpr, GATE_HEAD_STRIDE: tl.constexpr,
    BETA_BATCH_STRIDE: tl.constexpr, BETA_HEAD_STRIDE: tl.constexpr,
    NUM_K_HEADS: tl.constexpr, NUM_V_HEADS: tl.constexpr,
    HEAD_K_DIM: tl.constexpr, HEAD_V_DIM: tl.constexpr,
    FULL_K: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    FRAME_BYTES: tl.constexpr, FRAMES_PER_SLOT: tl.constexpr,
    HEAD_PAYLOAD_BYTES: tl.constexpr, DECAY_RECORD_BYTES: tl.constexpr,
    LAYER: tl.constexpr, PAGE_SIZE: tl.constexpr, REPLAY_WINDOW: tl.constexpr,
    LAYERS: tl.constexpr, GRANULARITY: tl.constexpr,
):
    value_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = batch_head // NUM_V_HEADS
    value_head = batch_head % NUM_V_HEADS
    key_head = value_head // (NUM_V_HEADS // NUM_K_HEADS)
    v_offset = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_offset < HEAD_V_DIM
    work_slot = tl.load(work_slots + batch_idx)
    log_slot = tl.load(log_slots + batch_idx)
    payload = storage.to(tl.pointer_type(query.dtype.element_ty))
    decay_frame = tl.load(decay_frames + log_slot * 2)
    decay_index = tl.load(decay_frames + log_slot * 2 + 1)
    decay_head = (
        storage + _frame_byte_offset(decay_frame, FRAME_BYTES)
        + decay_index * DECAY_RECORD_BYTES
    ).to(tl.pointer_type(tl.float32))
    position = tl.load(token_offsets + batch_idx)
    count = tl.load(replay_counts + batch_idx)
    replay_index = tl.arange(0, REPLAY_WINDOW)
    prior_offsets = position - count + replay_index
    valid = replay_index < count
    log_decays = tl.load(
        decay_head + (LAYER * PAGE_SIZE + prior_offsets) * NUM_V_HEADS + value_head,
        mask=valid, other=0.0,
    )
    total_log_decay = tl.sum(log_decays)
    total_decay = tl.exp(total_log_decay)
    suffix_decay = tl.exp(total_log_decay - tl.cumsum(log_decays))
    logged_values = tl.load(
        log_pointer(
            payload, payload_frames, log_slot, LAYER, value_head,
            prior_offsets[:, None], v_offset[None, :],
            LAYERS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, GRANULARITY, PAGE_SIZE,
            HEAD_PAYLOAD_BYTES, FRAME_BYTES, FRAMES_PER_SLOT, True, NUM_K_HEADS,
        ),
        mask=valid[:, None] & v_mask[None, :], other=0.0,
    )
    scaled_values = (
        logged_values.to(tl.float32) * suffix_decay[:, None]
    ).to(query.dtype.element_ty)

    q_ptr = query + (batch_idx * NUM_K_HEADS + key_head) * HEAD_K_DIM
    k_ptr = key + (batch_idx * NUM_K_HEADS + key_head) * HEAD_K_DIM
    full_k = tl.arange(0, FULL_K)
    q_full = tl.load(q_ptr + full_k, mask=full_k < HEAD_K_DIM, other=0.0).to(tl.float32)
    k_full = tl.load(k_ptr + full_k, mask=full_k < HEAD_K_DIM, other=0.0).to(tl.float32)
    q_norm = tl.rsqrt(tl.sum(q_full * q_full) + 1e-6) * HEAD_K_DIM ** -0.5
    k_norm = tl.rsqrt(tl.sum(k_full * k_full) + 1e-6)
    A = tl.load(A_log + value_head).to(tl.float32)
    bias = tl.load(dt_bias + value_head).to(tl.float32)
    a = tl.load(gate + batch_idx * GATE_BATCH_STRIDE + value_head * GATE_HEAD_STRIDE)
    b = tl.load(beta + batch_idx * BETA_BATCH_STRIDE + value_head * BETA_HEAD_STRIDE)
    log_decay = -exp(A) * _softplus(a.to(tl.float32) + bias)
    decay = exp(log_decay)
    beta_value = tl.sigmoid(b.to(tl.float32))
    state_q = tl.zeros((BLOCK_V,), tl.float32)
    state_k = tl.zeros((BLOCK_V,), tl.float32)
    kq = tl.full((), 0.0, tl.float32)

    for block in range(triton.cdiv(HEAD_K_DIM, BLOCK_K)):
        k_offset = block * BLOCK_K + tl.arange(0, BLOCK_K)
        q = tl.load(q_ptr + k_offset, mask=k_offset < HEAD_K_DIM, other=0.0).to(tl.float32) * q_norm
        k = tl.load(k_ptr + k_offset, mask=k_offset < HEAD_K_DIM, other=0.0).to(tl.float32) * k_norm
        state = _reconstruct_tile(
            payload, payload_frames, work_slot, log_slot, value_head, scaled_values, prior_offsets, valid,
            k_offset, v_offset, total_decay, HEAD_K_DIM, HEAD_V_DIM,
            LAYER, LAYERS, NUM_V_HEADS, GRANULARITY, PAGE_SIZE, HEAD_PAYLOAD_BYTES,
            FRAME_BYTES, FRAMES_PER_SLOT,
            NUM_K_HEADS,
        )
        state_q += tl.sum(state * q[:, None], axis=0)
        state_k += tl.sum(state * k[:, None], axis=0)
        kq += tl.sum(k * q)


        if value_block == 0 and value_head % (NUM_V_HEADS // NUM_K_HEADS) == 0:
            tl.store(log_pointer(
                payload, payload_frames, log_slot, LAYER, value_head, position, k_offset,
                LAYERS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, GRANULARITY, PAGE_SIZE,
                HEAD_PAYLOAD_BYTES, FRAME_BYTES, FRAMES_PER_SLOT, False, NUM_K_HEADS,
            ), k, mask=k_offset < HEAD_K_DIM)

    v = tl.load(value + batch_head * HEAD_V_DIM + v_offset, mask=v_mask, other=0.0).to(tl.float32)
    delta = (v - state_k * decay) * beta_value
    result = state_q * decay + delta * kq
    tl.store(output + batch_head * HEAD_V_DIM + v_offset, result, mask=v_mask)
    tl.store(
        log_pointer(
            payload, payload_frames, log_slot, LAYER, value_head, position, v_offset,
            LAYERS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, GRANULARITY, PAGE_SIZE,
            HEAD_PAYLOAD_BYTES, FRAME_BYTES, FRAMES_PER_SLOT, True, NUM_K_HEADS,
        ),
        delta, mask=v_mask,
    )
    if value_block == 0:
        tl.store(decay_head + (LAYER * PAGE_SIZE + position) * NUM_V_HEADS + value_head, log_decay)

    if (position + 1) % REPLAY_WINDOW == 0:


        for block in range(triton.cdiv(HEAD_K_DIM, BLOCK_K)):
            k_offset = block * BLOCK_K + tl.arange(0, BLOCK_K)
            k = tl.load(k_ptr + k_offset, mask=k_offset < HEAD_K_DIM, other=0.0).to(tl.float32) * k_norm
            state = _reconstruct_tile(
                payload, payload_frames, work_slot, log_slot, value_head, scaled_values, prior_offsets, valid,
                k_offset, v_offset, total_decay, HEAD_K_DIM, HEAD_V_DIM,
                LAYER, LAYERS, NUM_V_HEADS, GRANULARITY, PAGE_SIZE, HEAD_PAYLOAD_BYTES,
                FRAME_BYTES, FRAMES_PER_SLOT,
                NUM_K_HEADS,
            )
            updated = state * decay + k[:, None] * delta[None, :]
            tl.store(
                payload_pointer(
                    payload, payload_frames, work_slot, LAYER, value_head,
                    k_offset[:, None] * HEAD_V_DIM + v_offset[None, :],
                    LAYERS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, GRANULARITY, PAGE_SIZE,
                    HEAD_PAYLOAD_BYTES, FRAME_BYTES, FRAMES_PER_SLOT,
                ),
                updated,
                mask=(k_offset[:, None] < HEAD_K_DIM) & v_mask[None, :],
            )
