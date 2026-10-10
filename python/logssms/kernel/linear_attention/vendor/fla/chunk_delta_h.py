# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .index import prepare_chunk_indices, prepare_chunk_offsets
from .op import exp2
from .runtime import IS_NVIDIA_BLACKWELL


@dataclass(frozen=True)
class ChunkStateIO:
    """Persistent state addresses; -1 means a zero input or an omitted chunk write."""

    storage           : torch.Tensor
    read_slots        : torch.Tensor
    chunk_write_slots : torch.Tensor
    final_write_slots : torch.Tensor | None = None
    payload_frames    : torch.Tensor | None = None
    frame_bytes       : int = 0
    head_payload_bytes: int = 0
    layer             : int = 0


@dataclass(frozen=True)
class ChunkLogIO:
    """Framed log destinations; gate contains per-token, not cumulative, log-decay."""

    slots             : torch.Tensor
    token_offsets     : torch.Tensor
    decay_frames      : torch.Tensor
    gate              : torch.Tensor
    page_size         : int
    decay_record_bytes: int


@triton.jit
def _chunk_state_pointer(
    storage, payload_frames, slot, head,
    SLOT_STRIDE: tl.constexpr, LAYER_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr, LAYER: tl.constexpr,
    HV: tl.constexpr, FRAMED: tl.constexpr,
    FRAME_BYTES: tl.constexpr, FRAMES_PER_SLOT: tl.constexpr,
    HEAD_PAYLOAD_BYTES: tl.constexpr,
):
    if FRAMED:
        head_bytes = (LAYER * HV + head) * HEAD_PAYLOAD_BYTES
        frame = tl.load(
            payload_frames + slot * FRAMES_PER_SLOT + head_bytes // FRAME_BYTES,
            mask=slot >= 0, other=0,
        ).to(tl.int64)
        address = storage.to(tl.pointer_type(tl.uint8))
        address += frame * FRAME_BYTES + head_bytes % FRAME_BYTES
        return address.to(storage.dtype)
    return storage + slot.to(tl.int64) * SLOT_STRIDE + LAYER * LAYER_STRIDE + head * HEAD_STRIDE


@triton.jit
def _store_chunk_state_tile(
    destination, state, slot, key_offsets, value_offsets,
    K: tl.constexpr, V: tl.constexpr,
):
    offsets = key_offsets[:, None] * V + value_offsets[None, :]
    mask = (key_offsets[:, None] < K) & (value_offsets[None, :] < V)
    tl.store(destination + offsets, state.to(destination.dtype.element_ty), mask=mask & (slot >= 0))


@triton.jit
def _store_chunk_log(
    storage, payload_frames, decay_frames, slots, token_offsets, gate, key,
    residual, rows, token_mask, head, value_offsets, value_tile,
    K: tl.constexpr, V: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
    LAYER: tl.constexpr, PAGE_SIZE: tl.constexpr, FRAME_BYTES: tl.constexpr,
    FRAMES_PER_SLOT: tl.constexpr, HEAD_PAYLOAD_BYTES: tl.constexpr,
    DECAY_RECORD_BYTES: tl.constexpr, BK: tl.constexpr,
):
    slot = tl.load(slots + rows, token_mask, other=-1)
    position = tl.load(token_offsets + rows, token_mask, other=0).to(tl.int64)
    active = token_mask & (slot >= 0)
    destination = _chunk_state_pointer(
        storage, payload_frames, slot, head,
        0, 0, 0, LAYER, HV, True, FRAME_BYTES, FRAMES_PER_SLOT, HEAD_PAYLOAD_BYTES,
    )
    tl.store(
        destination[:, None] + PAGE_SIZE * K + position[:, None] * V + value_offsets[None, :],
        residual, active[:, None] & (value_offsets[None, :] < V),
    )
    # V tiles partition the residual output; K and scalar decay have one writer.
    if value_tile == 0:
        columns = tl.arange(0, BK)
        keys = tl.load(
            key + rows[:, None].to(tl.int64) * H * K + (head // (HV // H)) * K + columns[None, :],
            token_mask[:, None] & (columns[None, :] < K), other=0,
        )
        tl.store(destination[:, None] + position[:, None] * K + columns[None, :],
                 keys, active[:, None] & (columns[None, :] < K))
        frame = tl.load(decay_frames + slot * 2, active, other=0).to(tl.int64)
        record = tl.load(decay_frames + slot * 2 + 1, active, other=0).to(tl.int64)
        decay = storage.to(tl.pointer_type(tl.uint8)) + frame * FRAME_BYTES + record * DECAY_RECORD_BYTES
        decay = decay.to(tl.pointer_type(tl.float32)) + (LAYER * PAGE_SIZE + position) * HV + head
        values = tl.load(gate + rows.to(tl.int64) * HV + head, token_mask, other=0)
        tl.store(decay, values, active)


@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    h,
    state_storage,
    read_slots,
    chunk_write_slots,
    final_write_slots,
    payload_frames,
    log_slots,
    log_token_offsets,
    decay_frames,
    log_gate,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    FRAMED_STATE: tl.constexpr,
    STATE_SLOT_STRIDE: tl.constexpr,
    STATE_LAYER_STRIDE: tl.constexpr,
    STATE_HEAD_STRIDE: tl.constexpr,
    LAYER: tl.constexpr,
    FRAME_BYTES: tl.constexpr,
    FRAMES_PER_SLOT: tl.constexpr,
    HEAD_PAYLOAD_BYTES: tl.constexpr,
    STORE_WORK_STATE: tl.constexpr,
    STORE_LOG: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DECAY_RECORD_BYTES: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    NT = tl.cdiv(T, BT)
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * HV + i_h).to(tl.int64) * K*V
    v += (bos * HV + i_h).to(tl.int64) * V
    original_k = k
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    v_new += (bos * HV + i_h).to(tl.int64) * V

    read_slot = tl.load(read_slots + i_n)
    h0 = _chunk_state_pointer(
        state_storage, payload_frames, read_slot, i_h,
        STATE_SLOT_STRIDE, STATE_LAYER_STRIDE, STATE_HEAD_STRIDE, LAYER,
        HV, FRAMED_STATE, FRAME_BYTES, FRAMES_PER_SLOT, HEAD_PAYLOAD_BYTES,
    )

    # load initial state
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K
    p_h0_1 = h0 + o_k1[:, None] * V + o_v[None, :]
    m_h0_1 = m_k1[:, None] & m_v[None, :]
    m_h0_1 &= read_slot >= 0
    b_h1 += tl.load(p_h0_1, mask=m_h0_1, other=0.0).to(tl.float32)
    if K > 64:
        p_h0_2 = h0 + o_k2[:, None] * V + o_v[None, :]
        m_h0_2 = m_k2[:, None] & m_v[None, :]
        m_h0_2 &= read_slot >= 0
        b_h2 += tl.load(p_h0_2, mask=m_h0_2, other=0.0).to(tl.float32)
    if K > 128:
        p_h0_3 = h0 + o_k3[:, None] * V + o_v[None, :]
        m_h0_3 = m_k3[:, None] & m_v[None, :]
        m_h0_3 &= read_slot >= 0
        b_h3 += tl.load(p_h0_3, mask=m_h0_3, other=0.0).to(tl.float32)
    if K > 192:
        p_h0_4 = h0 + o_k4[:, None] * V + o_v[None, :]
        m_h0_4 = m_k4[:, None] & m_v[None, :]
        m_h0_4 &= read_slot >= 0
        b_h4 += tl.load(p_h0_4, mask=m_h0_4, other=0.0).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        p_h1 = h + i_t_int64 * HV*K*V + o_k1[:, None] * V + o_v[None, :]
        m_h1 = m_k1[:, None] & m_v[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_h1)
        if K > 64:
            p_h2 = h + i_t_int64 * HV*K*V + o_k2[:, None] * V + o_v[None, :]
            m_h2 = m_k2[:, None] & m_v[None, :]
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_h2)
        if K > 128:
            p_h3 = h + i_t_int64 * HV*K*V + o_k3[:, None] * V + o_v[None, :]
            m_h3 = m_k3[:, None] & m_v[None, :]
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_h3)
        if K > 192:
            p_h4 = h + i_t_int64 * HV*K*V + o_k4[:, None] * V + o_v[None, :]
            m_h4 = m_k4[:, None] & m_v[None, :]
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_h4)

        p_w = w + o_t[:, None] * (HV*K) + o_k1[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = w + o_t[:, None] * (HV*K) + o_k2[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = w + o_t[:, None] * (HV*K) + o_k3[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = w + o_t[:, None] * (HV*K) + o_k4[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = v + o_t[:, None] * (HV*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0) - b_v

        p_v = v_new + o_t[:, None] * (HV*V) + o_v[None, :]
        tl.store(p_v, b_v.to(p_v.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])

        if STORE_LOG:
            _store_chunk_log(
                state_storage, payload_frames, decay_frames, log_slots,
                log_token_offsets, log_gate, original_k,
                b_v.to(k.dtype.element_ty), bos + o_t, m_t, i_h, o_v, i_v,
                K, V, H, HV, LAYER, PAGE_SIZE, FRAME_BYTES,
                FRAMES_PER_SLOT, HEAD_PAYLOAD_BYTES, DECAY_RECORD_BYTES,
                triton.next_power_of_2(K),
            )

        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
        p_g = g + (bos * HV + i_h).to(tl.int64) + o_t * HV
        b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
        b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
        b_g_last = exp2(b_g_last)
        b_h1 *= b_g_last
        if K > 64:
            b_h2 *= b_g_last
        if K > 128:
            b_h3 *= b_g_last
        if K > 192:
            b_h4 *= b_g_last

        b_v = b_v.to(k.dtype.element_ty)

        p_k = k + o_k1[:, None] + o_t[None, :] * (H*K)
        b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = k + o_k2[:, None] + o_t[None, :] * (H*K)
            b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = k + o_k3[:, None] + o_t[None, :] * (H*K)
            b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = k + o_k4[:, None] + o_t[None, :] * (H*K)
            b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            b_h4 += tl.dot(b_k, b_v)

        write_slot = tl.load(chunk_write_slots + boh + i_t)
        if write_slot >= 0:
            destination = _chunk_state_pointer(
                state_storage, payload_frames, write_slot, i_h,
                STATE_SLOT_STRIDE, STATE_LAYER_STRIDE, STATE_HEAD_STRIDE, LAYER,
                HV, FRAMED_STATE, FRAME_BYTES, FRAMES_PER_SLOT, HEAD_PAYLOAD_BYTES,
            )
            _store_chunk_state_tile(destination, b_h1, write_slot, o_k1, o_v, K, V)
            if K > 64:
                _store_chunk_state_tile(destination, b_h2, write_slot, o_k2, o_v, K, V)
            if K > 128:
                _store_chunk_state_tile(destination, b_h3, write_slot, o_k3, o_v, K, V)
            if K > 192:
                _store_chunk_state_tile(destination, b_h4, write_slot, o_k4, o_v, K, V)

    if STORE_WORK_STATE:
        write_slot = tl.load(final_write_slots + i_n)
        destination = _chunk_state_pointer(
            state_storage, payload_frames, write_slot, i_h,
            STATE_SLOT_STRIDE, STATE_LAYER_STRIDE, STATE_HEAD_STRIDE, LAYER,
            HV, FRAMED_STATE, FRAME_BYTES, FRAMES_PER_SLOT, HEAD_PAYLOAD_BYTES,
        )
        _store_chunk_state_tile(destination, b_h1, write_slot, o_k1, o_v, K, V)
        if K > 64:
            _store_chunk_state_tile(destination, b_h2, write_slot, o_k2, o_v, K, V)
        if K > 128:
            _store_chunk_state_tile(destination, b_h3, write_slot, o_k3, o_v, K, V)
        if K > 192:
            _store_chunk_state_tile(destination, b_h4, write_slot, o_k4, o_v, K, V)



def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    state_io: ChunkStateIO,
    cu_seqlens: torch.LongTensor,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    chunk_offsets: torch.LongTensor | None = None,
    log_io: ChunkLogIO | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    BT = chunk_size
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    N, NT = len(cu_seqlens) - 1, len(chunk_indices)
    if chunk_offsets is None:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."
    h = k.new_empty(B, NT, HV, K, V)
    v_new = torch.empty_like(u)
    framed = state_io.payload_frames is not None
    assert state_io.read_slots.shape == (N,)
    assert state_io.chunk_write_slots.shape == (B * NT,)
    assert state_io.storage.device == k.device
    if state_io.final_write_slots is not None:
        assert state_io.final_write_slots.shape == (N,)
    if framed:
        assert state_io.storage.ndim == 2
        assert state_io.frame_bytes > 0 and state_io.head_payload_bytes > 0
    else:
        assert state_io.storage.ndim == 5
        assert state_io.storage.shape[2:] == (HV, K, V)
    if log_io is not None:
        assert framed
        assert log_io.slots.shape == log_io.token_offsets.shape == (B * T,)
        assert log_io.gate.numel() == B * T * HV
    # Never autotune an in-place work update against its own previous output.
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[(triton.cdiv(V, 32), N * HV)](
        k=k, v=u, w=w, v_new=v_new, g=g, h=h,
        state_storage=state_io.storage,
        read_slots=state_io.read_slots,
        chunk_write_slots=state_io.chunk_write_slots,
        final_write_slots=state_io.final_write_slots,
        payload_frames=state_io.payload_frames if framed else None,
        log_slots=log_io.slots if log_io is not None else None,
        log_token_offsets=log_io.token_offsets if log_io is not None else None,
        decay_frames=log_io.decay_frames if log_io is not None else None,
        log_gate=log_io.gate if log_io is not None else None,
        cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets,
        T=T, H=H, HV=HV, K=K, V=V, BT=BT, BV=32,
        FRAMED_STATE=framed,
        STATE_SLOT_STRIDE=state_io.storage.stride(0) if not framed else 0,
        STATE_LAYER_STRIDE=state_io.storage.stride(1) if not framed else 0,
        STATE_HEAD_STRIDE=state_io.storage.stride(2) if not framed else 0,
        LAYER=state_io.layer,
        FRAME_BYTES=state_io.frame_bytes if framed else 0,
        FRAMES_PER_SLOT=state_io.payload_frames.shape[1] if framed else 0,
        HEAD_PAYLOAD_BYTES=state_io.head_payload_bytes if framed else 0,
        STORE_WORK_STATE=state_io.final_write_slots is not None,
        STORE_LOG=log_io is not None,
        PAGE_SIZE=log_io.page_size if log_io is not None else 0,
        DECAY_RECORD_BYTES=log_io.decay_record_bytes if log_io is not None else 0,
        num_warps=2 if IS_NVIDIA_BLACKWELL else 4, num_stages=2,
    )
    return h, v_new
