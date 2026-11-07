from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from logssms.cache.logstate.layout import LOG_STATE_REPLAY_WINDOW
from .layout import log_pointer, state_pointer
from .replay import reconstruct_tile
from .tuning import decode_config, prefill_state_config
from .state_io import (
    ChunkStateIO, ChunkLogIO, _chunk_state_pointer, _store_chunk_state_tile, _store_chunk_log,
)


@triton.jit(do_not_specialize=["ROWS", "DS0"])
def _mamba2_prepare_kernel(
    XBC, DT, A, Bias, Keys, Queries, Values, Decay,
    H: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, C: tl.constexpr,
    DS0, DS1: tl.constexpr, ROWS,
    BR: tl.constexpr, BD: tl.constexpr, DT_SCALE: tl.constexpr,
):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    row, head = rows // H, rows % H
    group = head // (H // G)
    col = tl.arange(0, BD)
    active = rows < ROWS
    dt = tl.load(DT + row * DS0 + head * DS1, active, other=0)
    if DT_SCALE != 1.:
        dt = (dt.to(tl.float32) * DT_SCALE).to(dt.dtype)
    dt = dt.to(tl.float32)
    dt += tl.load(Bias + head, active, other=0).to(tl.float32)
    dt = tl.where(dt > 20, dt, libdevice.log1p(tl.exp(tl.minimum(dt, 20.))))
    decay = -tl.exp(tl.load(A + head, active, other=0).to(tl.float32)) * dt
    key_mask = active[:, None] & (col[None, :] < K) & (head % (H // G) == 0)[:, None]
    key = tl.load(XBC + row[:, None] * C + H * V + group[:, None] * K + col[None, :],
                  key_mask, other=0)
    query = tl.load(XBC + row[:, None] * C + H * V + G * K + group[:, None] * K + col[None, :],
                    key_mask, other=0)
    tl.store(Keys + (row[:, None] * G + group[:, None]) * K + col[None, :], key, key_mask)
    tl.store(Queries + (row[:, None] * G + group[:, None]) * K + col[None, :], query, key_mask)
    value_mask = active[:, None] & (col[None, :] < V)
    value = tl.load(XBC + row[:, None] * C + head[:, None] * V + col[None, :],
                    value_mask, other=0).to(tl.float32) * dt[:, None]
    tl.store(Values + rows[:, None].to(tl.int64) * V + col[None, :], value, value_mask)
    tl.store(Decay + rows, decay, active)


def prepare_mamba2(xbc, dt, A_log, dt_bias, config):
    tokens = len(xbc)
    keys = torch.empty((tokens, config.num_groups, config.state_dim), dtype=xbc.dtype, device=xbc.device)
    queries = torch.empty_like(keys)
    values = torch.empty((tokens, config.num_heads, config.head_dim), dtype=xbc.dtype, device=xbc.device)
    decay = torch.empty((tokens, config.num_heads), dtype=torch.float32, device=xbc.device)
    rows = tokens * config.num_heads
    _mamba2_prepare_kernel[(triton.cdiv(rows, 4),)](
        xbc, dt, A_log, dt_bias, keys, queries, values, decay,
        config.num_heads, config.num_groups, config.state_dim,
        config.head_dim, config.conv_dim, dt.stride(0), dt.stride(1), rows,
        4, triton.next_power_of_2(max(config.state_dim, config.head_dim)), config.dt_scale,
        num_warps=4, enable_fp_fusion=False,
    )
    return keys, queries, values, decay


@triton.jit(
    do_not_specialize=['LAYER'],
    do_not_specialize_on_alignment=[
        'Offsets', 'ChunkOffsets', 'Read', 'Writes', 'Final', 'Payload', 'LogSlots', 'Positions',
        'DecayFrames',
    ],
)
def _mamba2_prefill_state_kernel(
    Keys, Values, Decay, Storage, Offsets, ChunkOffsets,
    Hidden, Read, Writes, Final, Payload, LogSlots, Positions, DecayFrames, LogGate,
    H: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr,
    SS: tl.constexpr, LS: tl.constexpr, HS: tl.constexpr, LAYER,
    LOGSTATE: tl.constexpr, FRAME_BYTES: tl.constexpr, FRAMES: tl.constexpr,
    HEAD_BYTES: tl.constexpr, LAYERS: tl.constexpr, GRANULARITY: tl.constexpr,
    PAGE: tl.constexpr, STORE_FINAL: tl.constexpr, STORE_LOG: tl.constexpr,
    DECAY_BYTES: tl.constexpr, KEY_HEADS: tl.constexpr,
):
    req = tl.program_id(1) // H
    head = tl.program_id(1) % H
    group = head // (H // G)
    v = tl.program_id(0) * BV + tl.arange(0, BV)
    read = tl.load(Read + req)
    states = ()
    for ki in tl.static_range(triton.cdiv(K, BK)):
        k = ki * BK + tl.arange(0, BK)
        initial = _chunk_state_pointer(
            Storage, Payload, read, head, k[:, None] * V + v[None, :],
            SS, LS, HS, LAYER, H, LOGSTATE, FRAME_BYTES, FRAMES, HEAD_BYTES,
            K, V, LAYERS, GRANULARITY, PAGE,
        )
        state = tl.load(initial, (k[:, None] < K) & (v[None, :] < V) & (read >= 0), other=0).to(tl.float32)
        states += (state,)
    begin = tl.load(Offsets + req).to(tl.int32)
    end = tl.load(Offsets + req + 1).to(tl.int32)
    chunk = tl.load(ChunkOffsets + req).to(tl.int32)
    t = tl.arange(0, BT)
    for start in range(begin, end, BT):
        row = start + t
        active = row < end
        value = tl.load(Values + (row[:, None] * H + head) * V + v[None, :],
                        active[:, None] & (v[None, :] < V), other=0).to(tl.float32)
        if STORE_LOG:
            _store_chunk_log(
                Storage, Payload, DecayFrames, LogSlots, Positions, LogGate, Keys,
                value, row, active, head, v, tl.program_id(0),
                K, V, G, H, LAYER, PAGE, FRAME_BYTES, FRAMES, HEAD_BYTES,
                DECAY_BYTES, triton.next_power_of_2(K), LAYERS, GRANULARITY, KEY_HEADS,
            )
        prefix = tl.load(Decay + row * H + head, active, other=0)
        total = tl.load(Decay + (tl.minimum(start + BT, end) - 1) * H + head)
        scaled_value = (value * tl.exp2(total - prefix)[:, None]).to(Keys.dtype.element_ty)
        slot = tl.load(Writes + chunk)
        next_states = ()
        for ki in tl.static_range(triton.cdiv(K, BK)):
            k = ki * BK + tl.arange(0, BK)
            state = states[ki]
            tl.store(Hidden + ((chunk * H + head) * K + k[:, None]) * V + v[None, :],
                     state, (k[:, None] < K) & (v[None, :] < V))
            key = tl.load(Keys + (row[None, :] * G + group) * K + k[:, None],
                          active[None, :] & (k[:, None] < K), other=0)
            state = state * tl.exp2(total) + tl.dot(key, scaled_value)
            if slot >= 0:
                _store_chunk_state_tile(
                    Storage, Payload, state, slot, head, k, v, K, V, SS, LS, HS,
                    LAYER, H, LOGSTATE, FRAME_BYTES, FRAMES, HEAD_BYTES, LAYERS, GRANULARITY, PAGE,
                )
            next_states += (state,)
        states = next_states
        chunk += 1
    if STORE_FINAL:
        slot = tl.load(Final + req)
        for ki in tl.static_range(triton.cdiv(K, BK)):
            k = ki * BK + tl.arange(0, BK)
            _store_chunk_state_tile(
                Storage, Payload, states[ki], slot, head, k, v, K, V, SS, LS, HS,
                LAYER, H, LOGSTATE, FRAME_BYTES, FRAMES, HEAD_BYTES, LAYERS, GRANULARITY, PAGE,
            )


def mamba2_prefill(xbc, keys, queries, values, decay, D, state_io: ChunkStateIO,
                  indices, chunk_size, *, log_io: ChunkLogIO | None = None, out=None):
    from .vendor.fla.chunk_o import chunk_fwd_o
    from .vendor.fla.cumsum import chunk_local_cumsum

    _, groups, state_dim = keys.shape
    heads = values.shape[1]
    head_dim = values.shape[-1]
    output = torch.empty_like(values) if out is None else out
    storage = state_io.storage
    logstate = state_io.payload_frames is not None
    requests = len(indices.cu_seqlens) - 1
    chunks = len(indices.chunk_indices)
    cumulative_gate = chunk_local_cumsum(
        decay.unsqueeze(0), chunk_size=chunk_size, scale=1.4426950216,
        cu_seqlens=indices.cu_seqlens, chunk_indices=indices.chunk_indices,
    )
    hidden = torch.empty((chunks, heads, state_dim, head_dim), device=xbc.device, dtype=keys.dtype)
    state_config = prefill_state_config(requests, chunk_size)
    bv = state_config.pop("BV")
    _mamba2_prefill_state_kernel[(triton.cdiv(head_dim, bv), requests * heads)](
        keys, values, cumulative_gate, storage,
        indices.cu_seqlens, indices.chunk_offsets, hidden,
        state_io.read_slots, state_io.chunk_write_slots, state_io.final_write_slots,
        state_io.payload_frames, None if log_io is None else log_io.slots,
        None if log_io is None else log_io.token_offsets,
        None if log_io is None else log_io.decay_frames,
        decay,
        heads, groups, state_dim, head_dim,
        64, bv, chunk_size,
        0 if logstate else storage.stride(0), 0 if logstate else storage.stride(1),
        0 if logstate else storage.stride(2), state_io.layer,
        logstate, state_io.frame_bytes,
        0 if not logstate else state_io.payload_frames.shape[1],
        state_io.head_payload_bytes, state_io.num_layers, state_io.granularity,
        state_io.page_size, state_io.final_write_slots is not None, log_io is not None,
        0 if log_io is None else log_io.decay_record_bytes,
        heads if log_io is None else log_io.num_key_heads,
        **state_config,
    )
    chunk_fwd_o(
        queries.unsqueeze(0), keys.unsqueeze(0), values.unsqueeze(0), hidden.unsqueeze(0),
        g=cumulative_gate, scale=1., cu_seqlens=indices.cu_seqlens,
        chunk_indices=indices.chunk_indices, chunk_size=chunk_size, out=output.unsqueeze(0),
        skip=xbc[:, :heads * head_dim].view(1, len(xbc), heads, head_dim), skip_weight=D,
    )
    return output


@triton.jit(
    do_not_specialize=['LAYER'],
    do_not_specialize_on_alignment=['Payload', 'DecayFrames', 'Read', 'Write', 'LogSlots', 'Positions', 'Counts'],
)
def _mamba2_decode_kernel(
    XBC, DT, A, Bias, D, State, Storage, Payload, DecayFrames,
    Read, Write, LogSlots, Positions, Counts, Output,
    H: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.constexpr, C: tl.constexpr,
    DS0: tl.constexpr, DS1: tl.constexpr, RS: tl.constexpr, WS: tl.constexpr,
    LAYERS: tl.constexpr, LAYER, LOGSTATE: tl.constexpr, EAGER: tl.constexpr,
    FRAME_BYTES: tl.constexpr, PAYLOAD_FRAMES: tl.constexpr, HEAD_BYTES: tl.constexpr,
    DECAY_BYTES: tl.constexpr, PAGE: tl.constexpr, WINDOW: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    STATE_SLOT_STRIDE: tl.constexpr, STATE_LAYER_STRIDE: tl.constexpr,
    GRANULARITY: tl.constexpr = 64,
    DT_SCALE: tl.constexpr = 1.,
):
    value_tile = tl.program_id(0)
    req = tl.program_id(1) // H
    head = tl.program_id(1) % H
    group = head // (H // G)
    v = value_tile * BV + tl.arange(0, BV)
    read = tl.load(Read + req * RS)
    if LOGSTATE:
        payload = Storage.to(tl.pointer_type(XBC.dtype.element_ty))
        state_head = state_pointer(
            payload, Payload, read, LAYER, head, 0,
            H, HEAD_BYTES, FRAME_BYTES, PAYLOAD_FRAMES,
        )
        log_slot = tl.load(LogSlots + req)
        decay_frame = tl.load(DecayFrames + log_slot * 2).to(tl.int64)
        decay_index = tl.load(DecayFrames + log_slot * 2 + 1).to(tl.int64)
        decay_head = (Storage + decay_frame * FRAME_BYTES + decay_index * DECAY_BYTES).to(tl.pointer_type(tl.float32))
        position = tl.load(Positions + req)
    else:
        state_head = State + read.to(tl.int64) * STATE_SLOT_STRIDE + LAYER * STATE_LAYER_STRIDE + head * K * V
        write = tl.load(Write + req * WS)
        destination = State + write.to(tl.int64) * STATE_SLOT_STRIDE + LAYER * STATE_LAYER_STRIDE + head * K * V

    if LOGSTATE and not EAGER:
        count = tl.load(Counts + req)
        t = tl.arange(0, WINDOW)
        previous = position - count + t
        valid = t < count
        key_rows = log_pointer(
            payload, Payload, log_slot, LAYER, head, previous, 0,
            LAYERS, H, K, V, GRANULARITY, PAGE, HEAD_BYTES, FRAME_BYTES, PAYLOAD_FRAMES,
            False, G, position - count, position,
        )
        value_rows = log_pointer(
            payload, Payload, log_slot, LAYER, head, previous, 0,
            LAYERS, H, K, V, GRANULARITY, PAGE, HEAD_BYTES, FRAME_BYTES, PAYLOAD_FRAMES,
            True, G, position - count, position,
        )
        decays = tl.load(decay_head + (LAYER * PAGE + previous) * H + head, valid, other=0)
        total = tl.sum(decays)
        total_decay = tl.exp(total)
        suffix = tl.exp(total - tl.cumsum(decays))
        values = tl.load(value_rows[:, None] + v[None, :],
                         valid[:, None] & (v[None, :] < V), other=0)
        scaled = (values.to(tl.float32) * suffix[:, None]).to(XBC.dtype.element_ty)

    dt = tl.load(DT + req * DS0 + head * DS1)
    if DT_SCALE != 1.:
        dt = (dt.to(tl.float32) * DT_SCALE).to(dt.dtype)
    dt = dt.to(tl.float32) + tl.load(Bias + head).to(tl.float32)
    dt = tl.where(dt > 20, dt, tl.log(1.0 + tl.exp(dt)))
    log_decay = -tl.exp(tl.load(A + head).to(tl.float32)) * dt
    decay = tl.exp(log_decay)
    x = tl.load(XBC + req * C + head * V + v, v < V, other=0).to(tl.float32)
    update = x * dt
    state_q = tl.zeros((BV,), tl.float32)
    kq = tl.full((), 0., tl.float32)
    key_head = XBC + req * C + H * V + group * K
    query_head = key_head + G * K
    for ki in range(tl.cdiv(K, BK)):
        k = ki * BK + tl.arange(0, BK)
        mask = (k[:, None] < K) & (v[None, :] < V)
        if LOGSTATE and not EAGER:
            state = reconstruct_tile(state_head, key_rows, scaled, valid, k, v, total_decay, K, V)
        else:
            state = tl.load(state_head + k[:, None] * V + v[None, :], mask, other=0).to(tl.float32)
        key = tl.load(key_head + k, k < K, other=0).to(tl.float32)
        query = tl.load(query_head + k, k < K, other=0).to(tl.float32)
        state_q += tl.sum(state * query[:, None], 0)
        kq += tl.sum(key * query)
        if not LOGSTATE or EAGER:
            updated = state * decay + key[:, None] * update[None, :]
            if LOGSTATE:
                tl.store(state_head + k[:, None] * V + v[None, :], updated, mask)
            else:
                tl.store(destination + k[:, None] * V + v[None, :], updated, mask)
    y = decay * state_q + kq * update + x * tl.load(D + head).to(tl.float32)
    tl.store(Output + (req * H + head) * V + v, y, v < V)
    if LOGSTATE:
        if value_tile == 0:
            if head % (H // G) == 0:
                columns = tl.arange(0, triton.next_power_of_2(K))
                key = tl.load(key_head + columns, columns < K, other=0)
                tl.store(log_pointer(
                    payload, Payload, log_slot, LAYER, head, position, columns,
                    LAYERS, H, K, V, GRANULARITY, PAGE, HEAD_BYTES, FRAME_BYTES, PAYLOAD_FRAMES, False, G,
                ), key, columns < K)
            tl.store(decay_head + (LAYER * PAGE + position) * H + head, log_decay)
        tl.store(log_pointer(
            payload, Payload, log_slot, LAYER, head, position, v,
            LAYERS, H, K, V, GRANULARITY, PAGE, HEAD_BYTES, FRAME_BYTES, PAYLOAD_FRAMES, True, G,
        ), update, v < V)
        if not EAGER:
            if (position + 1) % WINDOW == 0:
                for ki in range(tl.cdiv(K, BK)):
                    k = ki * BK + tl.arange(0, BK)
                    mask = (k[:, None] < K) & (v[None, :] < V)
                    state = reconstruct_tile(state_head, key_rows, scaled, valid, k, v, total_decay, K, V)
                    key = tl.load(key_head + k, k < K, other=0).to(tl.float32)
                    updated = state * decay + key[:, None] * update[None, :]
                    tl.store(state_head + k[:, None] * V + v[None, :], updated, mask)


def mamba2_decode(xbc, dt, A_log, dt_bias, D, config, metadata, pool, *, logstate, eager=False, out=None):
    output = (torch.empty((len(xbc), config.num_heads, config.head_dim), dtype=xbc.dtype, device=xbc.device)
              if out is None else out)
    if logstate:
        read = write = metadata.work_slots
        log_slots, offsets, counts = metadata.log_slots, metadata.token_offsets, metadata.replay_counts
        state, storage, payload, decay = pool.storage, pool.storage, pool.payload_frame_table, pool.decay_frame_table
        layers, frame_bytes, frames = pool.layout.num_linear_layers, pool.frame_bytes, pool.payload_frame_table.shape[1]
        head_bytes, decay_bytes, page = pool.head_payload_bytes, pool.layout.log_decay_bytes, pool.page_size
    else:
        read, write = metadata.read_slots, metadata.write_slots
        log_slots = offsets = counts = read
        state = storage = payload = decay = pool.recurrent_state
        layers = pool.recurrent_state.shape[1]
        frame_bytes = frames = head_bytes = decay_bytes = page = 1
    bk, bv, warps, stages = decode_config(len(xbc))
    bk = min(bk, triton.next_power_of_2(config.state_dim))
    bv = min(bv, triton.next_power_of_2(config.head_dim))
    _mamba2_decode_kernel[(triton.cdiv(config.head_dim, bv), len(xbc) * config.num_heads)](
        xbc, dt, A_log, dt_bias, D, state, storage, payload, decay,
        read, write, log_slots, offsets, counts, output,
        config.num_heads, config.num_groups, config.state_dim, config.head_dim, config.conv_dim,
        dt.stride(0), dt.stride(1), read.stride(0), write.stride(0),
        layers, config.layer_idx, logstate, eager,
        frame_bytes, frames, head_bytes, decay_bytes, page, LOG_STATE_REPLAY_WINDOW,
        bk, bv, num_warps=warps, num_stages=stages,
        STATE_SLOT_STRIDE=0 if logstate else state.stride(0),
        STATE_LAYER_STRIDE=0 if logstate else state.stride(1),
        GRANULARITY=pool.layout.granularity if logstate else 64,
        DT_SCALE=config.dt_scale,
    )
    return output
