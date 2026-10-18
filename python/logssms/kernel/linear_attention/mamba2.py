from __future__ import annotations

import torch
import triton
import triton.language as tl

from logssms.cache.logstate_pool.layout import LOG_STATE_REPLAY_WINDOW
from .layout import log_pointer, state_pointer


@triton.jit
def _mamba2_prepare_kernel(
    XBC, DT, A, Bias, Keys, Queries, Values, Decay,
    H: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, C: tl.constexpr,
    DS0: tl.constexpr, DS1: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    group = head // (H // G)
    k = tl.arange(0, BK)
    v = tl.arange(0, BV)
    dt = tl.load(DT + row * DS0 + head * DS1).to(tl.float32)
    dt += tl.load(Bias + head).to(tl.float32)
    dt = tl.where(dt > 20, dt, tl.log(1.0 + tl.exp(dt)))
    decay = -tl.exp(tl.load(A + head).to(tl.float32)) * dt
    key = tl.load(XBC + row * C + H * V + group * K + k, k < K, other=0)
    query = tl.load(XBC + row * C + H * V + G * K + group * K + k, k < K, other=0)
    value = tl.load(XBC + row * C + head * V + v, v < V, other=0).to(tl.float32) * dt
    tl.store(Keys + (row * H + head) * K + k, key, k < K)
    tl.store(Queries + (row * H + head) * K + k, query, k < K)
    tl.store(Values + (row * H + head) * V + v, value, v < V)
    tl.store(Decay + row * H + head, decay)


def prepare_mamba2(xbc, dt, A_log, dt_bias, config):
    tokens = len(xbc)
    keys = torch.empty((tokens, config.num_heads, config.state_dim), dtype=xbc.dtype, device=xbc.device)
    queries = torch.empty_like(keys)
    values = torch.empty((tokens, config.num_heads, config.head_dim), dtype=xbc.dtype, device=xbc.device)
    decay = torch.empty((tokens, config.num_heads), dtype=torch.float32, device=xbc.device)
    _mamba2_prepare_kernel[(tokens, config.num_heads)](
        xbc, dt, A_log, dt_bias, keys, queries, values, decay,
        config.num_heads, config.num_groups, config.state_dim,
        config.head_dim, config.conv_dim, dt.stride(0), dt.stride(1),
        triton.next_power_of_2(config.state_dim), triton.next_power_of_2(config.head_dim),
    )
    return keys, queries, values, decay


@triton.jit
def _mamba2_prefill_kernel(
    Keys, Queries, Values, Decay, XBC, D, Initial, Offsets, ChunkOffsets,
    Output, ChunkStates,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, C: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr,
):
    req = tl.program_id(0)
    head = tl.program_id(1)
    k = tl.arange(0, BK)
    v = tl.program_id(2) * BV + tl.arange(0, BV)
    state_mask = (k[:, None] < K) & (v[None, :] < V)
    state = tl.load(Initial + ((req * H + head) * K + k[:, None]) * V + v[None, :],
                    state_mask, other=0).to(tl.float32)
    begin = tl.load(Offsets + req).to(tl.int32)
    end = tl.load(Offsets + req + 1).to(tl.int32)
    chunk = tl.load(ChunkOffsets + req).to(tl.int32)
    t = tl.arange(0, BT)
    d = tl.load(D + head).to(tl.float32)
    for start in range(begin, end, BT):
        row = start + t
        active = row < end
        key = tl.load(Keys + (row[:, None] * H + head) * K + k[None, :],
                      active[:, None] & (k[None, :] < K), other=0).to(tl.float32)
        query = tl.load(Queries + (row[:, None] * H + head) * K + k[None, :],
                        active[:, None] & (k[None, :] < K), other=0).to(tl.float32)
        value = tl.load(Values + (row[:, None] * H + head) * V + v[None, :],
                        active[:, None] & (v[None, :] < V), other=0).to(tl.float32)
        g = tl.load(Decay + row * H + head, active, other=0)
        prefix = tl.cumsum(g)
        total = tl.sum(g)
        causal = t[:, None] >= t[None, :]
        weights = tl.exp(tl.minimum(prefix[:, None] - prefix[None, :], 0.0))
        scores = tl.dot(query, tl.trans(key), input_precision="tf32x3")
        scores = tl.where(causal, scores * weights, 0.0)
        result = tl.dot(query, state, input_precision="tf32x3") * tl.exp(prefix)[:, None]
        result += tl.dot(scores, value, input_precision="tf32x3")
        x = tl.load(XBC + row[:, None] * C + head * V + v[None, :],
                    active[:, None] & (v[None, :] < V), other=0).to(tl.float32)
        result += x * d
        tl.store(Output + (row[:, None] * H + head) * V + v[None, :], result,
                 active[:, None] & (v[None, :] < V))
        scaled_value = value * tl.exp(total - prefix)[:, None]
        state = state * tl.exp(total) + tl.dot(tl.trans(key), scaled_value, input_precision="tf32x3")
        tl.store(ChunkStates + ((chunk * H + head) * K + k[:, None]) * V + v[None, :],
                 state, state_mask)
        chunk += 1


def mamba2_prefill(xbc, keys, queries, values, decay, D, initial_states, indices, chunk_size, *, out=None):
    _, heads, state_dim = keys.shape
    head_dim = values.shape[-1]
    output = torch.empty_like(values) if out is None else out
    states = torch.empty(
        (len(indices.chunk_indices), heads, state_dim, head_dim),
        device=xbc.device, dtype=torch.float32,
    )
    _mamba2_prefill_kernel[(len(indices.cu_seqlens) - 1, heads, triton.cdiv(head_dim, 32))](
        keys, queries, values, decay, xbc, D, initial_states,
        indices.cu_seqlens, indices.chunk_offsets, output, states,
        heads, state_dim, head_dim, xbc.shape[1],
        triton.next_power_of_2(state_dim), 32, chunk_size,
        num_warps=8, num_stages=1,
    )
    return output, states.index_select(0, indices.endings), states


@triton.jit
def _mamba2_conv_decode_kernel(
    X, Weight, Bias, State, Storage, Frames, Read, Write, ReadGranules,
    WriteGranules, Work, Output,
    C: tl.constexpr, S: tl.constexpr, LAYERS: tl.constexpr, LAYER: tl.constexpr,
    XS0: tl.constexpr, XS1: tl.constexpr, RS: tl.constexpr, WS: tl.constexpr,
    LOGSTATE: tl.constexpr, HAS_BIAS: tl.constexpr,
    FRAME_ELEMS: tl.constexpr, GRANULES: tl.constexpr,
    BC: tl.constexpr, BS: tl.constexpr,
):
    req = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    s = tl.arange(0, BS)
    read = tl.load(Read + req * RS)
    write = tl.load(Write + req * WS)
    if LOGSTATE:
        rg = tl.load(ReadGranules + req)
        wg = tl.load(WriteGranules + req)
        work = tl.load(Work + req)
        read_frame = tl.load(Frames + read * GRANULES + rg)
        write_frame = tl.load(Frames + write * GRANULES + wg)
        work_frame = tl.load(Frames + work * GRANULES)
        ptr = Storage.to(tl.pointer_type(X.dtype.element_ty))
        old_base = read_frame.to(tl.int64) * FRAME_ELEMS + LAYER * C * S
        new_base = write_frame.to(tl.int64) * FRAME_ELEMS + LAYER * C * S
        work_base = work_frame.to(tl.int64) * FRAME_ELEMS + LAYER * C * S
    else:
        ptr = State
        old_base = (read.to(tl.int64) * LAYERS + LAYER) * C * S
        new_base = (write.to(tl.int64) * LAYERS + LAYER) * C * S
    old = tl.load(ptr + old_base + c[:, None] * S + s[None, :],
                  (c[:, None] < C) & (s[None, :] < S), other=0).to(tl.float32)
    w = tl.load(Weight + c[:, None] * (S + 1) + s[None, :],
                (c[:, None] < C) & (s[None, :] < S), other=0).to(tl.float32)
    x = tl.load(X + req * XS0 + c * XS1, c < C, other=0)
    last_w = tl.load(Weight + c * (S + 1) + S, c < C, other=0).to(tl.float32)
    y = tl.sum(old * w, 1) + x.to(tl.float32) * last_w
    if HAS_BIAS:
        y += tl.load(Bias + c, c < C, other=0).to(tl.float32)
    y = y.to(X.dtype.element_ty).to(tl.float32)
    y *= tl.sigmoid(y)
    shifted = tl.load(ptr + old_base + c[:, None] * S + s[None, :] + 1,
                      (c[:, None] < C) & (s[None, :] + 1 < S), other=0)
    shifted = tl.where(s[None, :] == S - 1, x[:, None], shifted)
    mask = (c[:, None] < C) & (s[None, :] < S)
    tl.store(ptr + new_base + c[:, None] * S + s[None, :], shifted, mask)
    if LOGSTATE:
        tl.store(ptr + work_base + c[:, None] * S + s[None, :], shifted, mask)
    tl.store(Output + req * C + c, y, c < C)


def mamba2_conv_decode(xbc, weight, bias, config, metadata, pool, *, logstate):
    output = torch.empty_like(xbc, memory_format=torch.contiguous_format)
    if logstate:
        assert pool.allocator.conv_frames == 1
        read, write = metadata.conv_read_slots, metadata.log_slots
        rg = metadata.conv_read_granules
        wg = metadata.token_offsets // pool.layout.granularity
        work = metadata.work_slots
        state, storage, frames = pool.storage, pool.storage, pool.conv_frame_table
        frame_elems = pool.frame_bytes // xbc.element_size()
        granules, layers = pool.layout.num_granules, pool.layout.num_linear_layers
    else:
        read, write = metadata.read_slots, metadata.write_slots
        rg = wg = work = read
        state = storage = frames = pool.conv_state
        frame_elems = granules = 1
        layers = pool.conv_state.shape[1]
    _mamba2_conv_decode_kernel[(len(xbc), triton.cdiv(config.conv_dim, 32))](
        xbc, weight, bias, state, storage, frames, read, write, rg, wg, work, output,
        config.conv_dim, config.state_len, layers, config.layer_idx,
        xbc.stride(0), xbc.stride(1), read.stride(0), write.stride(0),
        logstate, bias is not None, frame_elems, granules, 32,
        triton.next_power_of_2(config.state_len), num_warps=4,
    )
    return output


@triton.jit
def _mamba2_decode_kernel(
    XBC, DT, A, Bias, D, State, Storage, Payload, DecayFrames,
    Read, Write, LogSlots, Positions, Counts, Output,
    H: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.constexpr, C: tl.constexpr,
    DS0: tl.constexpr, DS1: tl.constexpr, RS: tl.constexpr, WS: tl.constexpr,
    LAYERS: tl.constexpr, LAYER: tl.constexpr, LOGSTATE: tl.constexpr, EAGER: tl.constexpr,
    FRAME_BYTES: tl.constexpr, PAYLOAD_FRAMES: tl.constexpr, HEAD_BYTES: tl.constexpr,
    DECAY_BYTES: tl.constexpr, PAGE: tl.constexpr, WINDOW: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    GRANULARITY: tl.constexpr = 64,
):
    req = tl.program_id(0)
    head = tl.program_id(1)
    group = head // (H // G)
    k = tl.arange(0, BK)
    v = tl.program_id(2) * BV + tl.arange(0, BV)
    mask = (k[:, None] < K) & (v[None, :] < V)
    read = tl.load(Read + req * RS)
    if LOGSTATE:
        payload = Storage.to(tl.pointer_type(XBC.dtype.element_ty))
        state_ptr = state_pointer(
            payload, Payload, read, LAYER, head, k[:, None] * V + v[None, :],
            H, HEAD_BYTES, FRAME_BYTES, PAYLOAD_FRAMES,
        )
        log_slot = tl.load(LogSlots + req)
        decay_frame = tl.load(DecayFrames + log_slot * 2)
        decay_index = tl.load(DecayFrames + log_slot * 2 + 1)
        decay_head = (Storage + decay_frame.to(tl.int64) * FRAME_BYTES + decay_index * DECAY_BYTES).to(tl.pointer_type(tl.float32))
        position = tl.load(Positions + req)
    else:
        state_head = State + (read.to(tl.int64) * LAYERS + LAYER) * H * K * V + head * K * V
        state_ptr = state_head + k[:, None] * V + v[None, :]
    state = tl.load(state_ptr, mask, other=0).to(tl.float32)
    if LOGSTATE and not EAGER:
        count = tl.load(Counts + req)
        t = tl.arange(0, WINDOW)
        previous = position - count + t
        valid = t < count
        decays = tl.load(decay_head + (LAYER * PAGE + previous) * H + head, valid, other=0)
        total = tl.sum(decays)
        suffix = tl.exp(total - tl.cumsum(decays))
        keys = tl.load(log_pointer(
            payload, Payload, log_slot, LAYER, head, previous[None, :], k[:, None],
            LAYERS, H, K, V, GRANULARITY, PAGE, HEAD_BYTES, FRAME_BYTES, PAYLOAD_FRAMES, False, G,
            position - count, position,
        ),
                       (k[:, None] < K) & valid[None, :], other=0)
        values = tl.load(log_pointer(
            payload, Payload, log_slot, LAYER, head, previous[:, None], v[None, :],
            LAYERS, H, K, V, GRANULARITY, PAGE, HEAD_BYTES, FRAME_BYTES, PAYLOAD_FRAMES, True, G,
            position - count, position,
        ),
                         valid[:, None] & (v[None, :] < V), other=0)
        scaled = (values.to(tl.float32) * suffix[:, None]).to(XBC.dtype.element_ty)
        state = state * tl.exp(total) + tl.dot(keys, scaled)
    dt = tl.load(DT + req * DS0 + head * DS1).to(tl.float32) + tl.load(Bias + head).to(tl.float32)
    dt = tl.where(dt > 20, dt, tl.log(1.0 + tl.exp(dt)))
    log_decay = -tl.exp(tl.load(A + head).to(tl.float32)) * dt
    key = tl.load(XBC + req * C + H * V + group * K + k, k < K, other=0).to(tl.float32)
    query = tl.load(XBC + req * C + H * V + G * K + group * K + k, k < K, other=0).to(tl.float32)
    x = tl.load(XBC + req * C + head * V + v, v < V, other=0).to(tl.float32)
    update = x * dt
    state = state * tl.exp(log_decay) + key[:, None] * update[None, :]
    y = tl.sum(state * query[:, None], 0) + x * tl.load(D + head).to(tl.float32)
    tl.store(Output + (req * H + head) * V + v, y, v < V)
    if LOGSTATE:
        if tl.program_id(2) == 0:
            if head % (H // G) == 0:
                tl.store(log_pointer(
                    payload, Payload, log_slot, LAYER, head, position, k,
                    LAYERS, H, K, V, GRANULARITY, PAGE, HEAD_BYTES, FRAME_BYTES, PAYLOAD_FRAMES, False, G,
                ), key, k < K)
            tl.store(decay_head + (LAYER * PAGE + position) * H + head, log_decay)
        tl.store(log_pointer(
            payload, Payload, log_slot, LAYER, head, position, v,
            LAYERS, H, K, V, GRANULARITY, PAGE, HEAD_BYTES, FRAME_BYTES, PAYLOAD_FRAMES, True, G,
        ), update, v < V)
        if EAGER or (position + 1) % WINDOW == 0:
            tl.store(state_ptr, state, mask)
    else:
        write = tl.load(Write + req * WS)
        destination = State + (write.to(tl.int64) * LAYERS + LAYER) * H * K * V + head * K * V
        tl.store(destination + k[:, None] * V + v[None, :], state, mask)


def mamba2_decode(xbc, dt, A_log, dt_bias, D, config, metadata, pool, *, logstate, eager=False, out=None):
    output = (torch.empty((len(xbc), config.num_heads, config.head_dim), dtype=xbc.dtype, device=xbc.device)
              if out is None else out)
    if logstate:
        read = write = metadata.work_slots
        log_slots, offsets, counts = metadata.log_slots, metadata.token_offsets, metadata.replay_counts
        state, storage, payload, decay = pool.storage, pool.storage, pool.payload_frame_table, pool.decay_frame_table
        layers, frame_bytes, frames = pool.layout.num_linear_layers, pool.frame_bytes, pool.allocator.payload_frames
        head_bytes, decay_bytes, page = pool.head_payload_bytes, pool.layout.log_decay_bytes, pool.page_size
    else:
        read, write = metadata.read_slots, metadata.write_slots
        log_slots = offsets = counts = read
        state = storage = payload = decay = pool.recurrent_state
        layers = pool.recurrent_state.shape[1]
        frame_bytes = frames = head_bytes = decay_bytes = page = 1
    _mamba2_decode_kernel[(len(xbc), config.num_heads, triton.cdiv(config.head_dim, 32))](
        xbc, dt, A_log, dt_bias, D, state, storage, payload, decay,
        read, write, log_slots, offsets, counts, output,
        config.num_heads, config.num_groups, config.state_dim, config.head_dim, config.conv_dim,
        dt.stride(0), dt.stride(1), read.stride(0), write.stride(0),
        layers, config.layer_idx, logstate, eager,
        frame_bytes, frames, head_bytes, decay_bytes, page, LOG_STATE_REPLAY_WINDOW,
        triton.next_power_of_2(config.state_dim), 32, num_warps=8,
        GRANULARITY=pool.layout.granularity if logstate else 64,
    )
    return output
