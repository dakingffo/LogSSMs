"""GPU LogState snapshot stores accepting negative (inactive) slot IDs."""
from __future__ import annotations

import triton
import triton.language as tl

from .layout import state_pointer


@triton.jit
def _state_store(storage, frames, slots, values,
                 HEADS: tl.constexpr, WIDTH: tl.constexpr, LAYER: tl.constexpr,
                 HEAD_BYTES: tl.constexpr, FRAME_BYTES: tl.constexpr,
                 FRAMES_PER_SLOT: tl.constexpr, BLOCK: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slots + row)
    if slot < 0:
        return
    offsets = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    destination = state_pointer(
        storage, frames, slot, LAYER, head, offsets,
        HEADS, HEAD_BYTES, FRAME_BYTES, FRAMES_PER_SLOT,
    )
    value = tl.load(values + (row * HEADS + head) * WIDTH + offsets, offsets < WIDTH, other=0)
    tl.store(destination, value, offsets < WIDTH)


@triton.jit
def _conv_store(storage, frames, slots, granules, values,
                WIDTH: tl.constexpr, LAYER: tl.constexpr, GRANULES: tl.constexpr,
                FRAME_ELEMS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(slots + row)
    if slot < 0:
        return
    granule = tl.load(granules + row)
    frame = tl.load(frames + slot * GRANULES + granule).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(values + row * WIDTH + offsets, offsets < WIDTH, other=0)
    tl.store(storage + frame * FRAME_ELEMS + LAYER * WIDTH + offsets, value, offsets < WIDTH)


def store_states(pool, slots, layer, values):
    assert slots.is_cuda and values.is_cuda
    if not len(slots):
        return
    width = pool.layout.head_k_dim * pool.layout.head_v_dim
    _state_store[(len(slots), pool.layout.num_value_heads, triton.cdiv(width, 256))](
        pool.storage.view(pool.dtype), pool.payload_frame_table, slots.contiguous(),
        values.contiguous(), pool.layout.num_value_heads, width, layer,
        pool.head_payload_bytes, pool.frame_bytes, pool.payload_frame_table.shape[1], 256,
    )


def store_conv(pool, slots, granules, layer, values):
    assert slots.is_cuda and granules.is_cuda and values.is_cuda
    assert granules.shape == slots.shape
    assert pool.conv_frame_table.shape[2] == 1
    if not len(slots):
        return
    width = pool.layout.conv_dim * pool.layout.state_len
    _conv_store[(len(slots), triton.cdiv(width, 256))](
        pool.storage.view(pool.dtype), pool.conv_frame_table, slots.contiguous(),
        granules.contiguous(), values.contiguous(), width, layer,
        pool.conv_frame_table.shape[1], pool.frame_bytes // pool.dtype.itemsize, 256,
    )
