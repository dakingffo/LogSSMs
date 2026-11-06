# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Licensed under the MIT license in the vendored FLA LICENSE file.

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .layout import log_pointer, state_pointer


@dataclass(frozen=True)
class ChunkStateIO:

    storage           : torch.Tensor
    read_slots        : torch.Tensor
    chunk_write_slots : torch.Tensor
    final_write_slots : torch.Tensor | None = None
    payload_frames    : torch.Tensor | None = None
    frame_bytes       : int = 0
    head_payload_bytes: int = 0
    layer             : int = 0
    num_layers        : int = 1
    granularity       : int = 64
    page_size         : int = 64


@dataclass(frozen=True)
class ChunkLogIO:

    slots             : torch.Tensor
    token_offsets     : torch.Tensor
    decay_frames      : torch.Tensor
    gate              : torch.Tensor
    page_size         : int
    decay_record_bytes: int
    num_key_heads     : int


@triton.jit
def _chunk_state_pointer(
    storage, payload_frames, slot, head, offsets,
    SLOT_STRIDE: tl.constexpr, LAYER_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr, LAYER,
    HV: tl.constexpr, LOGSTATE: tl.constexpr,
    FRAME_BYTES: tl.constexpr, FRAMES_PER_SLOT: tl.constexpr,
    HEAD_PAYLOAD_BYTES: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, LAYERS: tl.constexpr,
    GRANULARITY: tl.constexpr, PAGE_SIZE: tl.constexpr,
):
    if LOGSTATE:
        return state_pointer(
            storage, payload_frames, slot, LAYER, head, offsets,
            HV, HEAD_PAYLOAD_BYTES, FRAME_BYTES, FRAMES_PER_SLOT,
        )
    return storage + slot.to(tl.int64) * SLOT_STRIDE + LAYER * LAYER_STRIDE + head * HEAD_STRIDE + offsets


@triton.jit
def _store_chunk_state_tile(
    storage, payload_frames, state, slot, head, key_offsets, value_offsets,
    K: tl.constexpr, V: tl.constexpr,
    SLOT_STRIDE: tl.constexpr, LAYER_STRIDE: tl.constexpr, HEAD_STRIDE: tl.constexpr,
    LAYER, HV: tl.constexpr, LOGSTATE: tl.constexpr,
    FRAME_BYTES: tl.constexpr, FRAMES_PER_SLOT: tl.constexpr, HEAD_PAYLOAD_BYTES: tl.constexpr,
    LAYERS: tl.constexpr, GRANULARITY: tl.constexpr, PAGE_SIZE: tl.constexpr,
):
    offsets = key_offsets[:, None] * V + value_offsets[None, :]
    destination = _chunk_state_pointer(
        storage, payload_frames, slot, head, offsets,
        SLOT_STRIDE, LAYER_STRIDE, HEAD_STRIDE, LAYER, HV, LOGSTATE,
        FRAME_BYTES, FRAMES_PER_SLOT, HEAD_PAYLOAD_BYTES, K, V, LAYERS, GRANULARITY, PAGE_SIZE,
    )
    mask = (key_offsets[:, None] < K) & (value_offsets[None, :] < V)
    tl.store(destination, state.to(destination.dtype.element_ty), mask=mask & (slot >= 0))


@triton.jit
def _store_chunk_log(
    storage, payload_frames, decay_frames, slots, token_offsets, gate, key,
    residual, rows, token_mask, head, value_offsets, value_tile,
    K: tl.constexpr, V: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
    LAYER, PAGE_SIZE: tl.constexpr, FRAME_BYTES: tl.constexpr,
    FRAMES_PER_SLOT: tl.constexpr, HEAD_PAYLOAD_BYTES: tl.constexpr,
    DECAY_RECORD_BYTES: tl.constexpr, BK: tl.constexpr,
    LAYERS: tl.constexpr, GRANULARITY: tl.constexpr,
    LOG_KEY_HEADS: tl.constexpr,
):
    slot = tl.load(slots + rows, token_mask, other=-1)
    position = tl.load(token_offsets + rows, token_mask, other=0).to(tl.int64)
    active = token_mask & (slot >= 0)
    tl.store(
        log_pointer(
            storage, payload_frames, slot[:, None], LAYER, head,
            position[:, None], value_offsets[None, :], LAYERS, HV, K, V, GRANULARITY,
            PAGE_SIZE, HEAD_PAYLOAD_BYTES, FRAME_BYTES, FRAMES_PER_SLOT, True, LOG_KEY_HEADS,
        ),
        residual, active[:, None] & (value_offsets[None, :] < V),
    )

    if value_tile == 0:
        if head % (HV // LOG_KEY_HEADS) == 0:
            columns = tl.arange(0, BK)
            keys = tl.load(
                key + rows[:, None].to(tl.int64) * H * K + (head // (HV // H)) * K + columns[None, :],
                token_mask[:, None] & (columns[None, :] < K), other=0,
            )
            tl.store(log_pointer(
                storage, payload_frames, slot[:, None], LAYER, head,
                position[:, None], columns[None, :], LAYERS, HV, K, V, GRANULARITY,
                PAGE_SIZE, HEAD_PAYLOAD_BYTES, FRAME_BYTES, FRAMES_PER_SLOT, False, LOG_KEY_HEADS,
            ), keys, active[:, None] & (columns[None, :] < K))
        frame = tl.load(decay_frames + slot * 2, active, other=0).to(tl.int64)
        record = tl.load(decay_frames + slot * 2 + 1, active, other=0).to(tl.int64)
        decay = storage.to(tl.pointer_type(tl.uint8)) + frame * FRAME_BYTES + record * DECAY_RECORD_BYTES
        decay = decay.to(tl.pointer_type(tl.float32)) + (LAYER * PAGE_SIZE + position) * HV + head
        values = tl.load(gate + rows.to(tl.int64) * HV + head, token_mask, other=0)
        tl.store(decay, values, active)
