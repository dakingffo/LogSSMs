from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from .layout import conv_pointer


@dataclass(frozen=True)
class PackedConvStateIO:

    storage     : torch.Tensor
    slots       : torch.Tensor
    granules    : torch.Tensor | None = None
    final_slots : torch.Tensor | None = None
    frame_table : torch.Tensor | None = None
    frame_bytes : int = 0
    layer       : int = 0


@triton.jit
def _scale_conv_input(value, channel, END0: tl.constexpr, END1: tl.constexpr,
                      SCALE0: tl.constexpr, SCALE1: tl.constexpr, SCALE2: tl.constexpr):
    if (SCALE0 != 1.) | (SCALE1 != 1.) | (SCALE2 != 1.):
        scale = tl.where(channel < END0, SCALE0, tl.where(channel < END1, SCALE1, SCALE2))
        value = (value.to(tl.float32) * scale).to(value.dtype)
    return value


def _conv_input_scale_args(channels, boundaries, scales):
    if boundaries is None:
        assert tuple(scales) == (1., 1., 1.)
        boundaries = (channels, channels)
    assert len(scales) == 3 and 0 <= boundaries[0] <= boundaries[1] <= channels
    return (*boundaries, *scales)


@triton.jit
def _conv_destination(
    storage, frames, slot, granule,
    C: tl.constexpr, S: tl.constexpr, LAYER,
    SLOT_STRIDE: tl.constexpr, LAYER_STRIDE: tl.constexpr,
    LOGSTATE: tl.constexpr, NUM_GRANULES: tl.constexpr, FRAME_ELEMS: tl.constexpr,
):
    if LOGSTATE:
        frame = tl.load(frames + slot * NUM_GRANULES + granule, slot >= 0, other=0).to(tl.int64)
        return storage + frame * FRAME_ELEMS + LAYER * C * S
    return storage + slot.to(tl.int64) * SLOT_STRIDE + LAYER * LAYER_STRIDE


@triton.jit(
    do_not_specialize=['LAYER'],
    do_not_specialize_on_alignment=['offsets', 'chunks', 'write_slots', 'write_granules', 'final_slots', 'conv_frames'],
)
def _packed_conv_kernel(
    x, weight, bias, history, offsets, chunks, output,
    state_storage, write_slots, write_granules, final_slots, conv_frames,
    C: tl.constexpr, K: tl.constexpr, S: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr, X_COL_STRIDE: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    STORE_STATE: tl.constexpr, LOGSTATE: tl.constexpr, STORE_FINAL: tl.constexpr,
    LAYER, SLOT_STRIDE: tl.constexpr, LAYER_STRIDE: tl.constexpr,
    NUM_GRANULES: tl.constexpr, FRAME_ELEMS: tl.constexpr, CONV_FRAMES: tl.constexpr,
    END0: tl.constexpr, END1: tl.constexpr,
    SCALE0: tl.constexpr, SCALE1: tl.constexpr, SCALE2: tl.constexpr,
):
    chunk = tl.program_id(0)
    request = tl.load(chunks + chunk * 2).to(tl.int32)
    chunk_index = tl.load(chunks + chunk * 2 + 1).to(tl.int32)
    begin = tl.load(offsets + request).to(tl.int32)
    end = tl.load(offsets + request + 1).to(tl.int32)
    t = chunk_index * BT + tl.arange(0, BT)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    active = (t[:, None] < end - begin) & (c[None, :] < C)
    if STORE_STATE:
        slot = tl.load(write_slots + begin + t, t < end - begin, other=-1)
        if LOGSTATE:
            granule = tl.load(write_granules + begin + t, t < end - begin, other=0)
        else:
            granule = tl.full((BT,), 0, tl.int32)
        if not LOGSTATE or CONV_FRAMES == 1:
            destination = _conv_destination(
                state_storage, conv_frames, slot, granule, C, S, LAYER,
                SLOT_STRIDE, LAYER_STRIDE, LOGSTATE, NUM_GRANULES, FRAME_ELEMS,
            )
        if STORE_FINAL:
            final_slot = tl.load(final_slots + request)
            if not LOGSTATE or CONV_FRAMES == 1:
                final_destination = _conv_destination(
                    state_storage, conv_frames, final_slot, 0, C, S, LAYER,
                    SLOT_STRIDE, LAYER_STRIDE, LOGSTATE, NUM_GRANULES, FRAME_ELEMS,
                )
    result = tl.full((BT, BC), 0, tl.float32)
    for tap in range(K):
        source = t + tap - S
        old = source < 0
        previous = tl.load(
            history + request * C * S + c[None, :] * S + source[:, None] + S,
            mask=active & old[:, None], other=0,
        )
        current = tl.load(
            x + (begin + source[:, None]) * X_ROW_STRIDE + c[None, :] * X_COL_STRIDE,
            mask=active & ~old[:, None], other=0,
        )
        current = _scale_conv_input(current, c[None, :], END0, END1, SCALE0, SCALE1, SCALE2)
        w = tl.load(weight + c * K + tap, mask=c < C, other=0)
        values = tl.where(old[:, None], previous, current)
        result += values.to(tl.float32) * w[None, :].to(tl.float32)
        if STORE_STATE:
            if tap > 0:
                if LOGSTATE and CONV_FRAMES > 1:
                    target = conv_pointer(state_storage, conv_frames, slot[:, None], granule[:, None],
                                          LAYER * C * S + c[None, :] * S + tap - 1,
                                          NUM_GRANULES, CONV_FRAMES, FRAME_ELEMS, active)
                else:
                    target = destination[:, None] + c[None, :] * S + tap - 1
                tl.store(
                    target,
                    values, mask=active & (slot[:, None] >= 0),
                )
                if STORE_FINAL:
                    if LOGSTATE and CONV_FRAMES > 1:
                        final_target = conv_pointer(state_storage, conv_frames, final_slot, 0,
                                                    LAYER * C * S + c[None, :] * S + tap - 1,
                                                    NUM_GRANULES, CONV_FRAMES, FRAME_ELEMS, c[None, :] < C)
                    else:
                        final_target = final_destination + c[None, :] * S + tap - 1
                    final_target = final_target + tl.full((BT, 1), 0, tl.int32)
                    tl.store(
                        final_target,
                        values, mask=active & (t[:, None] == end - begin - 1) & (final_slot >= 0),
                    )
    if HAS_BIAS:
        result += tl.load(bias + c, mask=c < C, other=0)[None, :].to(tl.float32)
    result = result.to(output.dtype.element_ty).to(tl.float32)
    result = result / (1.0 + tl.exp(-result))
    tl.store(output + (begin + t[:, None]) * C + c[None, :], result, mask=active)


def packed_causal_conv1d(
    x            : torch.Tensor,
    weight       : torch.Tensor,
    history      : torch.Tensor,
    cu_seqlens   : torch.Tensor,
    chunk_indices: torch.Tensor,
    bias         : torch.Tensor | None = None,
    *,
    state_io     : PackedConvStateIO | None = None,
    input_boundaries: tuple[int, int] | None = None,
    input_scales : tuple[float, float, float] = (1., 1., 1.),
) -> torch.Tensor:
    total_tokens, channels = x.shape
    kernel_size = weight.shape[-1]
    state_len = kernel_size - 1
    assert history.shape == (len(cu_seqlens) - 1, channels, state_len)
    assert weight.shape == (channels, 1, kernel_size)
    direct = state_io is not None
    logstate = direct and state_io.frame_table is not None
    if direct:
        assert state_io.slots.shape == (total_tokens,)
        assert state_io.storage.dtype == x.dtype and state_io.storage.device == x.device
        if logstate:
            assert state_io.granules is not None and state_io.granules.shape == (total_tokens,)
            assert state_io.frame_bytes % x.element_size() == 0
        else:
            assert state_io.storage.ndim == 4 and state_io.storage.shape[2:] == (channels, state_len)
        if state_io.final_slots is not None:
            assert state_io.final_slots.shape == (len(cu_seqlens) - 1,)
    output = torch.empty((total_tokens, channels), device=x.device, dtype=x.dtype)
    _packed_conv_kernel[(len(chunk_indices), triton.cdiv(channels, 32))](
        x, weight, bias, history, cu_seqlens, chunk_indices, output,
        state_io.storage if direct else None,
        state_io.slots if direct else None,
        state_io.granules if logstate else None,
        state_io.final_slots if direct else None,
        state_io.frame_table if logstate else None,
        channels, kernel_size, state_len, x.stride(0), x.stride(1),
        64, 32, bias is not None, direct, logstate,
        direct and state_io.final_slots is not None,
        state_io.layer if direct else 0,
        state_io.storage.stride(0) if direct and not logstate else 0,
        state_io.storage.stride(1) if direct and not logstate else 0,
        state_io.frame_table.shape[1] if logstate else 0,
        state_io.frame_bytes // x.element_size() if logstate else 0,
        state_io.frame_table.shape[2] if logstate else 1,
        *_conv_input_scale_args(channels, input_boundaries, input_scales),
        num_warps=4,
    )
    return output


__all__ = ["PackedConvStateIO", "packed_causal_conv1d"]
