from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _packed_conv_kernel(
    x, weight, bias, history, offsets, chunks, output,
    C: tl.constexpr, K: tl.constexpr, S: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr, X_COL_STRIDE: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    chunk = tl.program_id(0)
    request = tl.load(chunks + chunk * 2).to(tl.int32)
    chunk_index = tl.load(chunks + chunk * 2 + 1).to(tl.int32)
    begin = tl.load(offsets + request).to(tl.int32)
    end = tl.load(offsets + request + 1).to(tl.int32)
    t = chunk_index * BT + tl.arange(0, BT)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    active = (t[:, None] < end - begin) & (c[None, :] < C)
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
        w = tl.load(weight + c * K + tap, mask=c < C, other=0)
        result += tl.where(old[:, None], previous, current).to(tl.float32) * w[None, :].to(tl.float32)
    if HAS_BIAS:
        result += tl.load(bias + c, mask=c < C, other=0)[None, :].to(tl.float32)
    result = result.to(output.dtype.element_ty).to(tl.float32)
    result = result / (1.0 + tl.exp(-result))
    tl.store(output + (begin + t[:, None]) * C + c[None, :], result, mask=active)


@triton.jit
def _packed_snapshots_kernel(
    x, history, offsets, rows, requests, output,
    C: tl.constexpr, S: tl.constexpr, BS: tl.constexpr, BC: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr, X_COL_STRIDE: tl.constexpr,
):
    snapshot = tl.program_id(0)
    request = tl.load(requests + snapshot).to(tl.int32)
    begin = tl.load(offsets + request).to(tl.int32)
    row = tl.load(rows + snapshot).to(tl.int32) - begin
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    s = tl.arange(0, BS)
    source = row - S + 1 + s
    old = source < 0
    previous = tl.load(
        history + request * C * S + c[:, None] * S + source[None, :] + S,
        mask=(c[:, None] < C) & (s[None, :] < S) & old[None, :], other=0,
    )
    current = tl.load(
        x + (begin + source[None, :]) * X_ROW_STRIDE + c[:, None] * X_COL_STRIDE,
        mask=(c[:, None] < C) & (s[None, :] < S) & ~old[None, :], other=0,
    )
    tl.store(
        output + (snapshot * C + c[:, None]) * S + s[None, :],
        tl.where(old[None, :], previous, current),
        mask=(c[:, None] < C) & (s[None, :] < S),
    )


def packed_causal_conv1d(
    x            : torch.Tensor,
    weight       : torch.Tensor,
    history      : torch.Tensor,
    cu_seqlens   : torch.Tensor,
    chunk_indices: torch.Tensor,
    bias         : torch.Tensor | None = None,
) -> torch.Tensor:
    """Depthwise causal conv over packed requests, without per-request launches."""
    total_tokens, channels = x.shape
    kernel_size = weight.shape[-1]
    state_len = kernel_size - 1
    assert history.shape == (len(cu_seqlens) - 1, channels, state_len)
    assert weight.shape == (channels, 1, kernel_size)
    output = torch.empty((total_tokens, channels), device=x.device, dtype=x.dtype)
    _packed_conv_kernel[(len(chunk_indices), triton.cdiv(channels, 32))](
        x, weight, bias, history, cu_seqlens, chunk_indices, output,
        channels, kernel_size, state_len, x.stride(0), x.stride(1),
        64, 32, bias is not None, num_warps=4,
    )
    return output


def packed_conv_snapshots(
    x         : torch.Tensor,
    history   : torch.Tensor,
    cu_seqlens: torch.Tensor,
    rows      : torch.Tensor,
    requests  : torch.Tensor,
) -> torch.Tensor:
    _, channels = x.shape
    state_len = history.shape[-1]
    output = torch.empty(
        len(rows), channels, state_len, device=x.device, dtype=x.dtype
    )
    _packed_snapshots_kernel[(len(rows), triton.cdiv(channels, 32))](
        x, history, cu_seqlens, rows, requests, output,
        channels, state_len, triton.next_power_of_2(state_len), 32,
        x.stride(0), x.stride(1),
        num_warps=4,
    )
    return output


__all__ = ["packed_causal_conv1d", "packed_conv_snapshots"]
