from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_branches_kernel(
    X, Other, Residual, Weight, Output0, Output1, ResidualOut,
    WIDTH: tl.constexpr, XS: tl.constexpr, OS: tl.constexpr,
    EPS: tl.constexpr, HAS_OTHER: tl.constexpr, HAS_RESIDUAL: tl.constexpr,
    IN0: tl.constexpr, IN1: tl.constexpr, OUT0: tl.constexpr, OUT1: tl.constexpr,
    TWO_OUTPUTS: tl.constexpr, ZERO_CENTERED: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < WIDTH
    dtype = X.dtype.element_ty
    x = tl.load(X + row * XS + col, mask, other=0).to(tl.float32)
    if HAS_OTHER:
        y = tl.load(Other + row * OS + col, mask, other=0).to(tl.float32)
        x = ((x * IN0).to(dtype).to(tl.float32)
             + (y * IN1).to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
    if HAS_RESIDUAL:
        x += tl.load(Residual + row * WIDTH + col, mask, other=0).to(tl.float32)
    tl.store(ResidualOut + row * WIDTH + col, x, mask)
    variance = tl.sum(x * x, 0) / WIDTH
    weight = tl.load(Weight + col, mask, other=0).to(tl.float32)
    if ZERO_CENTERED:
        weight += 1.
    normalized = (x * tl.rsqrt(variance + EPS) * weight).to(dtype).to(tl.float32)
    tl.store(Output0 + row * WIDTH + col, normalized * OUT0, mask)
    if TWO_OUTPUTS:
        tl.store(Output1 + row * WIDTH + col, normalized * OUT1, mask)


def rms_norm_branches(x, weight, eps, residual=None, *, other=None,
                      input_scales=(1., 1.), output_scales=(1.,), zero_centered=False):
    assert x.ndim == 2 and x.stride(1) == 1
    assert 1 <= len(output_scales) <= 2
    outputs = torch.empty((len(output_scales), *x.shape), device=x.device, dtype=x.dtype)
    residual_out = torch.empty(x.shape, device=x.device, dtype=x.dtype) if residual is None else residual
    _rms_norm_branches_kernel[(len(x),)](
        x, other, residual, weight, outputs[0], outputs[-1], residual_out,
        x.shape[1], x.stride(0), 0 if other is None else other.stride(0), eps,
        other is not None, residual is not None, *input_scales,
        output_scales[0], output_scales[-1], len(output_scales) == 2,
        zero_centered,
        triton.next_power_of_2(x.shape[1]),
    )
    return tuple(outputs.unbind(0)), residual_out


@triton.jit
def _grouped_rms_norm_gated_kernel(
    X, Gate, Weight, Output, Variance,
    WIDTH: tl.constexpr, GROUPS: tl.constexpr, XS: tl.constexpr, GS: tl.constexpr,
    EPS: tl.constexpr, BEFORE_GATE: tl.constexpr, STATS_ONLY: tl.constexpr,
    EXTERNAL_STATS: tl.constexpr, BLOCK: tl.constexpr,
    GATE_SCALE: tl.constexpr,
):
    row, group = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, BLOCK)
    mask = col < WIDTH
    offset = group * WIDTH + col
    x = tl.load(X + row * XS + offset, mask, other=0).to(tl.float32)
    gate = tl.load(Gate + row * GS + offset, mask, other=0)
    if GATE_SCALE != 1.:
        gate = (gate.to(tl.float32) * GATE_SCALE).to(gate.dtype)
    z = gate.to(tl.float32)
    z *= tl.sigmoid(z)
    if not BEFORE_GATE:
        x *= z
    if EXTERNAL_STATS:
        variance = tl.load(Variance + row * GROUPS + group)
    else:
        variance = tl.sum(x * x, 0) / WIDTH
    if STATS_ONLY:
        tl.store(Variance + row * GROUPS + group, variance)
    else:
        w = tl.load(Weight + offset, mask, other=0).to(tl.float32)
        y = x * tl.rsqrt(variance + EPS) * w
        if BEFORE_GATE:
            y *= z
        tl.store(Output + row * GROUPS * WIDTH + offset, y, mask)


def grouped_rms_norm_gated(x, gate, weight, eps, groups, before_gate, *, reduce_mean=None, gate_scale=1.):
    assert x.ndim == 2 and x.shape == gate.shape
    assert x.stride(1) == gate.stride(1) == 1
    assert x.shape[1] == weight.numel() and x.shape[1] % groups == 0
    width = x.shape[1] // groups
    block = triton.next_power_of_2(width)
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    variance = None
    args = (width, groups, x.stride(0), gate.stride(0), eps, before_gate)
    grid = (x.shape[0], groups)
    if reduce_mean is not None:
        variance = torch.empty(grid, device=x.device, dtype=torch.float32)
        _grouped_rms_norm_gated_kernel[grid](
            x, gate, weight, output, variance, *args, True, False, block, gate_scale,
            num_warps=8 if block >= 2048 else 4,
        )
        variance = reduce_mean(variance)
    _grouped_rms_norm_gated_kernel[grid](
        x, gate, weight, output, variance, *args, False, variance is not None, block, gate_scale,
        num_warps=8 if block >= 2048 else 4,
    )
    return output


@triton.jit
def _rms_norm_gated_kernel(
    hidden_states,
    gate,
    weight,
    output,
    num_cols,
    eps,
    GATE_ROWS: tl.constexpr,
    GATE_GROUP_STRIDE: tl.constexpr,
    GATE_ROW_STRIDE: tl.constexpr,
    GATE_COL_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_cols

    hidden = tl.load(
        hidden_states + row * num_cols + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate_value = tl.load(
        gate + (row // GATE_ROWS) * GATE_GROUP_STRIDE
        + (row % GATE_ROWS) * GATE_ROW_STRIDE + offsets * GATE_COL_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weight_value = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(hidden * hidden, axis=0) / num_cols
    hidden *= tl.rsqrt(variance + eps)
    gate_value *= tl.sigmoid(gate_value)
    result = hidden * weight_value * gate_value

    tl.store(output + row * num_cols + offsets, result, mask=mask)


def rms_norm_gated(
    hidden_states: torch.Tensor,
    gate         : torch.Tensor,
    weight       : torch.Tensor,
    eps          : float,
) -> torch.Tensor:
    assert hidden_states.shape == gate.shape
    assert hidden_states.shape[-1] == weight.numel()

    assert hidden_states.is_cuda and gate.is_cuda and weight.is_cuda
    num_cols = hidden_states.shape[-1]
    block_size = triton.next_power_of_2(num_cols)
    assert block_size <= 65536, f"RMSNorm width is too large for one Triton block: {num_cols}"

    hidden = hidden_states.reshape(-1, num_cols).contiguous()

    gate = (
        gate.reshape(-1, gate.shape[-2], num_cols)
        if gate.ndim > 1 else gate.view(1, 1, num_cols)
    )
    weight = weight.contiguous()
    output = torch.empty_like(hidden)
    _rms_norm_gated_kernel[(hidden.shape[0],)](
        hidden,
        gate,
        weight,
        output,
        num_cols,
        eps,
        GATE_ROWS=gate.shape[1],
        GATE_GROUP_STRIDE=gate.stride(0),
        GATE_ROW_STRIDE=gate.stride(1),
        GATE_COL_STRIDE=gate.stride(2),
        BLOCK_SIZE=block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return output.view(hidden_states.shape)
