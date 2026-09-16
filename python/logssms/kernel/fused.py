from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_and_mul_kernel(
    input,
    gate,
    output,
    gate_token_stride: tl.constexpr,
    gate_head_stride : tl.constexpr,
    num_heads        : tl.constexpr,
    num_cols         : tl.constexpr,
    BLOCK_SIZE       : tl.constexpr,
):
    row = tl.program_id(0)
    token = row // num_heads
    head = row % num_heads
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_cols

    value = tl.load(input + row * num_cols + offsets, mask=mask)
    gate_value = tl.load(
        gate + token * gate_token_stride + head * gate_head_stride + offsets,
        mask=mask,
    ).to(tl.float32)
    result = value.to(tl.float32) * tl.sigmoid(gate_value)
    tl.store(output + row * num_cols + offsets, result, mask=mask)


def sigmoid_and_mul(
    input: torch.Tensor,
    gate : torch.Tensor,
    out  : torch.Tensor | None = None,
) -> torch.Tensor:
    assert input.shape == gate.shape
    assert input.is_cuda and gate.is_cuda
    assert input.is_contiguous()
    assert gate.stride(-1) == 1
    assert gate.ndim in (2, 3)

    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape
        assert out.device == input.device and out.dtype == input.dtype
        assert out.is_contiguous()

    num_cols = input.shape[-1]
    block_size = triton.next_power_of_2(num_cols)
    assert block_size <= 65536, (
        f"sigmoid_and_mul width is too large for one Triton block: {num_cols}"
    )
    num_rows = input.numel() // num_cols
    num_heads = gate.shape[-2] if gate.ndim == 3 else 1
    gate_token_stride = gate.stride(-3) if gate.ndim == 3 else gate.stride(-2)
    gate_head_stride = gate.stride(-2) if gate.ndim == 3 else 0
    _sigmoid_and_mul_kernel[(num_rows,)](
        input,
        gate,
        out,
        gate_token_stride=gate_token_stride,
        gate_head_stride=gate_head_stride,
        num_heads=num_heads,
        num_cols=num_cols,
        BLOCK_SIZE=block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return out
