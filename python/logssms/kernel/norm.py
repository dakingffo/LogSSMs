from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _rms_norm_gated_kernel(
    hidden_states,
    gate,
    weight,
    output,
    num_cols,
    eps,
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
        gate + row * num_cols + offsets,
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

    if not hidden_states.is_cuda:
        hidden = hidden_states.float()
        variance = hidden.pow(2).mean(-1, keepdim=True)
        hidden *= torch.rsqrt(variance + eps)
        hidden *= weight.float()
        hidden *= F.silu(gate.float())
        return hidden.to(hidden_states.dtype)

    assert gate.is_cuda and weight.is_cuda
    num_cols = hidden_states.shape[-1]
    block_size = triton.next_power_of_2(num_cols)
    assert block_size <= 65536, f"RMSNorm width is too large for one Triton block: {num_cols}"

    hidden = hidden_states.reshape(-1, num_cols).contiguous()
    gate = gate.reshape(-1, num_cols).contiguous()
    weight = weight.contiguous()
    output = torch.empty_like(hidden)
    _rms_norm_gated_kernel[(hidden.shape[0],)](
        hidden,
        gate,
        weight,
        output,
        num_cols,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return output.view(hidden_states.shape)
