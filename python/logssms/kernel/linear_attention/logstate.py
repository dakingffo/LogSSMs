from __future__ import annotations

import functools

import torch

from logssms.tvm import load_jit, make_cpp_args


@functools.cache
def _jit_logstate_module(
    page_size: int,
    head_k_dim: int,
    head_v_dim: int,
):
    args = make_cpp_args(page_size, head_k_dim, head_v_dim)
    return load_jit(
        "logstate", *args,
        cuda_files=["logstate.cu"],
        cuda_wrappers=[
            ("store", f"LogStateKernel<{args}>::store"),
            ("construct", f"LogStateKernel<{args}>::construct"),
        ],
    )


def _module_for(
    key_log            : torch.Tensor,
    residual_value_log : torch.Tensor,
    recurrent_state    : torch.Tensor,
):
    assert key_log.is_cuda
    assert residual_value_log.is_cuda and recurrent_state.is_cuda
    assert key_log.shape[:4] == residual_value_log.shape[:4]
    assert key_log.shape[:2] == recurrent_state.shape[:2]
    assert key_log.shape[3] == recurrent_state.shape[2]
    assert key_log.shape[3] == residual_value_log.shape[3], (
        "The in-place LogState kernel requires one key head per value head."
    )
    return _jit_logstate_module(
        key_log.shape[2], key_log.shape[4], residual_value_log.shape[4]
    )


def _int32(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    assert values.device == device
    return values.to(torch.int32).contiguous()


def store_log_states(
    key_log           : torch.Tensor,
    residual_value_log: torch.Tensor,
    decay_log         : torch.Tensor,
    recurrent_state   : torch.Tensor,
    slots             : torch.Tensor,
    token_offsets     : torch.Tensor,
    keys              : torch.Tensor,
    residual_values   : torch.Tensor,
    decays            : torch.Tensor,
    layer             : int,
) -> None:
    """Store token-wise lossless GDN updates into arbitrary LogState pages."""
    if len(slots) == 0:
        return
    module = _module_for(key_log, residual_value_log, recurrent_state)
    device = key_log.device
    module.store(
        key_log,
        residual_value_log,
        decay_log,
        recurrent_state,
        _int32(slots, device),
        _int32(token_offsets, device),
        keys.contiguous(),
        residual_values.contiguous(),
        decays.float().contiguous(),
        layer,
    )


def construct_log_states(
    key_log             : torch.Tensor,
    residual_value_log  : torch.Tensor,
    decay_log           : torch.Tensor,
    recurrent_state     : torch.Tensor,
    base_slots          : torch.Tensor,
    task_segment_offsets: torch.Tensor,
    log_slots           : torch.Tensor,
    log_begins          : torch.Tensor,
    log_ends            : torch.Tensor,
    output_offsets      : torch.Tensor,
    output_slots        : torch.Tensor,
) -> None:
    """Construct work states and checkpoints from one shared affine scan."""
    if len(base_slots) == 0:
        return
    assert key_log.dtype in (torch.float16, torch.bfloat16)
    assert key_log.shape[-1] % 16 == 0
    assert residual_value_log.shape[-1] % 16 == 0
    module = _module_for(key_log, residual_value_log, recurrent_state)
    device = key_log.device
    num_segments = len(log_slots)
    affine_gamma = torch.empty(
        num_segments,
        key_log.shape[1],
        key_log.shape[3],
        dtype=torch.float32,
        device=device,
    )
    affine_bias = torch.empty(
        num_segments,
        key_log.shape[1],
        key_log.shape[3],
        key_log.shape[4],
        residual_value_log.shape[4],
        dtype=torch.float32,
        device=device,
    )
    module.construct(
        key_log,
        residual_value_log,
        decay_log,
        recurrent_state,
        _int32(base_slots, device),
        _int32(task_segment_offsets, device),
        _int32(log_slots, device),
        _int32(log_begins, device),
        _int32(log_ends, device),
        _int32(output_offsets, device),
        _int32(output_slots, device),
        affine_gamma,
        affine_bias,
    )


__all__ = [
    "construct_log_states",
    "store_log_states",
]
