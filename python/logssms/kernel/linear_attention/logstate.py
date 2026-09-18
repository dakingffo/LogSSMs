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
            ("restore", f"LogStateKernel<{args}>::restore"),
            ("materialize", f"LogStateKernel<{args}>::materialize"),
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


def reconstruct_log_states(
    key_log             : torch.Tensor,
    residual_value_log  : torch.Tensor,
    decay_log           : torch.Tensor,
    recurrent_state     : torch.Tensor,
    conv_state          : torch.Tensor,
    output_slots        : torch.Tensor,
    base_slots          : torch.Tensor,
    task_log_offsets    : torch.Tensor,
    log_slots           : torch.Tensor,
    log_begins          : torch.Tensor,
    log_ends            : torch.Tensor,
    source_conv_slots   : torch.Tensor,
    source_conv_granules: torch.Tensor,
    output_conv_granules: torch.Tensor,
) -> None:
    """Restore independent request work states from LogState intervals."""
    num_tasks = len(output_slots)
    if num_tasks == 0:
        return

    reconstruct_recurrent_states(
        key_log=key_log,
        residual_value_log=residual_value_log,
        decay_log=decay_log,
        recurrent_state=recurrent_state,
        output_slots=output_slots,
        base_slots=base_slots,
        task_log_offsets=task_log_offsets,
        log_slots=log_slots,
        log_begins=log_begins,
        log_ends=log_ends,
    )

    device = key_log.device
    layer_indices = torch.arange(
        conv_state.shape[1], dtype=torch.int64, device=device
    )
    source = conv_state[
        source_conv_slots.to(torch.int64)[:, None],
        layer_indices[None, :],
        source_conv_granules.to(torch.int64)[:, None],
    ]
    conv_state[
        output_slots.to(torch.int64)[:, None],
        layer_indices[None, :],
        output_conv_granules.to(torch.int64)[:, None],
    ] = source


def reconstruct_recurrent_states(
    key_log           : torch.Tensor,
    residual_value_log: torch.Tensor,
    decay_log         : torch.Tensor,
    recurrent_state   : torch.Tensor,
    output_slots      : torch.Tensor,
    base_slots        : torch.Tensor,
    task_log_offsets  : torch.Tensor,
    log_slots         : torch.Tensor,
    log_begins        : torch.Tensor,
    log_ends          : torch.Tensor,
) -> None:
    """Replay LogState intervals without changing convolution snapshots."""
    if len(output_slots) == 0:
        return

    module = _module_for(key_log, residual_value_log, recurrent_state)
    device = key_log.device
    module.restore(
        key_log,
        residual_value_log,
        decay_log,
        recurrent_state,
        _int32(output_slots, device),
        _int32(base_slots, device),
        _int32(task_log_offsets, device),
        _int32(log_slots, device),
        _int32(log_begins, device),
        _int32(log_ends, device),
    )


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


def materialize_log_states_inplace(
    key_log           : torch.Tensor,
    residual_value_log: torch.Tensor,
    decay_log         : torch.Tensor,
    recurrent_state   : torch.Tensor,
    base_slots        : torch.Tensor,
    chain_log_offsets : torch.Tensor,
    log_slots         : torch.Tensor,
    log_begins        : torch.Tensor,
    log_ends          : torch.Tensor,
    checkpoint_flags  : torch.Tensor,
) -> None:
    """Materialize private LogState chains directly into their source slots."""
    if len(base_slots) == 0:
        return

    module = _module_for(key_log, residual_value_log, recurrent_state)
    device = key_log.device
    module.materialize(
        key_log,
        residual_value_log,
        decay_log,
        recurrent_state,
        _int32(base_slots, device),
        _int32(chain_log_offsets, device),
        _int32(log_slots, device),
        _int32(log_begins, device),
        _int32(log_ends, device),
        _int32(checkpoint_flags, device),
    )


__all__ = [
    "materialize_log_states_inplace",
    "reconstruct_log_states",
    "reconstruct_recurrent_states",
    "store_log_states",
]
