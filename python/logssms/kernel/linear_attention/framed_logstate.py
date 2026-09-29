from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from logssms.tvm import call_cuda_ffi, load_jit, make_cpp_args

if TYPE_CHECKING:
    from logssms.cache.logstate_pool.frame_pool import FramedMHALogStatePool


@functools.cache
def _framed_module(page_size: int, head_k_dim: int, head_v_dim: int):
    args = make_cpp_args(page_size, head_k_dim, head_v_dim)
    return load_jit(
        "framed_logstate", *args,
        cuda_files=["framed_logstate.cu"],
        cuda_wrappers=[("store", f"FramedLogStateKernel<{args}>::store")],
    )


@functools.cache
def _construction_module(page_size: int, head_k_dim: int, head_v_dim: int):
    args = make_cpp_args(page_size, head_k_dim, head_v_dim)
    return load_jit(
        "framed_construction", *args,
        cuda_files=["framed_construction.cu"],
        cuda_wrappers=[
            ("restore", f"FramedConstructionKernel<{args}>::restore"),
            ("construct", f"FramedConstructionKernel<{args}>::construct"),
        ],
    )


def _construction_args(pool: FramedMHALogStatePool) -> tuple:
    return (
        pool.storage,
        pool.payload_frame_table,
        pool.decay_frame_table,
    )


def _construction_dims(pool: FramedMHALogStatePool) -> tuple:
    return (
        pool.layout.num_linear_layers,
        pool.layout.num_value_heads,
        pool.head_payload_bytes,
        pool.layout.log_decay_bytes,
        pool.dtype == torch.bfloat16,
    )


def _vector(values: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device=device)


def restore_framed_log_state_chains(
    pool  : FramedMHALogStatePool,
    chains: Sequence[tuple[int | None, Sequence[tuple[int, int, int]], int]],
) -> None:
    if not chains:
        return
    module = _construction_module(
        pool.page_size, pool.layout.head_k_dim, pool.layout.head_v_dim
    )
    segments = [entry for _base, logs, _out in chains for entry in logs]
    offsets = [0]
    for _base, logs, _out in chains:
        offsets.append(offsets[-1] + len(logs))
    call_cuda_ffi(module.restore,
        *_construction_args(pool),
        _vector([-1 if base is None else base for base, _logs, _out in chains], pool.device),
        _vector(offsets, pool.device),
        _vector([slot for slot, _begin, _end in segments], pool.device),
        _vector([begin for _slot, begin, _end in segments], pool.device),
        _vector([end for _slot, _begin, end in segments], pool.device),
        _vector([output for _base, _logs, output in chains], pool.device),
        *_construction_dims(pool),
    )


def construct_framed_log_state_chains(
    pool     : FramedMHALogStatePool,
    chains   : Sequence[
        tuple[int | None, Sequence[tuple[int, int, int]], Sequence[Sequence[int]]]
    ],
    *,
    wave_size: int = 8,
    max_tasks: int = 8,
) -> int:
    if not chains:
        return 0
    assert wave_size > 0 and max_tasks > 0
    module = _construction_module(
        pool.page_size, pool.layout.head_k_dim, pool.layout.head_v_dim
    )
    layout = pool.layout
    peak = 0
    for task_begin in range(0, len(chains), max_tasks):
        group = chains[task_begin:task_begin + max_tasks]
        carry = torch.empty(
            (len(group), layout.num_linear_layers, layout.num_value_heads,
             layout.head_k_dim, layout.head_v_dim),
            dtype=pool.dtype, device=pool.device,
        )
        carry_bytes = carry.numel() * carry.element_size()
        peak = max(peak, carry_bytes)
        for base, segments, outputs in group:
            assert len(segments) == len(outputs)
        max_segments = max(len(segments) for _base, segments, _outputs in group)
        for wave_begin in range(0, max_segments, wave_size):
            current = []
            current_outputs = []
            task_offsets = [0]
            base_slots = []
            for base, segments, outputs in group:
                wave = segments[wave_begin:wave_begin + wave_size]
                current.extend(wave)
                current_outputs.extend(outputs[wave_begin:wave_begin + wave_size])
                task_offsets.append(len(current))
                base_slots.append(
                    (-1 if base is None else base) if wave_begin == 0 else -2
                )
            count = len(current)
            gamma = torch.empty(
                (count, layout.num_linear_layers, layout.num_value_heads),
                dtype=torch.float32, device=pool.device,
            )
            bias = torch.empty(
                (count, layout.num_linear_layers, layout.num_value_heads,
                 layout.head_k_dim, layout.head_v_dim),
                dtype=torch.float32, device=pool.device,
            )
            output_offsets = [0]
            output_slots = []
            for values in current_outputs:
                output_slots.extend(values)
                output_offsets.append(len(output_slots))
            call_cuda_ffi(module.construct,
                *_construction_args(pool),
                _vector(base_slots, pool.device),
                _vector(task_offsets, pool.device),
                _vector([slot for slot, _begin, _end in current], pool.device),
                _vector([begin for _slot, begin, _end in current], pool.device),
                _vector([end for _slot, _begin, end in current], pool.device),
                _vector(output_offsets, pool.device),
                _vector(output_slots, pool.device),
                gamma,
                bias,
                carry,
                *_construction_dims(pool),
            )
            peak = max(
                peak,
                carry_bytes
                + gamma.numel() * gamma.element_size()
                + bias.numel() * bias.element_size(),
            )
    return peak


def store_framed_log_states(
    pool           : FramedMHALogStatePool,
    slots          : torch.Tensor,
    token_offsets  : torch.Tensor,
    keys           : torch.Tensor,
    residual_values: torch.Tensor,
    decays         : torch.Tensor,
    layer          : int,
) -> None:
    if len(slots) == 0:
        return
    assert keys.is_cuda and keys.device == pool.device
    module = _framed_module(
        pool.page_size, pool.layout.head_k_dim, pool.layout.head_v_dim
    )
    call_cuda_ffi(module.store,
        pool.storage,
        pool.payload_frame_table,
        pool.decay_frame_table,
        slots.to(torch.int32).contiguous(),
        token_offsets.to(torch.int32).contiguous(),
        keys.contiguous(),
        residual_values.contiguous(),
        decays.float().contiguous(),
        layer,
        pool.head_payload_bytes,
        pool.layout.log_decay_bytes,
    )
