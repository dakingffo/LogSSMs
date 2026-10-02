from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from logssms.tvm import call_cuda_ffi, load_jit, make_cpp_args
from logssms.utils.torch_utils import make_device_vectors

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
    vectors = make_device_vectors([
        [-1 if base is None else base for base, _logs, _out in chains],
        offsets,
        [slot for slot, _begin, _end in segments],
        [begin for _slot, begin, _end in segments],
        [end for _slot, _begin, end in segments],
        [output for _base, _logs, output in chains],
    ], pool.device)
    call_cuda_ffi(module.restore,
        *_construction_args(pool),
        *vectors,
        *_construction_dims(pool),
    )


def construct_framed_log_state_chains(
    pool  : FramedMHALogStatePool,
    chains: Sequence[
        tuple[int | None, Sequence[tuple[int, int, int]], Sequence[Sequence[int]]]
    ],
    *,
    max_tasks: int | None = None,
) -> int:
    if not chains:
        return 0
    if max_tasks is None:
        max_tasks = torch.cuda.get_device_properties(pool.device).multi_processor_count
    assert max_tasks > 0
    module = _construction_module(
        pool.page_size, pool.layout.head_k_dim, pool.layout.head_v_dim
    )
    for task_begin in range(0, len(chains), max_tasks):
        group = chains[task_begin:task_begin + max_tasks]
        segments = []
        output_slots = []
        task_offsets = [0]
        output_offsets = [0]
        for _base, logs, outputs in group:
            assert len(logs) == len(outputs)
            segments.extend(logs)
            task_offsets.append(len(segments))
            for slots in outputs:
                output_slots.extend(slots)
                output_offsets.append(len(output_slots))
        vectors = make_device_vectors([
            [-1 if base is None else base for base, _logs, _outputs in group],
            task_offsets,
            [slot for slot, _begin, _end in segments],
            [begin for _slot, begin, _end in segments],
            [end for _slot, _begin, end in segments],
            output_offsets,
            output_slots,
        ], pool.device)
        call_cuda_ffi(module.construct,
            *_construction_args(pool),
            *vectors,
            *_construction_dims(pool),
        )
    return 0


def store_framed_log_states(
    pool           : FramedMHALogStatePool,
    slots          : torch.Tensor,
    token_offsets  : torch.Tensor,
    keys           : torch.Tensor,
    residual_values: torch.Tensor,
    decays         : torch.Tensor,
    layer          : int,
) -> None:
    """Store normalized K, residual V and FP32 logarithmic decays."""
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
