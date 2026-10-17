from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from logssms.tvm import call_cuda_ffi, load_jit, make_cpp_args
from logssms.utils.torch_utils import make_device_vectors

if TYPE_CHECKING:
    from logssms.cache.logstate_pool.mha_pool import MHALogStatePool


@functools.cache
def _load_logstate_module(page_size: int, head_k_dim: int, head_v_dim: int, granularity: int):
    args = make_cpp_args(page_size, head_k_dim, head_v_dim, granularity)
    return load_jit(
        "logstate", *args,
        cuda_files=["logstate.cu"],
        cuda_wrappers=[("store", f"LogStateKernel<{args}>::store")],
    )


@functools.cache
def _load_construction_module(page_size: int, head_k_dim: int, head_v_dim: int, granularity: int):
    args = make_cpp_args(page_size, head_k_dim, head_v_dim, granularity)
    return load_jit(
        "logstate_construction", *args,
        cuda_files=["logstate_construction.cu"],
        cuda_wrappers=[
            ("restore", f"LogStateConstructionKernel<{args}>::restore"),
            ("copy", f"LogStateConstructionKernel<{args}>::copy"),
        ],
    )


def _construction_args(pool: MHALogStatePool) -> tuple:
    return (
        pool.storage,
        pool.payload_frame_table,
        pool.decay_frame_table,
    )


def _construction_dims(pool: MHALogStatePool) -> tuple:
    return (
        pool.layout.num_linear_layers,
        pool.layout.num_value_heads,
        pool.head_payload_bytes,
        pool.layout.log_decay_bytes,
        pool.layout.num_key_heads,
        pool.dtype == torch.bfloat16,
    )


def restore_log_state_chains(
    pool  : MHALogStatePool,
    chains: Sequence[tuple[int | None, Sequence[tuple[int, int, int]], int]],
) -> None:
    if not chains:
        return
    assert all(pool.allocator.slot(output).kind != "log" for _base, _logs, output in chains), (
        "State output required"
    )
    module = _load_construction_module(
        pool.page_size, pool.layout.head_k_dim, pool.layout.head_v_dim, pool.layout.granularity
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


def store_log_states(
    pool           : MHALogStatePool,
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
    module = _load_logstate_module(
        pool.page_size, pool.layout.head_k_dim, pool.layout.head_v_dim, pool.layout.granularity
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
        pool.layout.num_linear_layers,
        pool.layout.num_key_heads,
    )


def copy_states(pool: MHALogStatePool, sources: Sequence[int], destinations: Sequence[int]) -> None:
    if not sources:
        return
    assert pool.device.type == "cuda"
    module = _load_construction_module(pool.page_size, pool.layout.head_k_dim,
                                  pool.layout.head_v_dim, pool.layout.granularity)
    vectors = make_device_vectors([sources, destinations], pool.device)
    call_cuda_ffi(module.copy, *_construction_args(pool), *vectors, *_construction_dims(pool))
