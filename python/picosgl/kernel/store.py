from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from picosgl.tvm import load_jit, make_cpp_args

from .utils import KernelConfig

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(
    num_threads=128,
    max_occupancy=1,
    use_pdl=False
)


@functools.cache
def _jit_store_module(
    element_size: int,
    *,
    config      : KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, *config)
    return load_jit(
        "store", "kv",
        *args,
        cuda_files=["store.cu"],
        cuda_wrappers=[("launch", f"KVStoreKernel<{args}>::run")],
    )


@functools.cache
def _jit_state_store_module(
    element_size: int,
    *,
    config      : KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, *config)
    return load_jit(
        "store", "state",
        *args,
        cuda_files=["store.cu"],
        cuda_wrappers=[("launch", f"StateStoreKernel<{args}>::run")],
    )


def store_cache(
    k_cache: torch.Tensor, # [N, H, D]
    v_cache: torch.Tensor, # [N, H, D]
    indices: torch.Tensor, # [T]
    k      : torch.Tensor, # [T, H, D]
    v      : torch.Tensor, # [T, H ,D]
) -> None:
    num_tokens = k_cache.shape[0]
    k_cache = k_cache.view(num_tokens, -1)
    v_cache = v_cache.view(num_tokens, -1)
    k = k.view(k.shape[0], -1)
    v = v.view(v.shape[0], -1)
    element_size = k_cache.shape[1] * k_cache.element_size()
    module = _jit_store_module(element_size)
    module.launch(k_cache, v_cache, indices, k, v)


def store_state(
    pool   : torch.Tensor, # [N, ...]
    indices: torch.Tensor, # [T]
    states : torch.Tensor, # [T, ...]
) -> None:
    assert pool.device == states.device == indices.device
    assert pool.dtype == states.dtype
    num_slots = pool.shape[0]
    pool = pool.view(num_slots, -1)
    states = states.view(states.shape[0], -1)
    assert pool.shape[1] == states.shape[1]
    element_size = pool.shape[1] * pool.element_size()
    assert element_size % 128 == 0
    module = _jit_state_store_module(element_size)
    module.launch(pool, indices, states)
