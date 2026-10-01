from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from logssms.distributed import DistributedInfo, try_get_tp_info

from .kv_pool.base import BaseKVCachePool
from .linear_pool.state_pool import LinearStatePool

if TYPE_CHECKING:
    from logssms.models import ModelConfig
    from logssms.models.config import LinearCacheSpec


_TP_DEFAULT = DistributedInfo(rank=0, size=1)


def make_kvcache_pool(
    model_config: ModelConfig,
    num_pages   : int,
    page_size   : int,
    dtype       : torch.dtype,
    device      : torch.device,
) -> BaseKVCachePool:
    from .kv_pool.mha_pool import MHAKVCachePool

    return MHAKVCachePool(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_attention_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )

def make_linear_cache_spec(model_config: ModelConfig) -> LinearCacheSpec:
    tp_size = (try_get_tp_info() or _TP_DEFAULT).size
    spec = model_config.make_linear_cache_spec(tp_size)
    assert spec is not None, "Hybrid model must describe its linear cache storage."
    assert spec.num_layers == model_config.num_linear_layers
    return spec


def linear_state_slot_bytes_for_config(
    model_config: ModelConfig,
    dtype       : torch.dtype,
) -> int:
    if not model_config.is_hybrid:
        return 0
    return make_linear_cache_spec(model_config).slot_bytes(dtype.itemsize)


def make_linear_state_pool(
    model_config: ModelConfig,
    num_slots   : int,
    device      : torch.device,
    dtype       : torch.dtype,
) -> LinearStatePool:
    spec = make_linear_cache_spec(model_config)
    num_heads, rows, columns = spec.state_shape
    conv_dim, state_len = spec.conv_shape
    return LinearStatePool(
        num_slots=num_slots,
        num_linear_layers=spec.num_layers,
        conv_dim=conv_dim,
        kernel_size=state_len + 1,
        num_v_heads=num_heads,
        head_k_dim=rows,
        head_v_dim=columns,
        device=device,
        dtype=dtype,
    )
