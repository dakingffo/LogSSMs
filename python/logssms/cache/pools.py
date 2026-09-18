from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from logssms.distributed import DistributedInfo, try_get_tp_info
from logssms.utils import div_even

from .kv_pool.base import BaseKVCachePool
from .linear_pool.logstate_pool import LogStatePool, LogStateSlotLayout
from .linear_pool.state_pool import LinearStatePool

if TYPE_CHECKING:
    from logssms.models import ModelConfig


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


def local_linear_head_counts(model_config: ModelConfig) -> tuple[int, int]:
    tp_size = (try_get_tp_info() or _TP_DEFAULT).size
    return (
        div_even(model_config.linear_num_key_heads, tp_size),
        div_even(model_config.linear_num_value_heads, tp_size),
    )


def linear_state_slot_bytes_for_config(
    model_config: ModelConfig,
    dtype       : torch.dtype,
) -> int:
    if not model_config.is_hybrid:
        return 0
    num_k_heads, num_v_heads = local_linear_head_counts(model_config)
    conv_dim = (
        num_k_heads * model_config.linear_key_head_dim * 2
        + num_v_heads * model_config.linear_value_head_dim
    )
    per_layer = (
        conv_dim * (model_config.linear_conv_kernel_dim - 1)
        + num_v_heads
        * model_config.linear_key_head_dim
        * model_config.linear_value_head_dim
    )
    return model_config.num_linear_layers * per_layer * dtype.itemsize


def make_linear_state_pool(
    model_config: ModelConfig,
    num_slots   : int,
    device      : torch.device,
    dtype       : torch.dtype,
) -> LinearStatePool:
    num_k_heads, num_v_heads = local_linear_head_counts(model_config)
    conv_dim = (
        num_k_heads * model_config.linear_key_head_dim * 2
        + num_v_heads * model_config.linear_value_head_dim
    )
    return LinearStatePool(
        num_slots=num_slots,
        num_linear_layers=model_config.num_linear_layers,
        conv_dim=conv_dim,
        kernel_size=model_config.linear_conv_kernel_dim,
        num_v_heads=num_v_heads,
        head_k_dim=model_config.linear_key_head_dim,
        head_v_dim=model_config.linear_value_head_dim,
        device=device,
        dtype=dtype,
    )


def make_logstate_pool(
    model_config: ModelConfig,
    num_slots   : int,
    page_size   : int,
    granularity : int,
    device      : torch.device,
    dtype       : torch.dtype,
) -> LogStatePool:
    num_k_heads, num_v_heads = local_linear_head_counts(model_config)
    conv_dim = (
        num_k_heads * model_config.linear_key_head_dim * 2
        + num_v_heads * model_config.linear_value_head_dim
    )
    return LogStatePool(
        num_slots=num_slots,
        page_size=page_size,
        granularity=granularity,
        num_linear_layers=model_config.num_linear_layers,
        num_key_heads=num_k_heads,
        num_value_heads=num_v_heads,
        head_k_dim=model_config.linear_key_head_dim,
        head_v_dim=model_config.linear_value_head_dim,
        conv_dim=conv_dim,
        state_len=model_config.linear_conv_kernel_dim - 1,
        device=device,
        dtype=dtype,
    )


def logstate_slot_bytes_for_config(
    model_config: ModelConfig,
    page_size   : int,
    granularity : int,
    dtype       : torch.dtype,
) -> int:
    if not model_config.is_hybrid:
        return 0
    num_k_heads, num_v_heads = local_linear_head_counts(model_config)
    conv_dim = (
        num_k_heads * model_config.linear_key_head_dim * 2
        + num_v_heads * model_config.linear_value_head_dim
    )
    return LogStateSlotLayout(
        page_size=page_size,
        granularity=granularity,
        num_linear_layers=model_config.num_linear_layers,
        num_key_heads=num_k_heads,
        num_value_heads=num_v_heads,
        head_k_dim=model_config.linear_key_head_dim,
        head_v_dim=model_config.linear_value_head_dim,
        conv_dim=conv_dim,
        state_len=model_config.linear_conv_kernel_dim - 1,
        dtype=dtype,
    ).slot_bytes
