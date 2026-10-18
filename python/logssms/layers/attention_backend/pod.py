# Copyright (c) 2023 FlashInfer team. Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0
from __future__ import annotations

from flashinfer.pod import get_batch_pod_module
from flashinfer.utils import MaskMode, PosEncodingMode, TensorLayout, _get_cache_alibi_slopes_buf


def run_pod(wrapper, q, kv_cache, split, out):
    assert wrapper._kv_layout == "NHD" and wrapper._pos_encoding_mode == "NONE"
    assert wrapper._window_left == -1 and not wrapper._logits_soft_cap
    assert q.is_contiguous() and out.is_contiguous()
    k, v = kv_cache
    module = get_batch_pod_module(
        q.dtype, k.dtype, q.dtype, q.shape[-1], PosEncodingMode.NONE.value,
        False, False, False, wrapper._indptr_type, PosEncodingMode.NONE.value, False, False,
    )
    scale = wrapper._sm_scale if wrapper._sm_scale is not None else q.shape[-1] ** -0.5
    slopes = _get_cache_alibi_slopes_buf(q.shape[1], q.device)
    args = []
    for suffix, begin, end, mask in (
        ("p", 0, split, MaskMode.CAUSAL.value),
        ("d", split, len(q), MaskMode.NON_CAUSAL.value),
    ):
        args.extend((
            getattr(wrapper, f"_float_workspace_buffer_{suffix}"),
            getattr(wrapper, f"_int_workspace_buffer_{suffix}"),
            getattr(wrapper, f"_plan_info_{suffix}"), q[begin:end], k, v,
            getattr(wrapper, f"_qo_indptr_buf_{suffix}"),
            getattr(wrapper, f"_kv_indptr_buf_{suffix}"),
            getattr(wrapper, f"_kv_indices_buf_{suffix}"),
            getattr(wrapper, f"_kv_last_page_len_buf_{suffix}"),
            out[begin:end], None, mask, TensorLayout.NHD.value, -1, None, None,
            slopes, 0.0, scale, 1.0, 1e-4,
        ))
    module.run_tensor(*args, False, wrapper._sm_aware_sched)
    return out
