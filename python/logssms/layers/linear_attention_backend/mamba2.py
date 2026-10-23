from __future__ import annotations

import torch
from dataclasses import replace

from logssms.core import get_global_ctx
from logssms.kernel import store_state
from logssms.kernel.linear_attention.mamba2 import (
    mamba2_conv_decode, mamba2_decode, mamba2_prefill, prepare_mamba2,
)
from logssms.kernel.linear_attention.packed_conv import packed_causal_conv1d, packed_conv_snapshots

from .base import BaseMamba2Backend
from .standard import StandardLinearAttentionBackend


class Mamba2Backend(StandardLinearAttentionBackend, BaseMamba2Backend):
    def forward(self, xbc, dt, conv_weight, conv_bias, A_log, dt_bias, D, config, batch, out=None):
        metadata = batch.linear_attn_metadata
        pool = get_global_ctx().linear_state
        if hasattr(pool, "group_table"):
            metadata, lane = pool.map_metadata(metadata, config.layer_idx)
            config = replace(config, layer_idx=lane)
        if batch.is_verify:
            raise NotImplementedError("Mamba2 speculative verify is not implemented")
        if batch.is_decode:
            convolved = mamba2_conv_decode(
                xbc, conv_weight, conv_bias, config, metadata, pool, logstate=False,
            )
            return mamba2_decode(
                convolved, dt, A_log, dt_bias, D, config, metadata, pool, logstate=False, out=out,
            )
        indices = metadata.prefill_indices
        slots = metadata.baseline_slots
        live = slots >= 0
        history = pool.conv_state[:, config.layer_idx].index_select(0, slots.clamp_min(0))
        history = torch.where(live[:, None, None], history, 0)
        initial = pool.recurrent_state[:, config.layer_idx].index_select(0, slots.clamp_min(0))
        initial = torch.where(live[:, None, None, None], initial, 0)
        snapshots = packed_conv_snapshots(
            xbc, history, indices.cu_seqlens, metadata.snapshot_rows, metadata.snapshot_reqs,
        )
        convolved = packed_causal_conv1d(
            xbc, conv_weight, history, indices.cu_seqlens, indices.conv_chunks, bias=conv_bias,
        )
        keys, queries, values, decay = prepare_mamba2(convolved, dt, A_log, dt_bias, config)
        output, _, states = mamba2_prefill(
            convolved, keys, queries, values, decay, D, initial, indices, 64, out=out,
        )
        store_state(pool.conv_state[:, config.layer_idx], metadata.checkpoint_slots, snapshots)
        store_state(
            pool.recurrent_state[:, config.layer_idx], metadata.checkpoint_slots,
            states.index_select(0, metadata.checkpoint_chunks).to(pool.dtype),
        )
        return output
