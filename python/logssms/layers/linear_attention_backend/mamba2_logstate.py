from __future__ import annotations

import torch

from logssms.core import get_global_ctx
from logssms.kernel.linear_attention.logstate import store_log_states
from logssms.kernel.linear_attention.mamba2 import (
    mamba2_conv_decode, mamba2_decode, mamba2_prefill, prepare_mamba2,
)
from logssms.kernel.linear_attention.packed_conv import packed_causal_conv1d, packed_conv_snapshots

from .base import BaseMamba2Backend
from .logstate import LogStateLinearAttentionBackend


class Mamba2LogStateBackend(LogStateLinearAttentionBackend, BaseMamba2Backend):
    def forward(self, xbc, dt, conv_weight, conv_bias, A_log, dt_bias, D, config, batch):
        metadata = batch.linear_attn_metadata
        pool = get_global_ctx().linear_state
        if batch.is_verify:
            raise NotImplementedError("Mamba2 LogState speculative verify is not implemented")
        if batch.is_decode:
            convolved = mamba2_conv_decode(
                xbc, conv_weight, conv_bias, config, metadata, pool, logstate=True,
            )
            return mamba2_decode(
                convolved, dt, A_log, dt_bias, D, config, metadata, pool,
                logstate=True, eager=self.enable_eager_state,
            )
        history = pool.read_conv(metadata.conv_read_slots, metadata.conv_read_granules, config.layer_idx)
        snapshots = packed_conv_snapshots(
            xbc, history, metadata.cu_seqlens, metadata.conv_snapshot_rows, metadata.conv_snapshot_reqs,
        )
        convolved = packed_causal_conv1d(
            xbc, conv_weight, history, metadata.cu_seqlens, metadata.prefill_indices.conv_chunks,
            bias=conv_bias,
        )
        keys, queries, values, decay = prepare_mamba2(convolved, dt, A_log, dt_bias, config)
        output, final, states = mamba2_prefill(
            convolved, keys, queries, values, decay, D,
            pool.read_states(metadata.work_slots, config.layer_idx),
            metadata.prefill_indices, pool.layout.granularity,
        )
        rows = metadata.log_row_indices
        if rows is None:
            slots, offsets = metadata.log_slots, metadata.token_offsets
        else:
            slots = metadata.log_slots.index_select(0, rows)
            offsets = metadata.token_offsets.index_select(0, rows)
            keys, values, decay = (tensor.index_select(0, rows) for tensor in (keys, values, decay))
        store_log_states(pool, slots, offsets, keys, values, decay, config.layer_idx)
        pool.write_states(metadata.work_slots, config.layer_idx, final.to(pool.dtype))
        if metadata.checkpoint_slots is not None:
            pool.write_states(
                metadata.checkpoint_slots, config.layer_idx,
                states.index_select(0, metadata.checkpoint_state_indices).to(pool.dtype),
            )
            pool.write_conv(
                metadata.checkpoint_slots, metadata.checkpoint_conv_granules, config.layer_idx,
                snapshots.index_select(0, metadata.checkpoint_snapshot_indices),
            )
        pool.write_conv(
            metadata.conv_write_slots, metadata.conv_write_granules, config.layer_idx,
            snapshots if metadata.conv_write_snapshot_indices is None
            else snapshots.index_select(0, metadata.conv_write_snapshot_indices),
        )
        pool.write_conv(
            metadata.work_slots, torch.zeros_like(metadata.work_slots), config.layer_idx,
            snapshots.index_select(0, metadata.conv_final_rows),
        )
        return output
