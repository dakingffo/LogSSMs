from __future__ import annotations

import torch
from dataclasses import replace

from logssms.core import get_global_ctx
from logssms.kernel.linear_attention.state_io import ChunkStateIO
from logssms.kernel.linear_attention.mamba2 import (
    mamba2_decode, mamba2_prefill, prepare_mamba2,
)
from logssms.kernel.linear_attention.conv_decode import conv_decode
from logssms.kernel.linear_attention.packed_conv import PackedConvStateIO, packed_causal_conv1d

from .base import BaseMamba2Backend
from .standard import LinearAttnMetadata, StandardLinearAttnBackend


class Mamba2Backend(StandardLinearAttnBackend, BaseMamba2Backend):
    def forward(self, xbc, dt, conv_weight, conv_bias, A_log, dt_bias, D, config, batch, out=None):
        metadata = batch.linear_attn_metadata
        pool = get_global_ctx().linear_state
        if hasattr(pool, "group_table"):
            metadata, lane = pool.map_metadata(metadata, config.layer_idx)
            config = replace(config, layer_idx=lane)
        assert isinstance(metadata, LinearAttnMetadata)
        assert metadata.phase == batch.phase
        if batch.is_prefill:
            return self._forward_prefill(
                xbc, dt, conv_weight, conv_bias, A_log, dt_bias, D, config, metadata, pool, out,
            )
        if batch.is_verify:
            raise NotImplementedError("Mamba2 speculative verify is not implemented")
        return self._forward_decode(
            xbc, dt, conv_weight, conv_bias, A_log, dt_bias, D, config, metadata, pool, out,
        )

    def _forward_prefill(self, xbc, dt, conv_weight, conv_bias, A_log, dt_bias, D,
                         config, metadata, pool, out):
        indices = metadata.prefill_indices
        slots = metadata.baseline_slots
        live = slots >= 0
        history = pool.conv_state[:, config.layer_idx].index_select(0, slots.clamp_min(0))
        history = torch.where(live[:, None, None], history, 0)
        convolved = packed_causal_conv1d(
            xbc, conv_weight, history, indices.cu_seqlens, indices.conv_chunks, bias=conv_bias,
            input_boundaries=config.conv_boundaries, input_scales=config.conv_scales,
            state_io=PackedConvStateIO(
                storage=pool.conv_state, slots=metadata.conv_token_slots, layer=config.layer_idx,
            ),
        )
        keys, queries, values, decay = prepare_mamba2(convolved, dt, A_log, dt_bias, config)
        return mamba2_prefill(
            convolved, keys, queries, values, decay, D,
            ChunkStateIO(storage=pool.recurrent_state, read_slots=slots,
                         chunk_write_slots=metadata.chunk_write_slots, layer=config.layer_idx),
            indices, 64, out=out,
        )

    def _forward_decode(self, xbc, dt, conv_weight, conv_bias, A_log, dt_bias, D,
                        config, metadata, pool, out):
        convolved = conv_decode(
            xbc, conv_weight, config, bias=conv_bias, round_conv=True,
            read_slots=metadata.read_slots, write_slots=metadata.write_slots,
            standard_state=pool.conv_state,
            input_boundaries=config.conv_boundaries, input_scales=config.conv_scales,
        )
        return mamba2_decode(
            convolved, dt, A_log, dt_bias, D, config, metadata, pool, logstate=False, out=out,
        )
