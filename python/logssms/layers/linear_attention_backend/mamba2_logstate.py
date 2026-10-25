from __future__ import annotations

from logssms.core import get_global_ctx
from logssms.kernel.linear_attention.state_io import ChunkStateIO, ChunkLogIO
from logssms.kernel.linear_attention.mamba2 import (
    mamba2_decode, mamba2_prefill, prepare_mamba2,
)
from logssms.kernel.linear_attention.conv_decode import conv_decode
from logssms.kernel.linear_attention.packed_conv import PackedConvStateIO, packed_causal_conv1d

from .base import BaseMamba2Backend
from .logstate import LogStateLinearAttnBackend, LogStateLinearAttnMetadata


class Mamba2LogStateBackend(LogStateLinearAttnBackend, BaseMamba2Backend):
    def forward(self, xbc, dt, conv_weight, conv_bias, A_log, dt_bias, D, config, batch, out=None):
        metadata = batch.linear_attn_metadata
        pool = get_global_ctx().linear_state
        assert isinstance(metadata, LogStateLinearAttnMetadata)
        assert metadata.phase == batch.phase
        if batch.is_prefill:
            return self._forward_prefill(
                xbc, dt, conv_weight, conv_bias, A_log, dt_bias, D, config, metadata, pool, out,
            )
        if batch.is_verify:
            raise NotImplementedError("Mamba2 LogState speculative verify is not implemented")
        return self._forward_decode(
            xbc, dt, conv_weight, conv_bias, A_log, dt_bias, D, config, metadata, pool, out,
        )

    def _forward_prefill(self, xbc, dt, conv_weight, conv_bias, A_log, dt_bias, D,
                         config, metadata, pool, out):
        history = pool.read_conv(metadata.conv_read_slots, metadata.conv_read_granules, config.layer_idx)
        convolved = packed_causal_conv1d(
            xbc, conv_weight, history, metadata.cu_seqlens, metadata.prefill_indices.conv_chunks,
            bias=conv_bias,
            input_boundaries=config.conv_boundaries, input_scales=config.conv_scales,
            state_io=PackedConvStateIO(
                storage=pool.storage.view(pool.dtype), slots=metadata.conv_token_slots,
                granules=metadata.conv_token_granules, final_slots=metadata.work_slots,
                frame_table=pool.conv_frame_table, frame_bytes=pool.frame_bytes,
                layer=config.layer_idx,
            ),
        )

        keys, queries, values, decay = prepare_mamba2(convolved, dt, A_log, dt_bias, config)
        return mamba2_prefill(
            convolved, keys, queries, values, decay, D,
            ChunkStateIO(
                storage=pool.storage.view(pool.dtype), read_slots=metadata.work_slots,
                chunk_write_slots=metadata.chunk_write_slots, final_write_slots=metadata.work_slots,
                payload_frames=pool.payload_frame_table, frame_bytes=pool.frame_bytes,
                head_payload_bytes=pool.head_payload_bytes, layer=config.layer_idx,
                num_layers=pool.layout.num_linear_layers, granularity=pool.layout.granularity,
                page_size=pool.page_size,
            ),
            metadata.prefill_indices, pool.layout.granularity, out=out,
            log_io=ChunkLogIO(
                slots=metadata.log_write_slots, token_offsets=metadata.token_offsets,
                decay_frames=pool.decay_frame_table, gate=decay, page_size=pool.page_size,
                decay_record_bytes=pool.layout.log_decay_bytes, num_key_heads=pool.layout.num_key_heads,
            ),
        )

    def _forward_decode(self, xbc, dt, conv_weight, conv_bias, A_log, dt_bias, D,
                        config, metadata, pool, out):
        convolved = conv_decode(
            xbc, conv_weight, config, bias=conv_bias, round_conv=True,
            read_slots=metadata.conv_read_slots, write_slots=metadata.log_slots,
            read_granules=metadata.conv_read_granules,
            write_granules=metadata.token_offsets // pool.layout.granularity,
            work_slots=metadata.work_slots, pool=pool,
            input_boundaries=config.conv_boundaries, input_scales=config.conv_scales,
        )
        return mamba2_decode(
            convolved, dt, A_log, dt_bias, D, config, metadata, pool,
            logstate=True, eager=self.enable_eager_state, out=out,
        )
