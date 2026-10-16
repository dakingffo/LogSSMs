from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from logssms.cache.logstate_pool.mha_pool import MHALogStatePool
from logssms.core import get_global_ctx
from logssms.kernel import prepare_gdn
from .base import BaseGatedDeltaBackend
from .logstate import LogStateLinearAttentionBackend, LogStateLinearAttentionMetadata

if TYPE_CHECKING:
    from logssms.core import Batch
    from logssms.layers.gated_delta import GatedDeltaConfig


class GatedDeltaLogStateBackend(LogStateLinearAttentionBackend, BaseGatedDeltaBackend):
    def forward(
        self,
        mixed_qkv  : torch.Tensor,
        gate       : torch.Tensor,
        beta       : torch.Tensor,
        conv_weight: torch.Tensor,
        A_log      : torch.Tensor,
        dt_bias    : torch.Tensor,
        config     : GatedDeltaConfig,
        batch      : Batch,
    ) -> torch.Tensor:
        metadata = batch.linear_attn_metadata
        assert isinstance(metadata, LogStateLinearAttentionMetadata)
        assert metadata.phase == batch.phase
        if batch.is_prefill:
            return self._forward_prefill(
                mixed_qkv,
                gate,
                beta,
                conv_weight,
                A_log,
                dt_bias,
                config,
                batch,
                metadata,
            )
        if batch.is_verify:
            raise NotImplementedError(
                "LogState speculative verify is not implemented yet."
            )
        return self._forward_decode(
            mixed_qkv,
            gate,
            beta,
            conv_weight,
            A_log,
            dt_bias,
            config,
            metadata,
        )

    @staticmethod
    def _split_qkv(
        mixed_qkv: torch.Tensor,
        config   : GatedDeltaConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = torch.split(
            mixed_qkv,
            [config.key_dim, config.key_dim, config.value_dim],
            dim=-1,
        )
        seq_len = mixed_qkv.shape[1]
        return (
            query.reshape(
                mixed_qkv.shape[0], seq_len, -1, config.head_k_dim
            ),
            key.reshape(
                mixed_qkv.shape[0], seq_len, -1, config.head_k_dim
            ),
            value.reshape(
                mixed_qkv.shape[0], seq_len, -1, config.head_v_dim
            ),
        )

    def _forward_prefill(
        self,
        mixed_qkv  : torch.Tensor,
        gate       : torch.Tensor,
        beta       : torch.Tensor,
        conv_weight: torch.Tensor,
        A_log      : torch.Tensor,
        dt_bias    : torch.Tensor,
        config     : GatedDeltaConfig,
        batch      : Batch,
        metadata   : LogStateLinearAttentionMetadata,
    ) -> torch.Tensor:
        from logssms.kernel.linear_attention.vendor.fla import (
            ChunkLogIO, ChunkStateIO, chunk_gated_delta_rule_to_cache,
        )
        from logssms.kernel.linear_attention.packed_conv import (
            PackedConvStateIO, packed_causal_conv1d,
        )

        pool = get_global_ctx().linear_state
        assert isinstance(pool, MHALogStatePool)
        layer_idx = config.layer_idx
        assert metadata.cu_seqlens is not None
        assert metadata.prefill_indices is not None
        assert metadata.chunk_write_slots is not None

        conv_history = pool.read_conv(
            metadata.conv_read_slots, metadata.conv_read_granules, layer_idx
        )
        assert metadata.conv_token_slots is not None
        assert metadata.conv_token_granules is not None
        assert metadata.log_write_slots is not None
        mixed_qkv = packed_causal_conv1d(
            mixed_qkv, conv_weight, conv_history,
            metadata.cu_seqlens, metadata.prefill_indices.conv_chunks,
            state_io=PackedConvStateIO(
                storage=pool.storage.view(pool.dtype),
                slots=metadata.conv_token_slots,
                granules=metadata.conv_token_granules,
                final_slots=metadata.work_slots,
                frame_table=pool.conv_frame_table, frame_bytes=pool.frame_bytes,
                layer=layer_idx,
            ),
        ).unsqueeze(0)
        query, key, value = self._split_qkv(mixed_qkv, config)
        query, key, value, prepared_gate, prepared_beta = prepare_gdn(
            query,
            key,
            value,
            gate.unsqueeze(0),
            beta.unsqueeze(0),
            A_log,
            dt_bias,
        )
        output, _ = (
            chunk_gated_delta_rule_to_cache(
                query,
                key,
                value,
                prepared_gate,
                prepared_beta,
                ChunkStateIO(
                    storage=pool.storage.view(pool.dtype),
                    read_slots=metadata.work_slots,
                    chunk_write_slots=metadata.chunk_write_slots,
                    final_write_slots=metadata.work_slots,
                    payload_frames=pool.payload_frame_table,
                    frame_bytes=pool.frame_bytes,
                    head_payload_bytes=pool.head_payload_bytes,
                    layer=layer_idx,
                    num_layers=pool.layout.num_linear_layers,
                    granularity=pool.layout.granularity,
                    page_size=pool.page_size,
                ),
                metadata.cu_seqlens,
                chunk_size=pool.layout.granularity,
                log_io=ChunkLogIO(
                    slots=metadata.log_write_slots,
                    token_offsets=metadata.token_offsets,
                    decay_frames=pool.decay_frame_table,
                    gate=prepared_gate,
                    page_size=pool.page_size,
                    decay_record_bytes=pool.layout.log_decay_bytes,
                    num_key_heads=pool.layout.num_key_heads,
                ),
                **metadata.prefill_indices.fla_kwargs(),
            )
        )
        return output.squeeze(0)

    def _forward_decode(
        self,
        mixed_qkv  : torch.Tensor,
        gate       : torch.Tensor,
        beta       : torch.Tensor,
        conv_weight: torch.Tensor,
        A_log      : torch.Tensor,
        dt_bias    : torch.Tensor,
        config     : GatedDeltaConfig,
        metadata   : LogStateLinearAttentionMetadata,
    ) -> torch.Tensor:
        from logssms.kernel.linear_attention.vendor.fla import (
            fused_recurrent_gated_delta_rule_to_logstate,
        )
        from logssms.kernel.linear_attention.gdn_conv_decode import conv_decode_qkv

        pool = get_global_ctx().linear_state
        assert isinstance(pool, MHALogStatePool)
        layer_idx = config.layer_idx
        query, key, value = conv_decode_qkv(
            mixed_qkv, conv_weight, config,
            read_slots=metadata.conv_read_slots,
            read_granules=metadata.conv_read_granules,
            write_slots=metadata.log_slots,
            write_granules=metadata.token_offsets // pool.layout.granularity,
            work_slots=metadata.work_slots,
            pool=pool,
        )
        output = fused_recurrent_gated_delta_rule_to_logstate(
            query,
            key,
            value,
            gate.unsqueeze(1),
            beta.unsqueeze(1),
            A_log,
            dt_bias,
            pool,
            metadata.work_slots,
            metadata.log_slots,
            metadata.token_offsets,
            layer_idx,
            metadata.replay_counts,
            enable_eager_state=self.enable_eager_state,
        )
        return output.squeeze(1)
