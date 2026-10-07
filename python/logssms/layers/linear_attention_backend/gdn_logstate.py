from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from logssms.cache.logstate_pool.frame_pool import FramedMHALogStatePool
from logssms.core import get_global_ctx
from logssms.kernel.linear_attention.framed_logstate import store_framed_log_states
from .base import BaseGatedDeltaBackend
from .gdn import _prepare_gdn
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
            ).contiguous(),
            key.reshape(
                mixed_qkv.shape[0], seq_len, -1, config.head_k_dim
            ).contiguous(),
            value.reshape(
                mixed_qkv.shape[0], seq_len, -1, config.head_v_dim
            ).contiguous(),
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
            chunk_gated_delta_rule_with_log,
        )
        from logssms.kernel.linear_attention.packed_conv import (
            packed_causal_conv1d, packed_conv_snapshots,
        )

        pool = get_global_ctx().linear_state
        assert isinstance(pool, FramedMHALogStatePool)
        layer_idx = config.layer_idx
        assert metadata.cu_seqlens is not None
        assert metadata.prefill_indices is not None
        assert metadata.conv_snapshot_rows is not None
        assert metadata.conv_snapshot_reqs is not None
        assert metadata.conv_final_rows is not None
        assert metadata.conv_write_slots is not None
        assert metadata.conv_write_granules is not None
        conv_history = pool.read_conv(
            metadata.conv_read_slots, metadata.conv_read_granules, layer_idx
        )
        conv_snapshots = packed_conv_snapshots(
            mixed_qkv, conv_history, metadata.cu_seqlens,
            metadata.conv_snapshot_rows, metadata.conv_snapshot_reqs,
        )
        mixed_qkv = packed_causal_conv1d(
            mixed_qkv, conv_weight, conv_history,
            metadata.cu_seqlens, metadata.prefill_indices.conv_chunks,
        ).unsqueeze(0)
        query, key, value = self._split_qkv(mixed_qkv, config)
        query, key, value, prepared_gate, prepared_beta = _prepare_gdn(
            query,
            key,
            value,
            gate.unsqueeze(0),
            beta.unsqueeze(0),
            A_log,
            dt_bias,
        )
        output, final_state, chunk_states, residual_value = (
            chunk_gated_delta_rule_with_log(
                query,
                key,
                value,
                prepared_gate,
                prepared_beta,
                pool.read_states(metadata.work_slots, layer_idx),
                metadata.cu_seqlens,
                chunk_size=pool.layout.granularity,
                **metadata.prefill_indices.fla_kwargs(),
            )
        )
        log_rows = metadata.log_row_indices
        log_decays = prepared_gate.squeeze(0)
        if log_rows is None:
            log_slots = metadata.log_slots
            token_offsets = metadata.token_offsets
            log_keys = key.squeeze(0)
            log_values = residual_value.squeeze(0)
        else:
            log_slots = metadata.log_slots.index_select(0, log_rows)
            token_offsets = metadata.token_offsets.index_select(0, log_rows)
            log_keys = key.squeeze(0).index_select(0, log_rows)
            log_values = residual_value.squeeze(0).index_select(0, log_rows)
            log_decays = log_decays.index_select(0, log_rows)
        store_framed_log_states(
            pool,
            log_slots, token_offsets, log_keys, log_values, log_decays,
            layer_idx,
        )
        pool.write_states(metadata.work_slots, layer_idx, final_state.to(pool.dtype))
        if metadata.checkpoint_slots is not None:
            assert metadata.checkpoint_state_indices is not None
            pool.write_states(
                metadata.checkpoint_slots, layer_idx,
                chunk_states.index_select(0, metadata.checkpoint_state_indices).to(pool.dtype),
            )
        pool.write_conv(
            metadata.conv_write_slots,
            metadata.conv_write_granules,
            layer_idx,
            conv_snapshots if metadata.conv_write_snapshot_indices is None
            else conv_snapshots.index_select(0, metadata.conv_write_snapshot_indices),
        )
        if metadata.checkpoint_slots is not None:
            assert metadata.checkpoint_snapshot_indices is not None
            pool.write_conv(
                metadata.checkpoint_slots,
                torch.zeros_like(metadata.checkpoint_slots),
                layer_idx,
                conv_snapshots.index_select(0, metadata.checkpoint_snapshot_indices),
            )
        pool.write_conv(
            metadata.work_slots,
            torch.zeros_like(metadata.work_slots),
            layer_idx,
            conv_snapshots.index_select(0, metadata.conv_final_rows),
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
            fused_recurrent_gated_delta_rule_to_framed_logstate,
        )
        from logssms.kernel.linear_attention.gdn_conv_decode import conv_decode_qkv

        pool = get_global_ctx().linear_state
        assert isinstance(pool, FramedMHALogStatePool)
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
        output = fused_recurrent_gated_delta_rule_to_framed_logstate(
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
