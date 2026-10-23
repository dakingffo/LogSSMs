from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import replace

import torch
import torch.nn.functional as F

from logssms.core import get_global_ctx
from logssms.kernel import prepare_gdn, store_state
from .base import BaseGatedDeltaBackend
from .standard import StandardLinearAttentionBackend, LinearAttentionMetadata

if TYPE_CHECKING:
    from logssms.core import Batch
    from logssms.layers.gated_delta import GatedDeltaConfig


class GatedDeltaBackend(StandardLinearAttentionBackend, BaseGatedDeltaBackend):
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
        out        : torch.Tensor | None = None,
    ) -> torch.Tensor:
        metadata = batch.linear_attn_metadata
        pool = get_global_ctx().linear_state
        if hasattr(pool, "group_table"):
            metadata, lane = pool.map_metadata(metadata, config.layer_idx)
            config = replace(config, layer_idx=lane)
        assert isinstance(metadata, LinearAttentionMetadata)
        assert metadata.phase == batch.phase
        if metadata.phase == "prefill":
            return self._forward_prefill(
                mixed_qkv, gate, beta, conv_weight, A_log, dt_bias, config, batch, out, metadata
            )
        elif metadata.phase == "verify":
            return self._forward_verify(
                mixed_qkv, gate, beta, conv_weight, A_log, dt_bias, config, metadata
            )
        return self._forward_decode(
            mixed_qkv, gate, beta, conv_weight, A_log, dt_bias, config, metadata, out
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
        out        : torch.Tensor | None = None,
        metadata   : LinearAttentionMetadata | None = None,
    ) -> torch.Tensor:
        from logssms.kernel.linear_attention.vendor.fla import (
            ChunkStateIO, chunk_gated_delta_rule_to_cache,
        )
        from logssms.kernel.linear_attention.packed_conv import (
            PackedConvStateIO, packed_causal_conv1d,
        )

        ctx = get_global_ctx()
        pool = ctx.linear_state
        layer_idx = config.layer_idx
        metadata = batch.linear_attn_metadata if metadata is None else metadata
        assert isinstance(metadata, LinearAttentionMetadata)
        indices = metadata.prefill_indices
        slots = metadata.baseline_slots
        assert indices is not None and slots is not None
        assert metadata.checkpoint_slots is not None
        assert metadata.checkpoint_chunks is not None
        assert metadata.chunk_write_slots is not None
        assert metadata.snapshot_rows is not None
        assert metadata.snapshot_reqs is not None
        live = slots >= 0
        conv_history = pool.conv_state[:, layer_idx].index_select(
            0, slots.clamp_min(0)
        )
        conv_history = torch.where(
            live[:, None, None], conv_history, 0
        )
        assert metadata.conv_token_slots is not None
        mixed_qkv = packed_causal_conv1d(
            mixed_qkv, conv_weight, conv_history,
            indices.cu_seqlens, indices.conv_chunks,
            state_io=PackedConvStateIO(
                storage=pool.conv_state, slots=metadata.conv_token_slots,
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
        output, _ = chunk_gated_delta_rule_to_cache(
            query,
            key,
            value,
            prepared_gate,
            prepared_beta,
            ChunkStateIO(
                storage=pool.recurrent_state,
                read_slots=slots,
                chunk_write_slots=metadata.chunk_write_slots,
                layer=layer_idx,
            ),
            indices.cu_seqlens,
            **indices.fla_kwargs(),
            out=None if out is None else out.unsqueeze(0),
        )
        return output.squeeze(0)

    def _forward_verify(
        self,
        mixed_qkv  : torch.Tensor,
        gate       : torch.Tensor,
        beta       : torch.Tensor,
        conv_weight: torch.Tensor,
        A_log      : torch.Tensor,
        dt_bias    : torch.Tensor,
        config     : GatedDeltaConfig,
        metadata   : LinearAttentionMetadata,
    ) -> torch.Tensor:
        from logssms.kernel.linear_attention.vendor.fla import (
            fused_recurrent_gated_delta_rule,
        )

        ctx = get_global_ctx()
        pool = ctx.linear_state
        layer_idx = config.layer_idx
        assert mixed_qkv.is_cuda and metadata.verify_groups is not None

        output = torch.empty(
            mixed_qkv.shape[0], config.num_v_heads, config.head_v_dim,
            dtype=mixed_qkv.dtype, device=mixed_qkv.device,
        )
        for seq_len, (row_indices, baseline_slots, write_slots) in (
            metadata.verify_groups.items()
        ):
            batch_size = len(baseline_slots)
            req_qkv = mixed_qkv.index_select(0, row_indices)
            req_qkv = req_qkv.view(batch_size, seq_len, config.conv_dim).transpose(1, 2)
            conv_state = pool.conv_state[baseline_slots, layer_idx]
            conv_input = torch.cat([conv_state, req_qkv], dim=-1)
            req_qkv = F.silu(
                F.conv1d(conv_input, conv_weight, groups=config.conv_dim)
            ).transpose(1, 2)

            conv_snapshots = (
                conv_input.unfold(-1, config.state_len, 1)[:, :, 1:]
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            pool.conv_state[:, layer_idx].index_copy_(
                0,
                write_slots.flatten().to(torch.int64),
                conv_snapshots.flatten(0, 1),
            )

            query, key, value = self._split_qkv(req_qkv, config)
            group_output, snapshots = fused_recurrent_gated_delta_rule(
                query,
                key,
                value,
                gate.index_select(0, row_indices).view(
                    batch_size, seq_len, config.num_v_heads
                ),
                beta.index_select(0, row_indices).view(
                    batch_size, seq_len, config.num_v_heads
                ),
                A_log,
                dt_bias,
                pool.recurrent_state[baseline_slots, layer_idx],
            )
            store_state(
                pool.recurrent_state[:, layer_idx],
                write_slots.flatten().to(torch.int64),
                snapshots.flatten(0, 1),
            )
            output.index_copy_(0, row_indices, group_output.flatten(0, 1))

        return output

    def _forward_decode(
        self,
        mixed_qkv  : torch.Tensor,
        gate       : torch.Tensor,
        beta       : torch.Tensor,
        conv_weight: torch.Tensor,
        A_log      : torch.Tensor,
        dt_bias    : torch.Tensor,
        config     : GatedDeltaConfig,
        metadata   : LinearAttentionMetadata,
        out        : torch.Tensor | None = None,
    ) -> torch.Tensor:
        from logssms.kernel.linear_attention.vendor.fla import (
            fused_recurrent_gated_delta_rule_to_slots,
        )
        from logssms.kernel.linear_attention.gdn_conv_decode import conv_decode_qkv

        ctx = get_global_ctx()
        pool = ctx.linear_state
        layer_idx = config.layer_idx
        read_slots = metadata.read_slots
        write_slots = metadata.write_slots
        assert read_slots is not None and write_slots is not None
        query, key, value = conv_decode_qkv(
            mixed_qkv, conv_weight, config,
            read_slots=read_slots,
            write_slots=write_slots,
            standard_state=pool.conv_state,
        )
        output = fused_recurrent_gated_delta_rule_to_slots(
            query,
            key,
            value,
            gate.unsqueeze(1),
            beta.unsqueeze(1),
            A_log,
            dt_bias,
            pool.recurrent_state,
            read_slots,
            write_slots,
            layer_idx,
            out=None if out is None else out.unsqueeze(1),
        )
        return output.squeeze(1)
