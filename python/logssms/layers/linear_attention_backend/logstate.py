from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from logssms.cache.linear_pool import LogStatePool
from logssms.core import get_global_ctx
from logssms.kernel.linear_attention import store_log_states

from .base import BaseLinearAttentionBackend, BaseLinearAttentionMetadata
from .native import _causal_conv1d_update, _prepare_gdn

if TYPE_CHECKING:
    from logssms.core import Batch
    from logssms.layers.gated_delta import GatedDeltaConfig


@dataclass
class LogStateLinearAttentionMetadata(BaseLinearAttentionMetadata):
    work_slots         : torch.Tensor
    log_slots          : torch.Tensor
    token_offsets      : torch.Tensor
    conv_read_slots    : torch.Tensor
    conv_read_granules : torch.Tensor
    cu_seqlens         : torch.Tensor | None = None
    conv_snapshot_rows : tuple[torch.Tensor, ...] | None = None
    conv_write_slots   : torch.Tensor | None = None
    conv_write_granules: torch.Tensor | None = None


@dataclass
class LogStateLinearAttentionCaptureData:
    work_slots        : torch.Tensor
    log_slots         : torch.Tensor
    token_offsets     : torch.Tensor
    conv_read_slots   : torch.Tensor
    conv_read_granules: torch.Tensor

    @classmethod
    def make(
        cls,
        max_bs: int,
        device: torch.device,
    ) -> LogStateLinearAttentionCaptureData:
        def tensor() -> torch.Tensor:
            return torch.zeros(max_bs, dtype=torch.int32, device=device)

        return cls(
            work_slots=tensor(),
            log_slots=tensor(),
            token_offsets=tensor(),
            conv_read_slots=tensor(),
            conv_read_granules=tensor(),
        )


class LogStateLinearAttentionBackend(BaseLinearAttentionBackend):
    """Gated DeltaNet backend backed by lossless per-token update logs."""

    def __init__(self) -> None:
        self.capture: LogStateLinearAttentionCaptureData | None = None
        self.capture_bs: list[int] = []

    def prepare_metadata(self, batch: Batch) -> None:
        if batch.is_verify:
            raise NotImplementedError(
                "LogState speculative verify is not implemented yet."
            )

        ctx = get_global_ctx()
        pool = ctx.linear_state
        state_table = ctx.state_table
        work_state_table = ctx.work_state_table
        assert isinstance(pool, LogStatePool)
        assert state_table is not None and work_state_table is not None
        assert work_state_table.shape[1] == 1

        reqs = batch.padded_reqs if batch.is_decode else batch.reqs
        table_indices = torch.tensor(
            [req.table_idx for req in reqs],
            dtype=torch.int64,
            device=state_table.device,
        )
        work_slots = work_state_table[table_indices, 0]

        row_table_indices = torch.repeat_interleave(
            table_indices,
            torch.tensor(
                [req.extend_len for req in reqs],
                dtype=torch.int64,
                device=state_table.device,
            ),
        )
        positions = batch.positions
        assert len(positions) == len(row_table_indices)
        log_slots = state_table[row_table_indices, positions]

        previous_positions = torch.tensor(
            [req.cached_len - 1 for req in reqs],
            dtype=torch.int64,
            device=state_table.device,
        )
        has_previous = previous_positions >= 0
        previous_slots = state_table[
            table_indices, previous_positions.clamp_min(0)
        ]
        conv_read_slots = torch.where(
            has_previous, previous_slots, work_slots
        )
        conv_read_granules = torch.where(
            has_previous,
            previous_positions.remainder(pool.page_size)
            // pool.layout.granularity,
            0,
        ).to(torch.int32)

        metadata = LogStateLinearAttentionMetadata(
            phase=batch.phase,
            work_slots=work_slots,
            log_slots=log_slots,
            token_offsets=positions.remainder(pool.page_size).to(torch.int32),
            conv_read_slots=conv_read_slots,
            conv_read_granules=conv_read_granules,
        )
        if batch.is_prefill:
            offsets = [0]
            snapshot_rows = []
            write_rows = []
            for req in reqs:
                seq_len = req.extend_len
                selected = [
                    index
                    for index in range(seq_len)
                    if (
                        req.cached_len + index + 1
                    ) % pool.layout.granularity == 0
                ]
                if not selected or selected[-1] != seq_len - 1:
                    selected.append(seq_len - 1)
                local_rows = torch.tensor(
                    selected, dtype=torch.int64, device=pool.device
                )
                snapshot_rows.append(local_rows)
                write_rows.append(local_rows + offsets[-1])
                offsets.append(offsets[-1] + seq_len)
            write_rows_tensor = torch.cat(write_rows)
            metadata.cu_seqlens = torch.tensor(
                offsets, dtype=torch.int64, device=pool.device
            )
            metadata.conv_snapshot_rows = tuple(snapshot_rows)
            metadata.conv_write_slots = log_slots.index_select(
                0, write_rows_tensor
            )
            metadata.conv_write_granules = (
                metadata.token_offsets.index_select(0, write_rows_tensor)
                // pool.layout.granularity
            )
        batch.linear_attn_metadata = metadata

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        del max_seq_len
        assert self.capture is None, "Capture already initialized."
        pool = get_global_ctx().linear_state
        assert isinstance(pool, LogStatePool)
        self.capture = LogStateLinearAttentionCaptureData.make(
            max(bs_list), pool.device
        )
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        bs = batch.size
        assert self.capture is not None and bs in self.capture_bs
        self.prepare_metadata(batch)
        metadata = batch.linear_attn_metadata
        assert isinstance(metadata, LogStateLinearAttentionMetadata)
        batch.linear_attn_metadata = self._copy_to_capture(metadata, bs)

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata = batch.linear_attn_metadata
        bs = batch.padded_size
        assert self.capture is not None and bs in self.capture_bs
        assert isinstance(metadata, LogStateLinearAttentionMetadata)
        self._copy_to_capture(metadata, bs)

    def _copy_to_capture(
        self,
        metadata: LogStateLinearAttentionMetadata,
        bs      : int,
    ) -> LogStateLinearAttentionMetadata:
        assert self.capture is not None
        for name in (
            "work_slots",
            "log_slots",
            "token_offsets",
            "conv_read_slots",
            "conv_read_granules",
        ):
            getattr(self.capture, name)[:bs].copy_(getattr(metadata, name))
        return LogStateLinearAttentionMetadata(
            phase="decode",
            work_slots=self.capture.work_slots[:bs],
            log_slots=self.capture.log_slots[:bs],
            token_offsets=self.capture.token_offsets[:bs],
            conv_read_slots=self.capture.conv_read_slots[:bs],
            conv_read_granules=self.capture.conv_read_granules[:bs],
        )

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

        pool = get_global_ctx().linear_state
        assert isinstance(pool, LogStatePool)
        layer_idx = config.layer_idx
        qkv_segments = []
        conv_snapshots = []
        assert metadata.cu_seqlens is not None
        assert metadata.conv_snapshot_rows is not None
        assert metadata.conv_write_slots is not None
        assert metadata.conv_write_granules is not None
        offset = 0
        for req_idx, req in enumerate(batch.reqs):
            seq_len = req.extend_len
            end_offset = offset + seq_len
            req_qkv = mixed_qkv[offset:end_offset].transpose(0, 1).unsqueeze(0)
            conv_state = pool.conv_state[
                metadata.conv_read_slots[req_idx],
                layer_idx,
                metadata.conv_read_granules[req_idx],
            ].unsqueeze(0)
            conv_input = torch.cat([conv_state, req_qkv], dim=-1)
            req_qkv = F.silu(
                F.conv1d(conv_input, conv_weight, groups=config.conv_dim)
            ).transpose(1, 2)
            qkv_segments.append(req_qkv.squeeze(0))
            snapshots = (
                conv_input.unfold(-1, config.state_len, 1)[:, :, 1:]
                .permute(0, 2, 1, 3)
                .squeeze(0)
            )
            conv_snapshots.append(
                snapshots.index_select(
                    0, metadata.conv_snapshot_rows[req_idx]
                )
            )
            offset = end_offset

        mixed_qkv = torch.cat(qkv_segments).unsqueeze(0)
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
        output, final_state, _chunk_states, residual_value = (
            chunk_gated_delta_rule_with_log(
                query,
                key,
                value,
                prepared_gate,
                prepared_beta,
                pool.recurrent_state[metadata.work_slots, layer_idx],
                metadata.cu_seqlens,
                chunk_size=pool.layout.granularity,
            )
        )
        store_log_states(
            key_log=pool.key_log,
            residual_value_log=pool.residual_value_log,
            decay_log=pool.decay_log,
            recurrent_state=pool.recurrent_state,
            slots=metadata.log_slots,
            token_offsets=metadata.token_offsets,
            keys=key.squeeze(0),
            residual_values=residual_value.squeeze(0),
            decays=prepared_gate.exp().squeeze(0),
            layer=layer_idx,
        )
        pool.recurrent_state[
            metadata.work_slots, layer_idx
        ] = final_state.to(pool.dtype)
        pool.conv_state[
            metadata.conv_write_slots,
            layer_idx,
            metadata.conv_write_granules,
        ] = torch.cat(conv_snapshots)
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
            fused_recurrent_gated_delta_rule_with_log,
        )

        pool = get_global_ctx().linear_state
        assert isinstance(pool, LogStatePool)
        layer_idx = config.layer_idx
        conv_state = pool.conv_state[
            metadata.conv_read_slots,
            layer_idx,
            metadata.conv_read_granules,
        ]
        mixed_qkv = mixed_qkv.unsqueeze(1).transpose(1, 2)
        mixed_qkv = _causal_conv1d_update(
            mixed_qkv, conv_state, conv_weight
        ).transpose(1, 2)
        conv_write_granules = (
            metadata.token_offsets // pool.layout.granularity
        )
        pool.conv_state[
            metadata.log_slots,
            layer_idx,
            conv_write_granules,
        ] = conv_state

        query, key, value = self._split_qkv(mixed_qkv, config)
        output, states, prepared_key, residual_value, decay = (
            fused_recurrent_gated_delta_rule_with_log(
                query,
                key,
                value,
                gate.unsqueeze(1),
                beta.unsqueeze(1),
                A_log,
                dt_bias,
                pool.recurrent_state[metadata.work_slots, layer_idx],
            )
        )
        store_log_states(
            key_log=pool.key_log,
            residual_value_log=pool.residual_value_log,
            decay_log=pool.decay_log,
            recurrent_state=pool.recurrent_state,
            slots=metadata.log_slots,
            token_offsets=metadata.token_offsets,
            keys=prepared_key.squeeze(1),
            residual_values=residual_value.squeeze(1),
            decays=decay.squeeze(1),
            layer=layer_idx,
        )
        pool.recurrent_state[
            metadata.work_slots, layer_idx
        ] = states.squeeze(1).to(pool.dtype)
        return output.squeeze(1)


__all__ = [
    "LogStateLinearAttentionBackend",
    "LogStateLinearAttentionCaptureData",
    "LogStateLinearAttentionMetadata",
]
