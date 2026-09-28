from __future__ import annotations

from dataclasses import dataclass
import os
from time import perf_counter_ns
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from logssms.cache.logstate_pool.frame_pool import FramedMHALogStatePool
from logssms.core import get_global_ctx
from logssms.kernel.linear_attention.framed_logstate import store_framed_log_states

from .base import BaseLinearAttentionBackend, BaseLinearAttentionMetadata
from .native import PackedLinearIndices, _causal_conv1d_update, _prepare_gdn

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
    replay_counts      : torch.Tensor
    cu_seqlens         : torch.Tensor | None = None
    prefill_indices    : PackedLinearIndices | None = None
    conv_snapshot_rows : torch.Tensor | None = None
    conv_snapshot_reqs : torch.Tensor | None = None
    conv_final_rows    : torch.Tensor | None = None
    conv_write_slots   : torch.Tensor | None = None
    conv_write_granules: torch.Tensor | None = None
    log_row_indices    : torch.Tensor | None = None
    checkpoint_slots   : torch.Tensor | None = None
    checkpoint_state_indices: torch.Tensor | None = None
    checkpoint_snapshot_indices: torch.Tensor | None = None
    conv_write_snapshot_indices: torch.Tensor | None = None


@dataclass
class LogStateLinearAttentionCaptureData:
    work_slots        : torch.Tensor
    log_slots         : torch.Tensor
    token_offsets     : torch.Tensor
    conv_read_slots   : torch.Tensor
    conv_read_granules: torch.Tensor
    replay_counts     : torch.Tensor

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
            replay_counts=tensor(),
        )


class LogStateLinearAttentionBackend(BaseLinearAttentionBackend):
    """Gated DeltaNet backend backed by lossless per-token update logs."""

    def __init__(self, *, enable_eager_state: bool = False) -> None:
        self.capture: LogStateLinearAttentionCaptureData | None = None
        self.capture_bs: list[int] = []
        self.enable_eager_state = enable_eager_state
        self._profile_enabled = os.getenv(
            "LOGSSMS_PROFILE_SCHEDULER", "0"
        ).lower() in ("1", "true", "yes")
        self._profile_prefill_samples: list[dict] = []

    def profile_result(self) -> dict | None:
        if not self._profile_enabled:
            return None
        samples = self._profile_prefill_samples
        fields = (
            "metadata_cpu_ms", "repeat_interleave_cpu_ms", "item_cpu_ms",
        )
        totals = {key: sum(sample[key] for sample in samples) for key in fields}
        return {
            "type": "logstate",
            "aggregate": {
                "prefill_batches": len(samples),
                "item_calls": sum(sample["item_calls"] for sample in samples),
                **totals,
            },
            "prefill_batches": samples,
        }

    def prepare_metadata(self, batch: Batch) -> None:
        profile = self._profile_enabled and batch.is_prefill
        metadata_start = perf_counter_ns() if profile else 0
        if batch.is_verify:
            raise NotImplementedError(
                "LogState speculative verify is not implemented yet."
            )

        ctx = get_global_ctx()
        pool = ctx.linear_state
        state_table = ctx.state_table
        work_state_table = ctx.work_state_table
        assert isinstance(pool, FramedMHALogStatePool)
        assert state_table is not None and work_state_table is not None
        assert ctx.page_table is not None
        assert work_state_table.shape[1] == 1

        reqs = batch.padded_reqs if batch.is_decode else batch.reqs
        table_ids = [req.table_idx for req in reqs]
        repeat_start = perf_counter_ns() if profile else 0
        if batch.is_decode:
            row_ids = table_ids
        else:
            row_ids = [req.table_idx for req in reqs for _ in range(req.extend_len)]
        repeat_cpu_ms = (
            (perf_counter_ns() - repeat_start) / 1e6 if profile else 0.0
        )
        previous_ids = (
            [] if batch.is_decode else [req.cached_len - 1 for req in reqs]
        )
        all_ids = torch.tensor(
            table_ids + ([] if batch.is_decode else row_ids + previous_ids),
            dtype=torch.int64,
            pin_memory=state_table.is_cuda,
        ).to(state_table.device, non_blocking=state_table.is_cuda)
        table_indices = all_ids[:len(reqs)]
        row_table_indices = (
            table_indices
            if batch.is_decode else all_ids[len(reqs):len(reqs) + len(row_ids)]
        )
        work_slots = work_state_table[table_indices, 0]
        positions = batch.positions
        assert len(positions) == len(row_table_indices)
        log_slots = state_table[row_table_indices, positions]
        token_offsets = batch.out_loc.remainder(pool.page_size).to(torch.int32)
        if batch.is_decode:
            counts = batch.logstate_replay_counts
            if counts is None:
                counts = [0] * len(reqs)  # direct/capture callers
            assert len(counts) == len(reqs)
            replay_counts = torch.tensor(
                counts, dtype=torch.int32, pin_memory=pool.device.type == "cuda"
            ).to(pool.device, non_blocking=pool.device.type == "cuda")
            conv_read_slots = work_slots
            conv_read_granules = torch.zeros_like(work_slots)
        else:
            replay_counts = torch.empty(0, dtype=torch.int32, device=pool.device)
            previous_positions = all_ids[len(reqs) + len(row_ids):]
            previous_slots = state_table[
                table_indices, previous_positions.clamp_min(0)
            ]
            previous_kv = ctx.page_table[
                table_indices, previous_positions.clamp_min(0)
            ]
            mapped_previous = (previous_positions >= 0) & (previous_slots >= 0)
            conv_read_slots = torch.where(
                mapped_previous, previous_slots, work_slots
            )
            conv_read_granules = torch.where(
                mapped_previous,
                previous_kv.remainder(pool.page_size) // pool.layout.granularity,
                0,
            ).to(torch.int32)

        metadata = LogStateLinearAttentionMetadata(
            phase=batch.phase,
            work_slots=work_slots,
            log_slots=log_slots,
            token_offsets=token_offsets,
            conv_read_slots=conv_read_slots,
            conv_read_granules=conv_read_granules,
            replay_counts=replay_counts,
        )
        if batch.is_prefill:
            offsets = [0]
            snapshot_rows = []
            snapshot_reqs = []
            final_rows = []
            write_rows = []
            write_snapshot_indices = []
            checkpoint_snapshot_indices = []
            checkpoint_state_indices = []
            checkpoint_slots = []
            log_rows = []
            checkpoint_by_row = dict(batch.logstate_checkpoint_writes)
            ranges_by_req: dict[int, list[tuple[int, int]]] = {}
            for req_index, begin, end in batch.logstate_checkpoint_ranges:
                ranges_by_req.setdefault(req_index, []).append((begin, end))
            chunk_offset = 0
            item_cpu_ns = 0
            item_calls = 0
            first_offsets = batch.logstate_first_offsets
            for req_idx, req in enumerate(reqs):
                seq_len = req.extend_len
                checkpoint_ranges = ranges_by_req.get(req_idx, ())
                def is_checkpoint_row(index: int) -> bool:
                    position = req.cached_len + index
                    return any(begin <= position < end for begin, end in checkpoint_ranges)

                if checkpoint_by_row:
                    log_rows.extend(
                        offsets[-1] + index for index in range(seq_len)
                        if not is_checkpoint_row(index)
                    )
                if first_offsets is None:
                    # Direct backend callers (e.g. isolated tests) may not
                    # have gone through CacheManager.prepare_batch.
                    item_start = perf_counter_ns() if profile else 0
                    first_offset = int(token_offsets[offsets[-1]].item())
                    item_calls += 1
                    if profile:
                        item_cpu_ns += perf_counter_ns() - item_start
                else:
                    first_offset = first_offsets[req_idx]
                if checkpoint_by_row:
                    selected = [
                        index for index in range(seq_len)
                        if (first_offset + index + 1) % pool.layout.granularity == 0
                        and (
                            not is_checkpoint_row(index)
                            or offsets[-1] + index in checkpoint_by_row
                        )
                    ]
                else:
                    selected = [
                        index for index in range(seq_len)
                        if (first_offset + index + 1) % pool.layout.granularity == 0
                    ]
                if not selected or selected[-1] != seq_len - 1:
                    selected.append(seq_len - 1)
                if checkpoint_by_row:
                    for row in selected:
                        global_row = row + offsets[-1]
                        snapshot_rows.append(global_row)
                        snapshot_reqs.append(req_idx)
                        if global_row in checkpoint_by_row:
                            checkpoint_snapshot_indices.append(len(snapshot_rows) - 1)
                            checkpoint_state_indices.append(
                                chunk_offset + row // pool.layout.granularity
                            )
                            checkpoint_slots.append(checkpoint_by_row[global_row])
                        elif not is_checkpoint_row(row):
                            write_rows.append(global_row)
                            write_snapshot_indices.append(len(snapshot_rows) - 1)
                else:
                    selected_rows = [row + offsets[-1] for row in selected]
                    snapshot_rows.extend(selected_rows)
                    snapshot_reqs.extend([req_idx] * len(selected))
                    write_rows.extend(selected_rows)
                final_rows.append(len(snapshot_rows) - 1)
                offsets.append(offsets[-1] + seq_len)
                chunk_offset += (seq_len + pool.layout.granularity - 1) // pool.layout.granularity
            metadata.prefill_indices = PackedLinearIndices.from_lengths(
                [req.extend_len for req in reqs], pool.device, pool.layout.granularity
            )
            metadata.cu_seqlens = metadata.prefill_indices.cu_seqlens
            metadata.conv_snapshot_rows = torch.tensor(
                snapshot_rows, dtype=torch.int64, device=pool.device
            )
            metadata.conv_snapshot_reqs = torch.tensor(
                snapshot_reqs, dtype=torch.int64, device=pool.device
            )
            metadata.conv_final_rows = torch.tensor(
                final_rows, dtype=torch.int64, device=pool.device
            )
            write_rows_tensor = torch.tensor(
                write_rows, dtype=torch.int64, device=pool.device
            )
            metadata.conv_write_slots = log_slots.index_select(
                0, write_rows_tensor
            )
            metadata.conv_write_granules = (
                metadata.token_offsets.index_select(0, write_rows_tensor)
                // pool.layout.granularity
            )
            if checkpoint_slots:
                metadata.conv_write_snapshot_indices = torch.tensor(
                    write_snapshot_indices, dtype=torch.int64, device=pool.device
                )
                metadata.log_row_indices = torch.tensor(
                    log_rows, dtype=torch.int64, device=pool.device
                )
                metadata.checkpoint_slots = torch.tensor(
                    checkpoint_slots, dtype=torch.int32, device=pool.device
                )
                metadata.checkpoint_state_indices = torch.tensor(
                    checkpoint_state_indices, dtype=torch.int64, device=pool.device
                )
                metadata.checkpoint_snapshot_indices = torch.tensor(
                    checkpoint_snapshot_indices, dtype=torch.int64, device=pool.device
                )
        batch.linear_attn_metadata = metadata
        if profile:
            self._profile_prefill_samples.append({
                "batch_size": batch.size,
                "metadata_cpu_ms": (perf_counter_ns() - metadata_start) / 1e6,
                "repeat_interleave_cpu_ms": repeat_cpu_ms,
                "item_calls": item_calls,
                "item_cpu_ms": item_cpu_ns / 1e6,
            })

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        del max_seq_len
        assert self.capture is None, "Capture already initialized."
        pool = get_global_ctx().linear_state
        assert isinstance(pool, FramedMHALogStatePool)
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
            "replay_counts",
        ):
            getattr(self.capture, name)[:bs].copy_(getattr(metadata, name))
        return LogStateLinearAttentionMetadata(
            phase="decode",
            work_slots=self.capture.work_slots[:bs],
            log_slots=self.capture.log_slots[:bs],
            token_offsets=self.capture.token_offsets[:bs],
            conv_read_slots=self.capture.conv_read_slots[:bs],
            conv_read_granules=self.capture.conv_read_granules[:bs],
            replay_counts=self.capture.replay_counts[:bs],
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
        decays = prepared_gate.exp().squeeze(0)
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
            decays = decays.index_select(0, log_rows)
        store_framed_log_states(
            pool,
            log_slots, token_offsets, log_keys, log_values, decays,
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
        from logssms.kernel.linear_attention.conv_decode import conv_decode_qkv

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


__all__ = [
    "LogStateLinearAttentionBackend",
    "LogStateLinearAttentionCaptureData",
    "LogStateLinearAttentionMetadata",
]
