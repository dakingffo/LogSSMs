from __future__ import annotations

from dataclasses import dataclass
import os
from time import perf_counter_ns
from typing import TYPE_CHECKING

import torch

from logssms.cache.logstate_pool.mha_pool import MHALogStatePool
from logssms.core import get_global_ctx

from .base import BaseLinearAttentionBackend, BaseLinearAttentionMetadata
from .standard import PackedLinearIndices

if TYPE_CHECKING:
    from logssms.core import Batch


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
    checkpoint_conv_granules: torch.Tensor | None = None
    checkpoint_state_indices: torch.Tensor | None = None
    chunk_write_slots: torch.Tensor | None = None
    checkpoint_snapshot_indices: torch.Tensor | None = None
    conv_write_snapshot_indices: torch.Tensor | None = None
    log_write_slots    : torch.Tensor | None = None
    conv_token_slots   : torch.Tensor | None = None
    conv_token_granules: torch.Tensor | None = None


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
    direct_prefill_metadata = False

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
        assert isinstance(pool, MHALogStatePool)
        assert state_table is not None and work_state_table is not None
        assert ctx.page_table is not None
        assert work_state_table.shape[1] == 1

        packet = batch.prepared_indices
        if batch.is_decode and packet is not None and state_table.is_cuda:
            from logssms.kernel.metadata import gather_linear_slots

            slots = gather_linear_slots(
                packet, state_table, work_table=work_state_table,
                out_loc=batch.out_loc, page_size=pool.page_size,
            )
            batch.linear_attn_metadata = LogStateLinearAttentionMetadata(
                phase=batch.phase, work_slots=slots[0], log_slots=slots[1],
                token_offsets=slots[2], conv_read_slots=slots[0],
                conv_read_granules=slots[3], replay_counts=packet.replay_counts,
            )
            return

        reqs = batch.padded_reqs if batch.is_decode else batch.reqs
        table_ids = [req.table_idx for req in reqs]
        repeat_start = perf_counter_ns() if profile else 0
        use_packet = packet is not None and packet.token_table_ids is not None
        if use_packet:
            row_ids = []
        elif batch.is_decode:
            row_ids = table_ids
        else:
            row_ids = [req.table_idx for req in reqs for _ in range(req.extend_len)]
        repeat_cpu_ms = (
            (perf_counter_ns() - repeat_start) / 1e6 if profile else 0.0
        )
        previous_ids = (
            [] if batch.is_decode else [req.cached_len - 1 for req in reqs]
        )
        if use_packet:
            table_indices = packet.table_ids
            row_table_indices = packet.token_table_ids
            previous_positions = packet.previous_positions
        else:
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
            previous_positions = all_ids[len(reqs) + len(row_ids):]
        work_slots = work_state_table[table_indices, 0]
        positions = batch.positions
        assert len(positions) == len(row_table_indices)
        log_slots = state_table[row_table_indices, positions]
        token_offsets = batch.out_loc.remainder(pool.page_size).to(torch.int32)
        if batch.is_decode:
            counts = batch.logstate_replay_counts
            if counts is None:
                counts = [0] * len(reqs)
            assert len(counts) == len(reqs)
            replay_counts = torch.tensor(
                counts, dtype=torch.int32, pin_memory=pool.device.type == "cuda"
            ).to(pool.device, non_blocking=pool.device.type == "cuda")
            conv_read_slots = work_slots
            conv_read_granules = torch.zeros_like(work_slots)
        else:
            replay_counts = torch.empty(0, dtype=torch.int32, device=pool.device)
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
        item_calls = item_cpu_ns = 0
        direct = (batch.is_prefill and batch.logstate_first_offsets is not None
                  and self.direct_prefill_metadata)
        if direct:
            self._prepare_direct_prefill(batch, metadata, pool)
        elif batch.is_prefill:
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

                if ranges_by_req:
                    log_rows.extend(
                        offsets[-1] + index for index in range(seq_len)
                        if not is_checkpoint_row(index)
                    )
                if first_offsets is None:


                    item_start = perf_counter_ns() if profile else 0
                    first_offset = int(token_offsets[offsets[-1]].item())
                    item_calls += 1
                    if profile:
                        item_cpu_ns += perf_counter_ns() - item_start
                else:
                    first_offset = first_offsets[req_idx]
                if ranges_by_req:
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
                if ranges_by_req:
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
            metadata.chunk_write_slots = torch.full(
                (chunk_offset,), -1, dtype=torch.int32, device=pool.device
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
            if ranges_by_req:
                metadata.conv_write_snapshot_indices = torch.tensor(
                    write_snapshot_indices, dtype=torch.int64, device=pool.device
                )
                metadata.log_row_indices = torch.tensor(
                    log_rows, dtype=torch.int64, device=pool.device
                )
            if checkpoint_slots:
                metadata.checkpoint_slots = torch.tensor(
                    checkpoint_slots, dtype=torch.int32, device=pool.device
                )
                metadata.checkpoint_state_indices = torch.tensor(
                    checkpoint_state_indices, dtype=torch.int64, device=pool.device
                )
                metadata.chunk_write_slots.index_copy_(
                    0, metadata.checkpoint_state_indices, metadata.checkpoint_slots
                )
                metadata.checkpoint_snapshot_indices = torch.tensor(
                    checkpoint_snapshot_indices, dtype=torch.int64, device=pool.device
                )
                metadata.checkpoint_conv_granules = torch.tensor(
                    [0 if pool.allocator.slot(slot).kind == "work"
                     else pool.layout.num_granules - 1 for slot in checkpoint_slots],
                    dtype=torch.int32, device=pool.device,
                )
            metadata.log_write_slots = log_slots
            if metadata.log_row_indices is not None:
                metadata.log_write_slots = torch.full_like(log_slots, -1)
                metadata.log_write_slots.index_copy_(
                    0, metadata.log_row_indices,
                    log_slots.index_select(0, metadata.log_row_indices),
                )
            metadata.conv_token_slots = torch.full_like(log_slots, -1)
            metadata.conv_token_granules = torch.zeros_like(log_slots)
            metadata.conv_token_slots.index_copy_(
                0, write_rows_tensor, metadata.conv_write_slots
            )
            metadata.conv_token_granules.index_copy_(
                0, write_rows_tensor, metadata.conv_write_granules
            )
            if metadata.checkpoint_slots is not None:
                rows = metadata.conv_snapshot_rows.index_select(
                    0, metadata.checkpoint_snapshot_indices
                )
                metadata.conv_token_slots.index_copy_(0, rows, metadata.checkpoint_slots)
                metadata.conv_token_granules.index_copy_(
                    0, rows, metadata.checkpoint_conv_granules,
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

    def _prepare_direct_prefill(self, batch, metadata, pool):
        from logssms.kernel.metadata import make_logstate_prefill_writes

        lengths = [req.extend_len for req in batch.reqs]
        granularity = pool.layout.granularity
        metadata.prefill_indices = PackedLinearIndices.from_lengths(lengths, pool.device, granularity)
        metadata.cu_seqlens = metadata.prefill_indices.cu_seqlens
        total = sum(lengths)
        host = torch.empty(3 * total + len(metadata.prefill_indices.chunk_indices),
                           dtype=torch.int32, pin_memory=pool.device.type == "cuda")
        conv_slots, conv_granules, log_mask = host[:3 * total].view(3, total).numpy()
        chunk_writes = host[3 * total:].numpy()
        conv_slots.fill(-1)
        conv_granules.fill(0)
        log_mask.fill(1)
        chunk_writes.fill(-1)
        checkpoints = iter(sorted(batch.logstate_checkpoint_writes))
        checkpoint = next(checkpoints, None)
        ranges = {}
        for req, begin, end in batch.logstate_checkpoint_ranges:
            ranges.setdefault(req, []).append((begin, end))
        token_begin = chunk_begin = 0
        for i, req in enumerate(batch.reqs):
            length, first = lengths[i], batch.logstate_first_offsets[i]
            token_end = token_begin + length
            conv_slots[token_begin + granularity - 1 - first % granularity:token_end:granularity] = -2
            if length:
                conv_slots[token_end - 1] = -2
            for begin, end in ranges.get(i, ()):
                begin = max(0, begin - req.cached_len)
                end = min(length, end - req.cached_len)
                if begin < end:
                    log_mask[token_begin + begin:token_begin + end] = 0
                    conv_slots[token_begin + begin:token_begin + end] = -1
            while checkpoint is not None and checkpoint[0] < token_end:
                global_row, slot = checkpoint
                row = global_row - token_begin
                assert row >= 0
                if (first + row + 1) % granularity == 0 or row + 1 == length:
                    conv_slots[global_row] = slot
                    conv_granules[global_row] = (
                        0 if pool.allocator.slot(slot).kind == "work" else pool.layout.num_granules - 1
                    )
                    chunk_writes[chunk_begin + row // granularity] = slot
                checkpoint = next(checkpoints, None)
            token_begin += length
            chunk_begin += (length + granularity - 1) // granularity
        packet = host.to(pool.device, non_blocking=pool.device.type == "cuda")
        metadata.chunk_write_slots = packet[3 * total:]
        metadata.log_write_slots, metadata.conv_token_slots, metadata.conv_token_granules = (
            make_logstate_prefill_writes(metadata.log_slots, metadata.token_offsets, packet, granularity)
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        del max_seq_len
        assert self.capture is None, "Capture already initialized."
        pool = get_global_ctx().linear_state
        assert isinstance(pool, MHALogStatePool)
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
        from logssms.kernel.metadata import copy_metadata

        names = (
            "work_slots",
            "log_slots",
            "token_offsets",
            "conv_read_slots",
            "conv_read_granules",
            "replay_counts",
        )
        copy_metadata(
            tuple(getattr(metadata, name) for name in names),
            tuple(getattr(self.capture, name)[:bs] for name in names),
        )
        return LogStateLinearAttentionMetadata(
            phase="decode",
            work_slots=self.capture.work_slots[:bs],
            log_slots=self.capture.log_slots[:bs],
            token_offsets=self.capture.token_offsets[:bs],
            conv_read_slots=self.capture.conv_read_slots[:bs],
            conv_read_granules=self.capture.conv_read_granules[:bs],
            replay_counts=self.capture.replay_counts[:bs],
        )
