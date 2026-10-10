from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from logssms.core import get_global_ctx
from .base import BaseLinearAttentionBackend, BaseLinearAttentionMetadata

if TYPE_CHECKING:
    from logssms.core import Batch


@dataclass
class LinearAttentionMetadata(BaseLinearAttentionMetadata):
    slots        : torch.Tensor | None = None
    read_slots   : torch.Tensor | None = None
    write_slots  : torch.Tensor | None = None
    verify_groups: dict[
        int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] | None = None
    prefill_indices : PackedLinearIndices | None = None
    baseline_slots  : torch.Tensor | None = None
    checkpoint_slots: torch.Tensor | None = None
    checkpoint_chunks: torch.Tensor | None = None
    chunk_write_slots: torch.Tensor | None = None
    snapshot_rows   : torch.Tensor | None = None
    snapshot_reqs   : torch.Tensor | None = None
    conv_token_slots: torch.Tensor | None = None


@dataclass
class PackedLinearIndices:
    cu_seqlens      : torch.Tensor
    chunk_indices   : torch.Tensor
    chunk_offsets   : torch.Tensor
    endings         : torch.Tensor
    conv_chunks     : torch.Tensor

    @classmethod
    def from_lengths(
        cls, lengths: list[int], device: torch.device, chunk_size: int
    ) -> PackedLinearIndices:
        offsets = [0]
        chunk_offsets = [0]
        chunks: list[tuple[int, int]] = []
        conv_chunks: list[tuple[int, int]] = []
        endings = []
        for req_idx, length in enumerate(lengths):
            assert length > 0
            offsets.append(offsets[-1] + length)
            n_chunks = (length + chunk_size - 1) // chunk_size
            chunk_offsets.append(chunk_offsets[-1] + n_chunks)
            chunks.extend((req_idx, chunk) for chunk in range(n_chunks))
            conv_chunks.extend((req_idx, chunk) for chunk in range((length + 63) // 64))
            endings.append(chunk_offsets[-1] - 1)

        def tensor(values) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int64, device=device)

        return cls(
            cu_seqlens=tensor(offsets),
            chunk_indices=tensor(chunks),
            chunk_offsets=tensor(chunk_offsets),
            endings=tensor(endings),
            conv_chunks=tensor(conv_chunks),
        )

    def fla_kwargs(self) -> dict[str, torch.Tensor]:
        return {
            "chunk_indices": self.chunk_indices,
            "chunk_offsets": self.chunk_offsets,
        }


@dataclass
class LinearAttentionCaptureData:
    slots: torch.Tensor

    @property
    def read_slots(self) -> torch.Tensor:
        return self.slots[:, 0]

    @property
    def write_slots(self) -> torch.Tensor:
        return self.slots[:, 1]

    @classmethod
    def make(cls, max_bs: int, device: torch.device) -> LinearAttentionCaptureData:
        return cls(
            slots=torch.zeros(max_bs, 2, dtype=torch.int64, device=device),
        )



class StandardLinearAttentionBackend(BaseLinearAttentionBackend):
    def __init__(self) -> None:
        self.capture: LinearAttentionCaptureData | None = None
        self.capture_bs: list[int] = []

    def prepare_metadata(self, batch: Batch) -> None:
        metadata = LinearAttentionMetadata(phase=batch.phase)
        ctx = get_global_ctx()
        state_table = ctx.state_table
        assert state_table is not None

        if batch.is_decode:
            reqs = batch.padded_reqs
            if batch.prepared_indices is not None and state_table.is_cuda:
                from logssms.kernel.metadata import gather_linear_slots

                metadata.slots = gather_linear_slots(batch.prepared_indices, state_table)
            else:
                descriptors = torch.tensor(
                    [[req.table_idx, req.cached_len - 1, req.cached_len] for req in reqs],
                    dtype=torch.int64,
                    pin_memory=state_table.is_cuda,
                ).to(state_table.device, non_blocking=state_table.is_cuda)
                metadata.slots = state_table[
                    descriptors[:, :1], descriptors[:, 1:]
                ].to(torch.int64)
            metadata.read_slots = metadata.slots[:, 0]
            metadata.write_slots = metadata.slots[:, 1]
        elif batch.is_prefill:
            ctx_page_size = ctx.page_size
            assert ctx_page_size % 64 == 0
            lengths = [req.extend_len for req in batch.reqs]
            metadata.prefill_indices = PackedLinearIndices.from_lengths(
                lengths, state_table.device, 64
            )
            table_indices = torch.tensor(
                [req.table_idx for req in batch.reqs],
                dtype=torch.int64, device=state_table.device,
            )
            previous = torch.tensor(
                [req.cached_len - 1 for req in batch.reqs],
                dtype=torch.int64, device=state_table.device,
            )
            baseline = state_table[table_indices, previous.clamp_min(0)]
            metadata.baseline_slots = torch.where(
                previous >= 0, baseline, -1
            ).to(torch.int64)

            checkpoint_tables = []
            checkpoint_positions = []
            checkpoint_chunks = []
            snapshot_rows = []
            snapshot_reqs = []
            chunk_offset = 0
            token_offset = 0
            for req_idx, req in enumerate(batch.reqs):
                assert req.cached_len % 64 == 0
                ends = list(range(ctx_page_size, req.extend_len, ctx_page_size))
                ends.append(req.extend_len)
                for end in ends:
                    checkpoint_tables.append(req.table_idx)
                    checkpoint_positions.append(req.cached_len + end - 1)
                    checkpoint_chunks.append(chunk_offset + (end - 1) // 64)
                    snapshot_rows.append(token_offset + end - 1)
                    snapshot_reqs.append(req_idx)
                chunk_offset += (req.extend_len + 63) // 64
                token_offset += req.extend_len
            checkpoint_tables_tensor = torch.tensor(
                checkpoint_tables, dtype=torch.int64, device=state_table.device
            )
            checkpoint_positions_tensor = torch.tensor(
                checkpoint_positions, dtype=torch.int64, device=state_table.device
            )
            metadata.checkpoint_slots = state_table[
                checkpoint_tables_tensor, checkpoint_positions_tensor
            ].to(torch.int64)
            metadata.checkpoint_chunks = torch.tensor(
                checkpoint_chunks, dtype=torch.int64, device=state_table.device
            )
            metadata.chunk_write_slots = torch.full(
                (chunk_offset,), -1, dtype=torch.int64, device=state_table.device
            )
            metadata.chunk_write_slots.index_copy_(
                0, metadata.checkpoint_chunks, metadata.checkpoint_slots
            )
            metadata.snapshot_rows = torch.tensor(
                snapshot_rows, dtype=torch.int64, device=state_table.device
            )
            metadata.snapshot_reqs = torch.tensor(
                snapshot_reqs, dtype=torch.int64, device=state_table.device
            )
            metadata.conv_token_slots = torch.full(
                (token_offset,), -1, dtype=torch.int64, device=state_table.device
            )
            metadata.conv_token_slots.index_copy_(
                0, metadata.snapshot_rows, metadata.checkpoint_slots
            )
        elif batch.is_verify:
            work_state_table = ctx.work_state_table
            assert work_state_table is not None

            entries_by_len: dict[int, list[tuple[int, int, int]]] = {}
            offset = 0
            for req in batch.reqs:
                entries_by_len.setdefault(req.extend_len, []).append(
                    (req.table_idx, req.cached_len - 1, offset)
                )
                offset += req.extend_len

            groups = {}
            for seq_len, entries in entries_by_len.items():
                row_indices = torch.tensor(
                    [
                        row
                        for _table_idx, _baseline_position, start in entries
                        for row in range(start, start + seq_len)
                    ],
                    dtype=torch.int64,
                    device=state_table.device,
                )
                table_indices = torch.tensor(
                    [table_idx for table_idx, _baseline_position, _start in entries],
                    dtype=torch.int64,
                    device=state_table.device,
                )
                baseline_positions = torch.tensor(
                    [position for _table_idx, position, _start in entries],
                    dtype=torch.int64,
                    device=state_table.device,
                )
                baseline_slots = state_table[
                    table_indices, baseline_positions
                ].to(torch.int64)
                write_slots = work_state_table[
                    table_indices, :seq_len
                ]
                groups[seq_len] = row_indices, baseline_slots, write_slots
            metadata.verify_groups = groups

        batch.linear_attn_metadata = metadata

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        del max_seq_len
        assert self.capture is None, "Capture already initialized."
        self.capture = LinearAttentionCaptureData.make(
            max(bs_list), get_global_ctx().linear_state.device
        )
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        bs = batch.size
        assert self.capture is not None and bs in self.capture_bs
        self.prepare_metadata(batch)
        metadata = batch.linear_attn_metadata
        assert metadata.slots is not None
        self.capture.slots[:bs].copy_(metadata.slots)
        batch.linear_attn_metadata = LinearAttentionMetadata(
            phase="decode",
            slots=self.capture.slots[:bs],
            read_slots=self.capture.read_slots[:bs],
            write_slots=self.capture.write_slots[:bs],
        )

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata = batch.linear_attn_metadata
        bs = batch.padded_size
        assert self.capture is not None and bs in self.capture_bs
        assert metadata.slots is not None
        self.capture.slots[:bs].copy_(metadata.slots)
