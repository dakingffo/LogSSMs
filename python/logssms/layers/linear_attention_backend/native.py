from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from logssms.core import get_global_ctx
from logssms.kernel import store_state

from .base import BaseLinearAttentionBackend, BaseLinearAttentionMetadata

if TYPE_CHECKING:
    from logssms.core import Batch
    from logssms.layers.gated_delta import GatedDeltaConfig


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
    snapshot_rows   : torch.Tensor | None = None
    snapshot_reqs   : torch.Tensor | None = None


@dataclass
class PackedLinearIndices:
    cu_seqlens      : torch.Tensor
    chunk_indices   : torch.Tensor
    chunk_offsets   : torch.Tensor
    continuing      : torch.Tensor
    endings         : torch.Tensor
    ending_sequences: torch.Tensor
    conv_chunks     : torch.Tensor

    @classmethod
    def from_lengths(
        cls, lengths: list[int], device: torch.device, chunk_size: int
    ) -> PackedLinearIndices:
        offsets = [0]
        chunk_offsets = [0]
        chunks: list[tuple[int, int]] = []
        conv_chunks: list[tuple[int, int]] = []
        continuing = []
        endings = []
        for req_idx, length in enumerate(lengths):
            assert length > 0
            offsets.append(offsets[-1] + length)
            n_chunks = (length + chunk_size - 1) // chunk_size
            chunk_offsets.append(chunk_offsets[-1] + n_chunks)
            chunks.extend((req_idx, chunk) for chunk in range(n_chunks))
            conv_chunks.extend((req_idx, chunk) for chunk in range((length + 63) // 64))
            continuing.extend(range(chunk_offsets[-2], chunk_offsets[-1] - 1))
            endings.append(chunk_offsets[-1] - 1)

        def tensor(values) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int64, device=device)

        return cls(
            cu_seqlens=tensor(offsets),
            chunk_indices=tensor(chunks),
            chunk_offsets=tensor(chunk_offsets),
            continuing=tensor(continuing),
            endings=tensor(endings),
            ending_sequences=tensor(list(range(len(lengths)))),
            conv_chunks=tensor(conv_chunks),
        )

    def fla_kwargs(self) -> dict[str, torch.Tensor]:
        return {
            "chunk_indices": self.chunk_indices,
            "chunk_offsets": self.chunk_offsets,
            "continuing": self.continuing,
            "endings": self.endings,
            "ending_sequences": self.ending_sequences,
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


def _causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state   : torch.Tensor,
    weight       : torch.Tensor,
) -> torch.Tensor:
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    output = F.conv1d(hidden_states_new, weight, groups=hidden_size)
    return F.silu(output[:, :, -seq_len:]).to(hidden_states.dtype)


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    x_float = x.float()
    output = x_float * torch.rsqrt(
        (x_float * x_float).sum(dim=dim, keepdim=True) + eps
    )
    return output.to(x.dtype)


def _prepare_gdn(
    query  : torch.Tensor,
    key    : torch.Tensor,
    value  : torch.Tensor,
    gate   : torch.Tensor,
    beta   : torch.Tensor,
    A_log  : torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    repeats = value.shape[2] // query.shape[2]
    if repeats > 1:
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)
    query = _l2norm(query)
    key = _l2norm(key)
    gate = -A_log.float().exp() * F.softplus(gate.float() + dt_bias)
    return (
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        gate.contiguous(),
        beta.float().sigmoid().contiguous(),
    )


class LinearAttentionBackend(BaseLinearAttentionBackend):
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
            metadata.snapshot_rows = torch.tensor(
                snapshot_rows, dtype=torch.int64, device=state_table.device
            )
            metadata.snapshot_reqs = torch.tensor(
                snapshot_reqs, dtype=torch.int64, device=state_table.device
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
        assert isinstance(metadata, LinearAttentionMetadata)
        assert metadata.phase == batch.phase
        if metadata.phase == "prefill":
            return self._forward_prefill(
                mixed_qkv, gate, beta, conv_weight, A_log, dt_bias, config, batch
            )
        elif metadata.phase == "verify":
            return self._forward_verify(
                mixed_qkv, gate, beta, conv_weight, A_log, dt_bias, config, metadata
            )
        return self._forward_decode(
            mixed_qkv, gate, beta, conv_weight, A_log, dt_bias, config, metadata
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
    ) -> torch.Tensor:
        from logssms.kernel.linear_attention.vendor.fla import chunk_gated_delta_rule
        from logssms.kernel.linear_attention.packed_conv import (
            packed_causal_conv1d, packed_conv_snapshots,
        )

        ctx = get_global_ctx()
        pool = ctx.linear_state
        layer_idx = config.layer_idx
        metadata = batch.linear_attn_metadata
        assert isinstance(metadata, LinearAttentionMetadata)
        indices = metadata.prefill_indices
        slots = metadata.baseline_slots
        assert indices is not None and slots is not None
        assert metadata.checkpoint_slots is not None
        assert metadata.checkpoint_chunks is not None
        assert metadata.snapshot_rows is not None
        assert metadata.snapshot_reqs is not None
        live = slots >= 0
        conv_history = pool.conv_state[:, layer_idx].index_select(
            0, slots.clamp_min(0)
        )
        conv_history = torch.where(
            live[:, None, None], conv_history, 0
        )
        initial_states = pool.recurrent_state[:, layer_idx].index_select(
            0, slots.clamp_min(0)
        )
        initial_states = torch.where(
            live[:, None, None, None], initial_states, 0
        )
        conv_snapshots = packed_conv_snapshots(
            mixed_qkv, conv_history, indices.cu_seqlens,
            metadata.snapshot_rows, metadata.snapshot_reqs,
        )
        mixed_qkv = packed_causal_conv1d(
            mixed_qkv, conv_weight, conv_history,
            indices.cu_seqlens, indices.conv_chunks,
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
        output, _final_state, chunk_states = chunk_gated_delta_rule(
            query,
            key,
            value,
            prepared_gate,
            prepared_beta,
            initial_states,
            indices.cu_seqlens,
            **indices.fla_kwargs(),
        )

        store_state(
            pool.conv_state[:, layer_idx],
            metadata.checkpoint_slots,
            conv_snapshots,
        )
        store_state(
            pool.recurrent_state[:, layer_idx],
            metadata.checkpoint_slots,
            chunk_states.index_select(0, metadata.checkpoint_chunks).to(pool.recurrent_state.dtype),
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
    ) -> torch.Tensor:
        from logssms.kernel.linear_attention.vendor.fla import (
            fused_recurrent_gated_delta_rule_to_slots,
        )
        from logssms.kernel.linear_attention.conv_decode import conv_decode_qkv

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
        )
        return output.squeeze(1)


__all__ = [
    "LinearAttentionBackend",
    "LinearAttentionCaptureData",
    "LinearAttentionMetadata",
]
