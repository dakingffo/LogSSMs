from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from picosgl.core import get_global_ctx
from picosgl.kernel import store_state

from .base import BaseLinearAttentionBackend, BaseLinearAttentionMetadata

if TYPE_CHECKING:
    from picosgl.core import Batch
    from picosgl.layers.gated_delta import GatedDeltaConfig


@dataclass
class LinearAttentionMetadata(BaseLinearAttentionMetadata):
    read_slots   : torch.Tensor | None = None
    write_slots  : torch.Tensor | None = None
    verify_groups: dict[
        int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] | None = None


@dataclass
class LinearAttentionCaptureData:
    read_slots : torch.Tensor
    write_slots: torch.Tensor

    @classmethod
    def make(cls, max_bs: int, device: torch.device) -> LinearAttentionCaptureData:
        return cls(
            read_slots=torch.zeros(max_bs, dtype=torch.int64, device=device),
            write_slots=torch.zeros(max_bs, dtype=torch.int64, device=device),
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
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


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
    return query, key, value, gate, beta.sigmoid()


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
            table_indices = torch.tensor(
                [req.table_idx for req in reqs],
                dtype=torch.int64,
                device=state_table.device,
            )
            read_pages = torch.tensor(
                [(req.cached_len - 1) // ctx.page_size for req in reqs],
                dtype=torch.int64,
                device=state_table.device,
            )
            write_pages = torch.tensor(
                [req.cached_len // ctx.page_size for req in reqs],
                dtype=torch.int64,
                device=state_table.device,
            )
            metadata.read_slots = state_table[table_indices, read_pages].to(torch.int64)
            metadata.write_slots = state_table[table_indices, write_pages].to(torch.int64)
        elif batch.is_verify:
            reserve_offset = ctx.draft_offset
            assert reserve_offset is not None

            entries_by_len: dict[int, list[tuple[int, int, int]]] = {}
            offset = 0
            for req in batch.reqs:
                entries_by_len.setdefault(req.extend_len, []).append(
                    (req.table_idx, req.baseline_slot, offset)
                )
                offset += req.extend_len

            groups = {}
            for seq_len, entries in entries_by_len.items():
                row_indices = torch.tensor(
                    [
                        row
                        for _table_idx, _baseline_slot, start in entries
                        for row in range(start, start + seq_len)
                    ],
                    dtype=torch.int64,
                    device=state_table.device,
                )
                table_indices = torch.tensor(
                    [table_idx for table_idx, _baseline_slot, _start in entries],
                    dtype=torch.int64,
                    device=state_table.device,
                )
                baseline_slots = torch.tensor(
                    [baseline_slot for _table_idx, baseline_slot, _start in entries],
                    dtype=torch.int64,
                    device=state_table.device,
                )
                write_slots = state_table[
                    table_indices, reserve_offset : reserve_offset + seq_len
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
        assert metadata.read_slots is not None and metadata.write_slots is not None
        self.capture.read_slots[:bs].copy_(metadata.read_slots)
        self.capture.write_slots[:bs].copy_(metadata.write_slots)
        batch.linear_attn_metadata = LinearAttentionMetadata(
            phase="decode",
            read_slots=self.capture.read_slots[:bs],
            write_slots=self.capture.write_slots[:bs],
        )

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata = batch.linear_attn_metadata
        bs = batch.padded_size
        assert self.capture is not None and bs in self.capture_bs
        assert metadata.read_slots is not None and metadata.write_slots is not None
        self.capture.read_slots[:bs].copy_(metadata.read_slots)
        self.capture.write_slots[:bs].copy_(metadata.write_slots)

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
            query.reshape(mixed_qkv.shape[0], seq_len, -1, config.head_k_dim),
            key.reshape(mixed_qkv.shape[0], seq_len, -1, config.head_k_dim),
            value.reshape(mixed_qkv.shape[0], seq_len, -1, config.head_v_dim),
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
        from picosgl.kernel.linear_attention.vendor.fla import chunk_gated_delta_rule

        ctx = get_global_ctx()
        pool = ctx.linear_state
        state_table = ctx.state_table
        page_size = ctx.page_size
        layer_idx = config.layer_idx
        assert state_table is not None
        assert page_size % 64 == 0
        lengths = []
        initial_states = []
        qkv_segments = []
        checkpoint_slots = []
        checkpoint_chunks = []
        conv_snapshots = []
        offset = 0
        chunk_offset = 0
        for req in batch.reqs:
            seq_len = req.extend_len
            end_offset = offset + seq_len
            use_state = req.cached_len > 0
            cached_len = req.cached_len
            # Chunked prefill boundaries are aligned to the GDN chunk. Prefix
            # cache hits are page-aligned, which is a stronger guarantee.
            assert cached_len % 64 == 0
            if use_state:
                baseline_slot = int(
                    state_table[req.table_idx, (cached_len - 1) // page_size]
                )

            req_qkv = mixed_qkv[offset:end_offset].transpose(0, 1).unsqueeze(0)
            conv_input = req_qkv
            if use_state:
                conv_input = torch.cat(
                    [pool.conv_state[baseline_slot, layer_idx].unsqueeze(0), conv_input],
                    dim=-1,
                )
            total_len = conv_input.shape[-1]
            conv_output = F.silu(
                F.conv1d(
                    conv_input,
                    conv_weight,
                    padding=config.state_len,
                    groups=config.conv_dim,
                )
            )
            req_qkv = conv_output[:, :, :total_len][:, :, -seq_len:].transpose(1, 2)
            qkv_segments.append(req_qkv.squeeze(0))
            lengths.append(seq_len)
            initial_states.append(
                pool.recurrent_state[baseline_slot, layer_idx]
                if use_state else torch.zeros_like(pool.recurrent_state[0, layer_idx])
            )

            history_prefix = config.state_len if use_state else 0
            first_page_end = page_size - cached_len % page_size
            for end in range(first_page_end, seq_len, page_size):
                page = (cached_len + end - 1) // page_size
                checkpoint_slots.append(state_table[req.table_idx, page])
                checkpoint_chunks.append(chunk_offset + (end - 1) // 64)
                history = conv_input[0, :, : history_prefix + end]
                if history.shape[-1] < config.state_len:
                    history = F.pad(history, (config.state_len - history.shape[-1], 0))
                conv_snapshots.append(history[:, -config.state_len:])

            end = seq_len
            page = (cached_len + end - 1) // page_size
            checkpoint_slots.append(state_table[req.table_idx, page])
            checkpoint_chunks.append(chunk_offset + (end - 1) // 64)
            history = conv_input[0, :, : history_prefix + end]
            if history.shape[-1] < config.state_len:
                history = F.pad(history, (config.state_len - history.shape[-1], 0))
            conv_snapshots.append(history[:, -config.state_len:])

            chunk_offset += (seq_len + 63) // 64
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
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int64,
            device=mixed_qkv.device,
        )
        output, _final_state, chunk_states = chunk_gated_delta_rule(
            query,
            key,
            value,
            prepared_gate,
            prepared_beta,
            torch.stack(initial_states),
            cu_seqlens,
        )

        slots = torch.stack(checkpoint_slots).to(torch.int64)
        chunks = torch.tensor(checkpoint_chunks, dtype=torch.int64, device=slots.device)
        store_state(
            pool.conv_state[:, layer_idx],
            slots,
            torch.stack(conv_snapshots),
        )
        store_state(
            pool.recurrent_state[:, layer_idx],
            slots,
            chunk_states.index_select(0, chunks).to(pool.recurrent_state.dtype),
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
        from picosgl.kernel.linear_attention.vendor.fla import (
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
        from picosgl.kernel.linear_attention.vendor.fla import (
            fused_recurrent_gated_delta_rule,
        )

        ctx = get_global_ctx()
        pool = ctx.linear_state
        layer_idx = config.layer_idx
        read_slots = metadata.read_slots
        write_slots = metadata.write_slots
        assert read_slots is not None and write_slots is not None
        conv_state = pool.conv_state[:, layer_idx].index_select(0, read_slots)
        recurrent_state = pool.recurrent_state[:, layer_idx].index_select(0, read_slots)

        mixed_qkv = mixed_qkv.unsqueeze(1).transpose(1, 2)
        mixed_qkv = _causal_conv1d_update(mixed_qkv, conv_state, conv_weight)
        pool.conv_state[:, layer_idx].index_copy_(0, write_slots, conv_state)

        query, key, value = self._split_qkv(mixed_qkv.transpose(1, 2), config)
        output, states = fused_recurrent_gated_delta_rule(
            query,
            key,
            value,
            gate.unsqueeze(1),
            beta.unsqueeze(1),
            A_log,
            dt_bias,
            recurrent_state,
        )
        store_state(
            pool.recurrent_state[:, layer_idx],
            write_slots,
            states.squeeze(1),
        )
        return output.squeeze(1)


__all__ = [
    "LinearAttentionBackend",
    "LinearAttentionCaptureData",
    "LinearAttentionMetadata",
]
