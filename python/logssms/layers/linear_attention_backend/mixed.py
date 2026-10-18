"""Fixed-capacity metadata for mixed prefill/decode graph buckets.

Prepare ordinary metadata with actual token lengths first, then normalize outside
capture. Buckets must fix prefill request count, token capacity, padded decode
size, backend and device. Only tensor contents (including actual cu_seqlens) may
change on replay. No device tensor is inspected on the host.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import TYPE_CHECKING

import torch

from .base import BaseMamba2Backend
from .logstate import LogStateLinearAttentionMetadata
from .standard import LinearAttentionMetadata, PackedLinearIndices

if TYPE_CHECKING:
    from logssms.core import Batch
    from .base import BaseLinearAttentionBackend


def _pad(tensor: torch.Tensor, capacity: int, value: int = 0) -> torch.Tensor:
    if tensor.shape[0] > capacity:
        raise ValueError("metadata exceeds mixed graph capacity")
    result = tensor.new_full((capacity, *tensor.shape[1:]), value)
    result[:tensor.shape[0]].copy_(tensor)
    return result


def _packed_pad(
    indices: PackedLinearIndices, capacity: int, requests: int, chunk_size: int,
) -> PackedLinearIndices:
    # Maximum sum(ceil(length / chunk_size)) for positive lengths summing to P.
    chunks = requests + (capacity - requests) // chunk_size
    conv_chunks = requests + (capacity - requests) // 64

    def pad_chunks(source: torch.Tensor, count: int) -> torch.Tensor:
        result = _pad(source, count)
        # A valid request index plus an out-of-range local chunk, never duplicate
        # a live chunk. All token masks are false for these phantom programs.
        result[source.shape[0]:, 1] = capacity
        return result

    return replace(
        indices,
        chunk_indices=pad_chunks(indices.chunk_indices, chunks),
        conv_chunks=pad_chunks(indices.conv_chunks, conv_chunks),
    )


def prepare_mixed_linear_metadata(
    backend: BaseLinearAttentionBackend,
    prefill: Batch,
    decode: Batch,
    prefill_capacity: int,
) -> tuple[LinearAttentionMetadata | LogStateLinearAttentionMetadata,
           LinearAttentionMetadata | LogStateLinearAttentionMetadata]:
    """Normalize already-prepared metadata, install it, and return both dataclasses.

    Inputs to the prefill forward must have exactly ``prefill_capacity`` rows;
    only the first sum(req.extend_len) output rows are meaningful. Request-level
    arrays and cu_seqlens retain actual values and shapes. Decode must already be
    prepared for its padded bucket size, with ordinary dummy-request slots.

    GDN and Mamba2 support standard and LogState caches.
    Pure-path metadata objects/tensors are never modified. Call on fresh ordinary
    metadata before each replay, then copy into the captured dataclasses.
    """
    pm, dm = prefill.linear_attn_metadata, decode.linear_attn_metadata
    if not isinstance(pm, (LinearAttentionMetadata, LogStateLinearAttentionMetadata)):
        raise TypeError("mixed prefill requires prepared linear metadata")
    if type(dm) is not type(pm) or pm.phase != "prefill" or dm.phase != "decode":
        raise ValueError("mixed metadata requires matching prefill/decode backends")
    lengths = [req.extend_len for req in prefill.reqs]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("mixed prefill requires nonempty positive request lengths")
    if prefill_capacity < sum(lengths):
        raise ValueError("actual prefill length exceeds mixed graph capacity")
    indices = pm.prefill_indices
    if indices is None or len(indices.cu_seqlens) != len(lengths) + 1:
        raise ValueError("prepare ordinary actual-length metadata before normalization")

    if isinstance(pm, LogStateLinearAttentionMetadata):
        from logssms.core import get_global_ctx

        chunk_size = get_global_ctx().linear_state.layout.granularity
    else:
        chunk_size = 64
    packed = _packed_pad(indices, prefill_capacity, len(lengths), chunk_size)
    chunk_capacity = len(packed.chunk_indices)
    if pm.chunk_write_slots is None or pm.conv_token_slots is None:
        raise ValueError("mixed prefill requires direct-write metadata")
    updates = dict(
        prefill_indices=packed,
        chunk_write_slots=_pad(pm.chunk_write_slots, chunk_capacity, -1),
        conv_token_slots=_pad(pm.conv_token_slots, prefill_capacity, -1),
    )
    if isinstance(pm, LinearAttentionMetadata):
        # Checkpoints are at most one per chunk. Index padding reads the first
        # real snapshot/state; sentinel destinations suppress all padded writes.
        for name in ("checkpoint_slots", "checkpoint_chunks", "snapshot_rows", "snapshot_reqs"):
            tensor = getattr(pm, name)
            if tensor is None:
                raise ValueError(f"missing prefill {name}")
            updates[name] = _pad(tensor, chunk_capacity, -1 if name == "checkpoint_slots" else 0)
    else:
        updates.update(
            cu_seqlens=packed.cu_seqlens,
            log_slots=_pad(pm.log_slots, prefill_capacity, -1),
            token_offsets=_pad(pm.token_offsets, prefill_capacity),
            log_write_slots=_pad(pm.log_write_slots, prefill_capacity, -1),
            conv_token_granules=_pad(pm.conv_token_granules, prefill_capacity),
        )
        # GDN consumes the direct-write arrays above. Remove unused eager gather
        # descriptors so optional checkpoints cannot change graph structure.
        for name in (
            "conv_snapshot_rows", "conv_snapshot_reqs", "conv_final_rows",
            "conv_write_slots", "conv_write_granules", "log_row_indices",
            "checkpoint_slots", "checkpoint_conv_granules",
            "checkpoint_state_indices", "checkpoint_snapshot_indices",
            "conv_write_snapshot_indices",
        ):
            updates[name] = None
        if isinstance(backend, BaseMamba2Backend):
            # Keep all gathers fixed-size, including the optional checkpoint
            # branch. Padding gathers row/state zero but never stores it.
            for name in (
                "conv_snapshot_rows", "conv_snapshot_reqs", "conv_write_slots",
                "conv_write_granules", "checkpoint_slots",
                "checkpoint_conv_granules", "checkpoint_state_indices",
                "checkpoint_snapshot_indices",
            ):
                tensor = getattr(pm, name)
                if tensor is None:
                    dtype = torch.int64 if name.endswith("indices") else pm.work_slots.dtype
                    tensor = torch.empty(0, dtype=dtype, device=pm.work_slots.device)
                updates[name] = _pad(
                    tensor, prefill_capacity,
                    -1 if name in ("conv_write_slots", "checkpoint_slots") else 0,
                )
            selection = pm.conv_write_snapshot_indices
            if selection is None:
                selection = torch.arange(
                    len(pm.conv_write_slots), device=pm.work_slots.device, dtype=torch.int64,
                )
            updates["conv_write_snapshot_indices"] = _pad(selection, prefill_capacity)
            updates["conv_final_rows"] = pm.conv_final_rows
            # The dense direct-log map already excludes checkpoint-only rows.
            # Using it avoids a variable-size log_row_indices gather entirely.
            updates["log_slots"] = updates["log_write_slots"]
    decode_updates = {}
    if isinstance(dm, LinearAttentionMetadata):
        for name in ("slots", "read_slots", "write_slots"):
            tensor = getattr(dm, name)
            if tensor is not None:
                decode_updates[name] = tensor.to(torch.int64)
    normalized = replace(pm, **updates), replace(dm, **decode_updates)
    prefill.linear_attn_metadata, decode.linear_attn_metadata = normalized
    return normalized


def copy_mixed_linear_metadata_(destination, source) -> None:
    """Copy a normalized metadata tree into captured buffers without rebinding.

    Validate the entire structure before writing; a different graph bucket is an
    error, never an implicit resize. Supports dataclasses, tuples/lists and dicts.
    """
    copies = []

    def visit(dst, src):
        if type(dst) is not type(src):
            raise ValueError("mixed metadata structure changed")
        if isinstance(dst, torch.Tensor):
            if (dst.shape, dst.dtype, dst.device) != (src.shape, src.dtype, src.device):
                raise ValueError("mixed metadata tensor shape/dtype/device changed")
            copies.append((dst, src))
        elif is_dataclass(dst):
            for field in fields(dst):
                visit(getattr(dst, field.name), getattr(src, field.name))
        elif isinstance(dst, (tuple, list)):
            if len(dst) != len(src):
                raise ValueError("mixed metadata sequence length changed")
            for a, b in zip(dst, src):
                visit(a, b)
        elif isinstance(dst, dict):
            if dst.keys() != src.keys():
                raise ValueError("mixed metadata dictionary keys changed")
            for key in dst:
                visit(dst[key], src[key])
        elif dst != src:
            raise ValueError("mixed metadata non-tensor value changed")

    visit(destination, source)
    from logssms.kernel.metadata import copy_metadata

    groups = {}
    fallback = []
    seen = set()
    for dst, src in copies:
        identity = (dst.device, dst.data_ptr(), dst.shape, dst.stride(), src.data_ptr(), src.stride())
        if identity in seen or dst is src or dst.numel() == 0:
            continue
        seen.add(identity)
        if dst.is_cuda and dst.is_contiguous() and src.is_contiguous():
            key = (dst.device, dst.dtype, dst.numel())
            groups.setdefault(key, []).append((dst.view(-1), src.view(-1)))
        else:
            fallback.append((dst, src))
    for pairs in groups.values():
        copy_metadata(tuple(src for dst, src in pairs), tuple(dst for dst, src in pairs))
    for dst, src in fallback:
        dst.copy_(src)


__all__ = ["prepare_mixed_linear_metadata", "copy_mixed_linear_metadata_"]

