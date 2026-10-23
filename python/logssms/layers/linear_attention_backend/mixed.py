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


class _MetadataPadding:
    def __init__(self):
        self.copies = []

    def __call__(self, tensor, capacity, value=0, *, chunk_tail=-1):
        if tensor.shape[0] > capacity:
            raise ValueError("metadata exceeds mixed graph capacity")
        output = tensor.new_empty((capacity, *tensor.shape[1:]))
        self.copies.append((tensor.contiguous(), output, value, chunk_tail))
        return output

    def finish(self):
        from logssms.kernel.metadata import pad_metadata

        if self.copies:
            sources, outputs, fills, tails = zip(*self.copies)
            pad_metadata(sources, outputs, fills, tails)


def _packed_pad(
    indices: PackedLinearIndices, capacity: int, requests: int, chunk_size: int,
    pad: _MetadataPadding,
) -> PackedLinearIndices:
    # Maximum sum(ceil(length / chunk_size)) for positive lengths summing to P.
    chunks = requests + (capacity - requests) // chunk_size
    conv_chunks = requests + (capacity - requests) // 64

    def pad_chunks(source: torch.Tensor, count: int) -> torch.Tensor:
        return pad(source, count, chunk_tail=capacity)

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
    pad = _MetadataPadding()
    packed = _packed_pad(indices, prefill_capacity, len(lengths), chunk_size, pad)
    chunk_capacity = len(packed.chunk_indices)
    if pm.chunk_write_slots is None or pm.conv_token_slots is None:
        raise ValueError("mixed prefill requires direct-write metadata")
    updates = dict(
        prefill_indices=packed,
        chunk_write_slots=pad(pm.chunk_write_slots, chunk_capacity, -1),
        conv_token_slots=pad(pm.conv_token_slots, prefill_capacity, -1),
    )
    if isinstance(pm, LinearAttentionMetadata):
        # Checkpoints are at most one per chunk. Index padding reads the first
        # real snapshot/state; sentinel destinations suppress all padded writes.
        for name in ("checkpoint_slots", "checkpoint_chunks", "snapshot_rows", "snapshot_reqs"):
            tensor = getattr(pm, name)
            if tensor is None:
                raise ValueError(f"missing prefill {name}")
            updates[name] = pad(tensor, chunk_capacity, -1 if name == "checkpoint_slots" else 0)
    else:
        updates.update(
            cu_seqlens=packed.cu_seqlens,
            log_slots=pad(pm.log_slots, prefill_capacity, -1),
            token_offsets=pad(pm.token_offsets, prefill_capacity),
            log_write_slots=pad(pm.log_write_slots, prefill_capacity, -1),
            conv_token_granules=pad(pm.conv_token_granules, prefill_capacity),
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
                updates[name] = pad(
                    tensor, prefill_capacity,
                    -1 if name in ("conv_write_slots", "checkpoint_slots") else 0,
                )
            selection = pm.conv_write_snapshot_indices
            if selection is None:
                selection = torch.arange(
                    len(pm.conv_write_slots), device=pm.work_slots.device, dtype=torch.int64,
                )
            updates["conv_write_snapshot_indices"] = pad(selection, prefill_capacity)
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
    pad.finish()
    normalized = replace(pm, **updates), replace(dm, **decode_updates)
    prefill.linear_attn_metadata, decode.linear_attn_metadata = normalized
    return normalized


class MixedLinearCopyPlan:
    def __init__(self, destination):
        self.tensors = []
        self.constants = []
        self.structures = []

        def visit(dst, getter):
            if isinstance(dst, torch.Tensor):
                self.tensors.append((getter, dst, (dst.shape, dst.dtype, dst.device)))
            elif is_dataclass(dst):
                self.structures.append((getter, type(dst), None))
                for field in fields(dst):
                    visit(getattr(dst, field.name),
                          lambda root, get=getter, name=field.name: getattr(get(root), name))
            elif isinstance(dst, (tuple, list, dict)):
                keys = tuple(dst) if isinstance(dst, dict) else tuple(range(len(dst)))
                self.structures.append((getter, type(dst), keys if isinstance(dst, dict) else len(dst)))
                for key in keys:
                    visit(dst[key], lambda root, get=getter, key=key: get(root)[key])
            else:
                self.constants.append((getter, dst))

        visit(destination, lambda root: root)
        groups = {}
        self.fallback = []
        for index, (_, dst, _) in enumerate(self.tensors):
            if dst.numel() == 0:
                continue
            if dst.is_cuda and (dst.is_contiguous() or dst.ndim == 1):
                groups.setdefault(dst.device, []).append(index)
            else:
                self.fallback.append(index)
        self.groups = [(tuple(indices), tuple(self.tensors[i][1] for i in indices))
                       for indices in groups.values()]

    def copy(self, source):
        from logssms.kernel.metadata import copy_metadata

        try:
            for get, kind, size in self.structures:
                value = get(source)
                if type(value) is not kind:
                    raise ValueError("mixed metadata structure changed")
                if size is not None and (tuple(value) if kind is dict else len(value)) != size:
                    raise ValueError("mixed metadata sequence/dictionary changed")
            for get, expected in self.constants:
                if get(source) != expected:
                    raise ValueError("mixed metadata non-tensor value changed")
            sources = tuple(get(source) for get, _, _ in self.tensors)
            for src, (_, _, layout) in zip(sources, self.tensors):
                if not isinstance(src, torch.Tensor) or (src.shape, src.dtype, src.device) != layout:
                    raise ValueError("mixed metadata tensor shape/dtype/device changed")
            for indices, outputs in self.groups:
                if any(not sources[i].is_contiguous() and sources[i].ndim != 1 for i in indices):
                    raise ValueError("mixed metadata source strides changed")
        except (AttributeError, KeyError, IndexError, TypeError) as error:
            raise ValueError("mixed metadata structure changed") from error
        for indices, outputs in self.groups:
            copy_metadata(tuple(sources[i] for i in indices), outputs)
        for i in self.fallback:
            self.tensors[i][1].copy_(sources[i])


__all__ = ["prepare_mixed_linear_metadata", "MixedLinearCopyPlan"]
