"""Batch CPU planning primitives; tree and device resource ownership stay in Python."""

from __future__ import annotations

import functools
from array import array
from typing import TYPE_CHECKING, Sequence, TypeAlias

import torch

from logssms.tvm import load_aot

if TYPE_CHECKING:
    from tvm_ffi import Module


Rows: TypeAlias = Sequence[Sequence[int]]


@functools.cache
def _load_logstate_module() -> Module:
    return load_aot("logstate", cpp_files=["logstate.cpp"])


def _pack(rows: Rows) -> torch.Tensor:
    values = []
    for row in rows:
        values.append(len(row))
        values.extend(row)
    if not values:
        return torch.empty(0, dtype=torch.int64, device="cpu")
    # frombuffer keeps the array alive; avoid torch.tensor's per-scalar conversion.
    return torch.frombuffer(array("q", values), dtype=torch.int64)


def _unpack(tensor) -> list[list[int]]:
    values = torch.from_dlpack(tensor).tolist()
    rows = []
    index = 0
    while index < len(values):
        size = values[index]
        rows.append(values[index + 1:index + 1 + size])
        index += 1 + size
    return rows


def batch_branch_points(inputs: Sequence[torch.Tensor], granularity: int) -> list[list[int]]:
    if len(inputs) < 2:
        return []
    return _unpack(_load_logstate_module().batch_branch_points(inputs, granularity))


def plan_materialize(paths: Rows, boundaries: Rows, candidates: Rows,
                     page_size: int) -> list[list[int]]:
    return _unpack(_load_logstate_module().plan_materialize(
        _pack(paths), _pack(boundaries), _pack(candidates), page_size,
    ))


def plan_coalesce(paths: Rows, boundaries: Rows, locks: Rows, anchors: Rows,
                  materialized: Sequence[int], page_size: int,
                  adaptive: bool, diagnostics: bool) -> tuple[list[list[int]], dict[str, int]]:
    result = _load_logstate_module().plan_coalesce(
        _pack(paths), _pack(boundaries), _pack(locks), _pack(anchors),
        materialized, page_size, adaptive, diagnostics,
    )
    return _unpack(result["groups"]), dict(result["diagnostics"].items())


def schedule_construction_waves(chains: Rows, plans: Sequence[int]) -> list[list[int]]:
    outputs = torch.frombuffer(array("q", plans), dtype=torch.int64) if plans else torch.empty(
        0, dtype=torch.int64, device="cpu",
    )
    return _unpack(_load_logstate_module().schedule_construction_waves(_pack(chains), outputs))


def cover_logs(path: Sequence[int], begin: int, end: int) -> list[int]:
    return _unpack(_load_logstate_module().cover_logs(_pack([path]), begin, end))[0]
