from __future__ import annotations

import functools
from array import array
from itertools import chain
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


def branch_protections(inputs: Sequence[torch.Tensor], forced: Rows,
                       granularity: int) -> list[list[int]]:
    return _unpack(_load_logstate_module().branch_protections(inputs, _pack(forced), granularity))


def plan_eviction(nodes: Rows, kv: Rows, node_slots: Rows, slots: Rows,
                  occupied: Rows, config: Sequence[int], checkpoint_actions: Rows, models=None) -> list[list[int]] | None:
    """Return a plan, [] for insufficient capacity, or None to expand a cold path."""
    groups = (nodes, kv, node_slots, slots, occupied, checkpoint_actions)
    result = _unpack(_load_logstate_module().plan_eviction(
        _pack(chain.from_iterable(groups)), [len(group) for group in groups], config,
        torch.tensor([] if models is None else models, dtype=torch.float64).flatten(),
    ))
    return None if result == [[-1]] else result


class EvictionPlanner:
    def __init__(self, nodes, kv, node_slots, slots, occupied, config, checkpoint_actions,
                 models, order):
        groups = (nodes, kv, node_slots, slots, occupied, checkpoint_actions)
        self._planner = _load_logstate_module().make_eviction_planner(
            _pack(chain.from_iterable(groups)), [len(group) for group in groups], config,
            torch.tensor(models, dtype=torch.float64).flatten(), _pack(order),
        )

    def refresh_models(self, indices, models):
        self._planner["refresh_models"](
            indices, torch.tensor(models, dtype=torch.float64).flatten(),
        )

    def plan(self, config, occupied):
        result = _unpack(self._planner["plan"](config, _pack(occupied)))
        return None if result == [[-1]] else result


def make_recycle_bin(capacity_bytes: int):
    return _load_logstate_module().make_recycle_bin(capacity_bytes)


def plan_materialize(paths: Rows, boundaries: Rows, candidates: Rows,
                     granularity: int, log_count: int) -> list[list[int]]:
    return _unpack(_load_logstate_module().plan_materialize(
        _pack(paths), _pack(boundaries), _pack(candidates), granularity, log_count,
    ))
