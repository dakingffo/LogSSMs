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


def _pack_models(models, past_tails) -> torch.Tensor:
    if past_tails:
        models = list(chain.from_iterable(chain(models, past_tails)))
    return torch.tensor(models, dtype=torch.float64).flatten()


def plan_eviction(nodes: Rows, kv: Rows, node_slots: Rows, slots: Rows,
                  occupied: Rows, config: Sequence[int], checkpoint_actions: Rows, models=None,
                  *, past_tails=()) -> list[list[int]]:
    """Plan reclamation with the default ordering."""
    groups = (nodes, kv, node_slots, slots, occupied, checkpoint_actions)
    result = _unpack(_load_logstate_module().plan_eviction(
        _pack(chain.from_iterable(groups)), [len(group) for group in groups], config,
        _pack_models(() if models is None else models, past_tails),
    ))
    return result


class EvictionPlanner:
    def __init__(self, nodes, kv, node_slots, slots, occupied, config, checkpoint_actions,
                 models, order, materializations=(), *, past_tails=()):
        groups = (nodes, kv, node_slots, slots, occupied, checkpoint_actions, materializations)
        self._planner = _load_logstate_module().make_eviction_planner(
            _pack(chain.from_iterable(groups)), [len(group) for group in groups], config,
            _pack_models(models, past_tails), _pack(order),
        )

    def update_locks(self, blocked, actions):
        self._planner["update_locks"](_pack(blocked), _pack(actions))

    def refresh_models(self, indices, models, *, past_tails=()):
        self._planner["refresh_models"](indices, _pack_models(models, past_tails))

    def plan(self, config, occupied, active_candidates=()):
        packed = _pack(occupied)
        if active_candidates:
            result = self._planner["plan_with_active"](
                config, packed, _pack([sources for _, sources in active_candidates]),
                [score for score, _ in active_candidates],
            )
        else:
            result = self._planner["plan"](config, packed)
        result = _unpack(result)
        return result


def make_recycle_bin(capacity_bytes: int):
    return _load_logstate_module().make_recycle_bin(capacity_bytes)


def plan_materialize(paths: Rows, boundaries: Rows, candidates: Rows,
                     granularity: int, log_count: int) -> list[list[int]]:
    return _unpack(_load_logstate_module().plan_materialize(
        _pack(paths), _pack(boundaries), _pack(candidates), granularity, log_count,
    ))
