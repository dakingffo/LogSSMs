from __future__ import annotations

import functools
from array import array
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Iterable, Sequence

import torch
from logssms.tvm import load_aot

if TYPE_CHECKING:
    from tvm_ffi import Module


@functools.cache
def _load_logstate_module() -> Module:
    return load_aot("logstate", cpp_files=["logstate.cpp"])


@functools.cache
def _empty(dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(0, dtype=dtype, device="cpu")


def _pack(rows: Iterable[Sequence[int]]) -> torch.Tensor:
    values = []
    for row in rows:
        values.append(len(row))
        values.extend(row)
    if not values:
        return _empty(torch.int64)

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


def branch_protections(inputs: Sequence[torch.Tensor], forced: Sequence[Sequence[int]],
                       granularity: int) -> list[list[int]]:
    return _unpack(_load_logstate_module().branch_protections(inputs, _pack(forced), granularity))


def _pack_models(models, past_tails) -> torch.Tensor:
    if past_tails:
        models = list(chain.from_iterable(chain(models, past_tails)))
    return torch.tensor(models, dtype=torch.float64).flatten()


@dataclass(frozen=True, slots=True)
class ActiveDensity:
    tail: int
    granularity: int
    bins: int
    amplitude: float
    exponent: float
    tail_probability: float
    prior: float
    past: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class ActiveCost:
    density: ActiveDensity
    begin: int
    boundary: int
    end: int
    replay: int
    frames: int
    decays: tuple[tuple[int, int], ...]
    dependencies: tuple[int, ...]


def _pack_active(candidates, occupied):
    if not candidates:
        return _empty(torch.int64), _empty(torch.float64)
    models = {}
    parameters, headers, dependencies, decays = [], [], [], []
    for candidate in candidates:
        density = candidate.density
        model_id = models.get(id(density))
        if model_id is None:
            model_id = models[id(density)] = len(models)
            parameters.extend((
                density.tail, density.granularity, density.bins, density.amplitude,
                density.exponent, density.tail_probability, density.prior, len(density.past),
            ))
            parameters.extend(value for point in density.past for value in point)
        dependencies.extend(candidate.dependencies)
        for frame, count in candidate.decays:
            decays.extend((count, occupied.get(frame, 0)))
        headers.extend((
            model_id, candidate.begin, candidate.boundary, candidate.end, candidate.replay,
            candidate.frames, len(dependencies), len(decays),
        ))
    packed = array("q", (len(candidates), len(dependencies)))
    packed.extend(headers)
    packed.extend(dependencies)
    packed.extend(decays)
    return (torch.frombuffer(packed, dtype=torch.int64),
            torch.frombuffer(array("d", parameters), dtype=torch.float64))


class EvictionPlanner:
    def __init__(self, nodes, kv, node_slots, slots, occupied, config, checkpoint_actions,
                 models, order, materializations=(), *, past_tails=()):
        groups = (nodes, kv, node_slots, slots, occupied, checkpoint_actions, materializations)
        self._planner = _load_logstate_module().make_eviction_planner(
            _pack(chain.from_iterable(groups)), [len(group) for group in groups], config,
            _pack_models(models, past_tails), _pack(order),
        )

    def rebase(
        self, size, removed, geometry, indices, models, order, config, active_candidates, *,
        past_tails, active_occupied,
    ):
        active_rows, active_models = _pack_active(active_candidates, active_occupied)
        return _unpack(self._planner["rebase"](
            size, removed, _pack(geometry), indices, _pack_models(models, past_tails), _pack(order),
            config, active_rows, active_models,
        ))

    def replace_materialized_node(self, index, timestamp, records, actions, frontiers, groups):
        return self._planner["replace_materialized_node"](
            index, timestamp, _pack(records), _pack(actions), _pack(frontiers), _pack(groups),
        )

    def plan(self, config, occupied, active_candidates=(), *, active_occupied=None, locks=None, models=None):
        num_blocked, lock_rows = 0, _empty(torch.int64)
        if locks is not None:
            blocked, actions = locks
            num_blocked = len(blocked)
            lock_rows = _pack(chain(blocked, actions))
        model_indices, model_rows = (), _empty(torch.float64)
        if models is not None:
            model_indices, density, past_tails = models
            model_rows = _pack_models(density, past_tails)
        active_rows, active_models = _pack_active(active_candidates, active_occupied)
        return _unpack(self._planner["plan"](
            config, _pack(occupied), active_rows, active_models,
            num_blocked, lock_rows, model_indices, model_rows,
        ))


def plan_materialize(paths: Sequence[Sequence[int]], boundaries: Sequence[Sequence[int]],
                     candidates: Sequence[Sequence[int]],
                     granularity: int, log_count: int) -> list[list[int]]:
    return _unpack(_load_logstate_module().plan_materialize(
        _pack(paths), _pack(boundaries), _pack(candidates), granularity, log_count,
    ))
