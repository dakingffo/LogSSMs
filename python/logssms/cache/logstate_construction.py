from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Sequence

import torch

from logssms.utils.torch_utils import make_device_vectors

from .logstate_pool.mha_pool import MHALogStatePool
from .prefix_cache.logstate import LogStateEntry


@dataclass(frozen=True)
class RestoreTask:
    output_slot        : int
    base_slot          : int | None
    logs               : tuple[LogStateEntry, ...]
    source_conv_slot   : int
    source_conv_granule: int
    output_conv_granule: int


class LogStateConstruction:
    """Restore execution through the pool API, independent of cache policy."""

    def __init__(self, pool: MHALogStatePool) -> None:
        self.pool = pool

    def restore(
        self,
        restores: Sequence[RestoreTask],
        *,
        profile_enabled: bool,
        pending_cuda_events: list[tuple[dict, torch.cuda.Event, torch.cuda.Event]],
        record_cuda: bool = True,
    ) -> dict:
        """Submit independent restores; append timings to the caller-owned event list.

        The list is supplied per call because its owner may replace it when
        polling completed events. This component never retains or polls it.
        Disable record_cuda when the caller times a larger coalesce/restore step.
        """
        submit_timing_ms: dict[str, float] | None = (
            {} if profile_enabled else None
        )
        step_start = perf_counter_ns() if submit_timing_ms is not None else 0
        events = None
        event_setup_cpu_ms = 0.0
        if profile_enabled and record_cuda and restores:
            event_setup_start = perf_counter_ns()
            events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            events[0].record()
            event_setup_cpu_ms = (
                perf_counter_ns() - event_setup_start
            ) / 1e6
        pool = self.pool
        direct = [task for task in restores if not task.logs]
        if direct:
            assert all(task.base_slot is not None for task in direct)
            pool.copy_states([task.base_slot for task in direct],
                             [task.output_slot for task in direct])
        if submit_timing_ms is not None:
            submit_timing_ms["direct_state_copy_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6
            step_start = perf_counter_ns()
        # Each restore reads immutable cache entries and writes its own work slot.
        chains = [
            (
                task.base_slot,
                [
                    (entry.slot, entry.begin - entry.physical_begin,
                     entry.end - entry.physical_begin)
                    for entry in task.logs
                ],
                task.output_slot,
            )
            for task in restores if task.logs
        ]
        if submit_timing_ms is not None:
            submit_timing_ms["chain_planning_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6

        step_start = perf_counter_ns() if submit_timing_ms is not None else 0
        conv_vectors = make_device_vectors([
            [task.source_conv_slot for task in restores],
            [task.source_conv_granule for task in restores],
            [task.output_slot for task in restores],
            [task.output_conv_granule for task in restores],
        ], pool.device) if restores else ()
        if submit_timing_ms is not None:
            submit_timing_ms["conv_metadata_submit_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6

        construction = {
            "chains": len(chains),
            "waves": int(bool(chains)),
            "segments": sum(len(logs) for _, logs, _ in chains),
            "outputs": len(chains),
            "direct_checkpoint_copies": len(direct),
            "cuda_ms": None if chains else 0.0,
        }
        if chains:
            step_start = perf_counter_ns() if submit_timing_ms is not None else 0
            pool.kernels.restore(pool, chains)
            if submit_timing_ms is not None:
                submit_timing_ms["restore_submit_ms"] = (
                    perf_counter_ns() - step_start
                ) / 1e6
                submit_timing_ms["materialize_submit_ms"] = 0.0

        step_start = perf_counter_ns() if submit_timing_ms is not None else 0
        if restores:
            pool.kernels.copy_conv(pool, *conv_vectors)
        if submit_timing_ms is not None:
            submit_timing_ms["conv_restore_submit_ms"] = (
                perf_counter_ns() - step_start
            ) / 1e6
        if events is not None:
            event_end_setup_start = perf_counter_ns()
            events[1].record()
            pending_cuda_events.append(
                (construction, events[0], events[1])
            )
            event_setup_cpu_ms += (
                perf_counter_ns() - event_end_setup_start
            ) / 1e6
        construction["event_setup_cpu_ms"] = event_setup_cpu_ms
        if submit_timing_ms is not None:
            construction["submit_timing_ms"] = submit_timing_ms
        return construction
