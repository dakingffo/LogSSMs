from __future__ import annotations

import os
from collections import Counter, deque
from contextlib import contextmanager
from time import perf_counter_ns
from typing import TYPE_CHECKING

import torch

from logssms.utils import nvtx_annotate

if TYPE_CHECKING:
    from logssms.core import Batch


def batch_identity(batch: Batch) -> dict:
    return {
        "batch_id": getattr(batch, "profile_id", None),
        "request_tags": [req.profile_tag for req in batch.reqs],
    }


def batch_snapshot(batch: Batch) -> dict:
    """Copy CPU metadata before overlap advances the mutable Requests."""
    return {
        **batch_identity(batch),
        "phase": batch.phase,
        "batch_size": batch.size,
        "padded_size": len(getattr(batch, "padded_reqs", batch.reqs)),
        "input_tokens": sum(req.extend_len for req in batch.reqs),
        "cached_lens": [req.cached_len for req in batch.reqs],
        "extend_lens": [req.extend_len for req in batch.reqs],
    }


class EngineProfiler:
    """Host-stage timings and asynchronous CUDA timings outside graph capture.

    Every batch gets fresh events, including graph replays. No event is captured
    in a graph or reused by overlapping batches. GPU timings measure stream
    intervals, not a sum of kernel durations; CPU/GPU times must not be added.
    """

    def __init__(self) -> None:
        self.enabled = os.getenv("LOGSSMS_PROFILE_SCHEDULER", "0").lower() in (
            "1", "true", "yes",
        )
        self.sample_limit = int(os.getenv("LOGSSMS_PROFILE_MAX_BATCH_SAMPLES", "8192"))
        if self.sample_limit < 1:
            raise ValueError("LOGSSMS_PROFILE_MAX_BATCH_SAMPLES must be positive")
        self.origin_ns = perf_counter_ns()
        self.next_batch_id = 0
        self.samples = {
            kind: {phase: deque(maxlen=self.sample_limit) for phase in ("prefill", "decode", "verify")}
            for kind in ("prepare", "forward")
        }
        self.totals = {kind: {} for kind in self.samples}
        self.pending: list[tuple[dict, list, dict, torch.cuda.Event | None]] = []
        self._sample: dict | None = None
        self._events: list = []
        self._stage_name = ""
        self._batch_name = ""
        self._last_forward_end: torch.cuda.Event | None = None
        self._last_forward_cpu_end: int | None = None
        self._last_forward_identity: tuple[int, str] | None = None

    def record_batch(self, batch: Batch, kind: str, *, graph_used: bool):
        assert self.enabled and self._sample is None
        if kind == "prepare" or getattr(batch, "profile_id", None) is None:
            batch.profile_id = self.next_batch_id
            self.next_batch_id += 1
        self._batch_name = f"{batch.phase}::{kind}[batch={batch.profile_id},bs={batch.size}]"
        return self._record_batch(batch, kind, graph_used=graph_used)

    @contextmanager
    @nvtx_annotate("Engine::{}", layer_id_field="_batch_name")
    def _record_batch(self, batch: Batch, kind: str, *, graph_used: bool):
        tic = perf_counter_ns()
        sample = {
            **batch_snapshot(batch),
            "graph_used": graph_used,
            "start_ms": (tic - self.origin_ns) / 1e6,
            "cpu_total_ms": None,
            "gpu_total_ms": None,
            "cpu_stages_ms": {},
            "gpu_stages_ms": {},
        }
        previous_end = None
        if kind == "forward":
            previous_end = self._last_forward_end
            sample["gpu_gap_before_ms"] = None
            sample["cpu_gap_before_ms"] = (
                None if self._last_forward_cpu_end is None
                else (tic - self._last_forward_cpu_end) / 1e6
            )
            sample["previous_forward"] = (
                None if self._last_forward_identity is None else {
                    "batch_id": self._last_forward_identity[0],
                    "phase": self._last_forward_identity[1],
                }
            )
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self._sample = sample
        self._events = [("total", start, end)]
        try:
            yield sample
        finally:
            end.record()
            stop = perf_counter_ns()
            sample["cpu_total_ms"] = (stop - tic) / 1e6
            totals = self.totals[kind].setdefault(batch.phase, {
                "batches": 0, "gpu_batches": 0,
                "cpu_total_ms": 0.0, "gpu_total_ms": 0.0,
                "cpu_stages_ms": Counter(), "gpu_stages_ms": Counter(),
                "replay_count_rows": Counter(), "by_padded_size": {},
            })
            totals["batches"] += 1
            totals["cpu_total_ms"] += sample["cpu_total_ms"]
            totals["cpu_stages_ms"].update(sample["cpu_stages_ms"])
            if kind == "forward":
                totals["replay_count_rows"].update(sample.get("effective_replay_counts", ()))
                self._last_forward_end = end
                self._last_forward_cpu_end = stop
                self._last_forward_identity = (batch.profile_id, batch.phase)
            else:
                # Padding and cache preparation may add CPU-only metadata.
                sample.update(batch_snapshot(batch))
                sample["logical_tokens"] = sample["input_tokens"]
                sample["total_ms"] = sample["cpu_total_ms"]
                sample.update({name + "_ms": value for name, value in sample["cpu_stages_ms"].items()})
                mapping_names = ("positions", "input_mapping", "write_mapping", "out_loc", "positions_mappings")
                sample["positions_mappings_ms"] = sum(
                    sample["cpu_stages_ms"].get(name, 0.0) for name in mapping_names
                )
            self.samples[kind][batch.phase].append(sample)
            self.pending.append((sample, self._events, totals, previous_end))
            self._sample = None
            self._events = []

    def record_stage(self, name: str):
        self._stage_name = name
        return self._record_stage(name)

    @contextmanager
    @nvtx_annotate("Engine::{}", layer_id_field="_stage_name")
    def _record_stage(self, name: str):
        assert self._sample is not None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        tic = perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ms = (perf_counter_ns() - tic) / 1e6
            end.record()
            self._sample["cpu_stages_ms"][name] = elapsed_ms
            self._sample["gpu_stages_ms"][name] = None
            self._events.append((name, start, end))

    def poll(self) -> None:
        remaining = []
        for sample, events, totals, previous_end in self.pending:
            if not events[0][2].query():
                remaining.append((sample, events, totals, previous_end))
                continue
            totals["gpu_batches"] += 1
            group = totals["by_padded_size"].setdefault(str(sample["padded_size"]), {
                "batches": 0, "graph_batches": 0,
                "logical_rows": 0, "gpu_total_ms": 0.0, "gpu_stages_ms": Counter(),
            })
            group["batches"] += 1
            group["graph_batches"] += int(sample["graph_used"])
            group["logical_rows"] += sample["batch_size"]
            for name, start, end in events:
                elapsed = start.elapsed_time(end)
                if name == "total":
                    sample["gpu_total_ms"] = elapsed
                    totals["gpu_total_ms"] += elapsed
                    group["gpu_total_ms"] += elapsed
                else:
                    sample["gpu_stages_ms"][name] = elapsed
                    totals["gpu_stages_ms"][name] += elapsed
                    group["gpu_stages_ms"][name] += elapsed
            if previous_end is not None:
                # Forward scopes are ordered on the same engine stream.
                sample["gpu_gap_before_ms"] = previous_end.elapsed_time(events[0][1])
        self.pending = remaining

    def result(self, kind: str) -> dict | None:
        if not self.enabled:
            return None
        result = {
            "pending_gpu_events": len(self.pending),
            "sample_limit_per_phase": self.sample_limit,
            "phases": {},
        }
        for phase, totals in self.totals[kind].items():
            result["phases"][phase] = {
                **totals,
                "cpu_stages_ms": dict(totals["cpu_stages_ms"]),
                "gpu_stages_ms": dict(totals["gpu_stages_ms"]),
                "replay_count_rows": dict(sorted(totals["replay_count_rows"].items())),
                "by_padded_size": {
                    size: {**group, "gpu_stages_ms": dict(group["gpu_stages_ms"])}
                    for size, group in totals["by_padded_size"].items()
                },
                "dropped_samples": max(0, totals["batches"] - self.sample_limit),
            }
            if kind == "prepare":
                result["phases"][phase].update({
                    name + "_ms": value for name, value in totals["cpu_stages_ms"].items()
                })
                result["phases"][phase]["positions_mappings_ms"] = sum(
                    totals["cpu_stages_ms"].get(name, 0.0)
                    for name in ("positions", "input_mapping", "write_mapping", "out_loc", "positions_mappings")
                )
                result["phases"][phase]["total_ms"] = totals["cpu_total_ms"]
        for phase, samples in self.samples[kind].items():
            result[phase + "_batches"] = list(samples)
        return result
