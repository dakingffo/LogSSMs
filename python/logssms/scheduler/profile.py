from __future__ import annotations

import json
import os
from time import perf_counter_ns
from collections import Counter, deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from logssms.distributed import get_tp_info
from logssms.engine.profile import batch_identity, batch_snapshot
from logssms.utils import init_logger, nvtx_annotate

if TYPE_CHECKING:
    from logssms.cache import BaseCacheManager
    from logssms.core import Batch
    from logssms.engine import Engine
    from logssms.layers.linear_attention_backend import BaseLinearAttentionBackend


logger = init_logger(__name__)


def _env_bool(name: str) -> bool:
    return os.getenv(name, "0").lower() in ("1", "true", "yes")


def _env_int(name: str) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else 0


def _batch_counts(batch: Batch, phase: str) -> dict[str, int]:
    prefill = batch.prefill_count
    return {
        "prefill_count": prefill,
        "decode_count": batch.size - prefill if phase in ("decode", "mixed") else 0,
        "verify_count": batch.size if phase == "verify" else 0,
    }


@dataclass
class _PhaseProfile:
    batches        : int = 0
    logical_rows   : int = 0
    padded_rows    : int = 0
    logical_tokens : int = 0
    padded_tokens  : int = 0
    graph_batches  : int = 0
    prefill_count  : int = 0
    decode_count   : int = 0
    verify_count   : int = 0
    schedule_ns    : int = 0
    prepare_ns     : int = 0
    submit_ns      : int = 0
    wait_ns        : int = 0
    process_ns     : int = 0
    gpu_ns         : int = 0
    logical_bs     : Counter[int] = field(default_factory=Counter)
    padded_bs      : Counter[int] = field(default_factory=Counter)

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    def as_dict(self) -> dict:
        return {
            "batches": self.batches,
            "logical_rows": self.logical_rows,
            "padded_rows": self.padded_rows,
            "logical_tokens": self.logical_tokens,
            "padded_tokens": self.padded_tokens,
            "graph_batches": self.graph_batches,
            "prefill_count": self.prefill_count,
            "decode_count": self.decode_count,
            "verify_count": self.verify_count,
            "avg_logical_bs": self._ratio(self.logical_rows, self.batches),
            "avg_padded_bs": self._ratio(self.padded_rows, self.batches),
            "row_padding_efficiency": self._ratio(self.logical_rows, self.padded_rows),
            "token_padding_efficiency": self._ratio(
                self.logical_tokens, self.padded_tokens
            ),
            "graph_hit_rate": self._ratio(self.graph_batches, self.batches),
            "schedule_ms": self.schedule_ns / 1e6,
            "prepare_ms": self.prepare_ns / 1e6,
            "submit_ms": self.submit_ns / 1e6,
            "wait_ms": self.wait_ns / 1e6,
            "process_ms": self.process_ns / 1e6,
            "gpu_ms": self.gpu_ns / 1e6,
            "avg_gpu_ms": self._ratio(self.gpu_ns / 1e6, self.batches),
            "gpu_ms_per_logical_token": self._ratio(
                self.gpu_ns / 1e6, self.logical_tokens
            ),
            "logical_bs": dict(sorted(self.logical_bs.items())),
            "padded_bs": dict(sorted(self.padded_bs.items())),
        }


class SchedulerProfiler:

    def __init__(
        self,
        device       : torch.device,
        cache_manager: BaseCacheManager,
        linear_backend: BaseLinearAttentionBackend | None = None,
        engine       : Engine | None = None,
    ) -> None:
        self.enabled = _env_bool("LOGSSMS_PROFILE_SCHEDULER")
        self.device = device
        self.cache_manager = cache_manager
        self.linear_backend = linear_backend
        self.engine = engine
        self.path = os.getenv("LOGSSMS_PROFILE_SCHEDULER_PATH")
        self.flush_interval = _env_int("LOGSSMS_PROFILE_FLUSH_INTERVAL")
        self.ar_forward_count = 0
        self.ar_no_batch = 0
        self.last_flushed_forward = 0
        self.phases = {
            phase: _PhaseProfile() for phase in ("prefill", "decode", "verify", "mixed")
        }
        self.pending_gpu: list[tuple[
            str, torch.cuda.Event, torch.cuda.Event, dict | None,
        ]] = []
        self.sample_limit = int(os.getenv("LOGSSMS_PROFILE_MAX_BATCH_SAMPLES", "8192"))
        if self.sample_limit < 1:
            raise ValueError("LOGSSMS_PROFILE_MAX_BATCH_SAMPLES must be positive")
        self.batch_samples = {
            kind: {phase: deque(maxlen=self.sample_limit) for phase in self.phases}
            for kind in ("prepare", "forward", "process")
        }
        self.prefill_forward_batches = self.batch_samples["forward"]["prefill"]
        self.prefill_process_batches = self.batch_samples["process"]["prefill"]
        self.process_stage_ns = {
            phase: Counter() for phase in self.phases
        }
        self._current_process_sample: dict | None = None
        self._process_stage_name = ""
        self._last_wait_ms = 0.0
        self.schedule_ns = 0
        self.schedule_iterations = 0
        self.schedule_no_batch = 0
        self.schedule_stage_ns = Counter()
        self.schedule_exclusive_ns = Counter()
        self.schedule_stage_calls = Counter()
        self.schedule_no_batch_samples = deque(maxlen=self.sample_limit)
        self._schedule_sample: dict | None = None
        self._last_schedule_sample: dict | None = None
        self._schedule_request: dict | None = None
        self._schedule_stack: list[list[int]] = []
        self._schedule_stage_name = ""

    @contextmanager
    @nvtx_annotate("Scheduler::schedule")
    def record_schedule(self):
        if not self.enabled:
            yield None
            return
        sample = {
            "iteration": self.schedule_iterations,
            "phase": "no_batch",
            "stages_ms": {},
            "exclusive_stages_ms": {},
            "stage_calls": {},
            "requests": [],
            "dropped_requests": 0,
        }
        self._schedule_sample = sample
        previous_recorder = getattr(self.cache_manager, "_schedule_profile_step", None)
        self.cache_manager._schedule_profile_step = self.record_schedule_step
        tic = perf_counter_ns()
        try:
            yield sample
        finally:
            self.schedule_ns = perf_counter_ns() - tic
            self.cache_manager._schedule_profile_step = previous_recorder
            sample["total_ms"] = self.schedule_ns / 1e6
            sample["unaccounted_ms"] = (
                sample["total_ms"] - sum(sample["exclusive_stages_ms"].values())
            )
            self.schedule_iterations += 1
            if sample["phase"] == "no_batch":
                self.schedule_no_batch += 1
                self.schedule_no_batch_samples.append(sample)
            self._last_schedule_sample = sample
            self._schedule_sample = None

    def record_schedule_step(self, name: str):
        if self._schedule_sample is None:
            return nullcontext()
        self._schedule_stage_name = name
        return self._record_schedule_step(name)

    @contextmanager
    @nvtx_annotate("Scheduler::schedule::{}", layer_id_field="_schedule_stage_name")
    def _record_schedule_step(self, name: str):
        frame = [perf_counter_ns(), 0]
        self._schedule_stack.append(frame)
        try:
            yield
        finally:
            elapsed = perf_counter_ns() - frame[0]
            exclusive = elapsed - frame[1]
            self._schedule_stack.pop()
            if self._schedule_stack:
                self._schedule_stack[-1][1] += elapsed
            self.schedule_stage_ns[name] += elapsed
            self.schedule_exclusive_ns[name] += exclusive
            self.schedule_stage_calls[name] += 1
            for sample in (self._schedule_sample, self._schedule_request):
                if sample is None:
                    continue
                for key, value in (
                    ("stages_ms", elapsed / 1e6),
                    ("exclusive_stages_ms", exclusive / 1e6),
                    ("stage_calls", 1),
                ):
                    stages = sample[key]
                    stages[name] = stages.get(name, 0) + value

    @contextmanager
    def record_schedule_request(self, req):
        if self._schedule_sample is None:
            yield {}
            return
        sample = {
            "uid": req.uid,
            "tag": req.profile_tag,
            "input_len": req.input_len,
            "output_len": req.output_len,
            "chunked": req.chunked_req is not None,
            "stages_ms": {},
            "exclusive_stages_ms": {},
            "stage_calls": {},
        }
        self._schedule_request = sample
        tic = perf_counter_ns()
        try:
            yield sample
        finally:
            sample["total_ms"] = (perf_counter_ns() - tic) / 1e6
            self._schedule_request = None
            iteration = self._schedule_sample
            if iteration is not None:
                if len(iteration["requests"]) < self.sample_limit:
                    iteration["requests"].append(sample)
                else:
                    iteration["dropped_requests"] += 1

    @contextmanager
    @nvtx_annotate("Scheduler::prepare_batch")
    def record_prepare(
        self,
        batch: Batch,
        *,
        graph_used: bool,
    ):
        if not self.enabled:
            yield
            return
        tic = perf_counter_ns()
        try:
            yield
        finally:
            self._record_batch(batch, perf_counter_ns() - tic, graph_used)

    def _record_batch(
        self,
        batch       : Batch,
        prepare_ns  : int,
        graph_used  : bool,
    ) -> None:
        profile = self.phases[batch.phase]
        profile.batches += 1
        profile.logical_rows += batch.size
        profile.padded_rows += batch.padded_size
        profile.logical_tokens += sum(req.extend_len for req in batch.reqs)
        profile.padded_tokens += sum(req.extend_len for req in batch.padded_reqs)
        profile.graph_batches += int(graph_used)
        profile.schedule_ns += self.schedule_ns
        profile.prepare_ns += prepare_ns
        profile.logical_bs[batch.size] += 1
        profile.padded_bs[batch.padded_size] += 1
        counts = _batch_counts(batch, batch.phase)
        profile.prefill_count += counts["prefill_count"]
        profile.decode_count += counts["decode_count"]
        profile.verify_count += counts["verify_count"]
        sample = {
            **batch_snapshot(batch),
            **counts,
            "graph_used": graph_used,
            "schedule_ms": self.schedule_ns / 1e6,
            "prepare_ms": prepare_ns / 1e6,
        }
        if self._last_schedule_sample is not None:
            self._last_schedule_sample.update(batch_identity(batch))
            self._last_schedule_sample.update(counts)
            sample["schedule_detail"] = self._last_schedule_sample
        self.batch_samples["prepare"][batch.phase].append(sample)

    @contextmanager
    @nvtx_annotate("Scheduler::forward(submit)")
    def record_submit(self, batch: Batch):
        if not self.enabled:
            yield
            return
        sample = {
            **batch_snapshot(batch),
            **_batch_counts(batch, batch.phase),
            "submit_ms": None,
            "gpu_ms": None,
        }
        tic = perf_counter_ns()
        events = self._begin_submit(batch.phase)
        try:
            yield
        finally:
            self._finish_submit(batch.phase, events, perf_counter_ns() - tic, sample)

    def _begin_submit(self, phase: str) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        if phase in ("decode", "verify", "mixed"):
            self.ar_forward_count += 1
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return start, end

    def _finish_submit(
        self,
        phase    : str,
        events   : tuple[torch.cuda.Event, torch.cuda.Event],
        submit_ns: int,
        sample   : dict | None,
    ) -> None:
        self.phases[phase].submit_ns += submit_ns
        start, end = events
        end.record()
        if sample is not None:
            sample["submit_ms"] = submit_ns / 1e6
            self.batch_samples["forward"][phase].append(sample)
        self.pending_gpu.append((phase, start, end, sample))

    @contextmanager
    @nvtx_annotate("Scheduler::wait_output")
    def record_wait(self, phase: str):
        if not self.enabled:
            yield
            return
        tic = perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ns = perf_counter_ns() - tic
            self.phases[phase].wait_ns += elapsed_ns
            self._last_wait_ms = elapsed_ns / 1e6

    @contextmanager
    @nvtx_annotate("Scheduler::process_output")
    def record_process(self, phase: str, batch: Batch | None = None):
        if not self.enabled:
            yield
            return
        sample = None
        if batch is not None:
            sample = {
                **batch_identity(batch),
                **_batch_counts(batch, phase),
                "batch_size": batch.size,
                "wait_ms": self._last_wait_ms,
            }
            self._current_process_sample = sample
        tic = perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ns = perf_counter_ns() - tic
            self.phases[phase].process_ns += elapsed_ns
            if sample is not None:
                sample["total_ms"] = elapsed_ns / 1e6
                self.batch_samples["process"][phase].append(sample)
                self._current_process_sample = None

    def record_process_step(self, phase: str, name: str):
        if not self.enabled:
            return nullcontext()
        self._process_stage_name = name
        return self._record_process_step(phase, name)

    @contextmanager
    @nvtx_annotate("Scheduler::{}", layer_id_field="_process_stage_name")
    def _record_process_step(self, phase: str, name: str):
        if not self.enabled:
            yield
            return
        tic = perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ns = perf_counter_ns() - tic
            self.process_stage_ns[phase][name] += elapsed_ns
            if self._current_process_sample is not None:
                self._current_process_sample[name + "_ms"] = elapsed_ns / 1e6

    def record_ar_no_batch(self) -> None:
        if self.enabled:
            self.ar_no_batch += 1

    def poll(self) -> None:
        if not self.enabled:
            return
        self.cache_manager.poll_profile()
        if self.engine is not None:
            self.engine.profiler.poll()
        remaining = []
        for phase, start, end, sample in self.pending_gpu:
            if end.query():
                gpu_ms = start.elapsed_time(end)
                self.phases[phase].gpu_ns += int(gpu_ms * 1e6)
                if sample is not None:
                    sample["gpu_ms"] = gpu_ms
            else:
                remaining.append((phase, start, end, sample))
        self.pending_gpu = remaining
        if (
            self.path
            and self.flush_interval > 0
            and self.ar_forward_count - self.last_flushed_forward
            >= self.flush_interval
        ):
            self._write_result()
            self.last_flushed_forward = self.ar_forward_count

    def _make_result(self) -> dict:
        tp_info = get_tp_info()
        result = {
            "tp_rank": tp_info.rank,
            "ar_forward_count": self.ar_forward_count,
            "ar_no_batch": self.ar_no_batch,
            "pending_gpu_events": len(self.pending_gpu),
            "phases": {
                phase: profile.as_dict() for phase, profile in self.phases.items()
            },
            "schema_version": 2,
            "sample_limit_per_phase": self.sample_limit,
            "dropped_batch_samples": {
                phase: max(0, profile.batches - self.sample_limit)
                for phase, profile in self.phases.items()
            },
            "timing_semantics": {
                "cpu": "host wall time including any blocking API calls",
                "gpu": "CUDA stream intervals, including idle/wait/contention; not kernel-duration sums",
                "overlap": "CPU/GPU and prepare/forward intervals overlap; do not add them into TPOT",
                "graph": "events bracket replay at runtime; graph-internal operator attribution requires a node-level GPU trace",
            },
            "process_stages_ms": {
                phase: {name: elapsed / 1e6 for name, elapsed in stages.items()}
                for phase, stages in self.process_stage_ns.items()
            },
            "schedule_profile": {
                "iterations": self.schedule_iterations,
                "no_batch_iterations": self.schedule_no_batch,
                "stages_ms": {
                    name: elapsed / 1e6
                    for name, elapsed in self.schedule_stage_ns.items()
                },
                "exclusive_stages_ms": {
                    name: elapsed / 1e6
                    for name, elapsed in self.schedule_exclusive_ns.items()
                },
                "stage_calls": dict(self.schedule_stage_calls),
                "no_batch_samples": list(self.schedule_no_batch_samples),
                "dropped_no_batch_samples": max(
                    0, self.schedule_no_batch - self.sample_limit
                ),
                "timing_semantics": (
                    "Host wall time; stages_ms includes nested stages. "
                    "Use exclusive_stages_ms for non-overlapping attribution. "
                    "No CUDA events or synchronization are added by schedule profiling."
                ),
            },
        }
        for kind, phases in self.batch_samples.items():
            for phase, samples in phases.items():
                result[f"{phase}_{kind}_batches"] = list(samples)
        cache = self.cache_manager.profile_result()
        if cache is not None:
            result["cache"] = cache
        if self.linear_backend is not None:
            linear = self.linear_backend.profile_result()
            if linear is not None:
                result["linear_attention_backend"] = linear
        if self.engine is not None:
            prepare = self.engine.prepare_profile_result()
            if prepare is not None:
                result["engine_prepare"] = prepare
            forward = self.engine.forward_profile_result()
            if forward is not None:
                result["engine_forward"] = forward
        return result

    def _write_result(self) -> None:
        assert self.path is not None
        tp_info = get_tp_info()
        stem, suffix = os.path.splitext(self.path)
        path = f"{stem}.rank{tp_info.rank}{suffix or '.json'}"
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self._make_result(), f, indent=2)
        os.replace(tmp_path, path)

    def dump(self) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize(self.device)
        self.poll()
        assert not self.pending_gpu

        result = self._make_result()
        logger.info("Scheduler profile:\n%s", json.dumps(result, indent=2))
        if self.path:
            self._write_result()
