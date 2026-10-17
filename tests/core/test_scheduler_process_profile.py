from types import SimpleNamespace

import pytest
import torch

from logssms.scheduler.profile import SchedulerProfiler
from logssms.scheduler.prefill import PendingRequest, PrefillAdder
from logssms.core import SamplingParams
from logssms.cache.base import BaseCacheManager


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("phase", ["prefill", "decode", "verify", "mixed"])
def test_process_profile_breakdown(monkeypatch, enabled: bool, phase: str):
    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", str(int(enabled)))
    profiler = SchedulerProfiler(torch.device("cpu"), SimpleNamespace())
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(profile_tag="session/turn-1")], size=1, profile_id=42,
    )
    with profiler.record_wait(phase):
        pass
    with profiler.record_process(phase, batch):
        with profiler.record_process_step(phase, "ar_process"):
            pass
        with profiler.record_process_step(phase, "send_result"):
            pass

    if not enabled:
        assert not profiler.batch_samples["process"][phase]
        return
    sample = profiler.batch_samples["process"][phase][0]
    assert sample["batch_id"] == 42
    assert sample["wait_ms"] >= 0
    assert sample["request_tags"] == ["session/turn-1"]
    assert sample["total_ms"] >= sample["ar_process_ms"]
    assert sample["total_ms"] >= sample["send_result_ms"]
    assert profiler.process_stage_ns[phase]["ar_process"] >= 0


def test_mixed_profile_full_lifecycle_and_counts_without_cuda(monkeypatch):
    from logssms.core import Batch, Request, ChunkedRequest

    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", "1")
    monkeypatch.setattr(
        "logssms.scheduler.profile.get_tp_info", lambda: SimpleNamespace(rank=0),
    )
    class Event:
        def __init__(self, **_kwargs):
            pass

        def record(self):
            pass

        def query(self):
            return True

        def elapsed_time(self, _end):
            return 2.0

    monkeypatch.setattr(torch.cuda, "Event", Event)
    cache = SimpleNamespace(poll_profile=lambda: None, profile_result=lambda: None)
    profiler = SchedulerProfiler(torch.device("cpu"), cache)
    chunk = ChunkedRequest(torch.arange(64), 0, 0, 8, 1, None, None, 128)
    decode = Request(torch.arange(4), 1, 3, 8, 2, None, None, 12)
    batch = Batch([chunk, decode], "mixed")
    batch.padded_reqs = batch.reqs
    with profiler.record_schedule() as schedule:
        schedule["phase"] = "mixed"
    with profiler.record_prepare(batch, graph_used=False):
        pass
    with profiler.record_submit(batch):
        pass
    with profiler.record_wait("mixed"):
        pass
    with profiler.record_process("mixed", batch):
        with profiler.record_process_step("mixed", "ar_process"):
            pass
    profiler.poll()
    result = profiler._make_result()
    assert result["ar_forward_count"] == 1
    assert result["pending_gpu_events"] == 0
    phase = result["phases"]["mixed"]
    assert phase["batches"] == 1
    assert phase["logical_rows"] == 2
    assert phase["logical_tokens"] == 65
    assert phase["prefill_count"] == phase["decode_count"] == 1
    assert phase["verify_count"] == 0
    assert phase["gpu_ms"] == 2.0
    for kind in ("prepare", "forward", "process"):
        sample = result[f"mixed_{kind}_batches"][0]
        assert sample["prefill_count"] == sample["decode_count"] == 1
        assert sample["verify_count"] == 0
    detail = result["mixed_prepare_batches"][0]["schedule_detail"]
    assert detail["prefill_count"] == detail["decode_count"] == 1
    assert "ar_process" in result["process_stages_ms"]["mixed"]


def test_schedule_profile_nested_exclusive_time_and_request_identity(monkeypatch):
    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", "1")
    profiler = SchedulerProfiler(torch.device("cpu"), SimpleNamespace())
    # Use a deterministic clock; timing itself must not require CUDA events.
    ticks = iter([0, 10, 20, 30, 50, 80, 90, 100])
    monkeypatch.setattr("logssms.scheduler.profile.perf_counter_ns", lambda: next(ticks))
    req = PendingRequest(
        7, torch.arange(16), SamplingParams(max_tokens=8),
        profile_tag="session/turn-2",
    )
    with profiler.record_schedule() as sample:
        with profiler.record_schedule_request(req) as request:
            with profiler.record_schedule_step("prefill_acquire"):
                with profiler.cache_manager._schedule_profile_step("kv_capacity"):
                    pass
            request["admitted"] = True
        sample["phase"] = "prefill"

    assert sample["total_ms"] == 100 / 1e6
    assert sample["stages_ms"] == {
        "kv_capacity": 20 / 1e6, "prefill_acquire": 60 / 1e6,
    }
    assert sample["exclusive_stages_ms"] == {
        "kv_capacity": 20 / 1e6, "prefill_acquire": 40 / 1e6,
    }
    assert sample["unaccounted_ms"] == pytest.approx(40 / 1e6)
    assert sample["stage_calls"] == {"kv_capacity": 1, "prefill_acquire": 1}
    assert request["tag"] == "session/turn-2"
    assert request["stages_ms"] == sample["stages_ms"]
    assert request["total_ms"] == 80 / 1e6
    assert profiler.cache_manager._schedule_profile_step is None
    assert profiler._schedule_stack == []
    assert profiler.schedule_no_batch == 0


def test_schedule_profile_keeps_rejected_attempts_without_batch(monkeypatch):
    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", "1")
    manager = SimpleNamespace()
    profiler = SchedulerProfiler(torch.device("cpu"), manager)
    adder = PrefillAdder(
        64, 0, manager, SimpleNamespace(available_size=0), profiler=profiler,
    )
    req = PendingRequest(
        7, torch.arange(16), SamplingParams(max_tokens=8),
        profile_tag="session/turn-2",
    )
    with profiler.record_schedule():
        with profiler.record_schedule_step("prefill_select"):
            assert adder.try_add_one(req) is None
    assert profiler.schedule_no_batch == 1
    sample = profiler.schedule_no_batch_samples[0]
    assert sample["requests"][0]["admitted"] is False
    assert sample["requests"][0]["rejection_reason"] == "no free request table slot"
    assert sample["requests"][0]["reserved_size_after"] == 0


def test_schedule_profile_disabled_does_not_read_clock_or_attach_recorder(monkeypatch):
    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", "0")
    manager = SimpleNamespace()
    profiler = SchedulerProfiler(torch.device("cpu"), manager)
    def unexpected_clock():
        raise AssertionError("Disabled profiling must not read the clock")
    monkeypatch.setattr("logssms.scheduler.profile.perf_counter_ns", unexpected_clock)
    with profiler.record_schedule() as sample:
        with profiler.record_schedule_step("prefill_select"):
            pass
    assert sample is None
    assert not hasattr(manager, "_schedule_profile_step")
    assert not profiler.schedule_stage_calls


def test_schedule_profile_restores_recorder_after_exception(monkeypatch):
    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", "1")
    previous = object()
    manager = SimpleNamespace(_schedule_profile_step=previous)
    profiler = SchedulerProfiler(torch.device("cpu"), manager)
    with pytest.raises(ValueError, match="test"):
        with profiler.record_schedule():
            with profiler.record_schedule_step("match_req"):
                raise ValueError("test")
    assert manager._schedule_profile_step is previous
    assert profiler._schedule_stack == []
    assert profiler._schedule_sample is None
    assert profiler.schedule_stage_calls["match_req"] == 1


def test_schedule_profile_exports_batch_details_and_bounded_failures(monkeypatch):
    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", "1")
    monkeypatch.setenv("LOGSSMS_PROFILE_MAX_BATCH_SAMPLES", "1")
    monkeypatch.setattr(
        "logssms.scheduler.profile.get_tp_info", lambda: SimpleNamespace(rank=0),
    )
    manager = SimpleNamespace(profile_result=lambda: None)
    profiler = SchedulerProfiler(torch.device("cpu"), manager)
    req = SimpleNamespace(profile_tag="session/turn-2", extend_len=16, cached_len=64)
    batch = SimpleNamespace(
        profile_id=42, phase="prefill", size=1, padded_size=1,
        reqs=[req], padded_reqs=[req],
        prefill_count=1, prefill_reqs=[req], decode_reqs=[], prefill_token_capacity=0,
    )
    with profiler.record_schedule() as sample:
        with profiler.record_schedule_step("prefill_select"):
            pass
        sample["phase"] = "prefill"
    profiler._record_batch(batch, 100, False)
    for _ in range(2):
        with profiler.record_schedule():
            with profiler.record_schedule_step("prefill_select"):
                pass
    result = profiler._make_result()
    details = result["prefill_prepare_batches"][0]["schedule_detail"]
    assert details["batch_id"] == 42
    assert details["request_tags"] == ["session/turn-2"]
    assert details["phase"] == "prefill"
    assert details["stage_calls"]["prefill_select"] == 1
    aggregate = result["schedule_profile"]
    assert aggregate["iterations"] == 3
    assert aggregate["no_batch_iterations"] == 2
    assert aggregate["stage_calls"]["prefill_select"] == 3
    assert len(aggregate["no_batch_samples"]) == 1
    assert aggregate["dropped_no_batch_samples"] == 1


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    ("capacities", "admitted", "expected_calls"),
    [
        ([0], False, ["capacity"]),
        ([128, 0], False, ["capacity", "lock", "capacity", "unlock"]),
        ([128, 128], True, ["capacity", "lock", "capacity"]),
    ],
)
def test_schedule_profile_preserves_admission_lock_order(
    monkeypatch, enabled, capacities, admitted, expected_calls,
):
    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", str(int(enabled)))
    calls = []
    values = iter(capacities)
    class Admission:
        _schedule_profile_step = None
        record_schedule_step = BaseCacheManager.record_schedule_step
        admission_capacity = BaseCacheManager.admission_capacity

        @property
        def available_size(self):
            calls.append("capacity")
            return next(values)

        def lock(self, handle):
            calls.append("lock")

        def unlock(self, handle):
            calls.append("unlock")

    manager = Admission()
    profiler = SchedulerProfiler(torch.device("cpu"), manager)
    with profiler.record_schedule():
        assert BaseCacheManager.prefill_acquire(manager, 64, object()) is admitted
    assert calls == expected_calls
    assert manager._schedule_profile_step is None
