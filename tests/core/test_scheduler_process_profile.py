from types import SimpleNamespace

import pytest
import torch

from logssms.scheduler.profile import SchedulerProfiler


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("phase", ["prefill", "decode", "verify"])
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
