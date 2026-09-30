from types import SimpleNamespace

import pytest
import torch

from logssms.scheduler.profile import SchedulerProfiler


@pytest.mark.parametrize("enabled", [False, True])
def test_prefill_process_profile_breakdown(monkeypatch, enabled: bool):
    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", str(int(enabled)))
    profiler = SchedulerProfiler(torch.device("cpu"), SimpleNamespace())
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(profile_tag="session/turn-1")], size=1,
    )
    with profiler.record_process("prefill", batch):
        with profiler.record_process_step("prefill", "ar_process"):
            pass
        with profiler.record_process_step("prefill", "send_result"):
            pass

    if not enabled:
        assert profiler.prefill_process_batches == []
        return
    sample = profiler.prefill_process_batches[0]
    assert sample["request_tags"] == ["session/turn-1"]
    assert sample["total_ms"] >= sample["ar_process_ms"]
    assert sample["total_ms"] >= sample["send_result_ms"]
    assert profiler.process_stage_ns["prefill"]["ar_process"] >= 0
