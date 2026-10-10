from types import SimpleNamespace

import pytest
import torch

from logssms.engine.engine import Engine
from logssms.engine.profile import EngineProfiler


class _Event:
    def __init__(self, **kwargs): pass
    def record(self): pass
    def query(self): return True
    def elapsed_time(self, other): return 0.01


@pytest.mark.parametrize("enabled", [False, True])
def test_engine_prepare_profile(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", str(int(enabled)))
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    engine = object.__new__(Engine)
    engine.ctx = SimpleNamespace(prefill_graph_runner=None)
    engine.prefill_graph_runner = None
    engine.profiler = EngineProfiler()
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(model_config=SimpleNamespace(is_hybrid=False))
    engine.decode_graph_runner = SimpleNamespace(
        pad_batch=lambda batch: None, can_use_cuda_graph=lambda batch: False,
    )
    engine.attn_backend = SimpleNamespace(prepare_metadata=lambda batch: None)
    engine.sampler = SimpleNamespace(prepare=lambda batch: "sample args")
    engine.page_table = torch.zeros((1, 1), dtype=torch.int32)
    mapping = (torch.tensor([0]), torch.tensor([0]))
    monkeypatch.setattr(engine, "_prepare_indices", lambda batch: (mapping, mapping))
    allocation_phases = []
    cache_manager = SimpleNamespace(
        allocate=lambda reqs, *, phase=None: allocation_phases.append(phase),
        prepare_batch=lambda batch: None,
    )
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(extend_len=3, cached_len=0, profile_tag="test")],
        phase="prefill",
        is_prefill=True,
        size=1,
    )

    result = engine.prepare_batch(batch, cache_manager)
    engine.profiler.poll()

    assert result.sample_args == "sample args"
    assert allocation_phases == ["prefill"]
    profile = engine.prepare_profile_result()
    if not enabled:
        assert profile is None
        return
    assert profile is not None
    sample = profile["prefill_batches"][0]
    assert sample["logical_tokens"] == 3
    assert profile["phases"]["prefill"]["batches"] == 1
    stages = (
        "pad_batch_ms", "allocate_ms", "positions_mappings_ms",
        "cache_prepare_ms", "attention_metadata_ms", "linear_metadata_ms",
        "sampler_prepare_ms",
    )
    assert all(sample[stage] >= 0 for stage in stages)
    assert sample["total_ms"] >= sum(sample[stage] for stage in stages)
    assert sample["gpu_total_ms"] is not None
    assert sample["batch_id"] == batch.profile_id


@pytest.mark.parametrize("phase", ["prefill", "decode", "verify"])
def test_prepare_profiles_all_phases(monkeypatch, phase):
    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", "1")
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    profiler = EngineProfiler()
    batch = SimpleNamespace(
        phase=phase, size=1,
        reqs=[SimpleNamespace(extend_len=1, cached_len=64, profile_tag="session/turn")],
    )
    with profiler.record_batch(batch, "prepare", graph_used=phase == "decode"):
        with profiler.record_stage("allocate"):
            pass
    profiler.poll()
    result = profiler.result("prepare")
    assert result[phase + "_batches"][0]["cached_lens"] == [64]
    assert result["phases"][phase]["gpu_batches"] == 1
