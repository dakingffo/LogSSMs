from types import SimpleNamespace

import pytest
import torch

from logssms.core import Batch, ChunkedRequest, Request, SamplingParams
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
    batch = Batch(
        [SimpleNamespace(extend_len=3, cached_len=0, profile_tag="test")], "prefill",
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


@pytest.mark.parametrize("phase", ["prefill", "decode", "verify", "mixed"])
def test_prepare_profiles_all_phases(monkeypatch, phase):
    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", "1")
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    profiler = EngineProfiler()
    reqs = [SimpleNamespace(extend_len=1, cached_len=64, profile_tag="session/turn")]
    batch = Batch(reqs * (2 if phase == "mixed" else 1), phase)
    with profiler.record_batch(batch, "prepare", graph_used=phase == "decode"):
        with profiler.record_stage("allocate"):
            pass
    profiler.poll()
    result = profiler.result("prepare")
    assert result[phase + "_batches"][0]["cached_lens"] == [64] * batch.size
    assert result["phases"][phase]["gpu_batches"] == 1


@pytest.mark.parametrize("chunked", [False, True])
def test_mixed_prepare_profile_records_real_rows_capacity_and_compact_writes(monkeypatch, chunked):
    from logssms.utils.batch import BatchIndices

    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", "1")
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    engine = object.__new__(Engine)
    engine.profiler = EngineProfiler()
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(
        mixed_prefill_tokens=128, model_config=SimpleNamespace(is_hybrid=False),
    )
    from logssms.engine.mixed_graph import MixedGraphRunner

    engine.mixed_graph_runner = object.__new__(MixedGraphRunner)
    engine.mixed_graph_runner.graphs = {4: object()}
    engine.dummy_req = Request(torch.arange(1), 3, 0, 8, 3, SamplingParams(), None, 16)
    prefill = (ChunkedRequest if chunked else Request)(
        torch.arange(96), 0, 64, 8, 0, SamplingParams(), None, 128,
    )
    decodes = [Request(torch.arange(65), i, 64, 8, i, SamplingParams(), None, 128)
               for i in (1, 2)]
    batch = Batch([prefill, *decodes], "mixed")
    metadata = []
    engine.attn_backend = SimpleNamespace(
        prepare_metadata=lambda part: metadata.append(part.phase),
        prepare_mixed_metadata=lambda parts: metadata.append(tuple(p.phase for p in parts)),
    )
    params_seen = []
    engine.sampler = SimpleNamespace(prepare_params=lambda params: params_seen.extend(params))
    def indices(part):
        packet = BatchIndices.make(part, engine.device, 64)
        part.positions = torch.tensor([position for req in part.padded_reqs
                                       for position in range(req.cached_len, req.device_len)])
        part.out_loc = torch.arange(packet.num_tokens)
        rows = torch.tensor([req.table_idx for req in part.padded_reqs
                             for _ in range(req.extend_len)])
        return (rows, part.positions), packet.write_tuple
    engine._prepare_indices = indices
    allocations, preparations = [], []
    cache = SimpleNamespace(
        allocate=lambda reqs, *, phase: allocations.append((phase, len(reqs))),
        prepare_batch=lambda part: preparations.append(part.phase),
    )
    result = engine.prepare_batch(batch, cache)
    engine.profiler.poll()
    assert allocations == [("prefill", 1), ("decode", 2)]
    assert preparations == ["decode", "prefill"]
    assert metadata == ["prefill", "decode", ("prefill", "decode")]
    assert len(params_seen) == 3 - int(chunked)
    assert result.write_tuple[0].tolist() == ([1, 2] if chunked else [0, 1, 2])
    assert batch.sample_indices.tolist() == [31, 128, 129, 130]
    sample = engine.prepare_profile_result()["mixed_batches"][0]
    assert sample["batch_size"] == 3 and sample["padded_size"] == 4
    assert sample["prefill_requests"] == 1 and sample["decode_requests"] == 2
    assert sample["prefill_tokens"] == 32 and sample["prefill_capacity"] == 128
    assert sample["logical_tokens"] == 34
    assert sample["graph_used"] and sample["gpu_total_ms"] is not None
    assert sample["batch_id"] == batch.profile_id
    assert "allocate" in sample["cpu_stages_ms"]
