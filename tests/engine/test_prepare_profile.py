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
    engine.ctx = SimpleNamespace(piecewise_graph_runner=None)
    engine.piecewise_graph_runner = None
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
    batch = Batch(reqs * (2 if phase == "mixed" else 1), phase, num_prefill_reqs=int(phase == "mixed"))
    with profiler.record_batch(batch, "prepare", graph_used=phase == "decode"):
        with profiler.record_stage("allocate"):
            pass
    profiler.poll()
    result = profiler.result("prepare")
    assert result[phase + "_batches"][0]["cached_lens"] == [64] * batch.size
    assert result["phases"][phase]["gpu_batches"] == 1


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("lengths", [[32], [16, 32, 64]])
def test_mixed_prepare_profile_records_real_rows_and_compact_writes(monkeypatch, chunked, lengths):
    from logssms.utils.batch import BatchIndices

    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", "1")
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    engine = object.__new__(Engine)
    engine.profiler = EngineProfiler()
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(
        mixed_prefill_tokens=128, model_config=SimpleNamespace(is_hybrid=False),
    )
    engine.piecewise_graph_runner = SimpleNamespace(can_use_cuda_graph=lambda _: True)
    prefills = [(ChunkedRequest if chunked and i % 2 == 0 else Request)(
        torch.arange(64 + length), i, 64, 8, i, SamplingParams(), None, 192,
    ) for i, length in enumerate(lengths)]
    decodes = [Request(torch.arange(65), i, 64, 8, i, SamplingParams(), None, 128)
               for i in range(len(prefills), len(prefills) + 2)]
    batch = Batch(prefills + decodes, "mixed", num_prefill_reqs=len(prefills))
    metadata = []
    engine.attn_backend = SimpleNamespace(
        uses_paged_indices=False,
        prepare_metadata=lambda part: metadata.append(part.phase),
        prepare_mixed_metadata=lambda batch: metadata.append(tuple(p.phase for p in batch.mixed_parts)),
    )
    params_seen = []
    engine.sampler = SimpleNamespace(prepare_params=lambda params: params_seen.extend(params))
    engine.ctx = SimpleNamespace(kv_cache=SimpleNamespace(kv_page_size=64))
    engine.page_table = torch.zeros(len(batch.reqs), 192, dtype=torch.int32)
    def mapping(packet, table, *, decode=False):
        positions = torch.tensor([position for req in batch.reqs
                                  for position in range(req.cached_len, req.device_len)])
        rows = torch.tensor([req.table_idx for req in batch.reqs for _ in range(req.extend_len)])
        return positions, (rows, positions), torch.arange(packet.num_tokens)
    monkeypatch.setattr("logssms.kernel.metadata.make_batch_mapping", mapping)
    allocations, preparations = [], []
    cache = SimpleNamespace(
        allocate=lambda reqs, *, phase: allocations.append((phase, len(reqs))),
        prepare_batch=lambda part: preparations.append(part.phase),
    )
    result = engine.prepare_batch(batch, cache)
    engine.profiler.poll()
    assert allocations == [("prefill", len(prefills)), ("decode", 2)]
    assert preparations == ["decode", "prefill"]
    assert metadata == [("prefill", "decode")]
    sampled = [i for i, req in enumerate(batch.reqs) if not isinstance(req, ChunkedRequest)]
    ends = torch.tensor([req.extend_len for req in batch.reqs]).cumsum(0) - 1
    assert len(params_seen) == len(sampled)
    assert result.write_tuple[0].tolist() == sampled
    assert batch.sample_req_indices.tolist() == sampled
    assert batch.sample_indices.tolist() == ends[sampled].tolist()
    logits = torch.arange(len(sampled) * 4).view(len(sampled), 4)
    engine.sampler.sample = lambda values, args: values[:, 0]
    tokens = engine._sample_batch(logits, batch, None)
    assert tokens[sampled].tolist() == logits[:, 0].tolist()
    assert all(tokens[i] == -1 for i in range(batch.size) if i not in sampled)
    sample = engine.prepare_profile_result()["mixed_batches"][0]
    assert sample["batch_size"] == batch.size and sample["padded_size"] == batch.size
    assert sample["prefill_requests"] == len(prefills) and sample["decode_requests"] == 2
    assert sample["prefill_tokens"] == sum(lengths)
    assert sample["logical_tokens"] == sum(lengths) + 2
    assert sample["graph_used"] and sample["gpu_total_ms"] is not None
    assert sample["batch_id"] == batch.profile_id
    assert "allocate" in sample["cpu_stages_ms"]
