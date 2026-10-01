from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from logssms.engine.engine import Engine
from logssms.engine.graph import GraphCaptureBuffer, GraphRunner
from logssms.engine.profile import EngineProfiler


class _Event:
    clock = 0
    ready = False

    def __init__(self, **kwargs):
        self.time = None

    def record(self, *args):
        type(self).clock += 1
        self.time = type(self).clock

    def query(self): return type(self).ready
    def elapsed_time(self, other): return float(other.time - self.time)


def _batch(phase="decode"):
    req = SimpleNamespace(extend_len=1, cached_len=64, profile_tag="session/turn-2")
    return SimpleNamespace(
        reqs=[req], padded_reqs=[req, req], size=1, padded_size=2,
        phase=phase, is_verify=phase == "verify", is_decode=phase == "decode",
        logstate_replay_counts=[3, 0],
    )


@pytest.fixture
def profiler(monkeypatch):
    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", "1")
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    _Event.ready = False
    return EngineProfiler()


def test_async_timings_preserve_batch_identity_and_overlap(profiler):
    batches = [_batch(), _batch()]
    for batch in batches:
        with profiler.record_batch(batch, "forward", graph_used=True) as sample:
            sample["effective_replay_counts"] = [3]
            with profiler.record_stage("graph_replay"):
                pass
        batch.reqs[0].cached_len += 1
    profiler.poll()
    assert len(profiler.pending) == 2
    samples = profiler.result("forward")["decode_batches"]
    assert samples[0]["batch_id"] != samples[1]["batch_id"]
    assert samples[0]["cached_lens"] == [64]
    assert samples[0]["gpu_total_ms"] is None
    _Event.ready = True
    profiler.poll()
    result = profiler.result("forward")
    assert not profiler.pending
    assert all(sample["gpu_stages_ms"]["graph_replay"] > 0 for sample in samples)
    assert samples[1]["gpu_gap_before_ms"] > 0
    assert samples[1]["previous_forward"] == {"batch_id": batches[0].profile_id, "phase": "decode"}
    assert result["phases"]["decode"]["replay_count_rows"] == {3: 2}
    assert result["phases"]["decode"]["by_padded_size"]["2"]["batches"] == 2


def test_retained_samples_bounded_but_totals_include_all(monkeypatch):
    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", "1")
    monkeypatch.setenv("LOGSSMS_PROFILE_MAX_BATCH_SAMPLES", "2")
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    profiler = EngineProfiler()
    _Event.ready = True
    for _ in range(5):
        with profiler.record_batch(_batch(), "forward", graph_used=True):
            pass
        profiler.poll()
    result = profiler.result("forward")
    assert len(result["decode_batches"]) == 2
    assert result["phases"]["decode"]["batches"] == 5
    assert result["phases"]["decode"]["gpu_batches"] == 5
    assert result["phases"]["decode"]["dropped_samples"] == 3


@pytest.mark.parametrize("phase", ["prefill", "decode", "verify"])
@pytest.mark.parametrize("enabled", [False, True])
def test_forward_profile_does_not_change_sampling(monkeypatch, phase, enabled):
    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", str(int(enabled)))
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "stream")
    engine = object.__new__(Engine)
    engine.profiler = EngineProfiler()
    engine.stream = "stream"
    engine.config = SimpleNamespace(
        enable_specualtive_decoding=False, model_config=SimpleNamespace(is_hybrid=False),
    )
    @contextmanager
    def forward_batch(batch): yield
    engine.ctx = SimpleNamespace(forward_batch=forward_batch)
    logits = torch.arange(8).reshape(2, 4)
    engine.model = SimpleNamespace(forward=lambda: logits)
    calls = []
    def sample(logits, args):
        calls.append(("sample", logits.shape))
        return torch.tensor([3])
    def reject(logits, batch):
        calls.append(("reject", logits.shape))
        return torch.tensor([[3, 2]])
    engine.sampler = SimpleNamespace(sample=sample, reject_sample=reject)
    engine.graph_runner = SimpleNamespace(can_use_cuda_graph=lambda batch: False)
    result = engine.forward_batch(_batch(phase), None)
    assert result.next_tokens_gpu.dtype == torch.int32
    assert result.next_tokens_cpu.tolist() == ([[3, 2]] if phase == "verify" else [3])
    expected = ("reject", torch.Size([2, 4])) if phase == "verify" else ("sample", torch.Size([1, 4]))
    assert calls == [expected]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph timing test")
def test_graph_replay_timings_use_fresh_events_and_do_not_change_output(monkeypatch):
    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", "1")
    device = torch.device("cuda")
    runner = object.__new__(GraphRunner)
    runner.max_graph_bs = 2
    runner.buffer = GraphCaptureBuffer(2, 4, device)
    runner.attn_backend = SimpleNamespace(prepare_for_replay=lambda batch: None)
    runner.linear_attn_backend = None
    graph = torch.cuda.CUDAGraph()
    # Warm up before capturing. Profile events intentionally do not enter capture.
    runner.buffer.logits.copy_(runner.buffer.input_ids[:, None].expand(2, 4))
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        runner.buffer.logits.copy_(runner.buffer.input_ids[:, None].expand(2, 4))
    runner.graph_map = {2: graph}
    profiler = EngineProfiler()
    outputs = []
    for value in (7, 11):
        batch = _batch()
        batch.input_ids = torch.full((2,), value, dtype=torch.int32, device=device)
        batch.out_loc = torch.zeros(2, dtype=torch.int32, device=device)
        batch.positions = torch.zeros(2, dtype=torch.int32, device=device)
        with profiler.record_batch(batch, "forward", graph_used=True):
            outputs.append(runner.replay(batch, profiler=profiler).clone())
    # Synchronize only for final test assertions, not during profiling/replay.
    torch.cuda.synchronize()
    profiler.poll()
    assert outputs[0].tolist() == [[7.0] * 4]
    assert outputs[1].tolist() == [[11.0] * 4]
    result = profiler.result("forward")
    assert result["phases"]["decode"]["gpu_batches"] == 2
    for sample in result["decode_batches"]:
        stages = sample["gpu_stages_ms"]
        assert stages["graph_replay"] >= 0
        assert sample["gpu_total_ms"] >= sum(stages.values())
