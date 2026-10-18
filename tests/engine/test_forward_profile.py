from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from logssms.core import Batch, ChunkedRequest, Request, SamplingParams
from logssms.engine.engine import Engine
from logssms.engine.decode_graph import DecodeGraphCaptureBuffer, DecodeGraphRunner
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
    batch = Batch([req], phase)
    batch.padded_reqs = [req, req]
    batch.logstate_replay_counts = [3, 0]
    batch.attn_metadata = None
    return batch


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


def test_piecewise_prefill_profile_does_not_use_decode_runner(monkeypatch, profiler):
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "stream")
    engine = object.__new__(Engine)
    engine.profiler = profiler
    engine.stream = "stream"
    engine.config = SimpleNamespace(
        enable_specualtive_decoding=False, model_config=SimpleNamespace(is_hybrid=False),
    )
    runner = SimpleNamespace(can_use_cuda_graph=lambda batch: True, replays=0, padded_tokens=0,
                             token_bucket=lambda _: 1)
    @contextmanager
    def forward_batch(batch): yield
    engine.ctx = SimpleNamespace(forward_batch=forward_batch, piecewise_graph_runner=runner)
    engine.piecewise_graph_runner = runner
    engine.decode_graph_runner = SimpleNamespace(can_use_cuda_graph=lambda batch: False)
    def forward():
        runner.replays += 4
        runner.padded_tokens += 12
        return torch.ones(1, 4)
    engine.model = SimpleNamespace(forward=forward)
    engine.sampler = SimpleNamespace(sample=lambda logits, args: torch.tensor([2]))
    batch = _batch("prefill")
    batch.input_ids = torch.zeros(1, dtype=torch.int32)
    engine.forward_batch(batch, None)
    _Event.ready = True
    profiler.poll()
    sample = profiler.result("forward")["prefill_batches"][0]
    assert sample["graph_kind"] == "piecewise"
    assert sample["piece_replays"] == 4
    assert sample["piece_padding"] == 12


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
    engine.piecewise_graph_runner = None
    engine.decode_graph_runner = SimpleNamespace(can_use_cuda_graph=lambda batch: False)
    result = engine.forward_batch(_batch(phase), None)
    assert result.next_tokens_gpu.dtype == torch.int32
    assert result.next_tokens_cpu.tolist() == ([[3, 2]] if phase == "verify" else [3])
    expected = ("reject", torch.Size([2, 4])) if phase == "verify" else ("sample", torch.Size([1, 4]))
    assert calls == [expected]


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_mixed_forward_profile_preserves_sampling_and_metadata(monkeypatch, chunked, enabled):
    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", str(int(enabled)))
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "stream")
    _Event.ready = True
    prefill = (ChunkedRequest if chunked else Request)(
        torch.arange(96), 0, 64, 8, 0, SamplingParams(), None, 128,
        profile_tag="prefill",
    )
    decodes = [Request(torch.arange(65), i, 64, 8, i, SamplingParams(), None, 128,
                       profile_tag=f"decode-{i}") for i in (1, 2)]
    batch = Batch([prefill, *decodes], "mixed", num_prefill_reqs=1)
    batch.attn_metadata = None
    batch.padded_reqs = batch.reqs
    batch.sample_req_indices = torch.tensor([1, 2] if chunked else [0, 1, 2])
    engine = object.__new__(Engine)
    engine.profiler = EngineProfiler()
    engine.stream = "stream"
    engine.config = SimpleNamespace(
        enable_specualtive_decoding=False, model_config=SimpleNamespace(is_hybrid=False),
    )
    @contextmanager
    def forward_batch(_batch):
        yield
    engine.ctx = SimpleNamespace(forward_batch=forward_batch)
    logits = torch.arange(12).reshape(3, 4)[int(chunked):]
    engine.model = SimpleNamespace(forward=lambda: logits)
    engine.decode_graph_runner = SimpleNamespace(can_use_cuda_graph=lambda _batch: False)
    engine.piecewise_graph_runner = SimpleNamespace(can_use_cuda_graph=lambda _: True, replays=0,
                                                   padded_tokens=0, token_bucket=lambda _: 64)
    batch.input_ids = torch.zeros(34, dtype=torch.int32)
    sampled = []
    def sample(values, _args):
        sampled.append(values.clone())
        return values[:, 0]
    engine.sampler = SimpleNamespace(sample=sample)
    output = engine.forward_batch(batch, None)
    torch.testing.assert_close(sampled[0], logits)
    assert output.next_tokens_cpu.tolist() == ([-1, 4, 8] if chunked else [0, 4, 8])
    engine.profiler.poll()
    profile = engine.forward_profile_result()
    if not enabled:
        assert profile is None
        return
    record = profile["mixed_batches"][0]
    assert record["graph_kind"] == "piecewise" and record["graph_used"]
    assert record["batch_size"] == 3 and record["padded_size"] == 3
    assert record["prefill_requests"] == 1 and record["decode_requests"] == 2
    assert record["prefill_tokens"] == 32
    assert record["input_tokens"] == 34
    assert record["request_tags"] == ["prefill", "decode-1", "decode-2"]
    assert record["gpu_stages_ms"]["model"] > 0
    assert profile["phases"]["mixed"]["by_padded_size"]["3"]["logical_rows"] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph timing test")
def test_graph_replay_timings_use_fresh_events_and_do_not_change_output(monkeypatch):
    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", "1")
    device = torch.device("cuda")
    runner = object.__new__(DecodeGraphRunner)
    runner.max_graph_bs = 2
    runner.buffer = DecodeGraphCaptureBuffer(2, 4, device)
    runner.attn_backend = SimpleNamespace(prepare_for_replay=lambda batch: None)
    runner.linear_attn_backend = None
    graph = torch.cuda.CUDAGraph()
    # Warm up before capturing. Profile events intentionally do not enter capture.
    runner.buffer.logits.copy_(runner.buffer.input_ids[:, None].expand(2, 4))
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        runner.buffer.logits.copy_(runner.buffer.input_ids[:, None].expand(2, 4))
    runner.graphs = {2: graph}
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
