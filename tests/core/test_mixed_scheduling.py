from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from logssms.core import Batch, ChunkedRequest, Request, SamplingParams
from logssms.distributed import DistributedInfo
from logssms.engine import ForwardInput, ForwardOutput
from logssms.engine.config import EngineConfig
from logssms.scheduler.config import SchedulerConfig
from logssms.scheduler.decode import DecodeManager
from logssms.scheduler.prefill import PendingRequest, PrefillAdder, PrefillManager
from logssms.scheduler.scheduler import Scheduler
from logssms.server.args import parse_args


def request(uid, *, chunked=False):
    return (ChunkedRequest if chunked else Request)(
        input_ids=torch.tensor([1, 2], dtype=torch.int32),
        table_idx=uid, cached_len=1, output_len=20, uid=uid,
        sampling_params=SamplingParams(ignore_eos=True, max_tokens=20),
        cache_handle=None, max_device_len=22,
    )


def manager(count=0, budget=16):
    cache = SimpleNamespace(
        lazy_free_region=nullcontext,
        cache_req=lambda *_args, **_kwargs: None,
        reservation_size=lambda n, **_kwargs: n,
    )
    result = DecodeManager(
        SimpleNamespace(page_size=1, decode_batch_budget=budget),
        torch.device("cpu"), cache,
        SimpleNamespace(token_pool=torch.zeros((32, 64), dtype=torch.int32)), 0,
    )
    result.running_reqs = {i: request(i) for i in range(1, count + 1)}
    return result


def scheduler(monkeypatch, count=3, max_running=8, budget=16, policy="mixed", admitted=True):
    s = object.__new__(Scheduler)
    s.config = SimpleNamespace(
        scheduling_policy=policy, mixed_prefill_tokens=128, max_running_req=max_running,
    )
    s.ar_manager = manager(count, budget)
    s.cache_manager = s.ar_manager.cache_manager
    s.prefill_manager = PrefillManager(s.ar_manager.token_pool, 1, s.cache_manager)
    s.prefill_manager.pending_list = [
        PendingRequest(i, torch.arange(8), SamplingParams(max_tokens=20))
        for i in (24, 25, 26)
    ]
    s.prefill_budget = 1024
    s.table_manager = SimpleNamespace()
    s.profiler = SimpleNamespace(
        enabled=False, record_schedule=nullcontext,
        record_schedule_step=lambda *_args: nullcontext(),
        record_prepare=lambda *_args, **_kwargs: nullcontext(),
        record_ar_no_batch=lambda: None,
    )
    s.engine = SimpleNamespace(
        prepare_batch=lambda batch, _cache: batch,
        decode_graph_runner=SimpleNamespace(can_use_cuda_graph=lambda _batch: False),
        piecewise_graph_runner=SimpleNamespace(can_use_cuda_graph=lambda _batch: False),
    )
    s._report_stall = lambda _adder: None
    attempts = []

    def add(adder, pending):
        adder._select_budget(pending.input_len)
        attempts.append((pending.uid, adder.token_budget, adder.reserved_size))
        return request(pending.uid) if admitted else None

    monkeypatch.setattr("logssms.scheduler.prefill.PrefillAdder.try_add_one", add)
    return s, attempts


@pytest.mark.parametrize("count,limit,budget,size", [
    (1, 8, 16, 4), (2, 8, 16, 5), (3, 8, 16, 6),
    (8, 6, 16, 6), (15, 16, 16, 16), (8, 8, 2, 5),
    (20, 32, 32, 23), (20, 24, 32, 23),
])
def test_mixed_request_capacity_and_only_selected_prefill_analyzed(monkeypatch, count, limit, budget, size):
    s, attempts = scheduler(monkeypatch, count, limit, budget)
    s.ar_manager.inflight_uids[1] = {99}
    batch = s._schedule_next_batch()
    assert batch.phase == "mixed"
    assert batch.size == size
    selected = min(3, limit - min(limit - 1, count, budget))
    assert [req.uid for req in batch.prefill_reqs] == [24, 25, 26][:selected]
    assert [p.uid for p in s.prefill_manager.pending_list] == [24, 25, 26][selected:]
    assert len(attempts) == selected and attempts[0][1] == 128
    assert s.ar_manager.inflight_uids[0] == {99}  # Exactly one window rotation.
    assert s.ar_manager.inflight_uids[1] == {r.uid for r in batch.decode_reqs}


@pytest.mark.parametrize("policy,count", [("mixed", 0), ("prefill-first", 3)])
def test_no_decode_or_legacy_policy_uses_full_prefill_budget(monkeypatch, policy, count):
    s, attempts = scheduler(monkeypatch, count=count, policy=policy)
    batch = s._schedule_next_batch()
    assert batch.phase == "prefill" and batch.size == 3
    assert all(attempt[1] == 1024 for attempt in attempts)


def test_unavailable_prefill_falls_back_to_decode(monkeypatch):
    s, attempts = scheduler(monkeypatch, admitted=False)
    batch = s._schedule_next_batch()
    assert batch.phase == "decode" and batch.size == 3
    assert len(attempts) == 1
    assert len(s.prefill_manager.pending_list) == 3


def test_multiple_prefills_share_one_token_budget(monkeypatch):
    s, _ = scheduler(monkeypatch, count=2)
    s.prefill_manager.pending_list.append(
        PendingRequest(27, torch.arange(8), SamplingParams(max_tokens=20))
    )
    lengths = {24: 36, 25: 52, 26: 40, 27: 8}
    seen = []
    def add(adder, pending):
        adder._select_budget(lengths[pending.uid])
        seen.append((pending.uid, adder.token_budget))
        length = lengths[pending.uid]
        if length > adder.token_budget:
            return None
        adder.token_budget -= length
        return Request(torch.zeros(length, dtype=torch.int32), pending.uid, 0, 20,
                       pending.uid, pending.sampling_params, None, length + 20)
    monkeypatch.setattr("logssms.scheduler.prefill.PrefillAdder.try_add_one", add)
    batch = s._schedule_next_batch()
    assert batch.prefill_count == 3 and batch.prefill_tokens == 128
    assert len(batch.decode_reqs) == 2 and batch.size == 5
    assert seen == [(24, 128), (25, 92), (26, 40), (27, 0)]
    assert [p.uid for p in s.prefill_manager.pending_list] == [27]


@pytest.mark.parametrize("head_len,tail_len,cached_len,chunked,phase,lengths,pending", [
    (100, 30, 0, False, "mixed", [100, 28], [25]),
    (100, 8000, 0, False, "mixed", [100, 28], [25]),
    (128, 30, 0, False, "mixed", [128], [25]),
    (129, 30, 0, False, "prefill", [129, 30], []),
    (8000, 30, 0, False, "prefill", [8000, 30], []),
    (9000, 30, 0, False, "prefill", [8192], [24, 25]),
    (8000, 30, 7900, False, "mixed", [100, 28], [25]),
    (8000, 30, 7800, False, "prefill", [200, 30], []),
    (8000, 30, 7900, True, "mixed", [100, 28], [25]),
    (8000, 30, 7800, True, "prefill", [200, 30], []),
])
def test_head_remaining_tokens_choose_prefill_or_mixed(
    monkeypatch, head_len, tail_len, cached_len, chunked, phase, lengths, pending,
):
    try_add_one = PrefillAdder.try_add_one
    decode_count = 3
    s, _ = scheduler(monkeypatch, count=decode_count)
    monkeypatch.setattr(PrefillAdder, "try_add_one", try_add_one)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    s.prefill_budget = 8192
    s.prefill_manager.pending_list = [
        PendingRequest(24, torch.arange(head_len, dtype=torch.int32), SamplingParams(max_tokens=20)),
        PendingRequest(25, torch.arange(tail_len, dtype=torch.int32), SamplingParams(max_tokens=20)),
    ]
    if chunked:
        head = s.prefill_manager.pending_list[0]
        head.chunked_req = ChunkedRequest(
            head.input_ids[:cached_len], 24, cached_len - 64, 20, 24,
            head.sampling_params, None, head_len + 20,
        )
        head.chunked_req.cached_len = cached_len
    matched = []
    def match(req):
        matched.append(req.uid)
        return SimpleNamespace(cuda_handle=SimpleNamespace(
            cached_len=cached_len if req.uid == 24 else 0,
        ))
    cache = s.cache_manager
    cache.match_req = match
    cache.plan_prefill_chunk = min
    cache.prefill_acquire = lambda *_args, **_kwargs: True
    cache.record_match = lambda *_args, **_kwargs: None
    cache.record_prefill_wait = lambda *_args: None
    cache.init_with_handle = lambda *_args: None
    indices = iter([25] if chunked else [24, 25])
    s.table_manager = SimpleNamespace(
        available_size=2,
        allocate=lambda: next(indices),
        token_pool=torch.zeros((32, 10000), dtype=torch.int32),
    )

    batch = s._schedule_next_batch()
    assert batch.phase == phase
    assert [r.extend_len for r in batch.prefill_reqs] == lengths
    assert batch.prefill_tokens == sum(lengths)
    assert [req.uid for req in s.prefill_manager.pending_list] == pending
    assert len(batch.decode_reqs) == (decode_count if phase == "mixed" else 0)
    assert matched == ([24] if len(lengths) == 1 else [24, 25])[int(chunked):]
    assert set(s.ar_manager.running_reqs) == set(range(1, decode_count + 1))


@pytest.mark.parametrize("eligible", [False, True])
def test_mixed_prepare_profile_uses_runner_graph_eligibility(monkeypatch, eligible):
    s, _ = scheduler(monkeypatch)
    checked, recorded = [], []
    def can_use(batch):
        checked.append(batch)
        return eligible
    s.engine.piecewise_graph_runner.can_use_cuda_graph = can_use
    def record(batch, *, graph_used):
        recorded.append((batch, graph_used))
        return nullcontext()
    s.profiler.record_prepare = record
    batch = s._schedule_next_batch()
    assert batch.is_mixed
    assert checked == [batch]
    assert recorded == [(batch, eligible)]


@pytest.mark.parametrize("policy", ["prefill-first", "mixed"])
@pytest.mark.parametrize("pending", [False, True])
def test_reservation_scans_only_when_prefill_pending(monkeypatch, policy, pending):
    s, attempts = scheduler(monkeypatch, policy=policy)
    if not pending:
        s.prefill_manager.pending_list.clear()
    calls = []
    def reserve(name, value):
        calls.append(name)
        return value
    monkeypatch.setattr(DecodeManager, "need_capacity", property(lambda _: reserve("ar", 320)))
    monkeypatch.setattr(PrefillManager, "need_capacity", property(lambda _: reserve("prefill", 128)))
    batch = s._schedule_next_batch()
    if pending:
        assert calls == ["ar", "prefill"]
        assert attempts[0][2] == 448
        assert batch.phase == ("mixed" if policy == "mixed" else "prefill")
    else:
        assert calls == [] and attempts == []
        assert batch.phase == "decode"


def test_mixed_policy_decode_fallback_rotates(monkeypatch):
    s, _ = scheduler(monkeypatch, count=5, budget=1, admitted=False)
    seen = []
    for _ in range(5):
        seen.append(s._schedule_next_batch().reqs[0].uid)
        s.ar_manager.on_prefill_scheduled()
    assert seen == [1, 2, 3, 4, 5]


def test_mixed_chunk_continuation_keeps_pending_order_and_abort_frontier():
    p = PrefillManager(torch.zeros((32, 64), dtype=torch.int32), 1)
    p.pending_list = [
        PendingRequest(i, torch.arange(12), SamplingParams(max_tokens=20))
        for i in (0, 1)
    ]
    calls = []

    def add(pending):
        calls.append(pending.uid)
        cached = pending.chunked_req.cached_len if pending.chunked_req else 0
        return ChunkedRequest(
            pending.input_ids[:cached + 2], pending.uid, cached, 20, pending.uid,
            pending.sampling_params, None, 32,
        )

    first = p.schedule_next_batch(SimpleNamespace(try_add_one=add), max_requests=1)
    batch = Batch(first.reqs + [request(2)], "mixed", num_prefill_reqs=1)
    mapping = (torch.tensor([2]), torch.tensor([2]))
    p.advance_for_next_schedule(
        None, (batch, None, None, mapping),
        SimpleNamespace(next_tokens_gpu=torch.tensor([-1, 9], dtype=torch.int32)),
    )
    second = p.schedule_next_batch(SimpleNamespace(try_add_one=add), max_requests=1)
    assert calls == [0, 0]
    assert [req.uid for req in p.pending_list] == [0, 1]
    assert second.reqs[0].cached_len == 2
    assert second.reqs[0].device_len == 4
    aborted, inflight = p.abort_req(0)
    assert aborted is second.reqs[0] and inflight and aborted.aborted
    assert [req.uid for req in p.pending_list] == [1]


def test_bounded_decode_selection_rotates_even_after_inflight_window_clears():
    m = manager(5)
    seen = []
    for _ in range(10):
        batch = m.schedule_next_batch(max_requests=1)
        seen.append(batch.reqs[0].uid)
        m.on_prefill_scheduled()
    assert seen == [1, 2, 3, 4, 5] * 2
    m.running_reqs[2].aborted = True
    assert m.candidate_count == 4
    assert all(r.uid != 2 for r in m.schedule_next_batch(max_requests=5).reqs)


def test_unbounded_decode_retains_inflight_priority_and_insertion_order():
    m = manager(5, budget=3)
    m.inflight_uids[1] = {1, 3}
    original_order = list(m.running_reqs)
    assert [r.uid for r in m.schedule_next_batch().reqs] == [2, 4, 5]
    assert list(m.running_reqs) == original_order


@pytest.mark.parametrize("chunked", [False, True])
def test_mixed_advance_and_process_use_per_request_phase(chunked):
    m = manager(1)
    prefill, decode = request(0, chunked=chunked), m.running_reqs[1]
    batch = Batch([prefill, decode], "mixed", num_prefill_reqs=1)
    rows = [1] if chunked else [0, 1]
    mapping = (torch.tensor(rows), torch.tensor([2] * len(rows)))
    inp = ForwardInput(batch, None, (torch.empty(0), torch.empty(0)), mapping)
    tokens = torch.tensor([-1 if chunked else 7, 8], dtype=torch.int32)
    out = ForwardOutput(tokens, tokens, None)
    p = PrefillManager(m.token_pool, 1)
    p.advance_for_next_schedule(None, inp, out)
    assert prefill.cached_len == 2 and decode.cached_len == 1
    assert m.token_pool[0, 2].item() == (0 if chunked else 7)
    assert m.token_pool[1, 2].item() == 8
    m.advance_for_overlap(batch)
    assert prefill.device_len == 2 and decode.device_len == 3
    reply, finished = m.process(None, inp, out)
    assert not finished
    assert [r.uid for r in reply] == rows
    assert prefill.device_len == (2 if chunked else 3)
    assert decode.device_len == 3
    assert (0 in m.running_reqs) == (not chunked)
    if not chunked:
        assert reply[0].cached_tokens == 1
    assert reply[-1].cached_tokens is None


def test_finished_decode_in_overlapping_mixed_batch_is_aborted():
    m = manager(1)
    req = m.running_reqs[1]
    req.sampling_params.ignore_eos = False
    previous = Batch([req], "decode")
    m.advance_for_overlap(previous)
    ongoing = Batch([request(0), req], "mixed", num_prefill_reqs=1)
    m.inflight_uids[1] = {1}
    m.advance_for_overlap(ongoing)
    inp = ForwardInput(previous, None, None, None)
    out = ForwardOutput(torch.tensor([0]), torch.tensor([0]), None)
    reply, finished = m.process(None, inp, out)
    assert reply[0].finished and finished == {req}
    assert req.aborted and 1 not in m.running_reqs
    out = ForwardOutput(torch.tensor([5, 6]), torch.tensor([5, 6]), None)
    reply, _ = m.process(None, ForwardInput(ongoing, None, None, None), out)
    assert [r.uid for r in reply] == [0]


def config_kwargs():
    return dict(model_path="unused", tp_info=DistributedInfo(0, 1), dtype=torch.float32)


@pytest.mark.parametrize("granularity", [16, 32, 64])
def test_logstate_mixed_budget_alignment(tmp_path, granularity):
    from logssms.cache.config import LogStateCacheConfig

    for budget in (granularity, 3 * granularity):
        config = EngineConfig(
            **config_kwargs(), cache_type="logstate",
            cache_config=LogStateCacheConfig(granularity=granularity),
            scheduling_policy="mixed", mixed_prefill_tokens=budget,
        )
        assert config.mixed_prefill_tokens == budget
        args = parse_args([
            "--model", str(tmp_path), "--dummy-weight", "--dtype", "float32",
            "--cache-type", "logstate", "--granularity", str(granularity),
            "--scheduling-policy", "mixed", "--mixed-prefill-tokens", str(budget),
        ])
        assert args.mixed_prefill_tokens == budget
    with pytest.raises(ValueError, match="positive multiple"):
        EngineConfig(**config_kwargs(), cache_config=LogStateCacheConfig(granularity=granularity),
                     mixed_prefill_tokens=granularity + 1)
    with pytest.raises(SystemExit):
        parse_args(["--model", "unused", "--cache-type", "logstate",
                    "--granularity", str(granularity), "--mixed-prefill-tokens", str(granularity + 1)])


@pytest.mark.parametrize("kwargs", [
    {"mixed_prefill_tokens": 0}, {"mixed_prefill_tokens": -64},
    {"mixed_prefill_tokens": 65}, {"scheduling_policy": "other"},
    {"scheduling_policy": "mixed", "speculative_algorithm": "mtp"},
])
def test_engine_config_rejects_invalid_mixed_settings(kwargs):
    with pytest.raises(ValueError):
        EngineConfig(**config_kwargs(), **kwargs)


def test_scheduler_config_checks_prefill_budget():
    assert SchedulerConfig(**config_kwargs()).scheduling_policy == "prefill-first"
    assert EngineConfig(**config_kwargs()).mixed_prefill_tokens == 128
    with pytest.raises(ValueError, match="max_prefill_tokens"):
        SchedulerConfig(**config_kwargs(), scheduling_policy="mixed",
                        mixed_prefill_tokens=256, max_prefill_tokens=128)


def test_prefill_first_allows_small_prefill_and_sequence_budgets(tmp_path):
    config = SchedulerConfig(**config_kwargs(), max_prefill_tokens=64, max_seq_len_override=64)
    assert config.scheduling_policy == "prefill-first"
    assert config.mixed_prefill_tokens == 128
    args = parse_args([
        "--model", str(tmp_path), "--dummy-weight", "--dtype", "float32",
        "--max-prefill-tokens", "64", "--max-seq-len-override", "64",
    ])
    assert args.scheduling_policy == "prefill-first"
    assert args.max_prefill_tokens == args.max_seq_len == 64
    assert args.mixed_prefill_tokens == 128


def test_explicit_mixed_budget_fits_64_token_prefill(tmp_path):
    config = SchedulerConfig(
        **config_kwargs(), scheduling_policy="mixed", mixed_prefill_tokens=64, max_prefill_tokens=64,
    )
    assert config.mixed_prefill_tokens == 64
    args = parse_args([
        "--model", str(tmp_path), "--dummy-weight", "--dtype", "float32",
        "--scheduling-policy", "mixed", "--max-prefill-tokens", "64", "--mixed-prefill-tokens", "64",
    ])
    assert args.mixed_prefill_tokens == args.max_prefill_tokens == 64


@pytest.mark.parametrize("args", [
    ["--mixed-prefill-tokens", "0"], ["--mixed-prefill-tokens", "65"],
    ["--scheduling-policy", "mixed", "--mixed-prefill-tokens", "256", "--max-prefill-tokens", "128"],
    ["--scheduling-policy", "mixed", "--speculative-algorithm", "mtp"],
])
def test_cli_rejects_invalid_mixed_settings_before_model_resolution(args):
    with pytest.raises(SystemExit):
        parse_args(["--model", "unused", *args])


@pytest.mark.parametrize("policy", ["prefill-first", "mixed"])
def test_cli_scheduling_fields_reach_engine_config(tmp_path, policy):
    config = parse_args([
        "--model", str(tmp_path), "--dummy-weight", "--dtype", "float32",
        "--scheduling-policy", policy, "--mixed-prefill-tokens", "192",
    ])
    assert isinstance(config, EngineConfig)
    assert config.scheduling_policy == policy
    assert config.mixed_prefill_tokens == 192
