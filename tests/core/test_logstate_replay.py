import json

import pytest

from experiment.utils.cpu_replay.replay import replay


def make_case():
    return {
        "granularity": 32,
        "max_running_requests": 32,
        "cache_bytes_per_rank": 47605350400,
        "max_seq_len": 18432,
        "output_cap": 1024,
        "eos_token_id": 248046,
        "gpu_count": 2,
        "tp_size": 2,
        "model": {
            "attention_layers": 16,
            "linear_layers": 48,
            "kv_heads": 4,
            "head_dim": 256,
            "linear_key_heads": 16,
            "linear_value_heads": 48,
            "linear_key_dim": 128,
            "linear_value_dim": 128,
            "conv_width": 4,
        },
    }


@pytest.mark.parametrize("mode", ["default", "online"])
@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("hit_materialization", [False, True])
def test_replay_recovers_an_existing_kv_prefix(tmp_path, mode, repeat, hit_materialization):
    case = make_case()
    case["output_cap"] = 154
    eos = case["eos_token_id"]
    prompt = list(range(39))
    first_output = [*range(39, 192), eos]
    branch_output = [*range(1000, 1121), eos]
    requests = [
        ("a", None, prompt, first_output),
        ("a-child", "a", prompt + first_output + list(range(2000, 2032)), [*range(3000, 3031), eos]),
        ("b", None, prompt, branch_output),
        ("b-child", "b", prompt + branch_output + list(range(4000, 4032)), [*range(5000, 5063), eos]),
    ]
    if repeat:
        requests.insert(2, ("a-repeat", None, prompt, first_output))
    history = tmp_path / "history.json"
    history.write_text(json.dumps({"requests": [
        dict(uid=uid, parent=parent, tree_id="tree", depth=1 if parent is None else 2,
             dispatch_index=i, prompt_token_ids=ids, generated_token_ids=output)
        for i, (uid, parent, ids, output) in enumerate(requests)
    ]}))
    result = replay(case, history, mode, batch_size=1, enable_hit_materialization=hit_materialization)
    assert result["execution"] == "production-scheduler-v1"
    assert result["overlap"] and not result["numerical_values"]
    records = {r["uid"]: r for r in result["requests_detail"]}
    assert records["b"]["tree_match"] == 32
    assert records["b"]["recoverable"] == (0 if hit_materialization and not repeat else 32)
    assert records["b-child"]["recoverable"] == 160
    assert records["b-child"]["historical_state_gap"] == 0
    assert (result["totals"]["rebuilt_state_tokens_on_existing_kv"] >= 32) == hit_materialization
    assert result["totals"]["verified_parent_prefixes"] == 2
    assert result["eviction_events"] == []
    for node in result["final_layout"]:
        entries = node["entries"]
        assert all(a["end"] <= b["begin"] for a, b in zip(entries, entries[1:]))


@pytest.mark.parametrize("mode", ["default", "online"])
def test_replay_uses_production_lifecycle_under_pressure(tmp_path, monkeypatch, mode):
    from collections import Counter
    from logssms.cache.logstate.manager import LogStateCacheManager
    from logssms.scheduler.scheduler import Scheduler
    from experiment.utils.cpu_replay.scheduler import ReplayScheduler

    assert ReplayScheduler.overlap_loop is Scheduler.overlap_loop
    calls = Counter()
    for name in ("match_req", "admit_prefill", "init_with_handle", "allocate",
                 "prepare_batch", "cache_req"):
        original = getattr(LogStateCacheManager, name)

        def count(self, *args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(LogStateCacheManager, name, count)
    case = make_case()
    case.update(max_running_requests=2, cache_bytes_per_rank=800 * 1024**2, output_cap=96)
    requests = []
    for tree in range(5):
        parent, full = None, []
        for turn in range(3):
            prompt = full + list(range(100000 + tree * 10000 + turn * 32,
                                       100033 + tree * 10000 + turn * 32))
            count = 1 if tree == 1 and turn == 0 else 96
            start = 200000 + tree * 10000 + turn * 128
            output = [*range(start, start + count - 1), case["eos_token_id"]]
            uid = f"{tree}-{turn}"
            requests.append(dict(uid=uid, parent=parent, tree_id=str(tree), depth=turn + 1,
                                 dispatch_index=turn * 5 + tree, prompt_token_ids=prompt,
                                 generated_token_ids=output))
            parent, full = uid, prompt + output
    history = tmp_path / "history.json"
    history.write_text(json.dumps({"requests": requests}))
    timings = tmp_path / "timings.jsonl"
    result = replay(case, history, mode, batch_size=2, timing_path=timings)
    events = [json.loads(line) for line in timings.read_text().splitlines()]
    started = {event["call"] for event in events if event["event"] == "start"}
    finished = {event["call"]: event for event in events if event["event"] == "finish"}
    assert started == finished.keys()
    assert {"admit_prefill", "eviction_plan"} == {event["operation"] for event in finished.values()}
    assert all(event["elapsed_ms"] >= 0 for event in finished.values())
    assert all(calls[name] > 0 for name in calls)
    assert len(calls) == 6
    assert result["forward_batches"]["mixed"] > 0
    assert result["forward_batches"]["decode"] > 0
    assert result["evictions"]["eviction_rounds"] > 0
    assert result["totals"]["verified_parent_prefixes"] == 10
    assert result["hit_rate"] > 0
    assert calls["init_with_handle"] == 15
    # The one-token reply finishes during prefill and needs only its final commit.
    assert calls["cache_req"] == 2 * 15 - 1
    assert result["batches"][-1]["frames_after"]["work"] == 2 * result["checkpoint_frames"]


def test_replay_rejects_broken_parent_token_prefix(tmp_path):
    case = make_case()
    history = tmp_path / "history.json"
    history.write_text(json.dumps({"requests": [
        dict(uid="a", parent=None, tree_id="tree", depth=1, dispatch_index=0,
             prompt_token_ids=[1], generated_token_ids=[case["eos_token_id"]]),
        dict(uid="b", parent="a", tree_id="tree", depth=2, dispatch_index=1,
             prompt_token_ids=[1, 3, 4], generated_token_ids=[5]),
    ]}))
    with pytest.raises(ValueError, match="Broken parent token prefix"):
        replay(case, history, "online")


@pytest.mark.parametrize("granularity", [16, 64])
def test_replay_holds_tree_slot_across_turns(tmp_path, granularity):
    case = make_case()
    case.update(granularity=granularity, max_running_requests=2, output_cap=8)
    requests = []
    for tree in range(2):
        parent, prefix = None, []
        for turn in range(2):
            begin = 1000 * tree + 200 * turn
            prompt = prefix + list(range(begin, begin + 97))
            output = [begin + 97, begin + 98, case["eos_token_id"]]
            uid = f"{tree}-{turn}"
            requests.append(dict(
                uid=uid, parent=parent, tree_id=str(tree), depth=turn + 1,
                dispatch_index=2 * turn + tree, prompt_token_ids=prompt,
                generated_token_ids=output,
            ))
            parent, prefix = uid, prompt + output
    history = tmp_path / "history.json"
    history.write_text(json.dumps({"requests": requests}))
    result = replay(case, history, "online", tree_concurrency=1, record_layout=False)
    ordered = sorted(result["requests_detail"], key=lambda record: record["admission_index"])
    assert [record["uid"] for record in ordered] == ["0-0", "0-1", "1-0", "1-1"]
    assert all(record["output_cap"] == 8 for record in ordered)
    assert result["granularity"] == granularity
    assert result["peak_tree_concurrency"] == 1
    assert result["totals"]["verified_parent_prefixes"] == 2
    assert result["request_hit_rate"] == pytest.approx(result["historical_hit_rate"])


def test_virtual_clock_releases_parallel_subagents_and_preserves_prefix(tmp_path):
    case = make_case()
    case.update(max_running_requests=4, output_cap=64)

    def request(t, blocks, out):
        return dict(type="s", t=t, api_time=1., **{"in": len(blocks) * 64},
                    out=out, hash_ids=blocks)

    history = tmp_path / "agent.jsonl"
    session = dict(id="session", block_size=64, hash_id_scope="local", requests=[
        request(0., [0], 64),
        dict(type="subagent", requests=[request(0., [3], 64)]),
        request(180., [0, 1, 2], 4),
        request(182., [0, 4], 4),
    ])
    history.write_text(json.dumps(session) + "\n")
    result = replay(case, history, "default", trace_mode="virtual_clock", tree_concurrency=1)
    first, subagent, continuation, rewrite = result["requests_detail"]
    assert first["admitted_iteration"] == subagent["admitted_iteration"]
    assert continuation["arrival_time_s"] == 180.
    assert rewrite["arrival_time_s"] == 182.
    assert continuation["history_tokens"] == 128
    assert continuation["recoverable"] >= 96
    assert rewrite["history_tokens"] == 64
    assert result["totals"]["verified_parent_prefixes"] == 1
    assert result["peak_tree_concurrency"] == 1


@pytest.mark.parametrize("recover", [False, True])
def test_virtual_clock_waits_for_arrivals_with_blocked_prefill(tmp_path, monkeypatch, recover):
    from experiment.utils.cpu_replay.scheduler import ReplayScheduler
    from logssms.scheduler.prefill import PrefillAdder

    arrivals = []
    make_message = ReplayScheduler._make_message
    try_add_one = PrefillAdder.try_add_one

    def record_arrival(scheduler, uid):
        arrivals.append((uid, scheduler.clock.now))
        return make_message(scheduler, uid)

    def try_add(adder, pending):
        if pending.uid == 0 and (not recover or len(arrivals) == 1):
            return None
        return try_add_one(adder, pending)

    monkeypatch.setattr(ReplayScheduler, "_make_message", record_arrival)
    monkeypatch.setattr(PrefillAdder, "try_add_one", try_add)
    case = make_case()
    case["output_cap"] = 1
    history = tmp_path / "history.json"
    history.write_text(json.dumps({"requests": [
        dict(uid=str(i), parent=None, tree_id="session", depth=0, dispatch_index=i,
             t=t, prompt_token_ids=[10 + i], generated_token_ids=[20 + i])
        for i, t in enumerate((0, 180))
    ]}))

    if recover:
        result = replay(case, history, "default", trace_mode="virtual_clock")
        assert all(req["admission_time_s"] == 180 for req in result["requests_detail"])
    else:
        with pytest.raises(RuntimeError, match=r"stalled at iteration 2; pending=2, running=0"):
            replay(case, history, "default", trace_mode="virtual_clock")
    assert arrivals == [(0, 0), (1, 180_000_000_000)]
