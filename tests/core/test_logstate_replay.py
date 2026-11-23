import json
from pathlib import Path

import pytest

from experiment.ablation_online_analysis.replay_common import replay


@pytest.mark.parametrize("mode", ["default", "online"])
@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("hit_materialization", [False, True])
def test_replay_recovers_an_existing_kv_prefix(tmp_path, mode, repeat, hit_materialization):
    root = Path(__file__).resolve().parents[2]
    case = json.loads((root / "experiment/ablation_online_analysis/cases.json").read_text())["32"]
    prompt = list(range(39))
    first_output = list(range(39, 193))
    branch_output = list(range(1000, 1122))
    requests = [
        ("a", None, prompt, first_output),
        ("a-child", "a", prompt + first_output + list(range(2000, 2032)), list(range(3000, 3032))),
        ("b", None, prompt, branch_output),
        ("b-child", "b", prompt + branch_output + list(range(4000, 4032)), list(range(5000, 5064))),
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
    from experiment.ablation_online_analysis.cpu_common import ReplayScheduler

    assert ReplayScheduler.overlap_loop is Scheduler.overlap_loop
    calls = Counter()
    for name in ("match_req", "admit_prefill", "init_with_handle", "allocate",
                 "prepare_batch", "cache_req"):
        original = getattr(LogStateCacheManager, name)

        def count(self, *args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(LogStateCacheManager, name, count)
    root = Path(__file__).resolve().parents[2]
    case = json.loads((root / "experiment/ablation_online_analysis/cases.json").read_text())["32"]
    case.update(max_running_requests=2, cache_bytes_per_rank=800 * 1024**2)
    requests = []
    for tree in range(5):
        parent, full = None, []
        for turn in range(3):
            prompt = full + list(range(100000 + tree * 10000 + turn * 32,
                                       100033 + tree * 10000 + turn * 32))
            count = 1 if tree == 1 and turn == 0 else 96
            output = list(range(200000 + tree * 10000 + turn * 128,
                                200000 + tree * 10000 + turn * 128 + count))
            uid = f"{tree}-{turn}"
            requests.append(dict(uid=uid, parent=parent, tree_id=str(tree), depth=turn + 1,
                                 dispatch_index=turn * 5 + tree, prompt_token_ids=prompt,
                                 generated_token_ids=output))
            parent, full = uid, prompt + output
    history = tmp_path / "history.json"
    history.write_text(json.dumps({"requests": requests}))
    result = replay(case, history, mode, batch_size=2)
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
    root = Path(__file__).resolve().parents[2]
    case = json.loads((root / "experiment/ablation_online_analysis/cases.json").read_text())["32"]
    history = tmp_path / "history.json"
    history.write_text(json.dumps({"requests": [
        dict(uid="a", parent=None, tree_id="tree", depth=1, dispatch_index=0,
             prompt_token_ids=[1], generated_token_ids=[2]),
        dict(uid="b", parent="a", tree_id="tree", depth=2, dispatch_index=1,
             prompt_token_ids=[1, 3, 4], generated_token_ids=[5]),
    ]}))
    with pytest.raises(ValueError, match="Broken parent token prefix"):
        replay(case, history, "online")
