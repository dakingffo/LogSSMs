import json
from pathlib import Path

import pytest

from experiment.ablation_online_analysis.replay_common import replay


@pytest.mark.parametrize("mode", ["default", "online"])
@pytest.mark.parametrize("repeat", [False, True])
def test_replay_rebuilds_state_under_an_existing_kv_prefix(tmp_path, mode, repeat):
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
    result = replay(case, history, mode, batch_size=1)
    assert result["execution"] == "production-scheduler-v1"
    assert result["overlap"] and not result["numerical_values"]
    records = {r["uid"]: r for r in result["requests_detail"]}
    assert records["b"]["tree_match"] == 32
    assert records["b"]["recoverable"] == (32 if repeat else 0)
    assert records["b-child"]["recoverable"] == 160
    assert records["b-child"]["historical_state_gap"] == 0
    assert result["totals"]["rebuilt_state_tokens_on_existing_kv"] >= 32
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


@pytest.mark.parametrize("mode", ["default", "online"])
def test_replay_publishes_remembered_protection_after_prefill(tmp_path, monkeypatch, mode):
    import torch
    from logssms.cache.logstate.manager import LogStateCacheManager
    from logssms.cache.logstate.prefix_cache import LogRadixTree, LogStateEntry, LogStateKind
    from tests.logstate_helpers import remember_log_tail

    root = Path(__file__).resolve().parents[2]
    case = json.loads((root / "experiment/ablation_online_analysis/cases.json").read_text())["32"]
    history = tmp_path / "history.json"
    history.write_text(json.dumps({"requests": [
        dict(uid="a", parent=None, tree_id="tree", depth=1, dispatch_index=0,
             prompt_token_ids=list(range(33)), generated_token_ids=[33]),
        dict(uid="b", parent="a", tree_id="tree", depth=2, dispatch_index=1,
             prompt_token_ids=list(range(513)), generated_token_ids=[513]),
    ]}))
    original = LogStateCacheManager.cache_req
    restored = []

    def commit(self, req, *, finished):
        if finished and req.uid == 1:
            restored.append(self._recycled_restores[req.table_idx].plan)
        result = original(self, req, finished=finished)
        if finished and req.uid == 0:
            ghost = LogRadixTree(32, torch.device("cpu"))
            node = ghost.insert(
                torch.arange(512, dtype=torch.int32), torch.arange(512, dtype=torch.int32),
                [LogStateEntry(i, LogStateKind.LOG, i * 32, (i + 1) * 32) for i in range(16)],
                analysis_owner=self.tree.trace(req.input_ids[:32]).node.analysis_owner,
            ).node
            ghost.protect_at(node, 256)
            remember_log_tail(self.recycle_bin, ghost, node)
        return result

    monkeypatch.setattr(LogStateCacheManager, "cache_req", commit)
    result = replay(case, history, mode, batch_size=1)
    assert 256 in restored[0].protected_endpoints
    assert any(node["end"] == 256 and node["protected"] for node in result["final_layout"])
    assert not any(node["begin"] < 256 < node["end"] for node in result["final_layout"])
    for node in result["final_layout"]:
        kinds = [e["kind"] for e in node["entries"]]
        if "log" in kinds:
            assert "checkpoint" not in kinds[kinds.index("log"):]


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
