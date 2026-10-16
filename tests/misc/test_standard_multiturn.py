import asyncio
import json
import time

import httpx
import pytest

from datasets.parser import DatasetMessage, DatasetRecord
from experiment.random_multiturn.client import TokenStream
from experiment.random_multiturn.metrics import summarize
from experiment.standard_multiturn import client as standard


class Tokenizer:
    all_special_ids = [1, 2]
    chat_template = "test"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert not tokenize and add_generation_prompt
        assert kwargs["enable_thinking"] is False
        return "".join(
            f"<start>{message['role']}\n{message['content']}<end>\n"
            for message in messages
        ) + "<start>assistant\n"

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        result = []
        while text:
            for special, uid in (("<end>", 1), ("<start>", 2)):
                if text.startswith(special):
                    result.append(uid)
                    text = text[len(special):]
                    break
            else:
                result.append(ord(text[0]) + 100)
                text = text[1:]
        return result


def records(tree_id="r"):
    root = (tree_id, None, "user", "root")
    assistant = (f"{tree_id}a", tree_id, "assistant", "RECORDED ANSWER")
    left = (f"{tree_id}l", assistant[0], "user", "left")
    right = (f"{tree_id}r", assistant[0], "user", "right")
    paths = [
        [root, assistant, left, (f"{tree_id}la", left[0], "assistant", "OLD LEFT")],
        [root, assistant, left, (f"{tree_id}lb", left[0], "assistant", "ALTERNATIVE")],
        [root, assistant, right],
    ]
    return [DatasetRecord(
        uid=path[-1][0], category="en",
        messages=tuple(DatasetMessage(role=role, content=text) for _, _, role, text in path),
        metadata={
            "message_tree_id": tree_id,
            "message_ids": tuple(node[0] for node in path),
            "parent_ids": tuple(node[1] for node in path),
            "node_metadata": tuple({"deleted": False} for _ in path),
        },
    ) for path in paths]


def forest():
    return standard.reconstruct_trees(records() + records("s"))


def test_reconstruction_deduplicates_without_cropping_branches():
    tree = forest()[0]
    assert len(tree.messages) == 6
    requests = standard.make_requests(tree, standard.ChatProtocol(Tokenizer()))
    assert len(requests) == 4
    root = next(node for node in requests if node.parent is None)
    assert len([node for node in requests if node.parent == root.uid]) == 3
    assert sum(node.assistant_alternative for node in requests) == 2
    assert sum(node.user_fork for node in requests) == 3
    assert all(node.depth == (1 if node.parent is None else 2) for node in requests)
    recorded = Tokenizer().encode("RECORDED ANSWER", add_special_tokens=False)
    assert all(list(node.suffix_ids) != recorded for node in requests)


def test_deleted_subtree_is_not_grafted_to_another_parent():
    source = records()
    from dataclasses import replace

    altered = []
    for record in source:
        meta = dict(record.metadata)
        meta["node_metadata"] = tuple({"deleted": uid == "rl"} for uid in meta["message_ids"])
        altered.append(replace(record, metadata=meta))
    result = standard.reconstruct_trees(altered)[0]
    assert {node.uid for node in result.messages} == {"r", "ra", "rr"}


def test_nested_selection_and_topological_order_are_seeded():
    trees = standard.reconstruct_trees([record for i in range(10) for record in records(f"r{i}")])
    protocol = standard.ChatProtocol(Tokenizer())
    small, small_requests = standard.select_requests(trees, protocol, count=3, seed=9)
    large, large_requests = standard.select_requests(trees, protocol, count=6, seed=9)
    assert small == large[:3]
    ordered = standard.topological_order(large_requests, 9)
    assert ordered == standard.topological_order(large_requests, 9)
    positions = {node.uid: index for index, node in enumerate(ordered)}
    assert all(node.parent is None or positions[node.parent] < positions[node.uid] for node in ordered)
    assert len(small_requests) == 12
    assert ordered != standard.topological_order(large_requests, 10)


def test_context_check_rejects_whole_tree_without_truncation():
    protocol = standard.ChatProtocol(Tokenizer())
    with pytest.raises(ValueError, match="satisfy the filters"):
        standard.select_requests(forest(), protocol, count=1, seed=0, max_seq_len=1)


def test_topological_validation_rejects_cycles_and_missing_parents():
    def node(uid, parent):
        return standard.Request(uid, "tree", "user", uid, parent, 1, "root", (2,), None)

    with pytest.raises(ValueError, match="cycle"):
        standard.topological_order([node("a", "b"), node("b", "a")], 0)
    with pytest.raises(ValueError, match="Missing"):
        standard.topological_order([node("a", "missing")], 0)


@pytest.mark.parametrize("api", ["logssms", "sglang", "vllm"])
@pytest.mark.parametrize("include_eos", [False, True])
def test_three_apis_preserve_actual_generated_prefixes(api, include_eos, monkeypatch):
    protocol = standard.ChatProtocol(Tokenizer())
    requests = standard.topological_order([
        node for tree in forest() for node in standard.make_requests(tree, protocol)
    ], 7)
    prompts = []
    output = [501, 1] if include_eos else [501, 502]

    def respond(request):
        body = json.loads(request.content)
        if api == "sglang":
            prompt = body["input_ids"]
            assert body["sampling_params"]["ignore_eos"] is False
            events = [{"output_ids": output, "meta_info": {
                "completion_tokens": len(output), "cached_tokens": 0,
                "finish_reason": {"type": "stop"},
            }}]
        elif api == "vllm":
            prompt = body["prompt"]
            assert body["ignore_eos"] is False
            events = [{"choices": [{"index": 0, "prompt_token_ids": prompt,
                                    "token_ids": output, "finish_reason": "stop"}]},
                      {"choices": [], "usage": {"prompt_tokens": len(prompt),
                       "completion_tokens": len(output), "prompt_tokens_details": {"cached_tokens": 0}}}]
        else:
            prompt = body["prompt_token_ids"]
            assert body["ignore_eos"] is False
            events = [{"choices": [{"delta": {"token_ids": output}, "finish_reason": "stop"}],
                       "meta_info": {"cached_tokens": 0}}]
        prompts.append(prompt)
        text = "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n"
        return httpx.Response(200, text=text)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(respond), **kwargs,
    ))
    stats, history = asyncio.run(standard.run_point(
        "http://unused", "model", requests, api=api, output_len=8, request_timeout=3,
    ))
    by_uid = {record["uid"]: record for record in history}
    for record in history:
        assert record["generated_token_ids"] == output
        if record["parent"] is not None:
            parent = by_uid[record["parent"]]
            shared = parent["prompt_token_ids"] + output
            suffix = list(next(node for node in requests if node.uid == record["uid"]).suffix_ids)
            if include_eos:
                suffix = suffix[1:]
            assert record["prompt_token_ids"] == shared + suffix
            assert record["expected_reusable_len"] == len(shared)
            assert record["dispatched_at"] >= parent["tics"][-1]
    assert len(prompts) == stats["n"] == 8
    assert stats["chunks_ok"] and not stats["ignore_eos"]
    assert stats["out_tokens"] == 16 and stats["early_stops"] == 8
    assert stats["cache"]["overall_hit_fraction"] == 0


def test_scheduler_skips_blocked_priority_without_head_of_line_wait(monkeypatch):
    requests = [
        standard.Request("a", "a", "u", "a", None, 1, "root", (2,), None),
        standard.Request("aa", "a", "uu", "aa", "a", 2, "continuation", (1, 2), 1),
        standard.Request("b", "b", "v", "b", None, 1, "root", (3,), None),
    ]
    dispatched, completed = [], []

    async def execute(client, base, model, prompt, length, tag, api, *, ignore_eos):
        assert ignore_eos is False
        started = time.perf_counter()
        dispatched.append(tag)
        if tag == "a":
            await asyncio.sleep(.02)
            assert "b" in completed
        if tag == "aa":
            assert "a" in completed
        now = time.perf_counter()
        completed.append(tag)
        return TokenStream([started, now, now, now], [9], len(prompt) - 1)

    monkeypatch.setattr(standard, "stream_token_ids", execute)
    stats, history = asyncio.run(standard.run_point(
        "http://unused", "model", requests, api="logssms", output_len=4, request_timeout=3,
    ))
    assert dispatched == ["a", "b", "aa"]
    assert stats["peak_inflight"] == 2
    assert [record["dispatch_index"] for record in history] == [0, 2, 1]


def test_scheduler_serializes_each_tree_but_not_different_trees(monkeypatch):
    requests = [
        standard.Request("a", "tree", "u", "a", None, 1, "root", (2,), None),
        standard.Request("b", "tree", "v", "b", "a", 2, "continuation", (1, 3), 1),
        standard.Request("c", "tree", "w", "c", "a", 2, "continuation", (1, 4), 1),
        standard.Request("d", "tree", "u", "d", None, 1, "root", (5,), None),
        standard.Request("x", "other", "u", "x", None, 1, "root", (6,), None),
    ]
    dispatched, completed, prompts = [], [], {}
    running = set()

    async def execute(client, base, model, prompt, length, tag, api, **kwargs):
        tree = next(node.tree_id for node in requests if node.uid == tag)
        assert tree not in running
        running.add(tree)
        dispatched.append(tag)
        prompts[tag] = list(prompt)
        started = time.perf_counter()
        if tag == "a":
            await asyncio.sleep(.02)
            assert "x" in completed
        else:
            await asyncio.sleep(0)
        running.remove(tree)
        completed.append(tag)
        now = time.perf_counter()
        return TokenStream([started, now, now, now], [9], 0)

    monkeypatch.setattr(standard, "stream_token_ids", execute)
    stats, history = asyncio.run(standard.run_point(
        "http://unused", "model", requests, api="logssms", output_len=4, request_timeout=3,
    ))
    assert dispatched == ["a", "x", "b", "c", "d"]
    assert stats["peak_inflight"] == 2 and stats["dispatch_policy"] == "serial_per_tree"
    assert prompts["b"] == [2, 9, 1, 3]
    assert prompts["c"] == [2, 9, 1, 4]
    assert prompts["d"] == [5]
    by_uid = {record["uid"]: record for record in history}
    for left, right in zip(("a", "b", "c"), ("b", "c", "d")):
        assert by_uid[right]["dispatched_at"] >= by_uid[left]["tics"][-1]


def test_failure_cancels_remaining_tasks_and_retains_completed_history(monkeypatch):
    requests = [standard.Request(str(i), str(i), "u", "a", None, 1, "root", (2,), None) for i in range(3)]
    requests.append(standard.Request("3", "1", "u", "b", None, 1, "root", (3,), None))
    canceled = []
    dispatched = []

    async def execute(client, base, model, prompt, length, tag, api, **kwargs):
        dispatched.append(tag)
        if tag == "1":
            await asyncio.sleep(.01)
            raise RuntimeError("failed request")
        if tag == "2":
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                canceled.append(tag)
                raise
        now = time.perf_counter()
        return TokenStream([now, now, now, now], [9], 0)

    monkeypatch.setattr(standard, "stream_token_ids", execute)
    history = []
    with pytest.raises(RuntimeError, match="failed request"):
        asyncio.run(standard.run_point("http://unused", "model", requests, api="logssms",
                                      output_len=4, request_timeout=3, history=history))
    assert canceled == ["2"]
    assert dispatched == ["0", "1", "2"]
    assert [record["uid"] for record in history] == ["0"]


def test_missing_cache_is_unmeasured_not_zero(monkeypatch):
    async def execute(*args, **kwargs):
        now = time.perf_counter()
        return TokenStream([now, now, now, now], [9])

    monkeypatch.setattr(standard, "stream_token_ids", execute)
    node = standard.Request("a", "a", "u", "a", None, 1, "root", (2,), None)
    stats, _ = asyncio.run(standard.run_point("http://unused", "model", [node], api="logssms",
                                            output_len=4, request_timeout=3))
    assert stats["cache"] is None


def test_eos_metrics_use_actual_tokens_and_random_default_stays_strict():
    ticks = [[0, 1, 2, 2, 2]]
    assert summarize(ticks, [4], [8], ignore_eos=False)["chunks_ok"]
    assert not summarize(ticks, [4], [8])["chunks_ok"]
    assert summarize(ticks, [4], [8], ignore_eos=False)["out_tokens"] == 2


def test_empty_and_single_token_eos_responses_do_not_bias_tpot():
    ticks = [[0, 1, 1], [0, 1, 1, 1], [0, 1, 3, 3, 3]]
    stats = summarize(ticks, [4] * 3, [8] * 3, ignore_eos=False)
    assert stats["chunks_ok"]
    assert stats["ttft_requests"] == 2 and stats["tpot_requests"] == 1
    assert stats["tpot_ms"][0] == 2000


def test_cli_warmup_manifest_history_and_report(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda _: Tokenizer())
    monkeypatch.setattr(standard, "iter_dataset", lambda _: iter(
        [record for i in range(70) for record in records(f"r{i}")]
    ))
    seen = []

    async def execute(client, base, model, prompt, length, tag, api, **kwargs):
        seen.append((tag, length))
        now = time.perf_counter()
        return TokenStream([now, now + .001, now + .002, now + .002, now + .002], [501, 502], 0)

    monkeypatch.setattr(standard, "stream_token_ids", execute)
    args = ["--api", "logssms", "--base-url", "http://unused", "--tokenizer", "model",
            "--num-trees", "64", "--output-len", "128", "--max-seq-len", "1024",
            "--out-dir", str(tmp_path), "--name", "mock"]
    assert standard.main(args) == 0
    manifest = json.loads((tmp_path / "mock/manifest.json").read_text())
    report = json.loads((tmp_path / "mock/report.json").read_text())
    history = json.loads((tmp_path / "mock/history.json").read_text())["requests"]
    assert report["status"] == "complete"
    assert report["stats"]["trees"] == 64 and report["stats"]["n"] == 256
    assert report["warmup"]["trees"] == 4 and report["warmup"]["n"] == 16
    assert len(history) == 256
    assert set(manifest["tree_ids"]).isdisjoint(manifest["warmup_tree_ids"])
    assert all(length == 64 for _, length in seen[:16])
    assert all(length == 128 for _, length in seen[16:])


def test_figure_requires_matching_workloads_and_measured_cache(tmp_path):
    from experiment.standard_multiturn.plot import load_reports, main

    paths = []
    for api in ("logssms", "vllm", "sglang"):
        run = tmp_path / api
        run.mkdir()
        (run / "manifest.json").write_text(json.dumps({"seed": 0}))
        report = {"status": "complete", "api": api, "manifest_sha256": "same",
                  "stats": {"trees": 64, "ttft_ms": [10, 9, 11, 12, 13], "in_tokens": 100,
                            "cache": {"total_cached_tokens": 60}}}
        path = run / "report.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    assert len(load_reports(paths)) == 3
    import matplotlib

    matplotlib.use("Agg")
    output = tmp_path / "figure.png"
    assert main(["--reports", *map(str, paths), "--out", str(output)]) == 0
    assert output.is_file()
    report["manifest_sha256"] = "different"
    paths[-1].write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Different workloads"):
        load_reports(paths)
