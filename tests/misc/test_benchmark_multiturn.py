import json
import asyncio
import time

import pytest

from benchmarks import bench_common
from datasets.parser import DatasetMessage


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return " ".join(message["content"] for message in messages)

    def encode(self, text, *, add_special_tokens=True):
        return text.split()

    def decode(self, tokens, *, skip_special_tokens=False):
        return " ".join(tokens)


class PrefixTokenizer:
    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        ) + "<|im_start|>assistant\n"

    def encode(self, text, *, add_special_tokens=True):
        return list(text)

    def decode(self, tokens, *, skip_special_tokens=False):
        return "".join(tokens)


def test_multiturn_sessions_advance_on_their_own_completion(monkeypatch):
    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda _model: FakeTokenizer())
    conversations = [
        bench_common.BenchmarkConversation(
            uid=name,
            initial_messages=(),
            user_turns=tuple(
                DatasetMessage(role="user", content=f"{name}-{turn}")
                for turn in (1, 2)
            ),
        )
        for name in ("slow", "fast", "next")
    ]

    async def scenario():
        release_slow = asyncio.Event()
        next_started = asyncio.Event()
        sent: list[tuple[str, list[dict]]] = []

        async def fake_stream(
            _client, _base, _model, messages, _output_len, progress,
            profile_tag=None, ignore_eos=True,
        ):
            current = messages[-1]["content"]
            sent.append((current, [dict(message) for message in messages]))
            started = time.perf_counter()
            if current == "slow-1":
                await release_slow.wait()
            if current == "next-1":
                next_started.set()
            completed = time.perf_counter()
            if progress is not None:
                progress.tick()
            return bench_common.StreamResult(
                tics=[started, started + 0.001, completed + 0.001,
                      completed + 0.002, completed + 0.003],
                output=f"reply:{current}",
            )

        monkeypatch.setattr(bench_common, "stream_messages", fake_stream)
        task = asyncio.create_task(bench_common.run_multiturn_bench(
            "http://unused", "unused-model", conversations, 2,
            concurrency=2, pbar=False,
        ))
        try:
            await asyncio.wait_for(next_started.wait(), timeout=3)
            started = [name for name, _messages in sent]
            assert "slow-2" not in started
            assert started.index("fast-1") < started.index("fast-2") < started.index("next-1")
            fast_second = next(messages for name, messages in sent if name == "fast-2")
            assert fast_second[-2] == {
                "role": "assistant", "content": "reply:fast-1"
            }
        finally:
            release_slow.set()
        result = await asyncio.wait_for(task, timeout=3)
        assert result["n"] == 6
        assert result["sessions"] == 3
        assert result["concurrency"] == 2
        assert result["chunks_ok"]

    asyncio.run(scenario())


def test_multiturn_history_records_full_context_and_actual_reply(monkeypatch, tmp_path):
    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda _model: FakeTokenizer())

    async def fake_stream(
        _client, _base, _model, messages, _output_len, _progress,
        profile_tag=None, ignore_eos=True,
    ):
        current = messages[-1]["content"]
        return bench_common.StreamResult(
            tics=[0.0, 0.1, 0.2, 0.3, 0.4],
            output=f"reply {current}",
            nonempty_content_chunks=1,
        )

    monkeypatch.setattr(bench_common, "stream_messages", fake_stream)
    conversation = bench_common.BenchmarkConversation(
        uid="sample",
        initial_messages=(DatasetMessage(role="system", content="instruction"),),
        user_turns=(
            DatasetMessage(role="user", content="first"),
            DatasetMessage(role="user", content="second"),
        ),
    )
    records = []
    asyncio.run(bench_common.run_multiturn_bench(
        "http://unused", "unused-model", [conversation], 2,
        pbar=False, history_records=records,
    ))

    turns = records[0]["turns"]
    assert [turn["response"] for turn in turns] == ["reply first", "reply second"]
    assert turns[1]["messages"] == [
        {"role": "system", "content": "instruction"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply first"},
        {"role": "user", "content": "second"},
    ]
    assert turns[1]["input_tokens"] == 5
    assert turns[1]["response_tokens"] == 2
    assert turns[1]["content_chunks"] == 2
    assert turns[1]["nonempty_content_chunks"] == 1

    monkeypatch.setattr(bench_common, "_REPO", str(tmp_path))
    path = bench_common.resolve_result_path("multiturn/history/logstate16.json")
    assert path == str(tmp_path / "benchmark-results/multiturn/history/logstate16.json")
    report = bench_common.resolve_result_path("multiturn/report/logstate16.json")
    assert report == str(tmp_path / "benchmark-results/multiturn/report/logstate16.json")
    with pytest.raises(ValueError, match="explicit .json"):
        bench_common.resolve_result_path("logstate16")
    custom = tmp_path / "custom" / "history.json"
    assert bench_common.resolve_result_path(str(custom)) == str(custom)
    bench_common.save_json(str(custom), {"points": {"1": {"sessions": records}}})
    assert json.loads(custom.read_text())["points"]["1"]["sessions"] == records
    bench_common.save_json("multiturn/history/logstate16.json", {"sessions": records})
    assert json.loads(open(path).read())["sessions"] == records


def test_qwen36_generated_history_keeps_raw_reasoning(monkeypatch):
    class ThinkingTokenizer(FakeTokenizer):
        _logssms_preserve_generated_reasoning = True

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                                preserve_thinking):
            assert preserve_thinking
            return " ".join(message["content"] for message in messages)

    monkeypatch.setattr(
        bench_common, "_load_benchmark_tokenizer", lambda _model: ThinkingTokenizer()
    )
    conversation = bench_common.BenchmarkConversation(
        uid="thinking", initial_messages=(),
        user_turns=(
            DatasetMessage(role="user", content="first"),
            DatasetMessage(role="user", content="second"),
        ),
    )
    seen = []

    async def fake_stream(_client, _base, _model, messages, _output_len, _progress,
                          profile_tag=None, ignore_eos=True):
        seen.append([dict(message) for message in messages])
        return bench_common.StreamResult(
            tics=[0.0, 0.1, 0.2, 0.3, 0.4], output="unfinished reasoning",
        )

    monkeypatch.setattr(bench_common, "stream_messages", fake_stream)
    asyncio.run(bench_common.run_multiturn_bench(
        "http://unused", "unused-model", [conversation], 2, pbar=False,
    ))
    assert seen[1][-2] == {
        "role": "assistant", "content": "unfinished reasoning",
        "raw_generated": "unfinished reasoning",
    }


def test_fixed_history_keeps_request_inputs_independent_of_generated_reply(monkeypatch):
    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda _model: FakeTokenizer())
    conversation = bench_common.BenchmarkConversation(
        uid="sample",
        initial_messages=(),
        user_turns=(
            DatasetMessage(role="user", content="first"),
            DatasetMessage(role="user", content="second"),
        ),
    )

    async def run_with_first_reply(reply):
        seen = []

        async def fake_stream(
            _client, _base, _model, messages, _output_len, _progress,
            profile_tag=None, ignore_eos=True,
        ):
            seen.append([dict(message) for message in messages])
            return bench_common.StreamResult(
                tics=[0.0, 0.1, 0.2, 0.3, 0.4],
                output=reply if len(seen) == 1 else "final",
            )

        monkeypatch.setattr(bench_common, "stream_messages", fake_stream)
        stats = await bench_common.run_multiturn_bench(
            "http://unused", "unused-model", [conversation], 8,
            pbar=False, history_mode="fixed",
        )
        assert stats["history_mode"] == "fixed"
        return seen

    first = asyncio.run(run_with_first_reply("one reply"))
    second = asyncio.run(run_with_first_reply("a different reply"))
    assert first == second
    assert first[1][1]["role"] == "assistant"
    assert first[1][1]["content"] != "one reply"


@pytest.mark.parametrize("history_mode", ["prefix_generated", "prefix_fixed"])
def test_raw_prefix_history_preserves_every_previous_prompt(monkeypatch, history_mode):
    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda _model: PrefixTokenizer())
    conversation = bench_common.BenchmarkConversation(
        uid="sample",
        initial_messages=(),
        user_turns=tuple(
            DatasetMessage(role="user", content=f"turn {index}")
            for index in range(3)
        ),
    )
    seen = []

    async def fake_stream(
        _client, _base, _model, messages, _output_len, _progress,
        profile_tag=None, ignore_eos=True,
    ):
        assert isinstance(messages, str)
        seen.append(messages)
        return bench_common.StreamResult(
            tics=[0.0, 0.1, 0.2, 0.3, 0.4],
            output=f"generated {len(seen)}",
        )

    monkeypatch.setattr(bench_common, "stream_messages", fake_stream)
    records = []
    stats = asyncio.run(bench_common.run_multiturn_bench(
        "http://unused", "unused-model", [conversation], 2,
        pbar=False, history_mode=history_mode, history_records=records,
    ))
    assert stats["n"] == 3
    assert all(new.startswith(old) for old, new in zip(seen, seen[1:]))
    assert records[0]["turns"][1]["messages"] == seen[1]
    assert records[0]["turns"][1]["input_tokens"] == len(seen[1])
    if history_mode == "prefix_generated":
        assert "generated 1" in seen[1]
    else:
        assert "generated 1" not in seen[1]


def test_history_requires_an_explicit_path():
    with pytest.raises(SystemExit):
        bench_common.parse_full(["--history"])
    with pytest.raises(ValueError, match="explicit .json"):
        bench_common.parse_full(["--history", "logstate16"])
    with pytest.raises(ValueError, match="explicit .json"):
        bench_common.parse_full(["--out", "logstate16"])


def test_single_turn_history_collects_actual_response(monkeypatch):
    import httpx
    from transformers import AutoTokenizer

    class FakeClient:
        async def aclose(self):
            pass

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: FakeClient())
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda _model: FakeTokenizer())

    async def fake_stream(
        _client, _base, _model, messages, _output_len, _progress,
        profile_tag=None, ignore_eos=True,
    ):
        return bench_common.StreamResult(
            tics=[0.0, 0.1, 0.2, 0.3, 0.4],
            output=f"reply {messages[0]['content']}",
            nonempty_content_chunks=1,
        )

    monkeypatch.setattr(bench_common, "stream_messages", fake_stream)
    records = []
    stats = asyncio.run(bench_common.run_online_bench(
        "http://unused", "unused-model", ["hello"], [1], [2],
        pbar=False, history_records=records,
    ))
    assert stats["n"] == 1
    assert records[0]["turns"][0]["messages"] == [
        {"role": "user", "content": "hello"}
    ]
    assert records[0]["turns"][0]["response"] == "reply hello"
