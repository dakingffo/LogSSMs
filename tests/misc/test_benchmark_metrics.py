import asyncio
import json

import pytest

from benchmarks.bench_common import (
    BenchmarkConversation,
    StreamResult,
    _run_conversation,
    make_server_cmd,
    save_json,
    split_warmup_prompt,
    stream_messages,
    summarize,
)
from datasets.parser import DatasetMessage


def test_warmup_prompt_is_not_in_measured_batch():
    warmup, prompts, input_lens = split_warmup_prompt(
        ["measured-0", "measured-1", "warmup"], [1, 2, 3], True
    )
    assert warmup == "warmup"
    assert prompts == ["measured-0", "measured-1"]
    assert input_lens == [1, 2]


def test_ar_online_metrics_match_sglang_tpot():
    # start, three content chunks, finish_reason, DONE
    stats = summarize([[0.0, 2.0, 3.0, 4.0, 4.1, 4.2]], [8], [3], mtp=False)
    assert stats["ttft_ms"][0] == pytest.approx(2000.0)
    assert stats["tpot_ms"][0] == pytest.approx(1100.0)
    assert stats["output_tok_per_s"] == pytest.approx(3 / 4.2)


def test_eos_enabled_uses_delivered_length_for_tpot():
    stats = summarize([[0.0, 2.0, 3.0, 3.1, 3.2]], [8], [256], mtp=False, ignore_eos=False)
    assert stats["requested_out"] == 256
    assert stats["out_tokens"] == 2
    assert stats["early_stops"] == 1
    assert stats["tpot_ms"][0] == pytest.approx(1200.0)
    assert stats["chunks_ok"]


def test_mtp_acceptance_excludes_mandatory_target_token():
    # One prefill token plus two 2-token verify rounds => one accepted draft/round.
    stats = summarize([[0.0, 2.0, 3.0, 4.0, 4.1, 4.2]], [8], [5], mtp=True)
    assert stats["avg_accept"] == pytest.approx(1.0)
    assert stats["rounds"] == 2
    assert stats["tpot_ms"][0] == pytest.approx(550.0)


def test_dt_server_flag_is_rebuilt_once():
    cmd = make_server_cmd(
        ["--model-path", "model", "--dt-separation", "false"],
        port=1234,
        enable_specualtive_decoding=True,
        num_spec_tokens=3,
        enable_dt=True,
    )
    assert cmd.count("--dt-separation") == 1
    assert cmd[cmd.index("--dt-separation") + 1] == "true"


def test_save_json_persists_each_completed_point(tmp_path):
    path = tmp_path / "result.json"
    save_json(str(path), {"mtp": {"1": {"output_tok_per_s": 10.0}}})
    assert json.loads(path.read_text()) == {
        "mtp": {"1": {"output_tok_per_s": 10.0}}
    }
    assert not path.with_suffix(".json.tmp").exists()


def test_multiturn_session_appends_generated_assistant_output(monkeypatch):
    requests = []

    async def fake_stream(
        client, base, model, messages, output_len, progress,
        profile_tag=None, ignore_eos=True,
    ):
        requests.append(messages)
        return StreamResult(
            tics=[float(len(requests)), float(len(requests)) + 1],
            output=f"generated-{len(requests)}",
        )

    monkeypatch.setattr("benchmarks.bench_common.stream_messages", fake_stream)
    conversation = BenchmarkConversation(
        uid="session",
        initial_messages=(DatasetMessage(role="system", content="system"),),
        user_turns=(
            DatasetMessage(role="user", content="first"),
            DatasetMessage(role="user", content="second"),
        ),
    )
    asyncio.run(
        _run_conversation(None, "base", "model", conversation, 8, None)
    )

    assert requests == [
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "first"},
        ],
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "generated-1"},
            {"role": "user", "content": "second"},
        ],
    ]


def test_stream_messages_collects_openai_delta_content():
    class Response:
        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            for line in (
                'data: {"choices":[{"delta":{"role":"assistant","content":"one"}}]}',
                'data: {"choices":[{"delta":{"content":" two"}}]}',
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ):
                yield line

    class Context:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, exc_type, exc, traceback):
            pass

    class Client:
        def stream(self, method, url, json, headers=None):
            assert json["messages"] == [{"role": "user", "content": "hello"}]
            assert json["ignore_eos"] is False
            return Context()

    result = asyncio.run(
        stream_messages(
            Client(),
            "http://server",
            "model",
            [{"role": "user", "content": "hello"}],
            2,
            ignore_eos=False,
        )
    )
    assert result.output == "one two"
    assert result.nonempty_content_chunks == 2
    assert len(result.tics) == 5
