import asyncio
import json

import pytest
import torch

from experiment import random_multiturn
from logssms.core import SamplingParams
from logssms.message import BaseTokenizerMsg, TokenizeMsg, UserReply
from logssms.server.api_server import FrontendManager, OpenAICompletionRequest
from logssms.tokenizer.tokenizer import TokenizeManager


class FakeTokenizer:
    vocab_size = 100
    all_special_ids = [0, 1]

    def encode(self, text, return_tensors=None):
        raise AssertionError("token-ID request must bypass text tokenization")


def test_token_id_request_bypasses_template_and_tokenizer():
    msg = TokenizeMsg(
        uid=1, text=None, sampling_params=SamplingParams(),
        input_ids=torch.tensor([12, 31, 42], dtype=torch.int32),
    )
    tokens = TokenizeManager(FakeTokenizer()).tokenize([msg])[0]
    assert tokens.dtype == torch.int32
    assert tokens.tolist() == [12, 31, 42]
    decoded = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert decoded.input_ids.tolist() == [12, 31, 42]
    assert TokenizeManager(FakeTokenizer()).tokenize([decoded])[0].tolist() == [12, 31, 42]
    req = OpenAICompletionRequest(
        model="test", prompt_token_ids=[12, 31, 42], return_token_ids=True
    )
    assert req.prompt is None and req.messages is None


def test_token_id_stream_keeps_empty_text_chunks():
    async def fake_ack(_uid):
        yield UserReply(uid=1, incremental_output="", finished=False, token_ids=[17])
        yield UserReply(uid=1, incremental_output="x", finished=True, token_ids=[18])

    manager = object.__new__(FrontendManager)
    manager.wait_for_ack = fake_ack

    async def read_stream(include_ids):
        return [line async for line in manager.stream_chat_completions(1, include_ids)]

    with_ids = asyncio.run(read_stream(True))
    deltas = [json.loads(line[6:])["choices"][0]["delta"] for line in with_ids[:-2]]
    assert deltas == [
        {"role": "assistant", "token_ids": [17]},
        {"content": "x", "token_ids": [18]},
    ]
    without_ids = asyncio.run(read_stream(False))
    assert b"token_ids" not in b"".join(without_ids)


def test_random_sessions_are_seeded_and_exclude_special_ids():
    first = random_multiturn.make_sessions(
        FakeTokenizer(), count=3, turns=4, input_len=10,
        append_len=5, shared_len=4, seed=17,
    )
    again = random_multiturn.make_sessions(
        FakeTokenizer(), count=3, turns=4, input_len=10,
        append_len=5, shared_len=4, seed=17,
    )
    assert first == again
    assert first[0].initial[:4] == first[1].initial[:4]
    assert first[0].initial != first[1].initial
    assert all(
        token not in (0, 1)
        for session in first
        for segment in (session.initial, *session.increments)
        for token in segment
    )


def test_closed_loop_appends_actual_generated_ids(monkeypatch):
    sessions = random_multiturn.make_sessions(
        FakeTokenizer(), count=2, turns=3, input_len=3,
        append_len=2, shared_len=0, seed=1,
    )
    seen = []

    async def fake_stream(_client, _base, _model, prompt, output_len, tag):
        seen.append((tag, list(prompt)))
        token = 70 + int(tag.rsplit("-", 1)[1])
        # start, output_len chunks, finish_reason, DONE
        return random_multiturn.TokenStream(
            tics=[float(i) for i in range(output_len + 3)],
            token_ids=[token] * output_len,
        )

    monkeypatch.setattr(random_multiturn, "stream_token_ids", fake_stream)
    stats, records = asyncio.run(random_multiturn.run_point(
        "http://unused", "model", sessions, concurrency=2,
        output_len=2, request_timeout=3,
    ))
    for session, record in zip(sessions, records, strict=True):
        turns = record["turns"]
        assert turns[0]["prompt_token_ids"] == list(session.initial)
        for index in (1, 2):
            previous = turns[index - 1]
            expected = (
                previous["prompt_token_ids"]
                + previous["generated_token_ids"]
                + list(session.increments[index - 1])
            )
            assert turns[index]["prompt_token_ids"] == expected
            assert turns[index]["expected_reusable_len"] == len(expected) - 2
    assert stats["chunks_ok"]
    assert stats["n"] == 6
    assert len(stats["turn_stats"]) == 3
    from benchmarks.bench_common import print_stats

    print_stats("random_multiturn test", stats)


def test_token_stream_parses_ids_even_when_text_is_empty():
    class Response:
        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            for line in (
                'data: {"choices":[{"delta":{"role":"assistant","token_ids":[4]}}]}',
                'data: {"choices":[{"delta":{"content":"text","token_ids":[5]}}]}',
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
        def stream(self, method, url, json, headers):
            assert json["prompt_token_ids"] == [1, 2, 3]
            assert json["ignore_eos"] and json["return_token_ids"]
            assert "messages" not in json and "prompt" not in json
            return Context()

    result = asyncio.run(random_multiturn.stream_token_ids(
        Client(), "http://unused", "model", [1, 2, 3], 2, "s/turn-1"
    ))
    assert result.token_ids == [4, 5]
    assert len(result.tics) == 5


def test_cache_matches_distinguish_eligible_from_admitted(tmp_path):
    records = [{"uid": "random-0", "turns": [
        {"tag": "random-0/turn-1", "expected_reusable_len": 0},
        {"tag": "random-0/turn-2", "expected_reusable_len": 10},
    ]}]
    path = tmp_path / "profile.rank0.json"
    path.write_text(json.dumps({"cache": {"match_samples": [
        {"tag": "random-0/turn-1", "tree_match_len": 0,
         "admitted_cached_len": 0},
        {"tag": "random-0/turn-2", "tree_match_len": 8,
         "admitted_cached_len": 4},
    ]}}))
    summary = random_multiturn.attach_cache_matches(records, path)
    assert summary["expected_reusable_tokens"] == 10
    assert summary["tree_match_fraction"] == pytest.approx(0.8)
    assert summary["admitted_hit_fraction"] == pytest.approx(0.4)
    assert records[0]["turns"][1]["cache_match"]["tree_match_len"] == 8
