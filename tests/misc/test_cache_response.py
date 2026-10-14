import asyncio
import json
from contextlib import nullcontext
from types import SimpleNamespace

import httpx
import pytest
import torch

from logssms.core import Batch, Request, SamplingParams
from logssms.message import BaseFrontendMsg, BaseTokenizerMsg, DetokenizeMsg, UserReply
from logssms.scheduler.ar import ARManagerBase
from logssms.server import api_server


@pytest.mark.parametrize("cached_tokens", [None, 0, 64])
def test_cache_count_message_roundtrip(cached_tokens):
    msg = DetokenizeMsg(1, [17, 18], True, cached_tokens=cached_tokens)
    decoded = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert decoded.cached_tokens == cached_tokens
    reply = UserReply(1, "answer", True, cached_tokens=decoded.cached_tokens)
    decoded_reply = BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(reply))
    assert decoded_reply.cached_tokens == cached_tokens


@pytest.mark.parametrize("cached_tokens", [0, 4])
def test_prefill_reply_reports_initial_match_not_chunk_frontier(cached_tokens):
    manager = ARManagerBase(
        config=SimpleNamespace(decode_batch_budget=32, page_size=4),
        device=torch.device("cpu"),
        cache_manager=SimpleNamespace(
            lazy_free_region=nullcontext, cache_req=lambda *_args, **_kwargs: None,
        ),
        table_manager=SimpleNamespace(token_pool=torch.empty((1, 32), dtype=torch.int32)),
        eos_token_id=-1,
    )
    req = Request(
        input_ids=torch.arange(12, dtype=torch.int32), table_idx=0, cached_len=8,
        output_len=8, uid=0, sampling_params=SamplingParams(ignore_eos=True),
        cache_handle=None, max_device_len=20, prefix_cached_len=cached_tokens,
    )
    output = SimpleNamespace(next_tokens_cpu=torch.tensor([17], dtype=torch.int32))
    replies, _ = manager.process(
        None, SimpleNamespace(batch=Batch([req], phase="prefill")), output,
    )
    assert replies[0].cached_tokens == cached_tokens
    assert req.cached_len == 12
    replies, _ = manager.process(
        None, SimpleNamespace(batch=Batch([req], phase="decode")), output,
    )
    assert replies[0].cached_tokens is None
    assert req.prefix_cached_len == cached_tokens


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("return_ids", [False, True])
@pytest.mark.parametrize("cached_tokens", [0, 3])
def test_chat_response_exposes_cache_count_without_profile(
    stream, return_ids, cached_tokens, monkeypatch,
):
    monkeypatch.delenv("LOGSSMS_PROFILE_SCHEDULER", raising=False)
    state = api_server.FrontendManager(
        config=SimpleNamespace(model_config=SimpleNamespace(vocab_size=100)),
        send_tokenizer=None, recv_tokenizer=None,
    )

    async def send_one(_msg):
        pass

    async def wait_for_ack(uid):
        yield UserReply(
            uid, "first", False, token_ids=[17], cached_tokens=cached_tokens,
        )
        yield UserReply(uid, "second", True, token_ids=[18])

    state.send_one = send_one
    state.wait_for_ack = wait_for_ack
    monkeypatch.setattr(api_server, "_GLOBAL_STATE", state)

    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_server.app), base_url="http://test",
        ) as client:
            return await client.post("/v1/chat/completions", json={
                "model": "test", "prompt_token_ids": [10, 11, 12, 13],
                "stream": stream, "max_tokens": 2, "return_token_ids": return_ids,
            })

    response = asyncio.run(request())
    assert response.status_code == 200
    if stream:
        events = [
            json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert events[0]["meta_info"]["cached_tokens"] == cached_tokens
        assert "meta_info" not in events[1]
        assert events[-1]["meta_info"]["cached_tokens"] == cached_tokens
        if return_ids:
            assert [
                token for event in events
                for token in event["choices"][0]["delta"].get("token_ids", [])
            ] == [17, 18]
    else:
        body = response.json()
        assert body["meta_info"]["cached_tokens"] == cached_tokens
        assert body["choices"][0]["message"]["content"] == "firstsecond"
        if return_ids:
            assert body["choices"][0]["message"]["token_ids"] == [17, 18]


def test_detokenizer_forwards_cache_count(monkeypatch):
    from logssms.tokenizer import server

    messages = [
        DetokenizeMsg(0, 17, False, cached_tokens=8),
        DetokenizeMsg(0, 18, True),
    ]
    replies = []

    class PullQueue:
        def empty(self):
            return not messages

        def get(self):
            return messages.pop(0)

    class PushQueue:
        def put(self, batch):
            replies.extend(batch.data)
            raise KeyboardInterrupt

    class Tokenizer:
        eos_token_id = -1

        def batch_decode(self, ids):
            return ["word " * len(tokens) for tokens in ids]

    monkeypatch.setattr(server, "load_tokenizer", lambda _: Tokenizer())
    monkeypatch.setattr(server, "ZmqPullQueue", lambda *_, **__: PullQueue())
    monkeypatch.setattr(server, "ZmqPushQueue", lambda *_, **__: PushQueue())
    server.detokenize_worker(
        tokenizer_path="unused", detokenizer_addr="unused", backend_addr="unused",
        frontend_addr="unused", local_bs=2,
    )
    assert [reply.cached_tokens for reply in replies] == [8, None]
