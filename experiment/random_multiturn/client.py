"""Closed-loop, token-exact multi-turn HTTP benchmark client.

The next prompt is previous prompt + *actual* generated tokens + a seeded random
increment. The server is started and stopped separately; this client never imports
or launches an inference engine.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from experiment.random_multiturn.metrics import (  # noqa: E402
    parse_int_list,
    print_stats,
    save_json,
    summarize,
)


@dataclass(frozen=True)
class Session:
    uid       : str
    initial   : tuple[int, ...]
    increments: tuple[tuple[int, ...], ...]


def make_sessions(
    tokenizer  : Any,
    *,
    count      : int,
    turns      : int,
    input_len  : int,
    append_len : int,
    shared_len : int,
    seed       : int,
    vocab_limit: int | None = None,
) -> list[Session]:
    rng = random.Random(seed)
    special_ids = set(tokenizer.all_special_ids)
    vocab_size = min(tokenizer.vocab_size, vocab_limit or tokenizer.vocab_size)
    if vocab_size <= sum(token < vocab_size for token in special_ids):
        raise ValueError("Tokenizer has no ordinary token IDs")

    def draw(length: int) -> tuple[int, ...]:
        result = []
        while len(result) < length:
            token = rng.randrange(vocab_size)
            if token not in special_ids:
                result.append(token)
        return tuple(result)

    shared = draw(shared_len)
    return [
        Session(
            uid=f"random-{index}",
            initial=shared + draw(input_len - shared_len),
            increments=tuple(draw(append_len) for _ in range(turns - 1)),
        )
        for index in range(count)
    ]


def make_warmup_sessions(
    tokenizer: Any, sessions: list[Session], *, concurrency: int, seed: int,
) -> list[Session]:
    forbidden = {session.initial[0] for session in sessions}
    forbidden.update(tokenizer.all_special_ids)
    first = next(token for token in range(tokenizer.vocab_size) if token not in forbidden)
    warmup = make_sessions(
        tokenizer, count=concurrency, turns=2, input_len=len(sessions[0].initial),
        append_len=len(sessions[0].increments[0]), shared_len=0, seed=seed + 1_000_000,
    )
    return [
        Session(f"warmup-{index}", (first, *session.initial[1:]), session.increments)
        for index, session in enumerate(warmup)
    ]


@dataclass(frozen=True)
class TokenStream:
    tics         : list[float]
    token_ids    : list[int]
    cached_tokens: int | None = None


async def stream_token_ids(
    client     : Any,
    base       : str,
    model      : str,
    prompt_ids : list[int],
    output_len : int,
    tag        : str,
    api        : str = "logssms",
) -> TokenStream:
    if api == "logssms":
        endpoint = "/v1/chat/completions"
        payload = {
            "model": model,
            "prompt_token_ids": prompt_ids,
            "max_tokens": output_len,
            "temperature": 0.0,
            "top_k": 1,
            "stream": True,
            "ignore_eos": True,
            "return_token_ids": True,
        }
        headers = {"X-LogSSMs-Profile-Tag": tag}
    elif api == "sglang":
        endpoint = "/generate"
        payload = {
            "input_ids": prompt_ids,
            "sampling_params": {
                "max_new_tokens": output_len,
                "temperature": 0.0,
                "top_k": 1,
                "ignore_eos": True,
            },
            "stream": True,
        }
        headers = {}
    elif api == "vllm":
        endpoint = "/v1/completions"
        payload = {
            "model": model,
            "prompt": prompt_ids,
            "max_tokens": output_len,
            "temperature": 0.0,
            "top_k": 1,
            "n": 1,
            "echo": False,
            "add_special_tokens": False,
            "stream": True,
            "ignore_eos": True,
            "return_token_ids": True,
            "stream_options": {"include_usage": True},
        }
        headers = {}
    else:
        raise ValueError(f"Unknown generation API: {api}")

    started = time.perf_counter()
    token_tics: list[float] = []
    finish_tic = None
    done_tic = None
    generated: list[int] = []
    cached_tokens = None
    prompt_verified = False
    async with client.stream(
        "POST", f"{base.rstrip('/')}{endpoint}", json=payload, headers=headers,
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                done_tic = time.perf_counter()
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"{tag}: server error: {event['error']}")
            now = time.perf_counter()
            if api == "logssms":
                chunk = event["choices"][0]
                new_ids = chunk["delta"].get("token_ids") or []
                if chunk.get("finish_reason") is not None:
                    finish_tic = now
            elif api == "vllm":
                usage = event.get("usage")
                if usage is not None:
                    details = usage.get("prompt_tokens_details") or {}
                    cached_tokens = details.get("cached_tokens", cached_tokens)
                # vLLM emits a final usage-only event with choices=[].
                choices = event.get("choices") or []
                new_ids = []
                if choices:
                    if len(choices) != 1 or choices[0].get("index") != 0:
                        raise RuntimeError(f"{tag}: expected one vLLM completion choice")
                    chunk = choices[0]
                    echoed_prompt = chunk.get("prompt_token_ids")
                    if echoed_prompt is not None:
                        if echoed_prompt != prompt_ids:
                            raise RuntimeError(f"{tag}: vLLM rewrote the token-ID prompt")
                        prompt_verified = True
                    new_ids = chunk.get("token_ids")
                    if new_ids is None:
                        raise RuntimeError(f"{tag}: vLLM response has no token_ids")
                    if chunk.get("finish_reason") is not None:
                        finish_tic = now
                if usage is not None:
                    if usage.get("prompt_tokens") != len(prompt_ids):
                        raise RuntimeError(f"{tag}: vLLM usage prompt length differs from input")
                    if usage.get("completion_tokens") != len(generated) + len(new_ids):
                        raise RuntimeError(f"{tag}: vLLM usage output length differs from token IDs")
            else:
                meta = event.get("meta_info") or {}
                ids = event.get("output_ids")
                if ids is None:
                    raise RuntimeError(f"{tag}: SGLang response has no output_ids")
                count = meta.get("completion_tokens")
                if count is None:
                    raise RuntimeError(f"{tag}: SGLang response has no completion_tokens")
                # SGLang can send cumulative IDs (default) or disjoint segments.
                if len(ids) == count and ids[:len(generated)] == generated:
                    new_ids = ids[len(generated):]
                elif len(ids) == count - len(generated):
                    new_ids = ids
                else:
                    raise RuntimeError(f"{tag}: inconsistent SGLang output_ids")
                cached_tokens = meta.get("cached_tokens", cached_tokens)
                if meta.get("finish_reason") is not None:
                    finish_tic = now
            generated.extend(new_ids)
            token_tics.extend([now] * len(new_ids))

    if done_tic is None or len(generated) != output_len:
        raise RuntimeError(
            f"{tag}: expected {output_len} generated IDs and [DONE], "
            f"got {len(generated)} IDs, finished={done_tic is not None}"
        )
    if api == "vllm" and not prompt_verified:
        raise RuntimeError(f"{tag}: vLLM response has no prompt_token_ids to verify")
    # summarize() expects [start, one tick per output ID, finish, DONE]. It only
    # uses the first token and DONE ticks for TTFT/TPOT; batched SSE chunks are safe.
    return TokenStream(
        [started, *token_tics, finish_tic or done_tic, done_tic],
        generated, cached_tokens,
    )


async def run_point(
    base           : str,
    model          : str,
    sessions       : list[Session],
    *,
    concurrency   : int,
    output_len    : int,
    request_timeout: float,
    warmup        : list[Session] | None = None,
    api           : str = "logssms",
) -> tuple[dict, list[dict]]:
    import httpx

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30.0, read=None, write=None, pool=None),
        limits=httpx.Limits(max_connections=1024, max_keepalive_connections=128),
        trust_env=False,
    )
    records: list[dict | None] = [None] * len(sessions)
    queue: asyncio.Queue[tuple[int, Session]] = asyncio.Queue()
    for index, session in enumerate(sessions):
        queue.put_nowait((index, session))

    async def one_session(index: int, session: Session, *, warming: bool = False) -> None:
        prompt = list(session.initial)
        expected_len = 0
        turns = []
        for turn_index in range(len(session.increments) + 1):
            tag = f"{session.uid}/turn-{turn_index + 1}"
            result = await asyncio.wait_for(
                stream_token_ids(
                    client, base, model, prompt,
                    min(output_len, 64) if warming else output_len, tag, api,
                ),
                timeout=request_timeout,
            )
            turns.append({
                "tag": tag,
                "turn": turn_index + 1,
                "prompt_token_ids": prompt,
                "generated_token_ids": result.token_ids,
                "expected_reusable_len": expected_len,
                "tics": result.tics,
            })
            if result.cached_tokens is not None:
                turns[-1]["cached_tokens"] = result.cached_tokens
            if turn_index < len(session.increments):
                prior = prompt + result.token_ids
                prompt = prior + list(session.increments[turn_index])
                assert prompt[:len(prior)] == prior
                expected_len = min(len(prior), len(prompt) - 1)
        if not warming:
            records[index] = {"uid": session.uid, "turns": turns}

    async def worker() -> None:
        while not queue.empty():
            index, session = queue.get_nowait()
            await one_session(index, session)

    warmup_summary = {"requests": 0, "duration_s": 0.0, "output_tokens": 0}
    try:
        if warmup:
            print(f"Warmup: {len(warmup)} sessions, two turns; excluded from results.", flush=True)
            started = time.perf_counter()
            await asyncio.gather(*[
                one_session(index, session, warming=True)
                for index, session in enumerate(warmup)
            ])
            warmup_summary = {
                "requests": sum(len(session.increments) + 1 for session in warmup),
                "duration_s": time.perf_counter() - started,
                "output_tokens": sum(len(session.increments) + 1 for session in warmup)
                * min(output_len, 64),
            }
            print(f"Warmup complete ({warmup_summary['duration_s']:.2f}s); starting measurement.", flush=True)
        await asyncio.gather(*[worker() for _ in range(concurrency)])
    finally:
        await client.aclose()

    completed = [record for record in records if record is not None]
    if len(completed) != len(sessions):
        raise RuntimeError("A session did not complete")
    turns = [turn for record in completed for turn in record["turns"]]
    stats = summarize(
        [turn["tics"] for turn in turns],
        [len(turn["prompt_token_ids"]) for turn in turns],
        [output_len] * len(turns),
    )
    stats.update(sessions=len(sessions), concurrency=concurrency,
                 num_turns=len(sessions[0].increments) + 1)
    stats["warmup"] = warmup_summary
    stats["session_per_s"] = (
        len(sessions) / stats["duration_s"] if stats["duration_s"] else 0.0
    )
    stats["turn_stats"] = []
    for index in range(stats["num_turns"]):
        turn_records = [record["turns"][index] for record in completed]
        turn_stats = summarize(
            [turn["tics"] for turn in turn_records],
            [len(turn["prompt_token_ids"]) for turn in turn_records],
            [output_len] * len(turn_records),
        )
        stats["turn_stats"].append({"turn": index + 1, **turn_stats})
    return stats, completed


def summarize_response_cache(records: list[dict], api: str = "sglang") -> dict | None:
    """Use request-level cache counts when the HTTP API exposes them."""
    turns = [turn for record in records for turn in record["turns"]]
    if any("cached_tokens" not in turn for turn in turns):
        return None
    expected_total = 0
    cached_total = 0
    continuations = 0
    for turn in turns:
        expected = turn["expected_reusable_len"]
        if expected:
            continuations += 1
            expected_total += expected
            cached_total += min(expected, turn["cached_tokens"])
    return {
        "source": (
            "response.usage.prompt_tokens_details.cached_tokens"
            if api == "vllm" else "response.meta_info.cached_tokens"
        ),
        "continuations": continuations,
        "expected_reusable_tokens": expected_total,
        "admitted_cached_tokens": cached_total,
        "admitted_hit_fraction": (
            cached_total / expected_total if expected_total else 0.0
        ),
    }


def input_trace_digest(records: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps([
            [turn["prompt_token_ids"] for turn in record["turns"]]
            for record in records
        ], separators=(",", ":")).encode()
    ).hexdigest()


def output_trace_digest(records: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps([
            [turn["generated_token_ids"] for turn in record["turns"]]
            for record in records
        ], separators=(",", ":")).encode()
    ).hexdigest()


def _parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Already-running server URL")
    parser.add_argument("--api", choices=("logssms", "sglang", "vllm"), required=True)
    parser.add_argument("--tokenizer", "--model-path", dest="tokenizer_path", required=True)
    parser.add_argument("--model", help="Served model name; defaults to --tokenizer")
    parser.add_argument("--max-seq-len", type=int, help="Optional client-side length check")
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--append-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-turns", type=int, default=4)
    parser.add_argument("--num-sessions", type=int, default=16)
    parser.add_argument("--num-concurrent-sessions", default="1")
    parser.add_argument("--shared-prefix-len", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None)
    parser.add_argument("--out-dir", default=str(_REPO / "experiment/random_multiturn"))
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--no-warmup", action="store_true")
    bench = parser.parse_args(argv)
    concurrencies = parse_int_list(bench.num_concurrent_sessions)
    if not concurrencies:
        parser.error("--num-concurrent-sessions must contain one integer")
    if any(value < 1 for value in (
        bench.input_len, bench.append_len, bench.output_len,
        bench.num_sessions, bench.request_timeout,
    )):
        parser.error("lengths, turns, sessions, and timeout must be positive")
    if bench.max_seq_len is not None and bench.max_seq_len < 1:
        parser.error("--max-seq-len must be positive")
    if bench.num_turns < 2:
        parser.error("random_multiturn needs at least two turns to measure reuse")
    if not 0 <= bench.shared_prefix_len <= bench.input_len:
        parser.error("--shared-prefix-len must lie between 0 and --input-len")
    if any(c < 1 or c > bench.num_sessions for c in concurrencies):
        parser.error("concurrency must be between 1 and --num-sessions")
    if len(concurrencies) != 1:
        parser.error("run one concurrency point per invocation, with a fresh server")
    max_input = bench.input_len + (bench.num_turns - 1) * (
        bench.output_len + bench.append_len
    )
    if bench.max_seq_len is not None and max_input + bench.output_len > bench.max_seq_len:
        parser.error("workload exceeds --max-seq-len")
    return bench, concurrencies[0]


def main(argv: list[str] | None = None) -> int:
    bench, concurrency = _parse_args(
        sys.argv[1:] if argv is None else argv
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(bench.tokenizer_path)
    sessions = make_sessions(
        tokenizer, count=bench.num_sessions, turns=bench.num_turns,
        input_len=bench.input_len, append_len=bench.append_len,
        shared_len=bench.shared_prefix_len, seed=bench.seed,
    )
    warmup = None if bench.no_warmup else make_warmup_sessions(
        tokenizer, sessions, concurrency=concurrency, seed=bench.seed,
    )
    name = bench.name or f"{datetime.now():%Y%m%d-%H%M%S}-pid{os.getpid()}"
    run_dir = Path(bench.out_dir).resolve() / name
    run_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "api": bench.api,
        "base_url": bench.base_url,
        "tokenizer": bench.tokenizer_path,
        "model": bench.model or bench.tokenizer_path,
        "seed": bench.seed,
        "input_len": bench.input_len,
        "append_len": bench.append_len,
        "output_len": bench.output_len,
        "num_turns": bench.num_turns,
        "num_sessions": bench.num_sessions,
        "shared_prefix_len": bench.shared_prefix_len,
        "ignore_eos": True,
        "chat_template": False,
        "max_seq_len": bench.max_seq_len,
        "points": {},
    }
    stats, history = asyncio.run(run_point(
        bench.base_url, bench.model or bench.tokenizer_path, sessions,
        concurrency=concurrency, output_len=bench.output_len,
        request_timeout=bench.request_timeout, warmup=warmup, api=bench.api,
    ))
    if not stats["chunks_ok"]:
        raise RuntimeError("Output token count differs from requested length")
    cache = summarize_response_cache(history, bench.api)
    stats["cache"] = cache
    trace_path = run_dir / f"history-c{concurrency}.json"
    save_json(str(trace_path), {"sessions": history})
    stats["history_path"] = str(trace_path)
    stats["input_trace_sha256"] = input_trace_digest(history)
    stats["output_trace_sha256"] = output_trace_digest(history)
    report["points"][str(concurrency)] = stats
    save_json(str(run_dir / "report.json"), report)
    print_stats(
        f"random_multiturn {bench.api} c={concurrency} turns={bench.num_turns}",
        stats,
    )
    if cache is not None:
        print(
            f"  reusable: {cache['expected_reusable_tokens']} tok; "
            f"cached: {cache['admitted_hit_fraction']:.1%}"
        )
    else:
        print("  cache hits: unmeasured by this API")
        if bench.api == "vllm":
            print("  enable vLLM --enable-prompt-tokens-details to report cache hits")
    print(f"Results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
