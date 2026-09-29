"""Closed-loop, token-exact multi-turn cache benchmark.

The next prompt is previous prompt + *actual* generated tokens + a seeded random
increment. No chat template or decode/encode round trip is involved. Each point
starts a fresh server and writes request histories plus scheduler cache matches.

Example:
  python experiment/random_multiturn.py --model-path /path/to/model \
      --cache-type logstate --granularity 32 --tensor-parallel-size 2 \
      --num-concurrent-sessions 8,16 --num-sessions 32 --num-turns 8
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


_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "python"))

from benchmarks.bench_common import (  # noqa: E402
    kill_server,
    launch_server,
    parse_int_list,
    print_stats,
    resolve_port,
    save_json,
    summarize,
    wait_server_ready,
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


@dataclass(frozen=True)
class TokenStream:
    tics     : list[float]
    token_ids: list[int]


async def stream_token_ids(
    client     : Any,
    base       : str,
    model      : str,
    prompt_ids : list[int],
    output_len : int,
    tag        : str,
) -> TokenStream:
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
    tics = [time.perf_counter()]
    generated: list[int] = []
    finished = False
    async with client.stream(
        "POST", f"{base}/v1/chat/completions", json=payload,
        headers={"X-LogSSMs-Profile-Tag": tag},
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                tics.append(time.perf_counter())
                finished = True
                break
            chunk = json.loads(data)["choices"][0]
            ids = chunk["delta"].get("token_ids")
            if ids is not None:
                if len(ids) != 1:
                    raise RuntimeError(f"{tag}: expected one AR token per chunk, got {ids}")
                generated.extend(ids)
            tics.append(time.perf_counter())
    if not finished or len(generated) != output_len:
        raise RuntimeError(
            f"{tag}: expected {output_len} generated IDs and [DONE], "
            f"got {len(generated)} IDs, finished={finished}"
        )
    if len(tics) != output_len + 3:
        raise RuntimeError(f"{tag}: unexpected number of streaming chunks")
    return TokenStream(tics, generated)


async def run_point(
    base           : str,
    model          : str,
    sessions       : list[Session],
    *,
    concurrency   : int,
    output_len    : int,
    request_timeout: float,
    warmup        : tuple[int, ...] | None = None,
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

    async def one_session(index: int, session: Session) -> None:
        prompt = list(session.initial)
        expected_len = 0
        turns = []
        for turn_index in range(len(session.increments) + 1):
            tag = f"{session.uid}/turn-{turn_index + 1}"
            result = await asyncio.wait_for(
                stream_token_ids(client, base, model, prompt, output_len, tag),
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
            if turn_index < len(session.increments):
                prior = prompt + result.token_ids
                prompt = prior + list(session.increments[turn_index])
                assert prompt[:len(prior)] == prior
                expected_len = min(len(prior), len(prompt) - 1)
        records[index] = {"uid": session.uid, "turns": turns}

    async def worker() -> None:
        while not queue.empty():
            index, session = queue.get_nowait()
            await one_session(index, session)

    try:
        if warmup is not None:
            await asyncio.wait_for(
                stream_token_ids(
                    client, base, model, list(warmup), min(output_len, 8),
                    "warmup/turn-1",
                ),
                timeout=request_timeout,
            )
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
        mtp=False,
        ignore_eos=True,
    )
    stats.update(sessions=len(sessions), concurrency=concurrency,
                 num_turns=len(sessions[0].increments) + 1)
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
            mtp=False, ignore_eos=True,
        )
        stats["turn_stats"].append({"turn": index + 1, **turn_stats})
    return stats, completed


def attach_cache_matches(records: list[dict], profile_path: Path) -> dict:
    if not profile_path.is_file():
        raise RuntimeError(f"Scheduler cache profile was not written: {profile_path}")
    profile = json.loads(profile_path.read_text())
    samples = {
        sample["tag"]: sample
        for sample in profile["cache"]["match_samples"]
    }
    expected_total = 0
    matched_total = 0
    recoverable_total = 0
    continuations = 0
    for record in records:
        for turn in record["turns"]:
            sample = samples.get(turn["tag"])
            if sample is None:
                raise RuntimeError(f"No cache match sample for {turn['tag']}")
            turn["cache_match"] = {
                name: sample.get(name) for name in (
                    "tree_match_len", "recoverable_len", "admitted_cached_len",
                    "arrival_tree_match_len", "state_recovery_gap",
                    "admission_wait_ms",
                )
            }
            expected = turn["expected_reusable_len"]
            if expected:
                continuations += 1
                expected_total += expected
                matched_total += min(expected, sample["tree_match_len"])
                recoverable_total += min(expected, sample["admitted_cached_len"])
    return {
        "continuations": continuations,
        "expected_reusable_tokens": expected_total,
        "tree_matched_tokens": matched_total,
        "admitted_cached_tokens": recoverable_total,
        "tree_match_fraction": matched_total / expected_total if expected_total else 0.0,
        "admitted_hit_fraction": (
            recoverable_total / expected_total if expected_total else 0.0
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
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--append-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-turns", type=int, default=4)
    parser.add_argument("--num-sessions", type=int, default=16)
    parser.add_argument("--num-concurrent-sessions", default="1,2,4,8")
    parser.add_argument("--shared-prefix-len", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None)
    parser.add_argument("--out-dir", default=str(_REPO / "experiment/random_multiturn"))
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--profile-only", action="store_true",
        help="skip the separate unprofiled timing run (diagnostics only)",
    )
    bench, server_argv = parser.parse_known_args(argv)

    from logssms.server.args import parse_args

    server_args = parse_args(server_argv)
    concurrencies = parse_int_list(bench.num_concurrent_sessions, 1)
    if any(value < 1 for value in (
        bench.input_len, bench.append_len, bench.output_len,
        bench.num_sessions, bench.request_timeout,
    )):
        parser.error("lengths, turns, sessions, and timeout must be positive")
    if bench.num_turns < 2:
        parser.error("random_multiturn needs at least two turns to measure reuse")
    if not 0 <= bench.shared_prefix_len <= bench.input_len:
        parser.error("--shared-prefix-len must lie between 0 and --input-len")
    if any(c < 1 or c > bench.num_sessions for c in concurrencies):
        parser.error("concurrency must be between 1 and --num-sessions")
    if server_args.speculative_algorithm is not None:
        parser.error("random_multiturn currently supports AR only")
    max_input = bench.input_len + (bench.num_turns - 1) * (
        bench.output_len + bench.append_len
    )
    if max_input + bench.output_len > server_args.max_seq_len:
        parser.error("workload exceeds the model's maximum sequence length")
    return bench, server_args, server_argv, concurrencies


def main(argv: list[str] | None = None) -> int:
    bench, server_args, server_argv, concurrencies = _parse_args(
        sys.argv[1:] if argv is None else argv
    )
    from logssms.utils import load_tokenizer

    tokenizer = load_tokenizer(server_args.model_path)
    sessions = make_sessions(
        tokenizer, count=bench.num_sessions, turns=bench.num_turns,
        input_len=bench.input_len, append_len=bench.append_len,
        shared_len=bench.shared_prefix_len, seed=bench.seed,
        vocab_limit=server_args.model_config.vocab_size,
    )
    warmup = None if bench.no_warmup else make_sessions(
        tokenizer, count=1, turns=1, input_len=min(bench.input_len, 32),
        append_len=bench.append_len, shared_len=0, seed=bench.seed + 1_000_000,
        vocab_limit=server_args.model_config.vocab_size,
    )[0].initial
    name = bench.name or f"{datetime.now():%Y%m%d-%H%M%S}-pid{os.getpid()}"
    run_dir = Path(bench.out_dir).resolve() / name
    run_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "model": server_args.model_path,
        "cache_type": server_args.cache_type,
        "cache_config": repr(server_args.cache_config),
        "seed": bench.seed,
        "input_len": bench.input_len,
        "append_len": bench.append_len,
        "output_len": bench.output_len,
        "num_turns": bench.num_turns,
        "num_sessions": bench.num_sessions,
        "shared_prefix_len": bench.shared_prefix_len,
        "ignore_eos": True,
        "chat_template": False,
        "server_argv": server_argv,
        "points": {},
    }
    port = resolve_port(server_argv, server_args)

    for concurrency in concurrencies:
        profile_base = run_dir / f"profile-c{concurrency}.json"
        profile_path = run_dir / f"profile-c{concurrency}.rank0.json"
        def run_server(profile: bool) -> tuple[dict, list[dict]]:
            old_enabled = os.environ.get("LOGSSMS_PROFILE_SCHEDULER")
            old_path = os.environ.get("LOGSSMS_PROFILE_SCHEDULER_PATH")
            os.environ["LOGSSMS_PROFILE_SCHEDULER"] = "1" if profile else "0"
            if profile:
                os.environ["LOGSSMS_PROFILE_SCHEDULER_PATH"] = str(profile_base)
            try:
                proc = launch_server(
                    server_argv, port=port,
                    enable_specualtive_decoding=False, num_spec_tokens=0,
                )
            finally:
                if old_enabled is None:
                    os.environ.pop("LOGSSMS_PROFILE_SCHEDULER", None)
                else:
                    os.environ["LOGSSMS_PROFILE_SCHEDULER"] = old_enabled
                if old_path is None:
                    os.environ.pop("LOGSSMS_PROFILE_SCHEDULER_PATH", None)
                else:
                    os.environ["LOGSSMS_PROFILE_SCHEDULER_PATH"] = old_path
            try:
                base = wait_server_ready(port)
                return asyncio.run(run_point(
                    base, server_args.model_path, sessions,
                    concurrency=concurrency, output_len=bench.output_len,
                    request_timeout=bench.request_timeout, warmup=warmup,
                ))
            finally:
                kill_server(proc)

        profiled_stats, profiled_history = run_server(profile=True)
        if not profiled_stats["chunks_ok"]:
            raise RuntimeError("Profiled output token count differs from requested length")
        cache = attach_cache_matches(profiled_history, profile_path)
        profile_trace_path = run_dir / f"history-profile-c{concurrency}.json"
        save_json(str(profile_trace_path), {"sessions": profiled_history})

        if bench.profile_only:
            stats, history = profiled_stats, profiled_history
        else:
            stats, history = run_server(profile=False)
            if not stats["chunks_ok"]:
                raise RuntimeError("Unprofiled output token count differs from requested length")
        same_trace = input_trace_digest(history) == input_trace_digest(profiled_history)
        stats["cache"] = cache
        stats["profiled_timing"] = bench.profile_only
        stats["profile_vs_timing_input_equal"] = same_trace
        stats["profile_vs_timing_output_equal"] = (
            output_trace_digest(history) == output_trace_digest(profiled_history)
        )
        trace_path = run_dir / f"history-c{concurrency}.json"
        save_json(str(trace_path), {"sessions": history})
        stats["history_path"] = str(trace_path)
        stats["profile_history_path"] = str(profile_trace_path)
        stats["profile_path"] = str(profile_path)
        stats["input_trace_sha256"] = input_trace_digest(history)
        stats["output_trace_sha256"] = output_trace_digest(history)
        if not same_trace:
            print(
                "WARNING: profiled and unprofiled runs generated different input "
                "traces; cache-match figures are not paired with the timing run",
                file=sys.stderr,
            )
        report["points"][str(concurrency)] = stats
        save_json(str(run_dir / "report.json"), report)
        print_stats(
            f"random_multiturn tp={server_args.tp_info.size} "
            f"c={concurrency} turns={bench.num_turns}", stats,
        )
        cache = stats["cache"]
        print(
            f"  reusable: {cache['expected_reusable_tokens']} tok; "
            f"tree match: {cache['tree_match_fraction']:.1%}; "
            f"admitted: {cache['admitted_hit_fraction']:.1%}"
        )
    print(f"Results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
