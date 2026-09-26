"""Shared online throughput-benchmark harness for LogSSMs.

bench_prefill / bench_decode / bench_spec_decode all launch a real LogSSMs HTTP
server as a subprocess, wait for it to be ready, then drive a streaming
``/v1/chat/completions`` workload against it over httpx. The workload code is
byte-identical across tp sizes and MTP on/off -- the ONLY thing that changes is the
server's launch flags (``--tensor-parallel-size`` / ``--speculative-algorithm``) -- so every run
is a controlled-variable comparison.

Server arguments are NOT redefined here: the library's
``logssms.server.args.parse_args`` is the single source of truth. This module only
splits off the bench-only workload flags via ``parse_known_args`` and hands the rest
to the library unchanged.

Run (tp=1 on a single GPU, tp>1 needs N GPUs):
  python benchmarks/throughput/bench_decode.py --model-path <model> --tp-size 2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "python"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
if os.path.isdir("/home/daking/.conda/envs/daking"):
    os.environ.setdefault("CUDA_HOME", "/home/daking/.conda/envs/daking")

from datasets.parser import (
    DatasetMessage,
    DatasetRecord,
    load_dataset,
    resolve_dataset_path as _resolve_dataset_path,
    sample_dataset,
)
from logssms.utils.platform import configure_chat_tokenizer, render_chat_prompt


# -------------------------------------------------------------------------------------
# argument parsing: library parse_args is the single source of truth
# -------------------------------------------------------------------------------------

# Bench-only flags. Every other flag (--model-path, --tp-size, --dtype, --host/--port,
# --page-size, --num-pages, --max-running-requests, --memory-ratio,
# --speculative-num-draft-tokens, ...) is parsed by the library's
# logssms.server.args.parse_args.
BENCH_ONLY_FLAGS: dict[str, dict[str, Any]] = {
    "--num-concurrent-sessions": {
        "type": str,
        "default": None,
        "help": "comma-separated concurrent session counts (default 32)",
    },
    "--input-len": {"type": int, "default": None, "help": "prompt length in tokens"},
    "--output-len": {"type": int, "default": None, "help": "generation length in tokens"},
    "--seed": {"type": int, "default": 0, "help": "RNG seed for prompt generation (A/B uses one batch)"},
    "--dataset": {
        "type": str,
        "default": None,
        "help": (
            "Dataset name, directory, or JSON/JSONL/Parquet path understood by "
            "datasets/parser.py"
        ),
    },
    "--dataset-category": {
        "type": str,
        "default": None,
        "help": "comma-separated dataset categories to include",
    },
    "--num-turns": {
        "type": int,
        "default": 1,
        "help": (
            "number of sequential user turns per concurrent session; values greater "
            "than one enable the multi-turn workload"
        ),
    },
    "--out": {
        "type": str,
        "default": None,
        "help": (
            "incrementally write completed points to this file; relative paths "
            "are under benchmark-results/"
        ),
    },
    "--history": {
        "type": str,
        "default": None,
        "help": (
            "save every measured request's full context and response to this file; "
            "relative paths are under benchmark-results/"
        ),
    },
    "--no-warmup": {"action": "store_true", "help": "skip the warmup request before the timed batch"},
    "--no-pbar": {"action": "store_true", "help": "disable the progress counter"},
    "--request-timeout": {
        "type": float,
        "default": 300.0,
        "help": "fail a request that does not complete within this many seconds",
    },
}


def parse_full(argv: list[str], extra: dict[str, dict[str, Any]] | None = None):
    """Split bench-only flags from server flags, then let the library parse the rest.

    Returns (ServerArgs, bench namespace, leftover server argv). The leftover argv is
    what the library parsed -- the subprocess cmd is rebuilt from it so the bench never
    redefines a server flag.
    """
    flags = dict(BENCH_ONLY_FLAGS)
    flags.update(extra or {})  # scripts override --input-len/--output-len defaults
    p = argparse.ArgumentParser(add_help=False)
    for name, kw in flags.items():
        p.add_argument(name, **kw)
    bench_ns, server_argv = p.parse_known_args(argv)
    if bench_ns.num_turns < 1:
        p.error("--num-turns must be positive")
    bench_ns.out = resolve_result_path(bench_ns.out)
    bench_ns.history = resolve_result_path(bench_ns.history)

    from logssms.server.args import parse_args

    server_args = parse_args(server_argv)
    return server_args, bench_ns, server_argv


def parse_int_list(s: str | None, default: int) -> list[int]:
    """Comma-separated bench flag value -> list of ints (None/empty -> [default])."""
    if s is None:
        return [default]
    vals = [int(x) for x in s.split(",") if x.strip()]
    return vals if vals else [default]


# -------------------------------------------------------------------------------------
# prompt generation (deterministic, seeded)
# -------------------------------------------------------------------------------------

def resolve_dataset_path(dataset: str) -> str:
    return str(_resolve_dataset_path(dataset))


def load_dataset_prompts(
    dataset   : str,
    categories: str | None = None,
) -> list[str]:
    return [record.prompt for record in load_dataset(dataset, categories)]


def _take_dataset_prompts(
    dataset   : str,
    categories: str | None,
    count     : int,
    seed      : int,
) -> list[str]:
    return [
        record.prompt
        for record in sample_dataset(dataset, count, seed, categories)
    ]


def _load_benchmark_tokenizer(model_path: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return configure_chat_tokenizer(tokenizer, model_path)


def _dataset_prompt_len(tokenizer: Any, prompt: str) -> int:
    """Match the chat-template tokenization performed by ``TokenizeManager``."""
    text = render_chat_prompt(tokenizer, [{"role": "user", "content": prompt}])
    return len(tokenizer.encode(text))


def _chat_prompt_len(tokenizer: Any, messages: list[dict[str, Any]] | str) -> int:
    text = (
        messages if isinstance(messages, str) else
        render_chat_prompt(tokenizer, messages)
    )
    return len(tokenizer.encode(text))


def _message_dict(message: DatasetMessage) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role"   : message.role,
        "content": message.content,
    }
    if message.tool_calls is not None:
        result["tool_calls"] = list(message.tool_calls)
    if message.tool_call_id is not None:
        result["tool_call_id"] = message.tool_call_id
    return result


@dataclass(frozen=True)
class BenchmarkConversation:
    uid             : str
    initial_messages: tuple[DatasetMessage, ...]
    user_turns      : tuple[DatasetMessage, ...]


def _conversation_from_record(
    record   : DatasetRecord,
    num_turns: int,
) -> BenchmarkConversation:
    assert record.num_turns >= num_turns
    return BenchmarkConversation(
        uid=record.uid,
        initial_messages=record.initial_messages,
        user_turns=record.user_turns[:num_turns],
    )


def make_conversations(
    model_path        : str,
    input_len         : int,
    num_conversations : int,
    num_turns         : int,
    seed              : int,
    *,
    dataset           : str | None = None,
    dataset_categories: str | None = None,
) -> list[BenchmarkConversation]:
    """Build deterministic sessions for the closed-loop multi-turn workload."""
    if num_turns < 1:
        raise ValueError(f"num_turns must be positive, got {num_turns}")
    if dataset is not None:
        records = sample_dataset(
            dataset,
            num_conversations,
            seed,
            dataset_categories,
            min_turns=num_turns,
        )
        return [
            _conversation_from_record(record, num_turns)
            for record in records
        ]

    tokenizer = _load_benchmark_tokenizer(model_path)
    random.seed(seed)
    return [
        BenchmarkConversation(
            uid=f"random-{index}",
            initial_messages=(),
            user_turns=tuple(
                DatasetMessage(role="user", content=generate_prompt(tokenizer, input_len))
                for _ in range(num_turns)
            ),
        )
        for index in range(num_conversations)
    ]


def generate_prompt(tokenizer: Any, n: int) -> str:
    """Generate a prompt of approximately `n` tokens via decode/encode round-trip."""
    vocab_size = tokenizer.vocab_size // 2
    token_ids = [random.randint(0, vocab_size) for _ in range(n)]
    for _ in range(64):
        prompt = tokenizer.decode(token_ids)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) == n:
            return prompt
        if len(token_ids) < n:
            token_ids.extend([random.randint(0, vocab_size) for _ in range(n - len(token_ids))])
        else:
            token_ids = token_ids[:n]
    raise ValueError("Failed to generate a message of the desired length.")


def make_prompts(
    model_path        : str,
    input_len         : int,
    num_prompts       : int,
    seed              : int,
    *,
    dataset           : str | None = None,
    dataset_categories: str | None = None,
):
    """One seeded batch of prompts + their exact token lengths. Reused across A/B runs."""
    tokenizer = _load_benchmark_tokenizer(model_path)
    if dataset is not None:
        prompts = _take_dataset_prompts(
            dataset, dataset_categories, num_prompts, seed
        )
    else:
        random.seed(seed)
        prompts = [generate_prompt(tokenizer, input_len) for _ in range(num_prompts)]
    lens = (
        [_dataset_prompt_len(tokenizer, prompt) for prompt in prompts]
        if dataset is not None else [input_len] * num_prompts
    )
    return prompts, lens


def make_mixed_prompts(
    model_path        : str,
    segments          : list[tuple[int, int]],
    seed              : int,
    *,
    dataset           : str | None = None,
    dataset_categories: str | None = None,
):
    """One seeded mixed-length batch. ``segments`` is [(input_len, count), ...]; returns
    (prompts, lens) in segment order so group indices are just running offsets."""
    tokenizer = _load_benchmark_tokenizer(model_path)
    if dataset is not None:
        count = sum(count for _, count in segments)
        dataset_prompts = load_dataset_prompts(dataset, dataset_categories)
        if count > len(dataset_prompts):
            raise ValueError(
                f"Requested {count} prompts, but dataset selection only contains "
                f"{len(dataset_prompts)}"
            )
        random.Random(seed).shuffle(dataset_prompts)
        candidates = [
            (prompt, _dataset_prompt_len(tokenizer, prompt))
            for prompt in dataset_prompts
        ]
        prompts, lens = [], []
        for target_len, group_size in segments:
            candidates.sort(key=lambda item: abs(item[1] - target_len))
            selected, candidates = candidates[:group_size], candidates[group_size:]
            prompts.extend(prompt for prompt, _ in selected)
            lens.extend(length for _, length in selected)
    else:
        random.seed(seed)
        prompts, lens = [], []
        for in_len, count in segments:
            for _ in range(count):
                prompt = generate_prompt(tokenizer, in_len)
                prompts.append(prompt)
                lens.append(len(tokenizer.encode(prompt, add_special_tokens=False)))
    return prompts, lens


def input_label(dataset: str | None, input_lens: list[int], fallback: int) -> str:
    if dataset is None:
        return str(fallback)
    mean_len = sum(input_lens) / len(input_lens) if input_lens else 0
    return f"dataset(avg={mean_len:.0f})"


def split_warmup_prompt(
    prompts   : list[str],
    input_lens: list[int],
    warmup    : bool,
) -> tuple[str | None, list[str], list[int]]:
    """Remove one distinct prompt from the measured batch for warmup."""
    if not warmup:
        return None, prompts, input_lens
    assert prompts, "warmup requires one extra prompt"
    return prompts[-1], prompts[:-1], input_lens[:-1]


def split_warmup_conversation(
    conversations: list[BenchmarkConversation],
    warmup      : bool,
) -> tuple[BenchmarkConversation | None, list[BenchmarkConversation]]:
    if not warmup:
        return None, conversations
    assert conversations, "warmup requires one extra conversation"
    return conversations[-1], conversations[:-1]


def resolve_result_path(path: str | None) -> str | None:
    if path is None:
        return None
    if not path.endswith(".json"):
        raise ValueError("--out and --history require an explicit .json file path")
    return path if os.path.isabs(path) else os.path.join(_REPO, "benchmark-results", path)


def save_json(path: str | None, data: Any) -> None:
    """Atomically persist every completed point so an interrupted matrix is useful."""
    resolved = resolve_result_path(path)
    if resolved is None:
        return
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    tmp_path = resolved + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, resolved)
    print(f"  wrote {resolved}", flush=True)


def prepare_history(
    benchmark  : str,
    server_args: Any,
    bench      : Any,
) -> tuple[str | None, dict[str, Any] | None]:
    if bench.history is None:
        return None, None
    path = resolve_result_path(bench.history)
    return path, {
        "benchmark": benchmark,
        "model": server_args.model_path,
        "cache_type": server_args.cache_type,
        "cache_config": repr(server_args.cache_config),
        "dataset": bench.dataset,
        "dataset_category": bench.dataset_category,
        "seed": bench.seed,
        "num_turns": bench.num_turns,
        "output_len": bench.output_len,
        "ignore_eos": getattr(bench, "ignore_eos", True),
        "history_mode": getattr(bench, "history_mode", "generated"),
        "fixed_history_tokens": getattr(bench, "fixed_history_tokens", None),
        "points": {},
    }


# -------------------------------------------------------------------------------------
# server lifecycle: launch / ready / kill
# -------------------------------------------------------------------------------------

def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def resolve_port(server_argv: list[str], server_args: Any) -> int:
    """Honor an explicit --port; otherwise pick a free one."""
    if any(a == "--port" or a.startswith("--port=") for a in server_argv):
        return server_args.server_port
    return get_free_port()


_MTP_VALUE_FLAGS = (
    "--port", "--speculative-algorithm",
    "--speculative-num-draft-tokens", "--speculative-draft-model-path",
)
_MTP_BOOL_FLAGS = ("--enable-dt-separation",)


def _flag_value(argv: list[str], name: str) -> str | None:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def resolve_num_spec_tokens(server_argv: list[str]) -> int:
    from logssms.speculator.drafters.mtp import MTPSpeculatorConfig

    value = _flag_value(server_argv, "--speculative-num-draft-tokens")
    return int(value) if value is not None else MTPSpeculatorConfig.num_draft_tokens


def make_server_cmd(server_argv: list[str], *, port: int, enable_specualtive_decoding: bool, num_spec_tokens: int, enable_dt: bool = False) -> list[str]:
    """Rebuild the child cmd from the library-parsed argv, pinning what the bench controls.

    --port and every speculative/MTP flag are stripped and set deterministically so tp /
    MTP / DT comparisons differ by exactly the flag under test.
    """
    argv: list[str] = []
    i = 0
    while i < len(server_argv):
        a = server_argv[i]
        if a in _MTP_VALUE_FLAGS:
            i += 2  # skip flag + value
            continue
        if a in _MTP_BOOL_FLAGS:
            i += 1  # store_true
            continue
        if any(a.startswith(f + "=") for f in _MTP_VALUE_FLAGS + _MTP_BOOL_FLAGS):
            i += 1
            continue
        argv.append(a)
        i += 1
    argv += ["--port", str(port)]
    if enable_specualtive_decoding:
        model_path = (
            _flag_value(server_argv, "--model-path")
            or _flag_value(server_argv, "--model")
        )
        assert model_path, "MTP bench needs --model-path/--model"
        argv += ["--speculative-algorithm", "MTP",
                 "--speculative-num-draft-tokens", str(num_spec_tokens),
                 "--speculative-draft-model-path", model_path]
        if enable_dt:
            argv += ["--enable-dt-separation"]
    return [sys.executable, "-m", "logssms"] + argv


def launch_server(server_argv: list[str], *, port: int, enable_specualtive_decoding: bool, num_spec_tokens: int, enable_dt: bool = False):
    """Spawn the LogSSMs server in its own process group. Returns (Popen, log path)."""
    cmd = make_server_cmd(
        server_argv,
        port=port,
        enable_specualtive_decoding=enable_specualtive_decoding,
        num_spec_tokens=num_spec_tokens,
        enable_dt=enable_dt,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(_REPO, "python") + os.pathsep + env.get("PYTHONPATH", "")
    log_path = f"/tmp/logssms_bench_{port}.log"
    print(f"  launching: {' '.join(cmd)}", flush=True)
    print(f"  server log: {log_path}", flush=True)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd,
            env=env,
            start_new_session=True,  # own process group -> killpg teardown hits every worker
            stdout=log,
            stderr=log,
        )
    return proc


def wait_server_ready(port: int, timeout: float = 900.0) -> str:
    """Poll /v1/models. The server binds its port only after every worker has acked, so
    a 200 means the model is loaded and all ranks are synced."""
    import httpx

    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + timeout
    last_report = 0.0
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=5, trust_env=False) as c:
                if c.get(f"{base}/v1/models").status_code == 200:
                    return base
        except Exception:  # noqa: BLE001 -- server not up yet
            pass
        if time.time() - last_report >= 10:
            print(f"  waiting for server on :{port} ... {int(time.time() - deadline + timeout)}s", flush=True)
            last_report = time.time()
        time.sleep(2)
    raise TimeoutError(f"server on :{port} not ready within {timeout}s")


def kill_server(proc: subprocess.Popen) -> None:
    if proc is None or proc.poll() is not None:
        return
    pgid = os.getpgid(proc.pid)
    try:
        # The scheduler writes its opt-in profile in shutdown().  SIGTERM skips
        # that path; SIGINT is handled cooperatively by every TP rank.
        os.killpg(pgid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)


# -------------------------------------------------------------------------------------
# load driver (httpx streaming) + metrics
# -------------------------------------------------------------------------------------

class _Progress:
    def __init__(self, total: int, on: bool):
        self.total, self.done, self.on = total, 0, on

    def tick(self) -> None:
        self.done += 1
        if self.on:
            print(f"\r  requests done: {self.done}/{self.total}", end="", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class StreamResult:
    tics                   : list[float]
    output                 : str
    nonempty_content_chunks: int = 0


async def stream_messages(
    client    : Any,
    base      : str,
    model     : str,
    messages  : list[dict[str, Any]] | str,
    output_len: int,
    progress  : _Progress | None = None,
    profile_tag: str | None = None,
    ignore_eos: bool = True,
) -> StreamResult:
    """Stream one chat-completions request and retain its assistant text.

    Every ``data: {json}`` line is exactly one detokenizer ack (one token per ack on the
    non-MTP path, one verify round per ack on the MTP path). ``data: [DONE]`` is the
    terminator. Its arrival timestamp is retained as the request completion boundary.
    ``top_k=1 + temperature=0`` makes requests greedy; ``ignore_eos`` controls whether
    they must consume the full ``max_tokens`` budget.
    """
    payload = {
        "model": model,
        ("prompt" if isinstance(messages, str) else "messages"): messages,
        "max_tokens": output_len,
        "temperature": 0.0,
        "top_k": 1,
        "stream": True,
        "ignore_eos": ignore_eos,
    }
    headers = (
        {"X-LogSSMs-Profile-Tag": profile_tag}
        if profile_tag is not None else None
    )
    tics = [time.perf_counter()]
    output = []
    nonempty_content_chunks = 0
    async with client.stream(
        "POST", f"{base}/v1/chat/completions", json=payload, headers=headers
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                tics.append(time.perf_counter())
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"]
            if content := delta.get("content"):
                output.append(content)
                nonempty_content_chunks += 1
            tics.append(time.perf_counter())
    if progress is not None:
        progress.tick()
    return StreamResult(
        tics=tics,
        output="".join(output),
        nonempty_content_chunks=nonempty_content_chunks,
    )


async def stream_one(
    client    : Any,
    base      : str,
    model     : str,
    prompt    : str,
    output_len: int,
    progress  : _Progress | None = None,
    ignore_eos: bool = True,
) -> list[float]:
    result = await stream_messages(
        client,
        base,
        model,
        [{"role": "user", "content": prompt}],
        output_len,
        progress,
        ignore_eos=ignore_eos,
    )
    return result.tics


async def _run_conversation(
    client         : Any,
    base           : str,
    model          : str,
    conversation   : BenchmarkConversation,
    output_len     : int,
    progress       : _Progress | None,
    request_timeout: float | None = None,
    fixed_assistant: str | None = None,
    prefix_tokenizer: Any | None = None,
    ignore_eos: bool = True,
) -> list[tuple[StreamResult, list[dict[str, Any]] | str]]:
    history = [_message_dict(message) for message in conversation.initial_messages]
    results = []
    next_prefix_prompt = None
    for turn_index, user_turn in enumerate(conversation.user_turns):
        if prefix_tokenizer is None:
            history.append(_message_dict(user_turn))
            request_messages = [dict(message) for message in history]
        else:
            request_messages = next_prefix_prompt
            if request_messages is None:
                request_messages = render_chat_prompt(
                    prefix_tokenizer, history + [_message_dict(user_turn)]
                )
        try:
            result = await asyncio.wait_for(
                stream_messages(
                    client,
                    base,
                    model,
                    request_messages,
                    output_len,
                    progress,
                    profile_tag=f"{conversation.uid}/turn-{turn_index + 1}",
                    ignore_eos=ignore_eos,
                ),
                timeout=request_timeout,
            )
        except TimeoutError as error:
            raise RuntimeError(
                f"Conversation {conversation.uid!r} turn {turn_index + 1} "
                f"timed out after {request_timeout}s"
            ) from error
        results.append((result, request_messages))
        assistant = result.output if fixed_assistant is None else fixed_assistant
        if prefix_tokenizer is None:
            history.append({"role": "assistant", "content": assistant})
        elif turn_index + 1 < len(conversation.user_turns):
            next_user = _message_dict(conversation.user_turns[turn_index + 1])
            next_prefix_prompt = (
                request_messages + assistant + prefix_tokenizer.eos_token + "\n"
                + render_chat_prompt(prefix_tokenizer, [next_user])
            )
            old_ids = prefix_tokenizer.encode(request_messages)
            new_ids = prefix_tokenizer.encode(next_prefix_prompt)
            if new_ids[:len(old_ids)] != old_ids:
                raise AssertionError(
                    f"Raw prefix is not token-stable for {conversation.uid} "
                    f"at turn {turn_index + 1}"
                )
    return results


def _fixed_assistant_reply(tokenizer: Any, output_len: int) -> str:
    """A deterministic prior reply for an identical-input cache-policy A/B trace."""
    sentence = "The assistant summarizes this turn in a stable and reproducible way. "
    tokens = tokenizer.encode(sentence, add_special_tokens=False)
    assert tokens
    repeated = (tokens * ((output_len + len(tokens) - 1) // len(tokens)))[:output_len]
    return tokenizer.decode(repeated, skip_special_tokens=False)


async def run_multiturn_bench(
    base               : str,
    model              : str,
    conversations      : list[BenchmarkConversation],
    output_len         : int,
    *,
    concurrency        : int | None = None,
    mtp                : bool = False,
    warmup_conversation: BenchmarkConversation | None = None,
    pbar               : bool = True,
    request_timeout     : float | None = 300.0,
    history_records     : list[dict[str, Any]] | None = None,
    history_mode        : str = "generated",
    fixed_history_tokens: int | None = None,
    ignore_eos          : bool = True,
) -> dict:
    """Run closed-loop sessions with a bounded number of concurrent sessions."""
    import httpx
    if not conversations:
        raise ValueError("multi-turn benchmark requires at least one conversation")
    if concurrency is None:
        concurrency = len(conversations)
    if concurrency < 1:
        raise ValueError(f"concurrency must be positive, got {concurrency}")
    concurrency = min(concurrency, len(conversations))
    tokenizer = _load_benchmark_tokenizer(model)
    if history_mode not in ("generated", "fixed", "prefix_generated", "prefix_fixed"):
        raise ValueError(f"Unsupported history mode: {history_mode}")
    if fixed_history_tokens is not None and (
        history_mode not in ("fixed", "prefix_fixed") or fixed_history_tokens < 1
    ):
        raise ValueError("fixed_history_tokens requires fixed mode and a positive length")
    fixed_assistant = (
        _fixed_assistant_reply(tokenizer, fixed_history_tokens or output_len)
        if history_mode in ("fixed", "prefix_fixed") else None
    )
    prefix_tokenizer = tokenizer if history_mode.startswith("prefix_") else None
    if prefix_tokenizer is not None and tokenizer.eos_token != "<|im_end|>":
        raise ValueError("prefix history mode currently requires a ChatML tokenizer")
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30.0, read=None, write=None, pool=None),
        limits=httpx.Limits(max_connections=1024, max_keepalive_connections=128),
        trust_env=False,
    )
    try:
        if warmup_conversation is not None:
            await _run_conversation(
                client,
                base,
                model,
                warmup_conversation,
                min(8, output_len),
                None,
                request_timeout,
                fixed_assistant,
                prefix_tokenizer,
                ignore_eos,
            )
        num_turns = len(conversations[0].user_turns) if conversations else 0
        assert all(
            len(conversation.user_turns) == num_turns
            for conversation in conversations
        )
        progress = _Progress(
            len(conversations) * num_turns,
            on=pbar and len(conversations) * num_turns > 1,
        )
        queue = asyncio.Queue()
        for index, conversation in enumerate(conversations):
            queue.put_nowait((index, conversation))
        session_results: list[
            list[tuple[StreamResult, list[dict[str, Any]] | str]] | None
        ] = [None] * len(conversations)

        async def worker() -> None:
            while not queue.empty():
                index, conversation = queue.get_nowait()
                session_results[index] = await _run_conversation(
                    client,
                    base,
                    model,
                    conversation,
                    output_len,
                    progress,
                    request_timeout,
                    fixed_assistant,
                    prefix_tokenizer,
                    ignore_eos,
                )

        await asyncio.gather(*[worker() for _ in range(concurrency)])
        assert all(result is not None for result in session_results)
        completed_results = [
            result for result in session_results if result is not None
        ]
    finally:
        await client.aclose()
    if progress.on:
        print(file=sys.stderr, flush=True)

    flat = [request for session in completed_results for request in session]
    tics_all = [result.tics for result, _messages in flat]
    input_lengths = [
        _chat_prompt_len(tokenizer, messages)
        for _result, messages in flat
    ]
    output_lengths = [output_len] * len(flat)
    summary = summarize(tics_all, input_lengths, output_lengths, mtp=mtp, ignore_eos=ignore_eos)
    summary["sessions"] = len(conversations)
    summary["concurrency"] = concurrency
    summary["num_turns"] = num_turns
    summary["history_mode"] = history_mode
    summary["fixed_history_tokens"] = fixed_history_tokens
    summary["workload_uids"] = [
        conversation.uid for conversation in conversations
    ]
    summary["warmup_uid"] = (
        warmup_conversation.uid if warmup_conversation is not None else None
    )
    summary["session_per_s"] = (
        len(conversations) / summary["duration_s"]
        if summary["duration_s"] > 0 else 0.0
    )
    summary["turn_stats"] = []
    for turn in range(num_turns):
        turn_requests = [session[turn] for session in completed_results]
        turn_tics = [result.tics for result, _messages in turn_requests]
        turn_input_lengths = [
            _chat_prompt_len(tokenizer, messages)
            for _result, messages in turn_requests
        ]
        turn_stats = summarize(
            turn_tics,
            turn_input_lengths,
            [output_len] * len(turn_requests),
            mtp=mtp,
            ignore_eos=ignore_eos,
        )
        summary["turn_stats"].append({"turn": turn + 1, **turn_stats})
    if history_records is not None:
        for conversation, session in zip(conversations, completed_results, strict=True):
            history_records.append({
                "uid": conversation.uid,
                "turns": [
                    {
                        "turn": turn + 1,
                        "input_tokens": _chat_prompt_len(tokenizer, messages),
                        "messages": messages,
                        "response": result.output,
                        "response_chars": len(result.output),
                        "response_tokens": len(tokenizer.encode(
                            result.output, add_special_tokens=False
                        )),
                        "content_chunks": max(len(result.tics) - 3, 0),
                        "nonempty_content_chunks": result.nonempty_content_chunks,
                    }
                    for turn, (result, messages) in enumerate(session)
                ],
            })
    return summary


async def run_benchmark_point(
    base               : str,
    model              : str,
    input_len          : int,
    output_len         : int,
    concurrency        : int,
    seed               : int,
    *,
    num_turns          : int = 1,
    dataset            : str | None = None,
    dataset_categories : str | None = None,
    mtp                : bool = False,
    warmup             : bool = True,
    pbar               : bool = True,
    history_records    : list[dict[str, Any]] | None = None,
    ignore_eos         : bool = True,
) -> tuple[dict, list[int]]:
    """Construct and execute one single- or multi-turn benchmark point."""
    if num_turns < 1:
        raise ValueError(f"num_turns must be positive, got {num_turns}")
    if concurrency < 1:
        raise ValueError(f"concurrency must be positive, got {concurrency}")
    if num_turns > 1:
        conversations = make_conversations(
            model,
            input_len,
            concurrency + int(warmup),
            num_turns,
            seed,
            dataset=dataset,
            dataset_categories=dataset_categories,
        )
        warmup_conversation, conversations = split_warmup_conversation(
            conversations, warmup
        )
        stats = await run_multiturn_bench(
            base,
            model,
            conversations,
            output_len,
            mtp=mtp,
            warmup_conversation=warmup_conversation,
            pbar=pbar,
            history_records=history_records,
            ignore_eos=ignore_eos,
        )
        return stats, [round(stats["in_tokens"] / max(stats["n"], 1))]

    prompts, input_lengths = make_prompts(
        model,
        input_len,
        concurrency + int(warmup),
        seed,
        dataset=dataset,
        dataset_categories=dataset_categories,
    )
    warmup_prompt, prompts, input_lengths = split_warmup_prompt(
        prompts, input_lengths, warmup
    )
    stats = await run_online_bench(
        base,
        model,
        prompts,
        input_lengths,
        [output_len] * concurrency,
        mtp=mtp,
        warmup_prompt=warmup_prompt,
        pbar=pbar,
        history_records=history_records,
        ignore_eos=ignore_eos,
    )
    return stats, input_lengths


async def run_online_bench_tics(
    base: str,
    model: str,
    prompts: list[str],
    input_lengths: list[int],
    output_lengths: list[int],
    *,
    mtp: bool = False,
    warmup_prompt: str | None = None,
    pbar: bool = True,
    history_records: list[dict[str, Any]] | None = None,
    ignore_eos: bool = True,
) -> tuple[dict, list[list[float]]]:
    """Drive the workload, returning (summary, per-request tics) for split reporting."""
    import httpx

    # bounded connect so a dead server fails fast; unbounded read for long streams
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30.0, read=None, write=None, pool=None),
        limits=httpx.Limits(max_connections=1024, max_keepalive_connections=128),
        trust_env=False,
    )
    try:
        if warmup_prompt is not None:
            # flush Triton autotune / CUDA graph capture / lazy detokenizer init
            await stream_one(
                client, base, model, warmup_prompt, min(8, output_lengths[0]),
                ignore_eos=ignore_eos,
            )
        progress = _Progress(len(prompts), on=pbar and len(prompts) > 1)
        if history_records is None:
            tics_all = await asyncio.gather(*[
                stream_one(client, base, model, p, o, progress, ignore_eos=ignore_eos)
                for p, o in zip(prompts, output_lengths, strict=True)
            ])
        else:
            tokenizer = _load_benchmark_tokenizer(model)
            responses = await asyncio.gather(*[
                stream_messages(
                    client, base, model, [{"role": "user", "content": prompt}],
                    output_len, progress, ignore_eos=ignore_eos,
                )
                for prompt, output_len in zip(prompts, output_lengths, strict=True)
            ])
            tics_all = [response.tics for response in responses]
            for index, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
                history_records.append({
                    "uid": f"request-{index}",
                    "turns": [{
                        "turn": 1,
                        "input_tokens": _chat_prompt_len(
                            tokenizer, [{"role": "user", "content": prompt}]
                        ),
                        "messages": [{"role": "user", "content": prompt}],
                        "response": response.output,
                        "response_chars": len(response.output),
                        "response_tokens": len(tokenizer.encode(
                            response.output, add_special_tokens=False
                        )),
                        "content_chunks": max(len(response.tics) - 3, 0),
                        "nonempty_content_chunks": response.nonempty_content_chunks,
                    }],
                })
    finally:
        await client.aclose()
    if progress.on:
        print(file=sys.stderr, flush=True)
    return summarize(tics_all, input_lengths, output_lengths, mtp=mtp, ignore_eos=ignore_eos), tics_all


async def run_online_bench(
    base: str,
    model: str,
    prompts: list[str],
    input_lengths: list[int],
    output_lengths: list[int],
    *,
    mtp: bool = False,
    warmup_prompt: str | None = None,
    pbar: bool = True,
    history_records: list[dict[str, Any]] | None = None,
    ignore_eos: bool = True,
) -> dict:
    summary, _ = await run_online_bench_tics(
        base, model, prompts, input_lengths, output_lengths,
        mtp=mtp, warmup_prompt=warmup_prompt, pbar=pbar,
        history_records=history_records, ignore_eos=ignore_eos,
    )
    return summary


def summarize_group(
    tics_all: list[list[float]],
    input_lengths: list[int],
    output_lengths: list[int],
    indices: list[int],
    *,
    mtp: bool = False,
) -> dict:
    """Summarize a subset of requests (by index) -- short/long split reporting."""
    sub = [tics_all[i] for i in indices]
    return summarize(
        sub,
        [input_lengths[i] for i in indices],
        [output_lengths[i] for i in indices],
        mtp=mtp,
    )


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def _stats(xs: list[float], scale: float = 1000.0) -> list[float]:
    """[avg, p50, p90, p99, max] * scale (ms by default)."""
    if not xs:
        return [0.0] * 5
    return [
        scale * sum(xs) / len(xs),
        scale * _pct(xs, 0.5),
        scale * _pct(xs, 0.9),
        scale * _pct(xs, 0.99),
        scale * max(xs),
    ]


def summarize(
    tics_all: list[list[float]],
    input_lengths: list[int],
    output_lengths: list[int],
    *,
    mtp: bool = False,
    ignore_eos: bool = True,
) -> dict:
    n = len(tics_all)
    in_tokens = sum(input_lengths)
    requested = sum(output_lengths)
    # Each trace is [request start, content chunks..., finish_reason, DONE]. The two
    # trailing protocol chunks are not model output: AR has one content chunk per token;
    # MTP has one prefill-token chunk followed by one chunk per verify round.
    content_chunks = [max(len(t) - 3, 0) for t in tics_all]
    # Actual generated token count:
    #  - MTP: each verify round commits num_sampled tokens; the server delivers the full
    #    max_tokens budget. Verified on Qwen3.5: max_tokens=8/16 -> 8/16 tokens.
    #  - non-MTP: one content chunk == one token, so ground truth == content chunks. The
    #    scheduler delivers the full budget (DecodeManager snapshots `finished` at
    #    complete_n time instead of reading the pipeline-advanced device_len).
    if mtp:
        out_tokens = requested
        # The first content chunk of each request is sampled by prefill. Every later
        # chunk is one verify round containing one mandatory target token plus zero or
        # more accepted drafts.
        verify_rounds = sum(max(chunks - 1, 0) for chunks in content_chunks)
        verify_tokens = requested - sum(min(length, 1) for length in output_lengths)
        avg_accept = (
            verify_tokens / verify_rounds - 1 if verify_rounds > 0 else 0.0
        )
        chunks_ok = all(
            chunks >= min(length, 1)
            for chunks, length in zip(content_chunks, output_lengths, strict=True)
        ) and 0 <= verify_rounds <= verify_tokens
    else:
        verify_rounds = 0
        out_tokens = sum(content_chunks)
        avg_accept = 0.0  # one chunk == one token; avg_accept is MTP-only
        chunks_ok = (
            out_tokens == requested if ignore_eos else
            all(1 <= chunks <= length for chunks, length in zip(content_chunks, output_lengths, strict=True))
        )

    start = min(t[0] for t in tics_all)
    end = max(t[-1] for t in tics_all)
    dur = end - start
    first_times = [t[1] - t[0] for t in tics_all]
    e2e_times = [t[-1] - t[0] for t in tics_all]
    ttft_ms = _stats(first_times)
    # Match SGLang/vLLM serving semantics exactly: TPOT=(E2E-TTFT)/(O-1), where E2E
    # ends when [DONE] arrives. This remains well-defined when one MTP chunk delivers
    # several committed tokens at once.
    tpot_times = [
        (t[-1] - t[1]) / (tokens - 1) if tokens > 1 else 0.0
        for t, tokens in zip(tics_all, content_chunks if not mtp else output_lengths, strict=True)
    ]
    tpot_ms = _stats(tpot_times)
    # HTTP timestamps cannot isolate a pure decode phase under continuous batching.
    # Report client-observed output throughput over the complete batch makespan.
    output_tok_per_s = out_tokens / dur if dur > 0 else 0.0
    return {
        "n": n,
        "ignore_eos": ignore_eos,
        "in_tokens": in_tokens,
        "requested_out": requested,
        "early_stops": (
            sum(chunks < length for chunks, length in zip(content_chunks, output_lengths, strict=True))
            if not mtp else 0
        ),
        "out_tokens": out_tokens,
        "duration_s": dur,
        "prefill_tok_per_s": in_tokens / max(first_times) if first_times else 0.0,
        "output_tok_per_s": output_tok_per_s,
        # Compatibility alias for existing result readers. This is serving output
        # throughput, not an isolated decode-kernel throughput.
        "decode_tok_per_s": output_tok_per_s,
        "req_per_s": n / dur if dur > 0 else 0.0,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_s": _stats(e2e_times, scale=1.0),
        "rounds": verify_rounds,
        "avg_accept": avg_accept,
        "is_mtp": mtp,
        "chunks_ok": chunks_ok,
    }


# -------------------------------------------------------------------------------------
# reporting
# -------------------------------------------------------------------------------------

def _fmt_ms(x: float) -> str:
    if x >= 1000:
        return f"{int(x):>7}"
    if x >= 10:
        return f"{x:>7.2f}"
    return f"{x:>7.3f}"


def print_stats(title: str, s: dict, *, note: str = "") -> None:
    t, p, e = s["ttft_ms"], s["tpot_ms"], s["e2e_s"]
    short = "" if s["out_tokens"] == s["requested_out"] else f"  (delivered {s['out_tokens']} of {s['requested_out']})"
    unit = "requests" if "sessions" in s else "prompts"
    print("=" * 64)
    print(f"  {title}{'  [' + note + ']' if note else ''}")
    print(f"  {unit}: {s['n']}  input: {s['in_tokens']} tok  output: {s['out_tokens']} tok"
          f"{short}  duration: {s['duration_s']:.2f}s")
    print(f"  prefill: {s['prefill_tok_per_s']:10.1f} input-tok/s")
    print(f"  output:  {s['output_tok_per_s']:10.1f} output-tok/s   {s['req_per_s']:8.2f} req/s")
    print(f"  TTFT(ms): avg {_fmt_ms(t[0])}  p50 {_fmt_ms(t[1])}  p90 {_fmt_ms(t[2])}  "
          f"p99 {_fmt_ms(t[3])}  max {_fmt_ms(t[4])}")
    print(f"  TPOT(ms): avg {_fmt_ms(p[0])}  p50 {_fmt_ms(p[1])}  p90 {_fmt_ms(p[2])}  "
          f"p99 {_fmt_ms(p[3])}  max {_fmt_ms(p[4])}")
    print(f"  E2E(s):   avg {e[0]:>7.3f}  p50 {e[1]:>7.3f}  p90 {e[2]:>7.3f}  "
          f"p99 {e[3]:>7.3f}  max {e[4]:>7.3f}")
    if s.get("is_mtp"):
        print(f"  verify rounds: {s['rounds']}  avg_accept: {s['avg_accept']:.2f} drafts/round")
    if "sessions" in s:
        print(
            f"  sessions: {s['sessions']}  concurrency: {s['concurrency']}  "
            f"turns/session: {s['num_turns']}  "
            f"session/s: {s['session_per_s']:.2f}"
        )
        print(f"  {'turn':>6} {'input':>10} {'output':>10} {'TTFT p50':>10} {'E2E p50':>10}")
        for turn in s["turn_stats"]:
            print(
                f"  {turn['turn']:>6} {turn['in_tokens']:>10} "
                f"{turn['out_tokens']:>10} {turn['ttft_ms'][1]:>10.2f} "
                f"{turn['e2e_s'][1]:>10.3f}"
            )
    print("=" * 64)


def print_conc_summary(rows: list[tuple[int, dict]]) -> None:
    print("=" * 66)
    print("  concurrency sweep")
    print(f"  {'conc':>6} {'prefill':>10} {'output':>10} {'req/s':>8} "
          f"{'TTFT p50':>10} {'TPOT p50':>10}")
    for conc, s in rows:
        print(f"  {conc:>6} {s['prefill_tok_per_s']:>10.1f} {s['decode_tok_per_s']:>10.1f} "
              f"{s['req_per_s']:>8.2f} {s['ttft_ms'][1]:>10.2f} {s['tpot_ms'][1]:>10.2f}")
    print("=" * 66)


def print_compare(a: dict, b: dict) -> None:
    print("=" * 64)
    print(f"  {'':>18} {'non-mtp':>12} {'mtp':>12} {'speedup':>10}")
    for key, name in [
        ("output_tok_per_s", "output tok/s"),
        ("req_per_s", "req/s"),
    ]:
        va, vb = a[key], b[key]
        print(f"  {name:>18} {va:12.1f} {vb:12.1f} {(vb / va if va > 0 else 0):>10.2f}")
    print(f"  {'avg_accept':>18} {'—':>12} {b['avg_accept']:12.2f}")
    print(f"  {'TTFT p50(ms)':>18} {a['ttft_ms'][1]:12.2f} {b['ttft_ms'][1]:12.2f}")
    print(f"  {'TPOT p50(ms)':>18} {a['tpot_ms'][1]:12.2f} {b['tpot_ms'][1]:12.2f}")
    print(f"  {'E2E p50(s)':>18} {a['e2e_s'][1]:12.3f} {b['e2e_s'][1]:12.3f}")
    print("=" * 64)
