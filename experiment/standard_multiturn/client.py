"""Token-exact OASST2 tree client with randomized topological priorities."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import heapq
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets.parser import iter_dataset
from experiment.random_multiturn.client import (
    stream_token_ids,
    summarize_response_cache,
)
from experiment.random_multiturn.metrics import save_json, summarize


_ROOT = Path(__file__).resolve().parents[2]
_MARKER = "LOGSSMS_ASSISTANT_9cc7c282a4104ebd"


@dataclass(frozen=True)
class Message:
    uid      : str
    parent   : str | None
    role     : str
    text     : str
    deleted  : bool = False


@dataclass(frozen=True)
class Tree:
    uid      : str
    language : str | None
    messages : tuple[Message, ...]


@dataclass(frozen=True)
class Request:
    uid         : str
    tree_id     : str
    user_id     : str
    assistant_id: str | None
    parent      : str | None
    depth       : int
    kind        : str
    suffix_ids  : tuple[int, ...]
    closing_id  : int | None
    user_fork   : bool = False
    assistant_alternative: bool = False


def reconstruct_trees(records: Any) -> list[Tree]:
    trees: dict[str, dict[str, Message]] = {}
    languages = {}
    for record in records:
        meta = record.metadata
        tree_id = meta.get("message_tree_id")
        if tree_id is None:
            raise ValueError("standard_multiturn requires explicit OASST2 parent IDs")
        nodes = trees.setdefault(tree_id, {})
        languages[tree_id] = record.category
        for uid, parent, message, info in zip(
            meta["message_ids"], meta["parent_ids"], record.messages,
            meta["node_metadata"], strict=True,
        ):
            node = Message(uid, parent, message.role, message.content, bool(info.get("deleted")))
            if uid in nodes and nodes[uid] != node:
                raise ValueError(f"Conflicting OASST2 node {uid!r}")
            nodes[uid] = node
    result = []
    for uid, nodes in sorted(trees.items()):
        if uid not in nodes or nodes[uid].parent is not None:
            raise ValueError(f"Missing root for tree {uid!r}")
        children: dict[str | None, list[Message]] = {}
        for node in nodes.values():
            if node.parent is not None and node.parent not in nodes:
                raise ValueError(f"Missing parent for node {node.uid!r}")
            children.setdefault(node.parent, []).append(node)
        queue = [nodes[uid]]
        surviving = []
        seen = set()
        while queue:
            node = queue.pop()
            if node.uid in seen:
                raise ValueError(f"Cycle in tree {uid!r}")
            seen.add(node.uid)
            if node.deleted:
                continue
            expected = "user" if node.parent is None else (
                "assistant" if nodes[node.parent].role == "user" else "user"
            )
            if node.role != expected:
                raise ValueError(f"Invalid role for node {node.uid!r}")
            surviving.append(node)
            queue.extend(sorted(children.get(node.uid, ()), key=lambda n: n.uid, reverse=True))
        if surviving:
            result.append(Tree(uid, languages[uid], tuple(surviving)))
    return result


class ChatProtocol:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.kwargs = {
            "enable_thinking": False, "preserve_thinking": True,
            "date_string": "01 Jan 2024",
        }

    def render(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self.kwargs,
        )

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(self.tokenizer.encode(text, add_special_tokens=False))

    def root(self, user: str) -> tuple[int, ...]:
        return self.encode(self.render([{"role": "user", "content": user}]))

    def suffix(self, user: str) -> tuple[tuple[int, ...], int]:
        if _MARKER in user:
            raise ValueError("User text contains the chat scaffold marker")
        rendered = self.render([
            {"role": "user", "content": "Previous user turn"},
            {"role": "assistant", "content": _MARKER},
            {"role": "user", "content": user},
        ])
        if rendered.count(_MARKER) != 1:
            raise ValueError("Chat template does not preserve assistant content")
        suffix = self.encode(rendered.split(_MARKER, 1)[1])
        if not suffix or suffix[0] not in self.tokenizer.all_special_ids:
            raise ValueError("Chat template needs a special-token assistant closing boundary")
        return suffix, suffix[0]


def make_requests(tree: Tree, protocol: ChatProtocol) -> list[Request]:
    children: dict[str, list[Message]] = {}
    for node in tree.messages:
        if node.parent is not None:
            children.setdefault(node.parent, []).append(node)
    depths: dict[str, int] = {}
    result = []
    for user in tree.messages:
        if user.role != "user":
            continue
        parent = user.parent
        depth = 1 if parent is None else depths[parent] + 1
        suffix, closing = (protocol.root(user.text), None) if parent is None else protocol.suffix(user.text)
        replies = sorted(children.get(user.uid, ()), key=lambda n: n.uid)
        for reply in replies or [None]:
            uid = f"{tree.uid}/{reply.uid if reply is not None else 'reply:' + user.uid}"
            result.append(Request(
                uid, tree.uid, user.uid, reply.uid if reply is not None else None,
                f"{tree.uid}/{parent}" if parent is not None else None,
                depth, "assistant_alternative" if len(replies) > 1 else (
                    "user_fork" if parent is not None and len(children[parent]) > 1 else
                    "root" if parent is None else "continuation"
                ), suffix, closing,
                parent is not None and len(children[parent]) > 1, len(replies) > 1,
            ))
            if reply is not None:
                depths[reply.uid] = depth
    return result


def topological_order(requests: list[Request], seed: int) -> list[Request]:
    nodes = {node.uid: node for node in requests}
    if len(nodes) != len(requests):
        raise ValueError("Duplicate request ID")
    children: dict[str, list[str]] = {}
    ready = []
    for node in requests:
        if node.parent is None:
            ready.append(node.uid)
        elif node.parent not in nodes:
            raise ValueError(f"Missing request parent: {node.parent}")
        else:
            children.setdefault(node.parent, []).append(node.uid)
    ready.sort()
    rng = random.Random(seed)
    result = []
    while ready:
        uid = ready.pop(rng.randrange(len(ready)))
        result.append(nodes[uid])
        ready.extend(sorted(children.get(uid, ())))
    if len(result) != len(requests):
        raise ValueError("Request dependencies contain a cycle")
    return result


def select_requests(
    trees: list[Tree], protocol: ChatProtocol, *, count: int, seed: int,
    language: str | None = "en", max_seq_len: int | None = None, output_len: int = 1024,
) -> tuple[list[Tree], list[Request]]:
    candidates = sorted((tree for tree in trees if language is None or tree.language == language), key=lambda t: t.uid)
    random.Random(seed).shuffle(candidates)
    selected, requests = [], []
    for tree in candidates:
        nodes = make_requests(tree, protocol)
        bounds = {}
        for node in nodes:
            bounds[node.uid] = len(node.suffix_ids) + output_len + bounds.get(node.parent, 0)
        if max_seq_len is not None and max(bounds.values()) > max_seq_len:
            continue
        selected.append(tree)
        requests.extend(nodes)
        if len(selected) == count:
            return selected, requests
    raise ValueError(f"Requested {count} trees; only {len(selected)} satisfy the filters and context limit")


async def run_point(
    base: str, model: str, requests: list[Request], *, api: str,
    output_len: int, request_timeout: float, max_seq_len: int | None = None,
    history: list[dict] | None = None,
) -> tuple[dict, list[dict]]:
    import httpx

    history = [] if history is None else history
    positions = {node.uid: index for index, node in enumerate(requests)}
    children: dict[str, list[int]] = {}
    ready = []
    for index, node in enumerate(requests):
        if node.parent is None:
            ready.append(index)
        else:
            if node.parent not in positions or positions[node.parent] >= index:
                raise ValueError("Requests must be in topological order")
            children.setdefault(node.parent, []).append(index)
    heapq.heapify(ready)
    complete: dict[str, dict] = {}
    active: dict[asyncio.Task, int] = {}
    ready_at = {index: time.perf_counter() for index in ready}
    dispatch_count = 0
    peak_inflight = 0

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30, read=None, write=None, pool=None),
        limits=httpx.Limits(max_connections=len(requests), max_keepalive_connections=len(requests)),
        trust_env=False,
    ) as client:
        async def execute(index: int, dispatch_index: int) -> dict:
            node = requests[index]
            prior = complete.get(node.parent)
            shared = [] if prior is None else prior["prompt_token_ids"] + prior["generated_token_ids"]
            suffix = list(node.suffix_ids)
            if shared and shared[-1] == node.closing_id:
                suffix = suffix[1:]
            prompt = shared + suffix
            if max_seq_len is not None and len(prompt) + output_len > max_seq_len:
                raise ValueError(f"{node.uid}: request exceeds --max-seq-len")
            dispatched = time.perf_counter()
            result = await asyncio.wait_for(stream_token_ids(
                client, base, model, prompt, output_len, node.uid, api, ignore_eos=False,
            ), timeout=request_timeout)
            if prompt[:len(shared)] != shared:
                raise RuntimeError(f"{node.uid}: broken shared token prefix")
            if result.cached_tokens is not None and not 0 <= result.cached_tokens <= len(prompt):
                raise RuntimeError(f"{node.uid}: invalid cached-token count")
            record = {
                "uid": node.uid, "tag": node.uid, "tree_id": node.tree_id,
                "user_id": node.user_id, "assistant_id": node.assistant_id,
                "parent": node.parent, "depth": node.depth, "kind": node.kind,
                "user_fork": node.user_fork, "assistant_alternative": node.assistant_alternative,
                "priority": index, "dispatch_index": dispatch_index,
                "ready_at": ready_at[index], "dispatched_at": dispatched,
                "client_wait_s": dispatched - ready_at[index],
                "prompt_token_ids": prompt, "generated_token_ids": result.token_ids,
                "expected_reusable_len": len(shared), "prefix_exact": True,
                "parent_completed_at": None if prior is None else prior["tics"][-1],
                "tics": result.tics,
            }
            if result.cached_tokens is not None:
                record["cached_tokens"] = result.cached_tokens
            history.append(record)
            if len(history) % 16 == 0 or len(history) == len(requests):
                print(f"requests done: {len(history)}/{len(requests)}", flush=True)
            return record

        try:
            while ready or active:
                while ready:
                    index = heapq.heappop(ready)
                    task = asyncio.create_task(execute(index, dispatch_count))
                    active[task] = index
                    dispatch_count += 1
                peak_inflight = max(peak_inflight, len(active))
                done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                for task in sorted(done, key=lambda task: active[task]):
                    index = active.pop(task)
                    record = task.result()
                    complete[requests[index].uid] = record
                    for child in children.get(requests[index].uid, ()):
                        ready_at[child] = record["tics"][-1]
                        heapq.heappush(ready, child)
            if len(complete) != len(requests):
                raise RuntimeError("Some requests never became executable")
        finally:
            for task in active:
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
    ordered = sorted(history, key=lambda record: record["priority"])
    stats = summarize(
        [record["tics"] for record in ordered],
        [len(record["prompt_token_ids"]) for record in ordered],
        [output_len] * len(ordered), ignore_eos=False,
    )
    stats.pop("prefill_tok_per_s")
    cache = summarize_response_cache([{"turns": ordered}], api)
    if cache is not None:
        cache["total_cached_tokens"] = sum(record["cached_tokens"] for record in ordered)
        cache["overall_hit_fraction"] = cache["total_cached_tokens"] / stats["in_tokens"]
    stats.update(
        trees=len({node.tree_id for node in requests}), peak_inflight=peak_inflight,
        cache=cache, exact_continuations=sum(record["parent"] is not None for record in ordered),
        single_token_responses=sum(len(record["generated_token_ids"]) == 1 for record in ordered),
        empty_responses=sum(not record["generated_token_ids"] for record in ordered),
    )
    for field in ("prompt_token_ids", "generated_token_ids"):
        digest = hashlib.sha256(json.dumps(
            [(record["uid"], record[field]) for record in ordered], separators=(",", ":"),
        ).encode()).hexdigest()
        stats["input_trace_sha256" if field == "prompt_token_ids" else "output_trace_sha256"] = digest
    return stats, ordered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", choices=("logssms", "sglang", "vllm"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--tokenizer", "--model-path", dest="tokenizer", required=True)
    parser.add_argument("--model")
    parser.add_argument("--dataset", default="oasst2")
    parser.add_argument("--language", default="en", help="Root language; use all for no language filter")
    parser.add_argument("--num-trees", type=int, default=64, help="Figure 1 uses 64, 128 and 256")
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--warmup-trees", type=int, default=4)
    parser.add_argument("--name")
    parser.add_argument("--out-dir", type=Path, default=_ROOT / "experiment/standard_multiturn")
    parser.add_argument("--prepare-only", action="store_true", help="Save the workload without contacting a server")
    args = parser.parse_args(argv)
    if min(args.num_trees, args.output_len, args.max_seq_len, args.request_timeout) <= 0 or args.warmup_trees < 0:
        parser.error("Lengths and timeout must be positive; warmup tree count must be nonnegative")
    from transformers import AutoTokenizer

    protocol = ChatProtocol(AutoTokenizer.from_pretrained(args.tokenizer))
    trees = reconstruct_trees(iter_dataset(args.dataset))
    selected, all_requests = select_requests(
        trees, protocol, count=args.num_trees + args.warmup_trees, seed=args.seed,
        language=None if args.language == "all" else args.language,
        max_seq_len=args.max_seq_len, output_len=args.output_len,
    )
    measured_ids = {tree.uid for tree in selected[:args.num_trees]}
    requests = topological_order([node for node in all_requests if node.tree_id in measured_ids], args.seed)
    warmup = topological_order([node for node in all_requests if node.tree_id not in measured_ids], args.seed)
    name = args.name or f"{datetime.now():%Y%m%d-%H%M%S}-pid{os.getpid()}"
    if Path(name).name != name or name in (".", ".."):
        parser.error("--name must be a single directory name")
    run_dir = args.out_dir.resolve() / name
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "dataset": args.dataset, "language": args.language, "seed": args.seed,
        "tree_ids": [tree.uid for tree in selected[:args.num_trees]],
        "warmup_tree_ids": [tree.uid for tree in selected[args.num_trees:]],
        "chat_template_sha256": hashlib.sha256(protocol.tokenizer.chat_template.encode()).hexdigest(),
        "chat_template_kwargs": protocol.kwargs,
        "output_len": args.output_len, "max_seq_len": args.max_seq_len,
        "requests": [asdict(node) for node in requests],
        "warmup_requests": [asdict(node) for node in warmup],
    }
    save_json(run_dir / "manifest.json", manifest)
    print(f"Prepared {args.num_trees} trees, {len(requests)} requests: {run_dir}", flush=True)
    if args.prepare_only:
        return 0
    history: list[dict] = []
    warmup_history: list[dict] = []
    report = {
        "api": args.api, "base_url": args.base_url, "model": args.model or args.tokenizer,
        "ignore_eos": False, "manifest": str(run_dir / "manifest.json"),
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "status": "running",
    }
    save_json(run_dir / "report.json", report)

    async def run() -> dict:
        if warmup:
            warm_stats, _ = await run_point(
                args.base_url, args.model or args.tokenizer, warmup, api=args.api,
                output_len=min(args.output_len, 64), request_timeout=args.request_timeout,
                max_seq_len=args.max_seq_len, history=warmup_history,
            )
            report["warmup"] = warm_stats
        stats, _ = await run_point(
            args.base_url, args.model or args.tokenizer, requests, api=args.api,
            output_len=args.output_len, request_timeout=args.request_timeout,
            max_seq_len=args.max_seq_len, history=history,
        )
        return stats

    try:
        report["stats"] = asyncio.run(run())
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save_json(run_dir / "history.json", {"requests": sorted(history, key=lambda record: record["priority"])})
        save_json(run_dir / "warmup-history.json", {"requests": warmup_history})
        save_json(run_dir / "report.json", report)
    stats = report["stats"]
    print(f"TTFT avg/p50: {stats['ttft_ms'][0]:.2f}/{stats['ttft_ms'][1]:.2f} ms")
    print(f"TPOT avg/p50: {stats['tpot_ms'][0]:.2f}/{stats['tpot_ms'][1]:.2f} ms")
    print(f"Output: {stats['output_tok_per_s']:.2f} tok/s; requests: {stats['req_per_s']:.2f}/s")
    cache = stats["cache"]
    print("Cache: unmeasured" if cache is None else (
        f"Cache: {cache['overall_hit_fraction']:.2%}; reusable history: {cache['admitted_hit_fraction']:.2%}"
    ))
    print(f"Results: {run_dir}")
    return 0
