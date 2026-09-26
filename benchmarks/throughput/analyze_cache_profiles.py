"""Summarize comparable scheduler cache profiles from multi-turn benchmarks.

Example:
  python benchmarks/throughput/analyze_cache_profiles.py \
      --profile-a benchmark-results/multiturn/profile/controlled_radix64.rank0.json \
      --profile-b benchmark-results/multiturn/profile/controlled_logstate32.rank0.json \
      --history-a benchmark-results/multiturn/diagnostic/controlled_radix64_history.json \
      --history-b benchmark-results/multiturn/diagnostic/controlled_logstate32_history.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def summarize_profile(path: Path) -> dict:
    cache = json.loads(path.read_text())["cache"]
    events = cache["lifecycle_events"]
    evictions = [event for event in events if event["event"] == "eviction"]
    arrivals = {
        event["uid"]: event for event in events if event["event"] == "arrival"
    }
    admissions = [event for event in events if event["event"] == "admission"]
    first = evictions[0] if evictions else None
    before_first = (
        Counter(event["event"] for event in events[:first["seq"]])
        if first is not None else None
    )
    by_turn = defaultdict(lambda: {
        "requests": 0, "matchable_tokens": 0, "arrival_match_tokens": 0,
        "admission_match_tokens": 0, "admission_zero_matches": 0,
        "arrival_to_admission_lost_tokens": 0,
    })
    previous_to_arrival_drops = []
    for event in admissions:
        turn = int(event["tag"].rsplit("/turn-", 1)[1])
        arrival = arrivals[event["uid"]]
        row = by_turn[turn]
        row["requests"] += 1
        row["matchable_tokens"] += arrival["matchable_len"]
        row["arrival_match_tokens"] += arrival["tree_match_len"]
        row["admission_match_tokens"] += event["tree_match_len"]
        row["admission_zero_matches"] += event["tree_match_len"] == 0
        row["arrival_to_admission_lost_tokens"] += max(
            0, arrival["tree_match_len"] - event["tree_match_len"]
        )
        previous = arrival.get("previous_inserted_len")
        if previous is not None:
            previous_to_arrival_drops.append(
                previous - arrival["tree_match_len"]
            )

    evicted_checkpoint_spans = [
        entry["end"] - entry["begin"]
        for event in evictions for entry in event.get("entries", ())
        if entry["kind"] == "checkpoint"
    ]
    result = {
        "type": cache["type"],
        "capacity": cache.get("capacity"),
        "first_eviction": None if first is None else {
            "elapsed_s": round(
                (first["time_ns"] - events[0]["time_ns"]) / 1e9, 3
            ),
            "prior_arrivals": before_first["arrival"],
            "prior_admissions": before_first["admission"],
            "prior_completions": before_first["complete"],
            "phase": first.get("phase"),
            "needed_kv_pages": first.get("need_kv_pages"),
            "needed_log_slots": first.get("need_log_slots"),
            "needed_state_slots": first.get("need_state_slots"),
            "free_frames_before": first.get("free_frames_before"),
            "free_kv_pages_before": first.get("free_kv_pages_before"),
            "active_request_tags": first.get("request_tags", []),
        },
        "eviction_events": len(evictions),
        "evictions_by_phase": dict(Counter(
            event.get("phase", "unknown") for event in evictions
        )),
        "evicted_kv_tokens": sum(event["kv_tokens"] for event in evictions),
        "max_eviction_kv_tokens": max(
            (event["kv_tokens"] for event in evictions), default=0
        ),
        "max_evicted_node_tokens": max(
            (node["end"] - node["begin"]
             for event in evictions for node in event["nodes"]), default=0
        ),
        "max_evicted_checkpoint_span_tokens": (
            max(evicted_checkpoint_spans, default=0)
            if any("entries" in event for event in evictions) else None
        ),
        "previous_to_arrival_drop_median": (
            median(previous_to_arrival_drops)
            if previous_to_arrival_drops else None
        ),
        "previous_to_arrival_exact": sum(
            value == 0 for value in previous_to_arrival_drops
        ),
        "previous_to_arrival_samples": len(previous_to_arrival_drops),
        "turns": dict(sorted(by_turn.items())),
    }
    if cache["type"] == "logstate":
        batches = cache["prefill_batches"]
        materialized = sum(batch["materialized_slots"] for batch in batches)
        merged = sum(batch["released_state_slots"] for batch in batches)
        live = (batches[-1]["resources_after"]["tree_checkpoint_slots"]
                if batches else 0)
        result["logstate"] = {
            "prefill_batches": len(batches),
            "materialized_slots": materialized,
            "merged_slots": merged,
            "evicted_checkpoint_entries": len(evicted_checkpoint_spans),
            "live_checkpoint_slots_after_last_prefill": live,
            "checkpoint_balance_at_end": (
                materialized - merged - len(evicted_checkpoint_spans)
            ),
            "merge_batch_outcomes": cache["aggregate"]["merge_batch_outcomes"],
            "admission_rejections": cache["aggregate"]["admission_rejections"],
            "max_pending_wait_ms": cache["aggregate"]["max_pending_wait_ms"],
            "last_resources": batches[-1]["resources_after"] if batches else None,
            "checkpoint_spans_by_turn": summarize_checkpoint_spans(cache),
        }
    return result


def compare_histories(left: Path, right: Path) -> dict:
    a = json.loads(left.read_text())["points"]
    b = json.loads(right.read_text())["points"]
    compared = equal = 0
    for point, value in a.items():
        sessions = {session["uid"]: session for session in b[point]["sessions"]}
        for session in value["sessions"]:
            for turn_a, turn_b in zip(
                session["turns"], sessions[session["uid"]]["turns"],
                strict=True,
            ):
                compared += 1
                equal += turn_a["messages"] == turn_b["messages"]
    return {"identical_request_inputs": equal, "requests_compared": compared}


def summarize_checkpoint_spans(cache: dict) -> dict[str, dict]:
    """Show retained checkpoint spans after each observed prefill turn."""
    uid_turn = {
        event["uid"]: int(event["tag"].rsplit("/turn-", 1)[1])
        for event in cache["lifecycle_events"]
        if event["event"] == "admission" and "/turn-" in event["tag"]
    }
    per_turn = defaultdict(lambda: {
        "batches": 0,
        "materialized_slots": 0,
        "merged_slots": 0,
        "matched_path_checkpoint_spans": Counter(),
        "last_tree_checkpoint_spans": {},
    })
    first_paths = defaultdict(list)
    seen_uids = set()
    for batch in cache["prefill_batches"]:
        turns = [
            uid_turn[request["uid"]]
            for request in batch["requests"]
            if request["uid"] in uid_turn
        ]
        if not turns:
            continue
        turn = max(turns)
        row = per_turn[turn]
        row["batches"] += 1
        row["materialized_slots"] += batch["materialized_slots"]
        row["merged_slots"] += batch["released_state_slots"]
        resources = batch["resources_after"]
        row["last_tree_checkpoint_spans"] = resources[
            "checkpoint_span_pages"
        ]
        row["last_tree_log_slots"] = resources["tree_log_slots"]
        row["last_tree_checkpoint_slots"] = resources[
            "tree_checkpoint_slots"
        ]
        row["last_free_frames"] = resources["free_frames"]
        for request in batch["requests"]:
            request_turn = uid_turn.get(request["uid"])
            if request_turn is not None:
                per_turn[request_turn]["matched_path_checkpoint_spans"].update(
                    request.get("cached_checkpoint_span_pages", {})
                )
                if request["uid"] not in seen_uids:
                    seen_uids.add(request["uid"])
                    spans = request.get("cached_checkpoint_span_pages", {})
                    first_paths[request_turn].append({
                        "cached_len": request["cached_len"],
                        "checkpoint_slots": request["cached_checkpoint_slots"],
                        "log_slots": request["cached_log_slots"],
                        "state_slots": (
                            request["cached_checkpoint_slots"]
                            + request["cached_log_slots"]
                        ),
                        "max_span_pages": (
                            max(map(int, spans), default=0)
                            if "cached_checkpoint_span_pages" in request else None
                        ),
                    })
    return {
        str(turn): {
            **row,
            "matched_path_checkpoint_spans": dict(sorted(
                row["matched_path_checkpoint_spans"].items(),
                key=lambda item: int(item[0]),
            )),
            "first_request_path": {
                "requests": len(first_paths[turn]),
                "cached_len_median": median(
                    item["cached_len"] for item in first_paths[turn]
                ),
                "checkpoint_slots_median": median(
                    item["checkpoint_slots"] for item in first_paths[turn]
                ),
                "log_slots_median": median(
                    item["log_slots"] for item in first_paths[turn]
                ),
                "state_slots_median": median(
                    item["state_slots"] for item in first_paths[turn]
                ),
                "max_span_pages_hist": dict(sorted(Counter(
                    item["max_span_pages"] for item in first_paths[turn]
                    if item["max_span_pages"] is not None
                ).items())),
                "max_span_pages_median": median(
                    item["max_span_pages"] for item in first_paths[turn]
                    if item["max_span_pages"] is not None
                ) if any(
                    item["max_span_pages"] is not None
                    for item in first_paths[turn]
                ) else None,
            },
        }
        for turn, row in sorted(per_turn.items())
    }


def summarize_chat_prefixes(path: Path, tokenizer) -> dict:
    """Measure whether each rendered prompt extends the preceding prompt."""
    from logssms.utils.platform import render_chat_prompt

    points = json.loads(path.read_text())["points"]
    losses = []
    for point in points.values():
        for session in point["sessions"]:
            previous = None
            for turn in session["turns"]:
                messages = turn["messages"]
                rendered = (
                    messages if isinstance(messages, str) else
                    render_chat_prompt(tokenizer, messages)
                )
                current = tokenizer.encode(rendered)
                if previous is not None:
                    common = 0
                    for a, b in zip(previous, current):
                        if a != b:
                            break
                        common += 1
                    losses.append(len(previous) - common)
                previous = current
    return {
        "continuations": len(losses),
        "exact_prefix_continuations": sum(loss == 0 for loss in losses),
        "median_previous_prompt_suffix_lost_tokens": (
            median(losses) if losses else None
        ),
        "max_previous_prompt_suffix_lost_tokens": max(losses, default=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-a", type=Path, required=True)
    parser.add_argument("--profile-b", type=Path, required=True)
    parser.add_argument("--history-a", type=Path)
    parser.add_argument("--history-b", type=Path)
    parser.add_argument(
        "--tokenizer-model", type=Path,
        help="also measure exact token-prefix continuity of chat history",
    )
    parser.add_argument(
        "--preserve-thinking", action="store_true",
        help="analyze histories rendered with LogSSMs' Qwen3.5 template override",
    )
    args = parser.parse_args()
    result = {
        "a": summarize_profile(args.profile_a),
        "b": summarize_profile(args.profile_b),
    }
    if args.history_a is not None or args.history_b is not None:
        if args.history_a is None or args.history_b is None:
            parser.error("provide both --history-a and --history-b")
        result["input_comparison"] = compare_histories(
            args.history_a, args.history_b
        )
        if args.tokenizer_model is not None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
            if args.preserve_thinking:
                from logssms.utils.platform import configure_chat_tokenizer

                tokenizer = configure_chat_tokenizer(
                    tokenizer, str(args.tokenizer_model)
                )
            result["chat_prefixes"] = {
                "a": summarize_chat_prefixes(args.history_a, tokenizer),
                "b": summarize_chat_prefixes(args.history_b, tokenizer),
            }
    elif args.tokenizer_model is not None:
        parser.error("--tokenizer-model requires both history files")
    if args.preserve_thinking and args.tokenizer_model is None:
        parser.error("--preserve-thinking requires --tokenizer-model")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
