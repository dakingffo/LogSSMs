"""Engine-independent client-observed metrics for AR token-ID streams."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def save_json(path: str | Path, data: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _stats(values: list[float], scale: float = 1000.0) -> list[float]:
    if not values:
        return [0.0] * 5
    ordered = sorted(values)
    return [
        scale * sum(values) / len(values),
        *(scale * ordered[min(len(ordered) - 1, int(len(ordered) * p))]
          for p in (0.5, 0.9, 0.99)),
        scale * ordered[-1],
    ]


def summarize(
    tics_all      : list[list[float]],
    input_lengths : list[int],
    output_lengths: list[int],
) -> dict:
    if not tics_all:
        raise ValueError("Cannot summarize an empty request set")
    # Tics are normalized by the API adapter: start, one per generated token,
    # finish, DONE. The token count comes from actual IDs, not SSE chunk count.
    actual = [len(tics) - 3 for tics in tics_all]
    requested = sum(output_lengths)
    out_tokens = sum(actual)
    start = min(tics[0] for tics in tics_all)
    end = max(tics[-1] for tics in tics_all)
    duration = end - start
    ttft = [tics[1] - tics[0] for tics in tics_all]
    e2e = [tics[-1] - tics[0] for tics in tics_all]
    tpot = [
        (tics[-1] - tics[1]) / (count - 1) if count > 1 else 0.0
        for tics, count in zip(tics_all, actual, strict=True)
    ]
    output_rate = out_tokens / duration if duration > 0 else 0.0
    return {
        "n": len(tics_all),
        "ignore_eos": True,
        "in_tokens": sum(input_lengths),
        "requested_out": requested,
        "early_stops": sum(
            count < length
            for count, length in zip(actual, output_lengths, strict=True)
        ),
        "out_tokens": out_tokens,
        "duration_s": duration,
        "prefill_tok_per_s": sum(input_lengths) / max(ttft) if ttft else 0.0,
        "output_tok_per_s": output_rate,
        "decode_tok_per_s": output_rate,
        "req_per_s": len(tics_all) / duration if duration > 0 else 0.0,
        "ttft_ms": _stats(ttft),
        "tpot_ms": _stats(tpot),
        "e2e_s": _stats(e2e, scale=1.0),
        "rounds": 0,
        "avg_accept": 0.0,
        "is_mtp": False,
        "chunks_ok": actual == output_lengths,
    }


def print_stats(title: str, stats: dict) -> None:
    ttft = stats["ttft_ms"]
    tpot = stats["tpot_ms"]
    e2e = stats["e2e_s"]
    print("=" * 64)
    print(f"  {title}")
    print(
        f"  requests: {stats['n']}  input: {stats['in_tokens']} tok  "
        f"output: {stats['out_tokens']} tok  duration: {stats['duration_s']:.2f}s"
    )
    print(
        f"  output: {stats['output_tok_per_s']:.1f} tok/s  "
        f"{stats['req_per_s']:.2f} req/s"
    )
    print(f"  TTFT(ms): avg {ttft[0]:.2f}  p50 {ttft[1]:.2f}  p90 {ttft[2]:.2f}")
    print(f"  TPOT(ms): avg {tpot[0]:.2f}  p50 {tpot[1]:.2f}  p90 {tpot[2]:.2f}")
    print(f"  E2E(s):   avg {e2e[0]:.3f}  p50 {e2e[1]:.3f}  p90 {e2e[2]:.3f}")
    print(
        f"  sessions: {stats['sessions']}  concurrency: {stats['concurrency']}  "
        f"turns/session: {stats['num_turns']}  "
        f"session/s: {stats['session_per_s']:.2f}"
    )
    print(f"  {'turn':>6} {'input':>10} {'output':>10} {'TTFT p50':>10} {'E2E p50':>10}")
    for turn in stats["turn_stats"]:
        print(
            f"  {turn['turn']:>6} {turn['in_tokens']:>10} "
            f"{turn['out_tokens']:>10} {turn['ttft_ms'][1]:>10.2f} "
            f"{turn['e2e_s'][1]:>10.3f}"
        )
    print("=" * 64)
