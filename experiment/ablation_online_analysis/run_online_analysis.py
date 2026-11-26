"""Compare Default and Online eviction on a Claude Code workload."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from datasets import load_replay_history
from datasets.parser import replay_history_prefix


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VARIANTS = {
    "marconi": ("MARCONI", "#d55e00", "-.", 1.8),
    "default": ("LogState Default", "#66717d", "--", 1.8),
    "online": ("LogState Online", "#0072b2", "-", 1.9),
}


def run_variant(out, history, case, variant, heartbeat_seconds):
    concurrency = case["tree_concurrency"]
    destination = out / f"{variant}.json"
    log_path = destination.with_suffix(".log")
    if log_path.exists():
        previous = log_path.with_name(f"{log_path.stem}-{datetime.now(ZoneInfo('Asia/Shanghai')):%Y%m%d-%H%M%S}.log")
        log_path.rename(previous)
    cmd = [
        sys.executable, "-m", "experiment.utils.cpu_replay.replay", "--mode", variant,
        "--case", str(out / "case.json"), "--history", str(history), "--out", str(destination),
        "--batch-size", str(case["arrival_batch_size"]), "--tree-concurrency", str(concurrency),
        "--trace-mode", "virtual_clock", "--trace-sessions", str(case["tree_count"]),
        "--trace-max-context", str(case["max_seq_len"]), "--clock-step-ms", str(case["clock_step_ms"]),
        "--skip-layout-snapshots", "--record-state-usage", "--progress-every", "32",
        "--heartbeat-seconds", str(heartbeat_seconds),
        "--timings", str(out / f"{variant}-timings.jsonl"),
    ]
    if variant != "marconi":
        cmd.append("--hit-materialization")
    else:
        cmd.extend([
            "--block-size", str(case.get("marconi_block_size", 128)),
            "--bootstrap-multiplier", str(case.get("marconi_bootstrap_multiplier", 5)),
        ])
    with log_path.open("w") as log:
        result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{variant} replay failed. See {log_path}")
    print(f"Completed {variant}", flush=True)


def analyze(data, records, case, digest, variant):
    rows = data["requests_detail"]
    concurrency = case["tree_concurrency"]
    expected_gran = case.get("marconi_block_size", 128) if variant == "marconi" else case["granularity"]
    expected_hit_mat = variant != "marconi"
    if (data["trace_sha256"] != digest or data["mode"] != variant
            or data["trace_mode"] != "virtual_clock" or data["clock_step_ms"] != case["clock_step_ms"]
            or data["arrival_batch_size"] != case["arrival_batch_size"]
            or data["trace_max_context"] != case["max_seq_len"]
            or data["tree_count"] != case["tree_count"]
            or data["tree_concurrency"] != concurrency or data["peak_tree_concurrency"] != concurrency
            or data["totals"]["verified_parent_prefixes"] != sum(record[2] for record in records.values())
            or {row["uid"] for row in rows} != records.keys() or len(rows) != len(records)
            or any((row["history_tokens"], row["output_cap"]) != records[row["uid"]][:2] for row in rows)
            or data["enable_hit_materialization"] != expected_hit_mat
            or data["granularity"] != expected_gran
            or data["num_frames"] != case["cache_bytes_per_rank"] // data["frame_bytes"]):
        raise RuntimeError(f"Workload or cache configuration mismatch for {variant}")

    counts = sorted(row["historical_recompute"] for row in rows if row["history_tokens"])
    stats = data["totals"]
    if sum(counts) != stats["historical_recompute_tokens"]:
        raise RuntimeError(f"Inconsistent historical recomputation for {variant}")
    usage = data["state_usage"]
    if usage[0]["completed"] != 0 or usage[-1]["completed"] != len(rows):
        raise RuntimeError(f"Incomplete State usage samples for {variant}")

    frame_gib = data["frame_bytes"] / 1024**3
    summary = {
        "peak_state_gib": max(sample["log_frames"] + sample["checkpoint_frames"] for sample in usage) * frame_gib,
        "final_log_gib": usage[-1]["log_frames"] * frame_gib,
        "final_checkpoint_gib": usage[-1]["checkpoint_frames"] * frame_gib,
        "hit_rate": data["hit_rate"],
        "historical_hit_rate": data["historical_hit_rate"],
        "historical_recompute_tokens": sum(counts),
        "model_prefill_tokens": stats["prefill_tokens"],
        "exponent_fits": (data["analysis"] or {}).get("aggregate", {}).get("exponent_fits", 0),
        "reclamation_rounds": data["evictions"].get("eviction_rounds", 0),
        "evicted_kv_tokens": sum(event.get("kv_tokens", 0) for event in data["eviction_events"]),
        "min_free_frames": data["min_free_frames"],
    }
    for quantile in (95, 99):
        summary[f"historical_recompute_p{quantile}"] = counts[math.ceil(len(counts) * quantile / 100) - 1]
    return summary, counts


def plot(results, out, requests):
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # Smooth non-uniform x-axis (cubic expansion toward the high-pressure tail)
    power = 3.0
    scale = float(max(1, requests))

    def fwd(x):
        arr = np.asarray(x, dtype=float)
        return np.sign(arr) * (np.abs(arr) / scale) ** power

    def inv(u):
        arr = np.asarray(u, dtype=float)
        return np.sign(arr) * (np.abs(arr) ** (1.0 / power)) * scale

    fig, ax = plt.subplots(figsize=(6.8, 3.8), layout="constrained")
    xs = list(range(requests + 1))
    series = {}
    for variant, (label, color, style, width) in VARIANTS.items():
        rows = sorted(results[variant]["requests_detail"], key=lambda row: row["admission_index"])
        cumulative = [0]
        for row in rows:
            cumulative.append(cumulative[-1] + row["historical_recompute"])
        series[variant] = cumulative
        ax.step(xs, cumulative, where="post", color=color, linestyle=style,
                linewidth=width, label=label)
        ax.plot(requests, cumulative[-1], marker="o", markersize=4, color=color, zorder=4)

    default_y, online_y, marconi_y = series["default"], series["online"], series["marconi"]
    ymax = max(default_y[-1], online_y[-1], marconi_y[-1], 1)
    ax.fill_between(xs, online_y, default_y, step="post",
                    where=[o <= d for o, d in zip(online_y, default_y)],
                    color="#0072b2", alpha=0.14, linewidth=0)

    ax.set_xscale("function", functions=(fwd, inv))
    tick_fracs = (0.0, 0.45, 0.62, 0.73, 0.83, 0.90, 0.96, 1.0)
    ticks = sorted({0, requests, *(int(round(f * requests / 10) * 10) for f in tick_fracs[1:-1])})
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])

    u_num = 1.008
    u_gray = 1.055
    u_orange = 1.150
    x_num = float(inv(u_num))
    x_gray = float(inv(u_gray))
    x_orange = float(inv(u_orange))

    def fmt_k(val):
        return f"{val / 1000:.0f}k" if val >= 100_000 else f"{val / 1000:.1f}k"

    # Horizontal dashed guide extensions from x=requests out to the vertical comparison arrows
    ax.hlines(online_y[-1], requests, x_orange, colors="#005a8d", linestyles=":", linewidth=0.9, alpha=0.55)
    ax.hlines(default_y[-1], requests, x_gray, colors="#66717d", linestyles=":", linewidth=0.9, alpha=0.55)
    ax.hlines(marconi_y[-1], requests, x_orange, colors="#d55e00", linestyles=":", linewidth=0.9, alpha=0.55)

    y_off = ymax * 0.015
    ax.text(x_num, marconi_y[-1] + y_off, fmt_k(marconi_y[-1]),
            color="#b24d00", fontsize=8.3, va="bottom", ha="left", fontweight="bold")
    ax.text(x_num, default_y[-1] + y_off, fmt_k(default_y[-1]),
            color="#4a535c", fontsize=8.3, va="bottom", ha="left")
    ax.text(x_num, online_y[-1] - y_off, fmt_k(online_y[-1]),
            color="#005a8d", fontsize=8.3, va="top", ha="left", fontweight="bold")

    if default_y[-1] > 0 and online_y[-1] < default_y[-1]:
        delta_d = (default_y[-1] - online_y[-1]) / default_y[-1] * 100
        ax.annotate("", xy=(x_gray, online_y[-1]), xytext=(x_gray, default_y[-1]),
                    arrowprops=dict(arrowstyle="-|>", color="#66717d", lw=1.3, mutation_scale=10, shrinkA=0, shrinkB=0))
        ax.text(float(inv(u_gray + 0.010)), (default_y[-1] + online_y[-1]) / 2,
                f"−{delta_d:.1f}%", color="#4a535c", fontsize=8.3, va="center", ha="left", fontweight="bold")

    if marconi_y[-1] > 0 and online_y[-1] < marconi_y[-1]:
        delta_m = (marconi_y[-1] - online_y[-1]) / marconi_y[-1] * 100
        ax.annotate("", xy=(x_orange, online_y[-1]), xytext=(x_orange, marconi_y[-1]),
                    arrowprops=dict(arrowstyle="-|>", color="#d55e00", lw=1.3, mutation_scale=10, shrinkA=0, shrinkB=0))
        ax.text(float(inv(u_orange + 0.010)), (marconi_y[-1] + online_y[-1]) / 2,
                f"−{delta_m:.1f}%", color="#b24d00", fontsize=8.3, va="center", ha="left", fontweight="bold")

    ax.set(xlabel="Processed requests",
           ylabel="Historical recomputation (k tokens)",
           xlim=(0, float(inv(1.28))), ylim=(0, ymax * 1.10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / 1000:g}"))
    ax.grid(alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False)
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"online_analysis.{suffix}", dpi=200)
    plt.close(fig)


def write_report(out, summary):
    lines = [
        "# Online analysis ablation",
        "",
        "| Policy | Overall hit rate | Historical hit rate | Historical recompute (tokens) | P95 / P99 (tokens) | Model prefill (tokens) | Reclamation rounds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, (label, *_) in VARIANTS.items():
        stats = summary[variant]
        lines.append(
            f"| {label} | {stats['hit_rate']:.4%} | {stats['historical_hit_rate']:.4%} | "
            f"{stats['historical_recompute_tokens']:,} | "
            f"{stats['historical_recompute_p95']:,} / {stats['historical_recompute_p99']:,} | "
            f"{stats['model_prefill_tokens']:,} | {stats['reclamation_rounds']:,} |"
        )
    lines.extend([
        "",
        "| Policy | Peak sampled State (GiB/rank) | Final Log (GiB/rank) | Final Checkpoint (GiB/rank) |",
        "|---|---:|---:|---:|",
    ])
    for variant, (label, *_) in VARIANTS.items():
        stats = summary[variant]
        lines.append(
            f"| {label} | {stats['peak_state_gib']:.2f} | "
            f"{stats['final_log_gib']:.2f} | {stats['final_checkpoint_gib']:.2f} |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--case", type=Path, default=HERE / "case.json")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate figures and summaries from --out")
    parser.add_argument("--resume", action="store_true", help="Reuse completed policies and run missing ones")
    parser.add_argument("--heartbeat-seconds", type=float, default=30.)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if (args.plot_only or args.resume) and args.out is None:
        parser.error("--plot-only and --resume require --out")

    out = (args.out or HERE / "results" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")).resolve()
    case = json.loads((out / "case.json" if args.plot_only or args.resume else args.case).read_text())
    trace = load_replay_history(ROOT / case["history"], session_limit=case["tree_count"],
                                max_context=case["max_seq_len"])
    sessions = list(dict.fromkeys(record["tree_id"] for record in trace.requests))
    if len(sessions) != case["tree_count"]:
        raise RuntimeError(f"Expected {case['tree_count']} sessions, found {len(sessions)}")
    history, digest = trace.path, trace.sha256()
    by_uid = {record["uid"]: record for record in trace.requests}
    records = {}
    for uid, record in by_uid.items():
        parent = by_uid.get(record["parent"])
        shared = 0 if parent is None else replay_history_prefix(record, parent)
        full = parent is not None and shared == len(parent["prompt_token_ids"]) + len(parent["generated_token_ids"])
        records[uid] = shared, record["output_cap"], full
    if not any(record[0] for record in records.values()):
        raise RuntimeError("Selected workload has no reusable historical prefix")
    del trace, by_uid, record, parent
    print(f"{len(sessions)} sessions, {case['tree_concurrency']} concurrent, {len(records)} requests. Output: {out}",
          flush=True)
    if not args.plot_only:
        out.mkdir(parents=True, exist_ok=True)
        if not args.resume:
            (out / "case.json").write_text(json.dumps(case, indent=2) + "\n")
        variants = []
        for variant in VARIANTS:
            existing = out / f"{variant}.json"
            if args.resume and existing.is_file():
                analyze(json.loads(existing.read_text()), records, case, digest, variant)
                print(f"Reusing {variant}", flush=True)
            else:
                variants.append(variant)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda v: run_variant(out, history, case, v, args.heartbeat_seconds), variants))

    results = {variant: json.loads((out / f"{variant}.json").read_text()) for variant in VARIANTS}
    summary, recompute = {}, {}
    for variant, data in results.items():
        summary[variant], recompute[variant] = analyze(data, records, case, digest, variant)
    (out / "summary.json").write_text(json.dumps({
        "case": case, "history_sha256": digest, "session_ids": sessions, "policies": summary,
    }, indent=2) + "\n")
    plot(results, out, len(records))
    write_report(out, summary)
    print(out / "report.md")


if __name__ == "__main__":
    main()
