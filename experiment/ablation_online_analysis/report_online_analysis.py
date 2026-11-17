from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _prefill_statistics(trees, mode, scope, values):
    samples = np.asarray(values)
    return dict(
        trees=int(trees), policy=mode, scope=scope, count=len(samples),
        mean=float(samples.mean()) if len(samples) else None,
        **{f"p{q}": float(np.percentile(samples, q)) if len(samples) else None
           for q in (50, 90, 99)},
        maximum=int(samples.max()) if len(samples) else None,
        **{f"le{limit}": float(np.mean(samples <= limit)) if len(samples) else None
           for limit in (32, 64, 128, 256, 512)},
    )


def _write_prefill_statistics(folder, rows):
    (folder / "prefill-distribution.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n"
    )
    with (folder / "prefill-distribution.csv").open("w") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Actual prefill token distributions", "",
        "Forward counts include only valid prefill tokens before Engine.prepare_batch "
        "advances requests. Mixed forwards exclude decode tokens; graph padding, State "
        "reconstruction and Log materialization are excluded. The request scope counts "
        "each request's total uncached input, including all prefill chunks. Both totals "
        "must equal the replay's prefill-token total. Quantiles use NumPy linear "
        "interpolation. Empty scopes have no quantiles or threshold fractions.", "",
        "| Trees | Policy | Scope | Count | Mean | P50 | P90 | P99 | Max | <=32 | <=64 | <=128 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cells = [str(row[key]) for key in ("trees", "policy", "scope", "count")]
        cells += [f"{row[key]:.2f}" if row[key] is not None else "—"
                  for key in ("mean", "p50", "p90", "p99")]
        cells.append(str(row["maximum"]) if row["maximum"] is not None else "—")
        cells += [f"{row[key]:.2%}" if row[key] is not None else "—"
                  for key in ("le32", "le64", "le128")]
        lines.append("| " + " | ".join(cells) + " |")
    (folder / "prefill-distribution.md").write_text("\n".join(lines) + "\n")


def build_report(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("execution") != "production-scheduler-v1":
        raise ValueError("Expected production scheduler replay results")
    cases = manifest["cases"]
    fig, axes = plt.subplots(4, len(cases), figsize=(5 * len(cases), 11), squeeze=False)
    prefill_fig, prefill_axes = plt.subplots(
        1, len(cases), figsize=(4.3 * len(cases), 3.5), sharey=True, squeeze=False,
    )
    prefill_rows = []
    summary = []
    for col, (trees, case) in enumerate(cases.items()):
        data = {
            mode: json.loads((folder / f"{mode}-trees{trees}.json").read_text())
            for mode in ("default", "online")
        }
        a, b = data.values()
        assert a["execution"] == b["execution"] == manifest["execution"]
        assert a["trace_sha256"] == b["trace_sha256"]
        assert a["hardware"] == b["hardware"]
        assert [r["uid"] for r in a["requests_detail"]] == [r["uid"] for r in b["requests_detail"]]
        rows = []
        for mode, d in data.items():
            steps = d["prefill_steps"]
            assert len(steps) == sum(d["forward_batches"].get(phase, 0)
                                     for phase in ("prefill", "mixed"))
            assert sum(step["tokens"] for step in steps) == d["totals"]["prefill_tokens"]
            request_tokens = [r["prefill_tokens"] for r in d["requests_detail"]]
            assert sum(request_tokens) == d["totals"]["prefill_tokens"]
            for scope in ("all", "mixed", "prefill", "request"):
                values = request_tokens if scope == "request" else [
                    step["tokens"] for step in steps
                    if scope == "all" or step["phase"] == scope
                ]
                prefill_rows.append(_prefill_statistics(trees, mode, scope, values))
            tokens = np.sort([step["tokens"] for step in steps])
            if len(tokens):
                prefill_axes[0, col].step(
                    tokens, np.arange(1, len(tokens) + 1) / len(tokens),
                    where="post", label=mode,
                )
            batches = d["batches"]
            n = d["requests"]
            row = dict(
                trees=int(trees),
                mode=mode,
                requests=n,
                hit_rate=d["hit_rate"],
                historical_hit_rate=d["historical_hit_rate"],
                historical_recompute_tokens=d["totals"]["historical_recompute_tokens"],
                prefill_tokens=d["totals"]["prefill_tokens"],
                mean_checkpoint=sum(x["checkpoint_before"] * x["requests"] for x in batches) / n,
                mean_log=sum(x["log_before"] * x["requests"] for x in batches) / n,
                final_checkpoint=batches[-1]["checkpoint"],
                final_log=batches[-1]["log"],
                min_free_frames=d["min_free_frames"],
                checkpoint_drops=d["evictions"].get("pressure_dropped_checkpoints", 0),
                evicted_kv_tokens=d["evictions"].get("evicted_kv_tokens", 0),
            )
            summary.append(row)
            for key in (
                "historical_kv_gap",
                "historical_state_gap",
                "historical_cacheable_kv_gap",
                "historical_alignment_gap",
            ):
                row[key] = d["totals"][key]
            for kind in ("log", "log_payload_conv", "decay", "checkpoint", "kv", "work", "free"):
                row[f"mean_frames_{kind}"] = (
                    sum(b["frames_before"][kind] * b["requests"] for b in batches) / n
                )
                row[f"final_frames_{kind}"] = batches[-1]["frames_after"][kind]
            records = sorted(d["requests_detail"], key=lambda r: r["admission_index"])
            x = list(range(1, n + 1))
            recompute = []
            hit = []
            total_c = total_h = total_i = 0
            for r in records:
                total_c += r["historical_recompute"]
                total_h += r["recoverable"]
                total_i += r["input_tokens"]
                recompute.append(total_c)
                hit.append(100 * total_h / total_i)
            style = "--" if mode == "default" else "-"
            axes[0, col].plot(x, recompute, style, label=mode)
            axes[1, col].plot(x, hit, style, label=mode)
            bx = [b["iteration"] for b in batches]
            axes[2, col].plot(
                bx, [b["checkpoint"] for b in batches], style, label=f"{mode} Checkpoint"
            )
            axes[2, col].plot(
                bx, [b["log"] for b in batches], style, alpha=0.6, label=f"{mode} Log"
            )
            spans = batches[-1]["checkpoint_spans"]
            positions = sorted(map(int, spans))
            axes[3, col].plot(
                positions, [spans[str(i)] for i in positions], style, marker="o", label=mode
            )
            log_spans = batches[-1]["log_run_spans"]
            positions = sorted(map(int, log_spans))
            axes[3, col].plot(
                positions,
                [log_spans[str(i)] for i in positions],
                style,
                marker="x",
                alpha=0.6,
                label=f"{mode} Log run",
            )
            for batch in batches:
                rows.append(
                    dict(
                        mode=mode,
                        **{
                            k: v
                            for k, v in batch.items()
                            if k
                            not in (
                                "checkpoint_spans",
                                "log_run_spans",
                                "frames_before",
                                "frames_after",
                            )
                        },
                        **{
                            f"frames_{phase}_{key}": value
                            for phase in ("before", "after")
                            for key, value in batch[f"frames_{phase}"].items()
                        },
                    )
                )
        with (folder / f"batches-trees{trees}.csv").open("w") as f:
            writer = csv.DictWriter(f, rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        for i, label in enumerate(
            ("Historical recompute (tokens)", "Recoverable hit rate (%)", "Entries", "Intervals")
        ):
            axes[i, col].set_ylabel(label)
            axes[i, col].grid(alpha=0.2)
            axes[i, col].legend(fontsize=8)
        axes[0, col].set_title(f"{trees} trees")
        for i in (0, 1):
            axes[i, col].set_xlabel("Admitted requests")
        axes[2, col].set_xlabel("Scheduler iteration")
        axes[3, col].set_xlabel("Final interval span (tokens)")
        prefill_ax = prefill_axes[0, col]
        prefill_ax.set_title(f"{trees} trees")
        prefill_ax.set_xlabel("Actual prefill tokens per forward")
        prefill_ax.set_xscale("log")
        prefill_ax.grid(alpha=0.25)
        if prefill_ax.lines:
            prefill_ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "comparison.png", dpi=180)
    fig.savefig(folder / "comparison.pdf")
    plt.close(fig)
    prefill_axes[0, 0].set_ylabel("CDF")
    prefill_fig.tight_layout()
    prefill_fig.savefig(folder / "prefill-token-cdf.png", dpi=180)
    prefill_fig.savefig(folder / "prefill-token-cdf.pdf")
    plt.close(prefill_fig)
    _write_prefill_statistics(folder, prefill_rows)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (folder / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# Online-analysis ablation",
        "",
        "Qwen3.6-27B token playback through the production Scheduler overlap loop, Engine cache preparation, PrefillAdder, DecodeManager and LogStateCacheManager. Two A800 TP ranks have mirrored logical frame ownership. Both policies use the same recorded tokens and arrival rules. Work states are allocated on admission, and decode allocates incrementally. GPU numerical kernels and completion events are replaced with synchronous CPU metadata adapters. No model arithmetic or GPU timing is simulated.",
        "",
        "| Trees | Policy | Prefix hit | Historical prefix hit | Historical recompute tokens | Mean Checkpoints before matching | Mean Logs before matching |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['trees']} | {r['mode']} | {r['hit_rate']:.2%} | {r['historical_hit_rate']:.2%} | {r['historical_recompute_tokens']:,} | {r['mean_checkpoint']:.2f} | {r['mean_log']:.2f} |"
        )
    lines += [
        "",
        "Historical recomputation is max(0, parent prompt length + actual generated length - recoverable prefix). Prefill tokens include both historical recomputation and new input. Prefix hit rate divides recoverable tokens by all input tokens; historical prefix hit rate uses only inherited parent tokens. The 32/64/128-tree histories retain their original 1024/512/256 output caps and actual EOS lengths. Compare policies within each case.",
        "",
        "Each JSON includes per-request recovery, per-batch Checkpoint span histograms and Log counts, per-rank frame usage/free space, memory-layout constants, model-update samples and eviction events. Both TP ranks have identical logical ownership; their bytes are not pooled into a single allocator.",
        "",
        "## Historical recomputation breakdown",
        "",
        "KV gap is inherited history beyond the token-prefix match. It includes both missing cacheable KV and the parent's unretained alignment tail. State gap is inherited history whose KV prefix matches but whose State endpoint is unavailable. Both are model-prefill tokens; replaying retained Logs to reconstruct work State is not counted here.",
        "",
        "| Trees | Policy | KV gap tokens | Cacheable KV gap | Alignment tail | State gap tokens |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['trees']} | {r['mode']} | {r['historical_kv_gap']:,} | {r['historical_cacheable_kv_gap']:,} | {r['historical_alignment_gap']:,} | {r['historical_state_gap']:,} |"
        )
    lines += [
        "",
        "## Frame distribution",
        "",
        "Request-weighted means before admission batches, per TP rank. One frame is 1 MiB. Log includes payload, convolution snapshots and bitmap-shared decay frames, including the sentinel Log. Checkpoint includes payload and convolution. Work contains admitted requests' actual work slots and the sentinel work slot. KV includes the sentinel KV frame. Every snapshot reconciles with the case's frame budget.",
        "",
        "| Trees | Policy | Log frames | Decay subset | Checkpoint frames | KV frames | Work frames | Free frames |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['trees']} | {r['mode']} | {r['mean_frames_log']:.1f} | {r['mean_frames_decay']:.1f} | {r['mean_frames_checkpoint']:.1f} | {r['mean_frames_kv']:.1f} | {r['mean_frames_work']:.1f} | {r['mean_frames_free']:.1f} |"
        )
    lines += [
        "", "## Prefill token distributions", "",
        "[Statistics by forward phase and request](prefill-distribution.md) include "
        "means, quantiles and threshold fractions. "
        "[The forward-token CDF](prefill-token-cdf.png) compares the two policies.",
    ]
    (folder / "report.md").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = list(range(len(summary)))
    bottom = [0.0] * len(summary)
    for kind, label in (
        ("checkpoint", "Checkpoint"),
        ("log_payload_conv", "Log payload + conv"),
        ("decay", "Log decay"),
        ("kv", "KV"),
        ("work", "Work"),
        ("free", "Free"),
    ):
        values = [r[f"mean_frames_{kind}"] for r in summary]
        ax.bar(positions, values, bottom=bottom, label=label)
        bottom = [a + b for a, b in zip(bottom, values)]
    ax.set_xticks(positions, [f"{r['trees']}\n{r['mode']}" for r in summary])
    ax.set_ylabel("Mean frames per rank (1 MiB/frame)")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(folder / "frame-distribution.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    build_report(parser.parse_args().results)
