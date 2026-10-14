"""Plot mixed-workload TTFT and cache hits from completed tree-client reports."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


_SYSTEMS = ("logssms", "vllm", "sglang")
_LABELS = ("LogSSMs", "vLLM", "SGLang")
_COLORS = ("#3575b7", "#e28b36", "#4b9c68")


def load_reports(paths: list[Path]) -> dict:
    groups = defaultdict(list)
    fingerprints = {}
    seen = set()
    for path in paths:
        report = json.loads(path.read_text())
        if report["status"] != "complete":
            raise ValueError(f"Incomplete run: {path}")
        stats = report["stats"]
        if stats["cache"] is None:
            raise ValueError(f"Cache hits are unmeasured: {path}")
        if stats.get("ttft_requests", 1) == 0:
            raise ValueError(f"No first-token latency samples: {path}")
        manifest = json.loads((path.parent / "manifest.json").read_text())
        key = stats["trees"], manifest["seed"]
        if key in fingerprints and fingerprints[key] != report["manifest_sha256"]:
            raise ValueError(f"Different workloads for tree count/seed {key}")
        fingerprints[key] = report["manifest_sha256"]
        identity = (*key, report["api"])
        if identity in seen:
            raise ValueError(f"Duplicate system/tree-count/seed: {identity}")
        seen.add(identity)
        groups[(stats["trees"], report["api"])].append((manifest["seed"], stats))
    for trees in {key[0] for key in groups}:
        seeds = []
        for api in _SYSTEMS:
            if (trees, api) not in groups:
                raise ValueError(f"Missing {api} report for {trees} trees")
            seeds.append({seed for seed, _ in groups[(trees, api)]})
        if not all(seed_set == seeds[0] for seed_set in seeds):
            raise ValueError(f"Unpaired seeds for {trees} trees")
    return dict(groups)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    groups = load_reports(args.reports)
    import matplotlib.pyplot as plt

    counts = sorted({key[0] for key in groups})
    figure, ttft_axis = plt.subplots(figsize=(9, 4.5))
    hit_axis = ttft_axis.twinx()
    for index, (api, label, color) in enumerate(zip(_SYSTEMS, _LABELS, _COLORS, strict=True)):
        positions = [group + (index - 1) * .24 for group in range(len(counts))]
        samples = [[stats for _, stats in groups[(count, api)]] for count in counts]
        latencies = [[stats["ttft_ms"][0] for stats in sample] for sample in samples]
        ttft_axis.bar(
            positions, [mean(values) for values in latencies], width=.22,
            yerr=[stdev(values) if len(values) > 1 else 0 for values in latencies],
            color=color, alpha=.7, capsize=3, label=f"{label} TTFT",
        )
        hits = [
            sum(stats["cache"]["total_cached_tokens"] for stats in sample)
            / sum(stats["in_tokens"] for stats in sample)
            for sample in samples
        ]
        hit_axis.plot(positions, [100 * hit for hit in hits], color=color,
                      marker="o", label=f"{label} cache hits")
    ttft_axis.set_xticks(range(len(counts)), [str(count) for count in counts])
    ttft_axis.set_xlabel("Conversation trees")
    ttft_axis.set_ylabel("Mean TTFT (ms)")
    hit_axis.set_ylabel("Cached input tokens (%)")
    ttft_axis.set_ylim(bottom=0)
    hit_axis.set_ylim(0, 100)
    handles, labels = ttft_axis.get_legend_handles_labels()
    other_handles, other_labels = hit_axis.get_legend_handles_labels()
    figure.legend(handles + other_handles, labels + other_labels,
                  loc="upper center", ncol=3, fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, .88))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=180)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
