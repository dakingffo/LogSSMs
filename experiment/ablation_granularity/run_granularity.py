"""Replay recorded multiturn requests to compare LogState granularities."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GRANULARITIES = (16, 32, 64)
CONCURRENCIES = (16, 32, 48, 64)


def run_point(args, output, case, digest, granularity, concurrency):
    name = f"g{granularity}-c{concurrency}"
    config = output / f"{name}-case.json"
    config.write_text(json.dumps(dict(case, granularity=granularity), indent=2) + "\n")
    result_path = output / f"{name}.json"
    log_path = output / f"{name}.log"
    with log_path.open("w") as log:
        result = subprocess.run([
            sys.executable, "-m", "experiment.utils.cpu_replay.replay",
            "--case", str(config), "--history", str(args.history),
            "--out", str(result_path), "--mode", "online",
            "--batch-size", str(case["max_running_requests"]),
            "--tree-concurrency", str(concurrency), "--hit-materialization", "--skip-layout-snapshots",
        ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Replay failed for {name}. See {log_path}")
    return summarize_point(result_path, case, digest, granularity, concurrency)


def summarize_point(result_path, case, digest, granularity, concurrency):
    data = json.loads(result_path.read_text())
    if (data["trace_sha256"] != digest or data["tree_count"] != case["tree_count"]
            or data["peak_tree_concurrency"] != concurrency
            or data["request_hit_rate"] is None
            or data["num_frames"] * data["frame_bytes"] > case["cache_bytes_per_rank"]):
        raise RuntimeError(f"Unexpected replay result for {result_path.stem}: {result_path}")
    print(f"{result_path.stem}: history {data['historical_hit_rate']:.2%}, "
          f"request {data['request_hit_rate']:.2%}", flush=True)
    return {key: data[key] for key in (
        "granularity", "tree_concurrency", "historical_hit_rate", "request_hit_rate",
        "requests", "num_frames", "frame_bytes",
    )}


def smooth_boundary(x, y, y_limits):
    import numpy as np

    widths = np.diff(y)
    slopes = np.diff(x) / widths
    tangent = np.empty_like(y)
    tangent[0], tangent[-1] = slopes[0], slopes[-1]
    left, right = slopes[:-1], slopes[1:]
    shared = (left * right) > 0
    interior = tangent[1:-1]
    interior.fill(0)
    interior[shared] = 2 * left[shared] * right[shared] / (left[shared] + right[shared])
    # Cubic Hermite interpolation through the measured points.
    t = np.linspace(0, 1, 40, endpoint=False)
    ys = y[:-1, None] + widths[:, None] * t
    xs = ((2 * t**3 - 3 * t**2 + 1) * x[:-1, None]
          + (t**3 - 2 * t**2 + t) * (widths * tangent[:-1])[:, None]
          + (-2 * t**3 + 3 * t**2) * x[1:, None]
          + (t**3 - t**2) * (widths * tangent[1:])[:, None])
    low, high = y_limits
    left_edge = x[0] + tangent[0] * (low - y[0])
    right_edge = x[-1] + tangent[-1] * (high - y[-1])
    return (np.r_[left_edge, xs.ravel(), x[-1], right_edge],
            np.r_[low, ys.ravel(), y[-1], high])


def plot(points, output):
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    colors = {16: "#863da0", 32: "#00875f", 64: "#d56500"}
    markers = {16: "o", 32: "s", 48: "^", 64: "D"}
    concurrencies = sorted({point["tree_concurrency"] for point in points})
    granularities = sorted({point["granularity"] for point in points})

    def coordinates(point):
        return point["request_hit_rate"], point["historical_hit_rate"]

    for concurrency in concurrencies:
        series = sorted(
            (point for point in points if point["tree_concurrency"] == concurrency),
            key=lambda point: point["granularity"],
        )
        x, y = zip(*(coordinates(point) for point in series), strict=True)
        ax.plot(x, y, color="#687487", linewidth=1.25, alpha=0.65, zorder=3)
        ax.scatter(x, y, c=[colors[point["granularity"]] for point in series],
                   marker=markers[concurrency], s=45, edgecolors="white", linewidths=0.55, zorder=4)

    ax.set(xlabel="Request-level prefix hit rate", ylabel="Historical prefix hit rate")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.margins(x=0.12, y=0.12)
    x_limits, y_limits = ax.get_xlim(), ax.get_ylim()
    ax.set(xlim=x_limits, ylim=y_limits)

    for granularity in granularities:
        samples = sorted(
            {coordinates(point) for point in points if point["granularity"] == granularity},
            key=lambda xy: (xy[1], xy[0]),
        )
        if len(samples) < 2:
            continue
        x, y = np.array(samples).T
        if not np.all(np.diff(y) > 0):
            ax.plot(x, y, "--", color=colors[granularity], linewidth=1, zorder=2)
            continue
        ax.plot(*smooth_boundary(x, y, y_limits), "--",
                color=colors[granularity], linewidth=1.05, zorder=2)

    ax.grid(alpha=0.25)
    concurrency_legend = ax.legend(
        handles=[Line2D([], [], color="#465162", linestyle="None", marker=markers[c],
                        label=str(c)) for c in concurrencies],
        title="Concurrency", loc="upper left", ncol=2,
    )
    ax.add_artist(concurrency_legend)
    ax.legend(
        handles=[Line2D([], [], color=colors[g], linestyle="--", label=str(g))
                 for g in granularities],
        title="Granularity", loc="upper left", bbox_to_anchor=(0, 0.79),
    )
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"granularity.{suffix}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=HERE / "data/history.json")
    parser.add_argument("--case", type=Path, default=HERE / "data/case.json")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--granularity", nargs="+", type=int, choices=GRANULARITIES, default=GRANULARITIES)
    parser.add_argument("--concurrency", nargs="+", type=int, choices=CONCURRENCIES, default=CONCURRENCIES)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--plot-only", action="store_true", help="Regenerate the plot from reports in --out")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.plot_only and args.out is None:
        parser.error("--plot-only requires --out with existing reports")
    output = (args.out or HERE / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
    args.history = args.history.resolve()
    case = json.loads(args.case.read_text())
    digest = hashlib.sha256(args.history.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    matrix = [(g, c) for g in sorted(set(args.granularity)) for c in sorted(set(args.concurrency))]
    if args.plot_only:
        points = [
            summarize_point(output / f"g{g}-c{c}.json", case, digest, g, c)
            for g, c in matrix
        ]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            points = list(pool.map(lambda point: run_point(args, output, case, digest, *point), matrix))

    (output / "summary.json").write_text(json.dumps(dict(
        case=case, history_sha256=digest, points=points,
    ), indent=2) + "\n")
    plot(points, output)
    print(output / "granularity.pdf")


if __name__ == "__main__":
    main()
