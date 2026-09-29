"""Run a resumable TP=2 ShareGPT comparison of Radix and LogState.

Each point starts a fresh server, uses the same sampled sessions for every
cache policy, and keeps EOS handling enabled. Reports and logs are written
under benchmark-results/<result-root>. Use --dry-run to inspect the matrix.

Example:
    python benchmarks/throughput/compare_multiturn_matrix.py \
        --model-path /root/autodl-tmp/models/Qwen3.6-27B \
        --point 8:16:256 --point 16:32:512 \
        --result-root qwen36-tp2-multiturn
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


POLICIES = {
    "radix64": ("--cache-type", "radix", "--page-size", "64"),
    "radix128": ("--cache-type", "radix", "--page-size", "128"),
    "radix256": ("--cache-type", "radix", "--page-size", "256"),
    "logstate32": (
        "--cache-type", "logstate", "--granularity", "32",
        "--memorized-evict", "true", "--eager-state", "false",
    ),
}
DEFAULT_SERVER_FLAGS = (
    "--dtype", "bfloat16",
    "--tensor-parallel-size", "2",
    "--memory-ratio", "0.9",
    "--max-prefill-tokens", "8192",
)


def _ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(","))
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return result


def _points(args: argparse.Namespace) -> list[tuple[int, int, int]]:
    if args.point:
        try:
            points = [tuple(map(int, value.split(":"))) for value in args.point]
        except ValueError as error:
            raise ValueError("--point must be turns:concurrency:output_len") from error
        if any(len(point) != 3 for point in points):
            raise ValueError("--point must be turns:concurrency:output_len")
    else:
        points = list(itertools.product(
            args.turns, args.concurrencies, args.output_lens
        ))
    if any(turns < 2 or concurrency < 1 or output_len < 1
           for turns, concurrency, output_len in points):
        raise ValueError("turns must be >= 2; concurrency/output_len must be positive")
    if any(concurrency > args.num_sessions for _, concurrency, _ in points):
        raise ValueError("--num-sessions must cover every concurrency point")
    return list(dict.fromkeys(points))


def _complete(path: Path, key: str, requests: int) -> bool:
    if not path.exists():
        return False
    try:
        stats = json.loads(path.read_text())[key]
    except (OSError, ValueError, KeyError):
        return False
    return stats.get("n") == requests and stats.get("chunks_ok") is True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", default="sharegpt")
    parser.add_argument("--turns", type=_ints, default=(8, 16))
    parser.add_argument("--concurrencies", type=_ints, default=(32, 64))
    parser.add_argument("--output-lens", type=_ints, default=(512, 1024))
    parser.add_argument(
        "--point", action="append",
        help="one turns:concurrency:output_len point; repeatable",
    )
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--num-sessions", type=int, default=64)
    parser.add_argument("--max-running-requests", type=int, default=64)
    parser.add_argument("--max-seq-len-override", type=int, default=32768)
    parser.add_argument("--max-decode-tokens", type=int, default=32)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=28401)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--result-root", default="qwen36-tp2-multiturn")
    parser.add_argument(
        "--save-history", action="store_true",
        help="retain full request/response histories for input-equivalence audits",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    points = _points(args)
    policies = tuple(args.policies.split(","))
    if not policies or any(policy not in POLICIES for policy in policies):
        parser.error(f"--policies must be drawn from {','.join(POLICIES)}")

    root = (Path(__file__).resolve().parents[2] / "benchmark-results"
            / args.result_root).resolve()
    benchmark_root = (Path(__file__).resolve().parents[2] / "benchmark-results").resolve()
    if not root.is_relative_to(benchmark_root) or root == benchmark_root:
        parser.error("--result-root must be inside benchmark-results/")
    server_flags = [
        *DEFAULT_SERVER_FLAGS,
        "--max-running-requests", str(args.max_running_requests),
        "--max-seq-len-override", str(args.max_seq_len_override),
        "--max-decode-tokens", str(args.max_decode_tokens),
        "--cuda-graph-max-bs", str(args.cuda_graph_max_bs),
    ]
    config = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "num_sessions": args.num_sessions,
        "seed": args.seed,
        "port": args.port,
        "request_timeout": args.request_timeout,
        "history_mode": "generated",
        "ignore_eos": False,
        "save_history": args.save_history,
        "server_flags": server_flags,
    }
    if args.dry_run:
        for point in points:
            for policy in policies:
                print(f"turns={point[0]} concurrency={point[1]} "
                      f"output_len={point[2]} cache={policy}")
        print(f"{len(points) * len(policies)} cases; result root: {root}")
        return 0

    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        parser.error(f"{manifest} belongs to a different benchmark configuration")
    manifest.write_text(json.dumps(config, indent=2) + "\n")
    (root / "reports").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    if args.save_history:
        (root / "histories").mkdir(exist_ok=True)
    benchmark = Path(__file__).with_name("bench_decode.py")
    failed = False
    for turns, concurrency, output_len in points:
        for policy in policies:
            name = f"t{turns}_c{concurrency}_o{output_len}_{policy}"
            report = root / "reports" / f"{name}.json"
            log = root / "logs" / f"{name}.log"
            if _complete(report, str(concurrency), args.num_sessions * turns):
                print(f"SKIP {name}: complete report exists", flush=True)
                continue
            command = [
                sys.executable, str(benchmark),
                "--model-path", args.model_path,
                "--dataset", args.dataset,
                "--num-concurrent-sessions", str(concurrency),
                "--num-sessions", str(args.num_sessions),
                "--num-turns", str(turns),
                "--output-len", str(output_len),
                "--seed", str(args.seed),
                "--history-mode", "generated",
                "--request-timeout", str(args.request_timeout),
                "--no-pbar", "--port", str(args.port),
                "--out", str(report),
                *server_flags, *POLICIES[policy],
            ]
            if args.save_history:
                command.extend((
                    "--history", str(root / "histories" / f"{name}.json")
                ))
            print(f"RUN {name}", flush=True)
            with log.open("w") as output:
                result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
            if result.returncode or not _complete(
                report, str(concurrency), args.num_sessions * turns
            ):
                failed = True
                print(f"FAIL {name}: exit={result.returncode}; log={log}", flush=True)
                if not args.continue_on_error:
                    return 1
                continue
            stats = json.loads(report.read_text())[str(concurrency)]
            print(
                f"DONE {name}: {stats['out_tokens']} output tokens, "
                f"{stats['output_tok_per_s']:.1f} tok/s, "
                f"{stats['early_stops']} early stops",
                flush=True,
            )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
