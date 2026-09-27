"""Run a matched Radix256 / LogState32 multi-turn A/B workload.

All unrecognized flags are passed to bench_decode.py as common server flags.
Example::

    python benchmarks/throughput/compare_cache.py \
      --model-path /path/to/model --dataset sharegpt \
      --num-concurrent-sessions 1,16,32,128 --num-sessions 128 \
      --tensor-parallel-size 2 --max-running-requests 128 \
      --output-prefix comparisons/qwen36_ar

The fixed-history mode keeps the prompt tokens independent of the cache policy.
"""

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", default="sharegpt")
    parser.add_argument("--num-concurrent-sessions", default="1,16,32,128")
    parser.add_argument("--num-sessions", type=int, default=128)
    parser.add_argument("--num-turns", type=int, default=8)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-prefix", default="comparisons/cache_ar",
        help="relative to benchmark-results/; a policy suffix and .json are appended",
    )
    args, server_flags = parser.parse_known_args()
    if args.num_sessions < max(map(int, args.num_concurrent_sessions.split(","))):
        parser.error("--num-sessions must cover the largest concurrency")
    if args.num_turns <= 1:
        parser.error("the matched cache workload requires multiple turns")
    forbidden = {"--cache-type", "--page-size", "--granularity", "--out", "--history"}
    if any(flag.split("=")[0] in forbidden for flag in server_flags):
        parser.error("cache policy and output paths are controlled by this runner")

    benchmark = Path(__file__).with_name("bench_decode.py")
    common = [
        sys.executable, str(benchmark),
        "--model-path", args.model_path,
        "--dataset", args.dataset,
        "--num-concurrent-sessions", args.num_concurrent_sessions,
        "--num-sessions", str(args.num_sessions),
        "--num-turns", str(args.num_turns),
        "--output-len", str(args.output_len),
        "--seed", str(args.seed),
        "--history-mode", "fixed",
        "--ignore-eos",
        *server_flags,
    ]
    for name, cache_flags in (
        ("radix256", ["--cache-type", "radix", "--page-size", "256"]),
        ("logstate32", ["--cache-type", "logstate", "--granularity", "32"]),
    ):
        subprocess.run(
            [
                *common,
                *cache_flags,
                "--out", f"{args.output_prefix}_{name}.json",
                "--history", f"{args.output_prefix}_{name}_history.json",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
