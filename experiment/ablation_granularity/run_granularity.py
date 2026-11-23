"""Compare three LogState granularities on the same OASST2 workload."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from benchmarks.bench_common import kill_server, wait_server_ready
from experiment.standard_multiturn.client import main as run_multiturn

_ROOT = Path(__file__).resolve().parent
_GRANULARITIES = (16, 32, 64)


def server_args(model_path: str, granularity: int, port: int, max_seq_len: int) -> list[str]:
    return [
        "--model-path", model_path,
        "--dtype", "bfloat16",
        "--tensor-parallel-size", "2",
        "--memory-ratio", "0.9",
        "--max-running-requests", "64",
        "--max-seq-len-override", str(max_seq_len),
        "--max-prefill-tokens", "8192",
        "--cuda-graph-max-bs", "64",
        "--scheduling-policy", "mixed",
        "--mixed-prefill-tokens", "128",
        "--cache-type", "logstate",
        "--granularity", str(granularity),
        "--eager-state", "false",
        "--merge-adjustment", "true",
        "--host", "127.0.0.1",
        "--port", str(port),
    ]


def run_point(args: argparse.Namespace, granularity: int, output: Path) -> dict:
    name = f"g{granularity}"
    log = output / f"{name}-server.log"
    command = [
        sys.executable, "-m", "experiment.launch_server.launch_logssms",
        *server_args(args.model_path, granularity, args.port, args.max_seq_len),
    ]
    env = {**os.environ, "LOGSSMS_PROFILE_SCHEDULER": "0"}
    for port in (args.port, args.port + 1):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as error:
                raise RuntimeError(f"Port {port} is unavailable") from error

    print(f"Starting g={granularity}. Server log: {log}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        print(shlex.join(command), file=stream, flush=True)
        server = subprocess.Popen(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            base_url = wait_server_ready(args.port, timeout=args.startup_timeout, proc=server)
            run_multiturn([
                "--api", "logssms", "--base-url", base_url, "--tokenizer", args.model_path,
                "--dataset", "oasst2", "--num-trees", "64", "--output-len", "1024",
                "--max-seq-len", str(args.max_seq_len), "--seed", str(args.seed),
                "--out-dir", str(output), "--name", name,
            ])
        except (RuntimeError, TimeoutError) as error:
            raise RuntimeError(f"g={granularity}: {error}. Server log: {log}") from error
        finally:
            kill_server(server)
    return json.loads((output / name / "report.json").read_text(encoding="utf-8"))


def summary_row(granularity: int, report: dict) -> str:
    stats = report["stats"]
    cache = stats["cache"]
    if cache is None or not stats["exact_continuations"] or not cache["expected_reusable_tokens"]:
        raise RuntimeError(f"g={granularity}: cache metrics or exact token continuations are missing")
    if not stats["ttft_requests"] or not stats["tpot_requests"]:
        raise RuntimeError(f"g={granularity}: insufficient generated tokens for latency metrics")
    return (
        f"| {granularity} | {cache['admitted_hit_fraction']:.2%} | "
        f"{stats['ttft_ms'][1]:.2f} | {stats['ttft_ms'][3]:.2f} | {stats['tpot_ms'][1]:.2f} | "
        f"{stats['output_tok_per_s']:.2f} | {stats['out_tokens']:,} |"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--granularity", type=int, nargs="+", choices=_GRANULARITIES, default=_GRANULARITIES)
    parser.add_argument("--max-seq-len", type=int, default=18432)
    parser.add_argument("--port", type=int, default=28300, help="The server also uses port + 1 for TP")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--out-dir", type=Path, default=_ROOT / "results")
    args = parser.parse_args(argv)
    if len(args.granularity) != len(set(args.granularity)):
        parser.error("Select each granularity once per run")
    if not 0 < args.port < 65535 or args.max_seq_len <= 0 or args.startup_timeout <= 0:
        parser.error("Ports must be valid and lengths and timeouts must be positive")

    output = args.out_dir.resolve() / f"{datetime.now():%Y%m%d-%H%M%S}-pid{os.getpid()}"
    output.mkdir(parents=True)
    rows = [
        "| Granularity | History hit rate | TTFT P50 (ms) | TTFT P99 (ms) | TPOT P50 (ms) | Output (tok/s) | Generated tokens |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    manifest = None
    for granularity in args.granularity:
        report = run_point(args, granularity, output)
        if manifest is None:
            manifest = report["manifest_sha256"]
        elif manifest != report["manifest_sha256"]:
            raise RuntimeError(f"g={granularity}: the benchmark workload differs from the first run")
        rows.append(summary_row(granularity, report))
        (output / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
