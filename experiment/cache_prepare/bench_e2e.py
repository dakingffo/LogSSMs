"""Small graph-enabled local E2E with exact generated-token prefix continuity.

Compare a saved --source-root and current python/ using the same model and seed.
This checks the production path; laptop latency is not a server performance claim.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("python"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--port", type=int, default=28610)
    args = parser.parse_args()
    directory = Path("benchmark-results/metadata-study") / args.name
    directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.source_root.resolve()) + os.pathsep + str(Path.cwd())
    env["LOGSSMS_PROFILE_SCHEDULER"] = "1"
    env["LOGSSMS_PROFILE_SCHEDULER_PATH"] = str((directory / "scheduler.json").resolve())
    model = "/home/daking/models/huggingface/Qwen3.5-0.8B"
    command = [sys.executable, "-m", "logssms", "--model-path", model,
               "--tensor-parallel-size", "1", "--max-running-requests", "4",
               "--max-seq-len-override", "2048", "--max-decode-tokens", "4",
               "--cuda-graph-max-bs", "4", "--memory-ratio", "0.65",
               "--num-pages", "2048", "--cache-type", "logstate",
               "--granularity", "32", "--eager-state", "false",
               "--memorized-evict", "true", "--port", str(args.port)]
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "server.log").open("w") as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        try:
            deadline = time.monotonic() + 300
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited: {directory / 'server.log'}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=1):
                        break
                except (OSError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Server startup timed out")
                    time.sleep(1)
            subprocess.run([
                sys.executable, "-m", "experiment.random_multiturn",
                "--api", "logssms", "--base-url", f"http://127.0.0.1:{args.port}",
                "--tokenizer", model, "--input-len", "128", "--append-len", "64",
                "--output-len", "64", "--num-turns", "8", "--num-sessions", "4",
                "--num-concurrent-sessions", "4", "--max-seq-len", "2048",
                "--seed", "0", "--name", "client", "--out-dir", str(directory),
                "--no-warmup",
            ], env=env, check=True)
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGINT)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGTERM)
                    server.wait(timeout=15)
    subprocess.run([
        sys.executable, "-m", "experiment.random_multiturn.profile",
        "--run-dir", str(directory / "client"),
        "--profile", str(directory / "scheduler.rank0.json"),
    ], env=env, check=True)


if __name__ == "__main__":
    main()
