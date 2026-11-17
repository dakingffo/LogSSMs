from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay standard multiturn histories to ablate online eviction."
    )
    parser.add_argument(
        "--trees", type=int, nargs="+", choices=(32, 64, 128), default=[32, 64, 128]
    )
    parser.add_argument("--cases", type=Path, default=HERE / "cases.json")
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE
        / "results"
        / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Maximum new arrivals per production scheduler iteration")
    parser.add_argument("--min-sampling-len", type=int, default=8192)
    parser.add_argument("--hit-half-life", type=int, default=1024)
    args = parser.parse_args()
    if min(args.workers, args.batch_size, args.min_sampling_len) < 1:
        parser.error("workers, batch size and sampling window must be positive")
    cases = json.loads(args.cases.read_text())
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "execution": "production-scheduler-v1",
        "source": "recorded_model_tokens",
        "cases": {},
        "batch_size": args.batch_size,
        "min_sampling_len": args.min_sampling_len,
        "hit_half_life": args.hit_half_life,
        "python": sys.executable,
        "simulation_sha256": {
            name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in ("run_online_analysis.py", "replay_common.py", "cpu_common.py", "report_online_analysis.py")
        },
    }
    for trees in dict.fromkeys(args.trees):
        case = cases[str(trees)].copy()
        history = (ROOT / case["history"]).resolve()
        if not history.is_file():
            parser.error(f"Missing original history: {history}")
        case.update(history=str(history), sha256=hashlib.sha256(history.read_bytes()).hexdigest())
        manifest["cases"][str(trees)] = case
    manifest["commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    patch = subprocess.check_output(["git", "diff", "--binary", "--", "python", "csrc"], cwd=ROOT)
    manifest["production_patch_sha256"] = hashlib.sha256(patch).hexdigest()
    (args.out / "production.patch").write_bytes(patch)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def run(trees: str, mode: str) -> dict:
        case = manifest["cases"][trees]
        destination = args.out / f"{mode}-trees{trees}.json"
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=os.pathsep.join((str(ROOT / "python"), str(ROOT))))
        command = [
            sys.executable,
            "-m",
            "experiment.ablation_online_analysis.replay_common",
            "--history",
            case["history"],
            "--out",
            str(destination),
            "--case",
            str(args.out / f"case-trees{trees}.json"),
            "--mode",
            mode,
            "--batch-size",
            str(args.batch_size),
            "--min-sampling-len",
            str(args.min_sampling_len),
        ]
        command.extend(["--hit-half-life", str(args.hit_half_life)])
        (args.out / f"case-trees{trees}.json").write_text(json.dumps(case, indent=2))
        with destination.with_suffix(".log").open("w") as stream:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
        if result.returncode:
            raise RuntimeError(f"Replay failed: {destination.with_suffix('.log')}")
        data = json.loads(destination.read_text())
        if data["trace_sha256"] != case["sha256"]:
            raise RuntimeError("History changed during replay")
        return {
            "trees": int(trees),
            "mode": mode,
            "hit": data["hit_rate"],
            "verified_parent_prefixes": data["totals"]["verified_parent_prefixes"],
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(run, trees, mode)
            for trees in manifest["cases"]
            for mode in ("default", "online")
        ]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    subprocess.run(
        [sys.executable, str(HERE / "report_online_analysis.py"), "--results", str(args.out)], check=True
    )
    print(args.out / "report.md")


if __name__ == "__main__":
    main()
