"""Attach a LogSSMs scheduler profile after the server has shut down."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import save_json


def attach_cache_matches(records: list[dict], profile_path: Path) -> dict:
    if not profile_path.is_file():
        raise RuntimeError(f"Scheduler cache profile was not written: {profile_path}")
    profile = json.loads(profile_path.read_text())
    samples = {
        sample["tag"]: sample
        for sample in profile["cache"]["match_samples"]
    }
    expected_total = 0
    matched_total = 0
    recoverable_total = 0
    continuations = 0
    for record in records:
        for turn in record["turns"]:
            sample = samples.get(turn["tag"])
            if sample is None:
                raise RuntimeError(f"No cache match sample for {turn['tag']}")
            turn["cache_match"] = {
                name: sample.get(name) for name in (
                    "tree_match_len", "recoverable_len", "admitted_cached_len",
                    "arrival_tree_match_len", "state_recovery_gap",
                    "admission_wait_ms",
                )
            }
            expected = turn["expected_reusable_len"]
            if expected:
                continuations += 1
                expected_total += expected
                matched_total += min(expected, sample["tree_match_len"])
                recoverable_total += min(expected, sample["admitted_cached_len"])
    return {
        "source": "logssms.scheduler_profile",
        "continuations": continuations,
        "expected_reusable_tokens": expected_total,
        "tree_matched_tokens": matched_total,
        "admitted_cached_tokens": recoverable_total,
        "tree_match_fraction": matched_total / expected_total if expected_total else 0.0,
        "admitted_hit_fraction": (
            recoverable_total / expected_total if expected_total else 0.0
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args(argv)
    report_path = args.run_dir / "report.json"
    report = json.loads(report_path.read_text())
    if report["api"] != "logssms" or len(report["points"]) != 1:
        parser.error("expected a one-point LogSSMs client report")
    concurrency, stats = next(iter(report["points"].items()))
    history_path = args.run_dir / f"history-c{concurrency}.json"
    history = json.loads(history_path.read_text())
    stats["cache"] = attach_cache_matches(history["sessions"], args.profile)
    stats["profile_path"] = str(args.profile.resolve())
    save_json(str(history_path), history)
    save_json(str(report_path), report)
    print(f"Attached cache profile to {report_path}")


if __name__ == "__main__":
    main()
