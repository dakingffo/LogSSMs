"""CPU-only benchmark for LogState prefill merge planning."""

import argparse
import statistics
from time import perf_counter_ns

import torch

from logssms.cache.logstate.online_analysis_scheduler import (
    OnlineAnalysisScheduler,
)
from logssms.cache.logstate.prefix_cache import LogRadixTree, LogStateEntry, LogStateKind


GRANULARITY = 32
CHECKPOINT_GRANULES = 2


def make_workload(requests: int, intervals: int, granularity: int, checkpoint_granules: int):
    tree = LogRadixTree(
        granularity, torch.device("cpu"), checkpoint_log_count=checkpoint_granules
    )
    inputs = []
    checkpoint_tokens = granularity * checkpoint_granules
    length = intervals * checkpoint_tokens
    for request in range(requests):
        values = torch.arange(
            request * 100_000, request * 100_000 + length,
            dtype=torch.int32,
        )
        indices = torch.arange(
            request * length, (request + 1) * length,
            dtype=torch.int32,
        )
        entries = tuple(
            LogStateEntry(
                request * intervals + interval,
                LogStateKind.CHECKPOINT,
                interval * checkpoint_tokens,
                (interval + 1) * checkpoint_tokens,
                span_granules=checkpoint_granules,
            )
            for interval in range(intervals)
        )
        tree.insert(values, indices, entries)
        inputs.append(values)
    scheduler = OnlineAnalysisScheduler(granularity, checkpoint_granules)
    return tree, inputs, scheduler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--intervals", type=int, default=14)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--granularity", type=int, choices=(16, 32, 64), default=GRANULARITY)
    parser.add_argument("--checkpoint-granules", type=int, default=CHECKPOINT_GRANULES)
    args = parser.parse_args()
    if min(args.requests, args.intervals, args.iterations, args.checkpoint_granules) < 1:
        parser.error("Workload sizes must be positive")

    tree, inputs, scheduler = make_workload(args.requests, args.intervals, args.granularity, args.checkpoint_granules)
    traces = tuple(tree.trace(values) for values in inputs)
    round_info = scheduler.begin_round(tree, traces)
    expected = scheduler.analyze(tree, inputs, round_info)
    samples = []
    coalesce_samples = []
    for _ in range(args.iterations):
        step_timing_ms = {}
        start = perf_counter_ns()
        plan = scheduler.analyze(
            tree, inputs, round_info,
            collect_diagnostics=True,
            step_timing_ms=step_timing_ms,
        )
        samples.append((perf_counter_ns() - start) / 1e6)
        coalesce_samples.append(step_timing_ms["coalesce_plan_ms"])
        assert plan.coalesce == expected.coalesce

    print(
        f"requests={args.requests} intervals={args.intervals} "
        f"planned_groups={len(expected.coalesce)} "
        f"coalesce_p50_ms={statistics.median(coalesce_samples):.3f} "
        f"analyze_p50_ms={statistics.median(samples):.3f}"
    )


if __name__ == "__main__":
    main()
