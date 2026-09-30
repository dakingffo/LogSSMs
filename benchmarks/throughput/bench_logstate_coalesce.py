"""CPU-only benchmark for LogState prefill merge planning."""

import argparse
import statistics
from time import perf_counter_ns

import torch

from logssms.cache.logstate import OnlineAnalysisScheduler
from logssms.cache.prefix_cache.logstate import (
    LogRadixTree,
    LogStateEntry,
    LogStateKind,
)


PAGE_SIZE = 64
GRANULARITY = 32


def make_workload(requests: int, pages: int):
    tree = LogRadixTree(
        PAGE_SIZE, torch.device("cpu"), granularity=GRANULARITY
    )
    inputs = []
    length = pages * PAGE_SIZE
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
                request * pages + page,
                LogStateKind.CHECKPOINT,
                page * PAGE_SIZE,
                (page + 1) * PAGE_SIZE,
                span_pages=1,
            )
            for page in range(pages)
        )
        tree.insert(values, indices, entries)
        inputs.append(values)
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)
    return tree, inputs, scheduler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--pages", type=int, default=14)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    tree, inputs, scheduler = make_workload(args.requests, args.pages)
    round_info = scheduler.begin_round(tree)
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
        f"requests={args.requests} pages={args.pages} "
        f"planned_groups={len(expected.coalesce)} "
        f"coalesce_p50_ms={statistics.median(coalesce_samples):.3f} "
        f"analyze_p50_ms={statistics.median(samples):.3f}"
    )


if __name__ == "__main__":
    main()
