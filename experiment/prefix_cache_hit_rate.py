"""Measure where real multi-turn requests hit a page-based prefix cache.

Every assistant response represents one model invocation: the preceding messages
form the query, and the history after the response forms the cache entry committed
by that invocation. Prefix endpoints are found by exact token-ID comparison in a
compact radix tree.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from array import array
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from datasets.parser import DatasetMessage, DatasetRecord, iter_dataset


@dataclass
class RadixNode:
    children: dict[int, "RadixEdge"] = field(default_factory=dict)
    nearest_tail: int = 0


@dataclass
class RadixEdge:
    tokens: array
    child : RadixNode


class TokenRadixTree:
    """A compact token trie retaining every completed cache entry."""

    def __init__(self):
        self.root = RadixNode()

    @staticmethod
    def _update_nearest_tail(node: RadixNode) -> None:
        node.nearest_tail = min(
            (
                len(edge.tokens) + edge.child.nearest_tail
                for edge in node.children.values()
            ),
            default=0,
        )

    @staticmethod
    def _common_prefix(lhs: array, lhs_begin: int, rhs: array) -> int:
        length = min(len(lhs) - lhs_begin, len(rhs))
        offset = 0
        while offset < length and lhs[lhs_begin + offset] == rhs[offset]:
            offset += 1
        return offset

    def lookup(self, tokens: array) -> tuple[int, int | None]:
        """Return exact token LCP length and distance to its nearest cache leaf."""
        if not self.root.children:
            return 0, None

        node = self.root
        offset = 0
        while offset < len(tokens):
            edge = node.children.get(tokens[offset])
            if edge is None:
                return offset, node.nearest_tail

            matched = self._common_prefix(tokens, offset, edge.tokens)
            offset += matched
            if matched != len(edge.tokens):
                return (
                    offset,
                    len(edge.tokens) - matched + edge.child.nearest_tail,
                )
            node = edge.child

        return offset, node.nearest_tail

    def insert(self, tokens: array) -> None:
        node = self.root
        path = [node]
        offset = 0
        while offset < len(tokens):
            first = tokens[offset]
            edge = node.children.get(first)
            if edge is None:
                child = RadixNode()
                node.children[first] = RadixEdge(array("I", tokens[offset:]), child)
                for ancestor in reversed(path):
                    self._update_nearest_tail(ancestor)
                return

            matched = self._common_prefix(tokens, offset, edge.tokens)
            if matched == len(edge.tokens):
                offset += matched
                node = edge.child
                path.append(node)
                continue

            old_child = edge.child
            old_suffix = array("I", edge.tokens[matched:])
            middle = RadixNode()
            middle.children[old_suffix[0]] = RadixEdge(old_suffix, old_child)
            node.children[first] = RadixEdge(
                array("I", edge.tokens[:matched]),
                middle,
            )

            offset += matched
            if offset < len(tokens):
                new_suffix = array("I", tokens[offset:])
                middle.children[new_suffix[0]] = RadixEdge(
                    new_suffix,
                    RadixNode(),
                )
            self._update_nearest_tail(middle)
            for ancestor in reversed(path):
                self._update_nearest_tail(ancestor)
            return


@dataclass
class PageStatistics:
    hits       : int = 0
    branch_hits: int = 0
    all_bins   : Counter[int] = field(default_factory=Counter)
    branch_bins: Counter[int] = field(default_factory=Counter)


class HitDensity:
    def __init__(self, page_sizes: Sequence[int], bin_width: int):
        if any(page_size <= 0 for page_size in page_sizes):
            raise ValueError("page sizes must be positive")
        if bin_width <= 0:
            raise ValueError("bin width must be positive")
        self.page_sizes = tuple(page_sizes)
        self.bin_width = bin_width
        self.num_queries = 0
        self.raw_hits = 0
        self.stats = {page_size: PageStatistics() for page_size in self.page_sizes}

    def observe(
        self,
        matched      : int,
        tail_distance: int | None,
    ) -> None:
        self.num_queries += 1
        if matched == 0 or tail_distance is None:
            return

        self.raw_hits += 1
        raw_end = matched + tail_distance
        is_branch = tail_distance > 0

        for page_size, stats in self.stats.items():
            hit_end = matched // page_size * page_size
            if hit_end == 0:
                continue
            cache_end = raw_end // page_size * page_size
            distance = cache_end - hit_end
            bucket = distance // self.bin_width
            stats.hits += 1
            stats.all_bins[bucket] += 1
            if is_branch:
                stats.branch_hits += 1
                stats.branch_bins[bucket] += 1

    def rows(self) -> Iterator[dict[str, int | float | str]]:
        for page_size, stats in self.stats.items():
            for scope, bins in (
                ("all", stats.all_bins),
                ("branch", stats.branch_bins),
            ):
                for bucket, count in sorted(bins.items()):
                    begin = bucket * self.bin_width
                    yield {
                        "page_size"     : page_size,
                        "scope"         : scope,
                        "bin_begin"     : begin,
                        "bin_end"       : begin + self.bin_width,
                        "hit_count"     : count,
                        "num_queries"   : self.num_queries,
                        "density"       : count / self.bin_width / self.num_queries,
                    }

    def summary(self) -> dict[str, Any]:
        return {
            "num_queries": self.num_queries,
            "raw_hits"   : self.raw_hits,
            "raw_hit_rate": self.raw_hits / self.num_queries,
            "bin_width"  : self.bin_width,
            "page_sizes" : {
                str(page_size): {
                    "hits"           : stats.hits,
                    "hit_rate"       : stats.hits / self.num_queries,
                    "branch_hits"    : stats.branch_hits,
                    "branch_hit_rate": stats.branch_hits / self.num_queries,
                }
                for page_size, stats in self.stats.items()
            },
        }

    def ccdf_rows(self) -> Iterator[dict[str, int | float | str]]:
        for page_size, stats in self.stats.items():
            for scope, bins in (
                ("all", stats.all_bins),
                ("branch", stats.branch_bins),
            ):
                remaining = sum(bins.values())
                for bucket in range(max(bins, default=-1) + 1):
                    yield {
                        "page_size"  : page_size,
                        "scope"      : scope,
                        "distance"   : bucket * self.bin_width,
                        "tail_count" : remaining,
                        "ccdf"       : remaining / self.num_queries,
                        "num_queries": self.num_queries,
                    }
                    remaining -= bins.get(bucket, 0)


def _messages(messages: Sequence[DatasetMessage]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {
            "role"   : message.role,
            "content": message.content,
        }
        if message.tool_calls is not None:
            item["tool_calls"] = list(message.tool_calls)
        if message.tool_call_id is not None:
            item["tool_call_id"] = message.tool_call_id
        result.append(item)
    return result


def _tokenize(tokenizer: Any, messages: Sequence[DatasetMessage], generate: bool) -> array:
    token_ids = tokenizer.apply_chat_template(
        _messages(messages),
        tokenize=True,
        add_generation_prompt=generate,
    )
    return array("I", token_ids)


def _model_calls(
    record   : DatasetRecord,
    max_turns: int | None,
) -> Iterator[tuple[tuple[DatasetMessage, ...], tuple[DatasetMessage, ...]]]:
    history: list[DatasetMessage] = []
    turns = 0
    for message in record.messages:
        if message.role == "system":
            continue
        if message.role == "assistant":
            if history:
                yield tuple(history), tuple((*history, message))
                turns += 1
                if max_turns is not None and turns >= max_turns:
                    return
        history.append(message)


def _ordered_calls(
    records  : Sequence[DatasetRecord],
    order    : str,
    max_turns: int | None,
) -> Iterator[tuple[int, tuple[DatasetMessage, ...], tuple[DatasetMessage, ...]]]:
    if order == "sequential":
        for session_id, record in enumerate(records):
            for query, completed in _model_calls(record, max_turns):
                yield session_id, query, completed
        return

    calls = [
        iter(_model_calls(record, max_turns))
        for record in records
    ]
    pending = set(range(len(calls)))
    while pending:
        for session_id in tuple(sorted(pending)):
            try:
                query, completed = next(calls[session_id])
            except StopIteration:
                pending.remove(session_id)
                continue
            yield session_id, query, completed


def run_experiment(
    tokenizer : Any,
    records   : Sequence[DatasetRecord],
    page_sizes: Sequence[int],
    bin_width : int,
    order     : str,
    max_turns : int | None,
) -> HitDensity:
    tree = TokenRadixTree()
    density = HitDensity(page_sizes, bin_width)

    for _session_id, query_messages, completed_messages in _ordered_calls(
        records, order, max_turns
    ):
        query = _tokenize(tokenizer, query_messages, generate=True)
        matched, tail_distance = tree.lookup(query)
        density.observe(matched, tail_distance)

        completed = _tokenize(tokenizer, completed_messages, generate=False)
        tree.insert(completed)

    return density


def _select_records(
    dataset         : str,
    seed            : int,
    max_trajectories: int | None,
    min_isl         : int | None,
    max_isl         : int | None,
) -> list[DatasetRecord]:
    generator = random.Random(seed)
    records: list[DatasetRecord] = []
    eligible = 0
    for record in iter_dataset(dataset):
        max_input_len = record.metadata.get("max_isl")
        if min_isl is not None and (max_input_len is None or max_input_len < min_isl):
            continue
        if max_isl is not None and (max_input_len is None or max_input_len > max_isl):
            continue

        eligible += 1
        if max_trajectories is None or len(records) < max_trajectories:
            records.append(record)
        else:
            replace = generator.randrange(eligible)
            if replace < max_trajectories:
                records[replace] = record

    generator.shuffle(records)
    return records


def _write_results(
    density      : HitDensity,
    output_dir   : Path,
    configuration: dict[str, Any],
    plot_max_distance: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(density.rows())

    with (output_dir / "density.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    _write_ccdf(density, output_dir / "ccdf.csv")

    summary = density.summary()
    summary["configuration"] = configuration
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    dataset_title = {
        "swe_agent"                  : "SWE-agent",
        "agentic_coding_trajectories": "Thoughtworks agentic-coding",
    }.get(configuration["dataset"], configuration["dataset"])

    _plot(
        density,
        output_dir / "hit_density.png",
        dataset_title=dataset_title,
        branch_only=False,
        max_distance=plot_max_distance,
    )
    _plot(
        density,
        output_dir / "branch_hit_density.png",
        dataset_title=dataset_title,
        branch_only=True,
        max_distance=plot_max_distance,
    )
    _plot_ccdf(
        density,
        output_dir / "branch_hit_ccdf.png",
        dataset_title=dataset_title,
        max_distance=plot_max_distance,
    )


def _write_ccdf(density: HitDensity, path: Path) -> None:
    rows = list(density.ccdf_rows())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def _plot(
    density      : HitDensity,
    path         : Path,
    dataset_title: str,
    branch_only  : bool,
    max_distance : int | None = None,
) -> None:
    import matplotlib.pyplot as plt

    attribute = "branch_bins" if branch_only else "all_bins"
    max_bucket = max(
        (
            max(getattr(stats, attribute), default=0)
            for stats in density.stats.values()
        ),
        default=0,
    )
    x = [(bucket + 0.5) * density.bin_width for bucket in range(max_bucket + 1)]

    figure, axis = plt.subplots(figsize=(8, 4.8))
    for page_size, stats in density.stats.items():
        bins = getattr(stats, attribute)
        y = [
            bins.get(bucket, 0) / density.bin_width / density.num_queries
            for bucket in range(max_bucket + 1)
        ]
        label = (
            "exact match (page size = 1)"
            if page_size == 1 else f"page size = {page_size}"
        )
        axis.plot(x, y, label=label)

    axis.set_xlabel("Distance from hit endpoint to cached tail (tokens)")
    axis.set_ylabel("Hit density (hits / token / query)")
    axis.set_title(
        f"{dataset_title} branch-hit density"
        if branch_only else f"{dataset_title} hit density"
    )
    if max_distance is not None:
        axis.set_xlim(0, max_distance)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_ccdf(
    density      : HitDensity,
    path         : Path,
    dataset_title: str,
    max_distance : int | None = None,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.8))
    reference: tuple[list[int], list[float]] | None = None
    for page_size, stats in density.stats.items():
        remaining = sum(stats.branch_bins.values())
        x: list[int] = []
        y: list[float] = []
        for bucket in range(max(stats.branch_bins, default=-1) + 1):
            distance = bucket * density.bin_width
            if distance > 0 and remaining > 0:
                x.append(distance)
                y.append(remaining / density.num_queries)
            remaining -= stats.branch_bins.get(bucket, 0)
        label = (
            "exact match (page size = 1)"
            if page_size == 1 else f"page size = {page_size}"
        )
        axis.plot(x, y, marker=".", label=label)
        if reference is None and x:
            reference_x = [x[0], max_distance or x[-1]]
            reference_y = [y[0], y[0] * x[0] / reference_x[1]]
            reference = reference_x, reference_y

    if reference is not None:
        axis.plot(*reference, "--", color="black", label=r"$d^{-1}$ reference")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Distance from hit endpoint to cached tail (tokens)")
    axis.set_ylabel(r"CCDF: $P(D \geq d)$")
    axis.set_title(f"{dataset_title} branch-hit distance CCDF")
    if max_distance is not None:
        axis.set_xlim(density.bin_width, max_distance)
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="Tokenizer path or model ID")
    parser.add_argument("--dataset", default="swe_agent")
    parser.add_argument("--page-sizes", default="1,16,32,64,128,256")
    parser.add_argument("--bin-width", type=int, default=256)
    parser.add_argument("--order", choices=("sequential", "round_robin"), default="round_robin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--min-isl", type=int)
    parser.add_argument("--max-isl", type=int)
    parser.add_argument(
        "--plot-max-distance",
        type=int,
        default=2000,
        help="Maximum x-axis distance in tokens; CSV data is never truncated",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment/results/prefix_cache_hit_rate"),
    )
    return parser.parse_args()


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    page_sizes = tuple(int(value) for value in args.page_sizes.split(","))
    records = _select_records(
        dataset=args.dataset,
        seed=args.seed,
        max_trajectories=args.max_trajectories,
        min_isl=args.min_isl,
        max_isl=args.max_isl,
    )
    if not records:
        raise ValueError("No trajectories selected")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    density = run_experiment(
        tokenizer=tokenizer,
        records=records,
        page_sizes=page_sizes,
        bin_width=args.bin_width,
        order=args.order,
        max_turns=args.max_turns,
    )
    configuration = {
        "tokenizer"            : args.tokenizer,
        "dataset"              : args.dataset,
        "num_trajectories"     : len(records),
        "page_sizes"           : page_sizes,
        "bin_width"            : args.bin_width,
        "order"                : args.order,
        "seed"                 : args.seed,
        "max_turns"            : args.max_turns,
        "min_isl"              : args.min_isl,
        "max_isl"              : args.max_isl,
        "exclude_system_prompt": True,
        "sources"              : dict(Counter(record.category for record in records)),
        "plot_max_distance"    : args.plot_max_distance,
    }
    _write_results(
        density,
        args.output_dir,
        configuration,
        args.plot_max_distance,
    )
    print(json.dumps(density.summary(), indent=2))
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
