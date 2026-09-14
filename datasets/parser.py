from __future__ import annotations

import json
import random
from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal


DatasetRole = Literal["system", "user", "assistant"]
RecordParser = Callable[[dict[str, Any]], "DatasetRecord"]


@dataclass(frozen=True)
class DatasetMessage:
    role   : DatasetRole
    content: str


@dataclass(frozen=True)
class DatasetRecord:
    uid     : str
    category: str | None
    messages: tuple[DatasetMessage, ...]

    @property
    def prompt(self) -> str:
        for message in self.messages:
            if message.role == "user":
                return message.content
        raise ValueError(f"Dataset record {self.uid!r} has no user message")


_DATASET_ROOT = Path(__file__).resolve().parent
_ALIASES = {
    "specbench"       : "spec_bench",
    "spec_bench"      : "spec_bench",
    "swebench"        : "swe_bench",
    "swe_bench"       : "swe_bench",
    "sharegpt"        : "sharegpt",
    "lmsys"           : "lmsys_chat_1m",
    "lmsys_chat"      : "lmsys_chat_1m",
    "lmsys_chat_1m"   : "lmsys_chat_1m",
}


def _normalize_name(name: str) -> str:
    return name.lower().replace("-", "_")


def _dataset_name(dataset: str | Path) -> str:
    path = Path(dataset)
    candidate = path.name if path.exists() and path.is_dir() else path.stem
    normalized = _normalize_name(candidate)
    if normalized in _ALIASES:
        return _ALIASES[normalized]

    for part in reversed(path.parts):
        normalized = _normalize_name(part)
        if normalized in _ALIASES:
            return _ALIASES[normalized]
    if path.is_file() and path.suffix in (".json", ".jsonl"):
        return "custom"
    raise ValueError(f"Cannot determine dataset type from {str(dataset)!r}")


def resolve_dataset_path(dataset: str | Path) -> Path:
    path = Path(dataset).expanduser()
    if path.exists():
        return path.resolve()

    normalized = _normalize_name(str(dataset))
    name = _ALIASES.get(normalized, normalized)
    path = _DATASET_ROOT / name
    if path.exists():
        return path

    raise FileNotFoundError(
        f"Dataset {str(dataset)!r} was not found as a path or under {_DATASET_ROOT}"
    )


def _message(role: str, content: Any) -> DatasetMessage | None:
    role = {
        "human": "user",
        "gpt"  : "assistant",
    }.get(role.lower(), role.lower())
    if role not in ("system", "user", "assistant"):
        return None
    content = str(content).strip()
    if not content:
        return None
    return DatasetMessage(role=role, content=content)  # type: ignore[arg-type]


def _parse_spec_bench(record: dict[str, Any]) -> DatasetRecord:
    messages = tuple(
        DatasetMessage(role="user", content=str(turn))
        for turn in record["turns"]
        if str(turn).strip()
    )
    return DatasetRecord(
        uid=str(record["question_id"]),
        category=record.get("category"),
        messages=messages,
    )


def _parse_sharegpt(record: dict[str, Any]) -> DatasetRecord:
    messages = tuple(
        message
        for turn in record.get("conversations", ())
        if (message := _message(
            str(turn.get("from", turn.get("role", ""))),
            turn.get("value", turn.get("content", "")),
        )) is not None
    )
    return DatasetRecord(
        uid=str(record.get("id", "")),
        category="sharegpt",
        messages=messages,
    )


def _parse_swe_bench(record: dict[str, Any]) -> DatasetRecord:
    return DatasetRecord(
        uid=str(record["instance_id"]),
        category=str(record.get("repo", "swe_bench")),
        messages=(
            DatasetMessage(role="user", content=str(record["problem_statement"])),
        ),
    )


def _parse_lmsys(record: dict[str, Any]) -> DatasetRecord:
    messages = tuple(
        message
        for turn in record.get("conversation", ())
        if (message := _message(
            str(turn.get("role", turn.get("from", ""))),
            turn.get("content", turn.get("value", "")),
        )) is not None
    )
    return DatasetRecord(
        uid=str(record.get("conversation_id", "")),
        category=record.get("language"),
        messages=messages,
    )


def _parse_custom(record: dict[str, Any]) -> DatasetRecord:
    if "turns" in record:
        messages = tuple(
            DatasetMessage(role="user", content=str(turn))
            for turn in record["turns"]
            if str(turn).strip()
        )
    elif "conversations" in record:
        return _parse_sharegpt(record)
    else:
        content = next(
            (
                record[key]
                for key in ("prompt", "question", "text", "instruction")
                if record.get(key)
            ),
            None,
        )
        if content is None:
            raise ValueError(f"Cannot find a prompt in dataset record: {record!r}")
        messages = (DatasetMessage(role="user", content=str(content)),)
    return DatasetRecord(
        uid=str(record.get("id", record.get("question_id", ""))),
        category=record.get("category"),
        messages=messages,
    )


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def _iter_json(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import ijson
    except ImportError as exc:
        raise RuntimeError(
            "Streaming JSON datasets requires ijson. Install the benchmark extras."
        ) from exc

    with path.open("rb") as file:
        yield from ijson.items(file, "item")


def _iter_parquet(paths: list[Path]) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "Parquet datasets require pyarrow. Install the benchmark extras."
        ) from exc

    for path in paths:
        parquet_file = parquet.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=4096):
            yield from batch.to_pylist()


def _parquet_source(
    path: Path,
    name: str,
) -> tuple[list[Path], RecordParser]:
    if not path.is_dir():
        paths = [path]
    elif name == "swe_bench":
        paths = sorted(path.rglob("test-*.parquet"))
    else:
        paths = sorted(path.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {path}")
    parser = _parse_swe_bench if name == "swe_bench" else _parse_lmsys
    return paths, parser


def _dataset_source(
    dataset: str | Path,
) -> tuple[Iterator[dict[str, Any]], RecordParser]:
    path = resolve_dataset_path(dataset)
    name = _dataset_name(path)

    if name == "spec_bench":
        source = path / "question.jsonl" if path.is_dir() else path
        return _iter_jsonl(source), _parse_spec_bench
    elif name == "sharegpt":
        source = (
            path / "ShareGPT_V3_unfiltered_cleaned_split.json"
            if path.is_dir() else path
        )
        return _iter_json(source), _parse_sharegpt
    elif name in ("swe_bench", "lmsys_chat_1m"):
        paths, parser = _parquet_source(path, name)
        return _iter_parquet(paths), parser
    elif name == "custom":
        records = _iter_jsonl(path) if path.suffix == ".jsonl" else _iter_json(path)
        return records, _parse_custom
    else:
        raise ValueError(f"Unsupported dataset: {name}")


def iter_dataset(dataset: str | Path) -> Iterator[DatasetRecord]:
    records, parser = _dataset_source(dataset)
    for raw_record in records:
        record = parser(raw_record)
        if record.messages:
            yield record


def _sample_parquet(
    path : Path,
    name : str,
    count: int,
    seed : int,
) -> list[DatasetRecord]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "Parquet datasets require pyarrow. Install the benchmark extras."
        ) from exc

    paths, parser = _parquet_source(path, name)
    locations: list[tuple[Path, int, int, int]] = []
    ends: list[int] = []
    total = 0
    for parquet_path in paths:
        metadata = parquet.ParquetFile(parquet_path).metadata
        for row_group in range(metadata.num_row_groups):
            begin = total
            total += metadata.row_group(row_group).num_rows
            locations.append((parquet_path, row_group, begin, total))
            ends.append(total)

    if total < count:
        raise ValueError(
            f"Requested {count} records, but dataset only contains {total}; "
            "refusing to repeat prompts"
        )

    generator = random.Random(seed)
    selected_rows = generator.sample(range(total), count)
    grouped: dict[tuple[Path, int], list[tuple[int, int]]] = {}
    for output_idx, row_idx in enumerate(selected_rows):
        location_idx = bisect_right(ends, row_idx)
        parquet_path, row_group, begin, _ = locations[location_idx]
        grouped.setdefault((parquet_path, row_group), []).append(
            (output_idx, row_idx - begin)
        )

    selected: list[DatasetRecord | None] = [None] * count
    for (parquet_path, row_group), rows in grouped.items():
        table = parquet.ParquetFile(parquet_path).read_row_group(row_group)
        for output_idx, row_idx in rows:
            selected[output_idx] = parser(table.slice(row_idx, 1).to_pylist()[0])

    result = [record for record in selected if record is not None and record.messages]
    if len(result) != count:
        raise ValueError("Sampled parquet records contain empty conversations")
    return result


def _category_set(categories: str | None) -> set[str] | None:
    if categories is None:
        return None
    return {category.strip() for category in categories.split(",") if category.strip()}


def load_dataset(
    dataset   : str | Path,
    categories: str | None = None,
) -> list[DatasetRecord]:
    selected = _category_set(categories)
    return [
        record
        for record in iter_dataset(dataset)
        if selected is None or record.category in selected
    ]


def sample_dataset(
    dataset   : str | Path,
    count     : int,
    seed      : int,
    categories: str | None = None,
) -> list[DatasetRecord]:
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")

    selected_categories = _category_set(categories)
    path = resolve_dataset_path(dataset)
    name = _dataset_name(path)
    if selected_categories is None and name in ("swe_bench", "lmsys_chat_1m"):
        return _sample_parquet(path, name, count, seed)

    generator = random.Random(seed)
    reservoir: list[DatasetRecord] = []
    seen = 0
    for record in iter_dataset(dataset):
        if (
            selected_categories is not None
            and record.category not in selected_categories
        ):
            continue
        seen += 1
        if len(reservoir) < count:
            reservoir.append(record)
        else:
            replace = generator.randrange(seen)
            if replace < count:
                reservoir[replace] = record

    if seen < count:
        suffix = (
            f" for categories {sorted(selected_categories)}"
            if selected_categories is not None else ""
        )
        raise ValueError(
            f"Requested {count} records, but dataset only contains {seen}{suffix}; "
            "refusing to repeat prompts"
        )
    generator.shuffle(reservoir)
    return reservoir
