import json

import pytest

from benchmarks.bench_common import (
    _dataset_prompt_len,
    _take_dataset_prompts,
    load_dataset_prompts,
    resolve_dataset_path,
)
from datasets.parser import iter_dataset, sample_dataset


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize
        assert add_generation_prompt
        return f"<user>{messages[0]['content']}<assistant>"

    def encode(self, text):
        return text.split()


def test_load_spec_bench():
    try:
        path = resolve_dataset_path("spec-bench")
    except FileNotFoundError:
        pytest.skip("Spec-Bench is an optional local benchmark dataset")
    assert path.endswith("datasets/spec_bench")
    assert len(load_dataset_prompts(path)) == 480
    assert len(load_dataset_prompts("spec_bench", "coding")) == 10
    assert len(load_dataset_prompts("spec_bench", "qa,rag")) == 160


def test_dataset_uses_first_turn(tmp_path):
    path = tmp_path / "multi_turn.jsonl"
    path.write_text(
        json.dumps({"category": "test", "turns": ["first", "second"]}) + "\n",
        encoding="utf-8",
    )
    assert load_dataset_prompts(str(path)) == ["first"]


def test_dataset_sampling_is_deterministic_and_never_repeats(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps([{"prompt": str(i)} for i in range(4)]),
        encoding="utf-8",
    )
    first = _take_dataset_prompts(str(path), None, 3, seed=7)
    assert first == _take_dataset_prompts(str(path), None, 3, seed=7)
    assert len(set(first)) == 3
    with pytest.raises(ValueError, match="refusing to repeat prompts"):
        _take_dataset_prompts(str(path), None, 5, seed=7)


def test_dataset_token_length_includes_chat_template():
    tokenizer = FakeTokenizer()
    assert _dataset_prompt_len(tokenizer, "one two three") == 3


def test_sharegpt_normalizes_messages(tmp_path):
    pytest.importorskip("ijson")
    path = tmp_path / "sharegpt"
    path.mkdir()
    (path / "ShareGPT_V3_unfiltered_cleaned_split.json").write_text(
        json.dumps([
            {
                "id": "conversation-1",
                "conversations": [
                    {"from": "human", "value": "hello"},
                    {"from": "gpt", "value": "hi"},
                ],
            }
        ]),
        encoding="utf-8",
    )
    record = next(iter_dataset(path))
    assert record.uid == "conversation-1"
    assert [(message.role, message.content) for message in record.messages] == [
        ("user", "hello"),
        ("assistant", "hi"),
    ]


def test_swe_agent_normalizes_trajectory(tmp_path):
    path = tmp_path / "swe_agent" / "trajs"
    path.mkdir(parents=True)
    (path / "org__repo-1.traj").write_text(
        json.dumps({
            "history": [
                {"role": "system", "content": "instructions"},
                {"role": "user", "content": "fix this"},
                {"role": "assistant", "content": "working"},
            ]
        }),
        encoding="utf-8",
    )

    record = next(iter_dataset(path.parent))
    assert (record.uid, record.category, record.prompt) == (
        "org__repo-1",
        "org/repo",
        "fix this",
    )


def test_agentic_coding_preserves_tool_messages(tmp_path):
    parquet = pytest.importorskip("pyarrow.parquet")
    pyarrow = pytest.importorskip("pyarrow")
    path = tmp_path / "agentic_coding_trajectories"
    path.mkdir()
    tool_calls = [{
        "function": {
            "arguments": json.dumps({"command": "pwd"}),
            "name": "execute_bash",
        },
        "id": "call-1",
        "type": "function",
    }]
    raw = {
        "session_id"           : "session-1",
        "source_dataset"       : "openhands",
        "source_id"            : "instance-1",
        "agent_framework"      : "OpenHands",
        "recorded_model"       : "model",
        "messages_json"        : json.dumps([
            {"role": "user", "content": "fix this"},
            {
                "role"           : "assistant",
                "content"        : "",
                "tool_calls_json": json.dumps(tool_calls),
            },
            {
                "role"        : "tool",
                "content"     : "/workspace",
                "tool_call_id": "call-1",
            },
        ]),
        "n_turns"              : 1,
        "max_isl"              : 4096,
        "total_tokens"         : 5000,
        "ground_truth_meta_json": "{}",
    }
    parquet.write_table(
        pyarrow.Table.from_pylist([raw]),
        path / "sessions.parquet",
    )

    record = next(iter_dataset(path))
    assert record.category == "openhands"
    assert record.metadata["max_isl"] == 4096
    assert record.messages[1].content == ""
    assert record.messages[1].tool_calls == ({
        "function": {
            "arguments": {"command": "pwd"},
            "name": "execute_bash",
        },
        "id": "call-1",
        "type": "function",
    },)
    assert record.messages[2].tool_call_id == "call-1"


@pytest.mark.parametrize(
    ("name", "raw", "expected"),
    [
        (
            "swe_bench",
            {
                "instance_id": "repo-1",
                "repo": "org/repo",
                "problem_statement": "fix this bug",
            },
            ("repo-1", "org/repo", "fix this bug"),
        ),
        (
            "lmsys_chat_1m",
            {
                "conversation_id": "chat-1",
                "language": "English",
                "conversation": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
            },
            ("chat-1", "English", "hello"),
        ),
    ],
)
def test_parquet_datasets_normalize_and_sample(tmp_path, name, raw, expected):
    parquet = pytest.importorskip("pyarrow.parquet")
    pyarrow = pytest.importorskip("pyarrow")
    path = tmp_path / name / "data"
    path.mkdir(parents=True)
    filename = "test-00000.parquet" if name == "swe_bench" else "train-00000.parquet"
    parquet.write_table(pyarrow.Table.from_pylist([raw]), path / filename)

    record = sample_dataset(path.parent, count=1, seed=7)[0]
    assert (record.uid, record.category, record.prompt) == expected
