import gzip
import json

import pytest

from benchmarks.bench_common import (
    _dataset_prompt_len,
    _take_dataset_prompts,
    load_dataset_prompts,
    resolve_dataset_path,
)
from datasets.parser import iter_dataset, load_dataset, sample_dataset


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


def test_dataset_exposes_user_turns_and_filters_multiturn_records(tmp_path):
    path = tmp_path / "multi_turn.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"id": "single", "turns": ["only"]}),
            json.dumps({"id": "multi", "turns": ["first", "second", "third"]}),
        ]) + "\n",
        encoding="utf-8",
    )

    record = sample_dataset(str(path), 1, seed=7, min_turns=3)[0]
    assert record.uid == "multi"
    assert record.num_turns == 3
    assert [message.content for message in record.user_turns] == [
        "first", "second", "third",
    ]

    with pytest.raises(ValueError, match="at least 4 user turns"):
        sample_dataset(str(path), 1, seed=7, min_turns=4)


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


def _oasst2_tree():
    def node(uid, parent, role, text, *replies):
        return {
            "message_id": uid,
            "parent_id": parent,
            "role": role,
            "text": text,
            "lang": "en",
            "rank": 0 if role == "assistant" else None,
            "deleted": False,
            "review_result": True,
            "replies": list(replies),
        }
    root = node("root", None, "prompter", "question",
        node("answer", "root", "assistant", "recorded answer",
            node("follow-a", "answer", "prompter", "follow-up A",
                node("leaf-a", "follow-a", "assistant", "answer A"),
                node("leaf-b", "follow-a", "assistant", "alternative A"),
            ),
            node("follow-b", "answer", "prompter", "follow-up B",
                node("leaf-c", "follow-b", "assistant", "answer B"),
            ),
        ),
    )
    return {"message_tree_id": "root", "tree_state": "ready_for_export", "prompt": root}


def _write_oasst2_tree(path, tree):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as file:
        file.write(json.dumps(tree) + "\n")


def test_oasst2_preserves_branch_ancestry_and_normalizes_messages(tmp_path):
    path = tmp_path / "oasst2" / "2023-11-05_oasst2_ready.trees.jsonl.gz"
    _write_oasst2_tree(path, _oasst2_tree())
    records = list(iter_dataset(path.parent))
    assert [record.uid for record in records] == ["leaf-a", "leaf-b", "leaf-c"]
    assert records == list(iter_dataset(path))
    first = records[0]
    assert first.category == "en"
    assert first.num_turns == 2
    assert [message.role for message in first.messages] == [
        "user", "assistant", "user", "assistant",
    ]
    assert [message.content for message in first.user_turns] == ["question", "follow-up A"]
    assert first.metadata["message_tree_id"] == "root"
    assert first.metadata["message_ids"] == ("root", "answer", "follow-a", "leaf-a")
    assert first.metadata["parent_ids"] == (None, "root", "answer", "follow-a")
    assert first.metadata["user_branch_ids"] == ("answer",)
    assert first.metadata["assistant_branch_ids"] == ("follow-a",)
    assert records[-1].metadata["assistant_branch_ids"] == ()
    assert first.metadata["node_metadata"][1]["rank"] == 0


def test_oasst2_sampling_filters_language_and_user_turns(tmp_path, monkeypatch):
    import datasets.parser as parser

    path = tmp_path / "oasst2" / "2023-11-05_oasst2_ready.trees.jsonl.gz"
    _write_oasst2_tree(path, _oasst2_tree())
    monkeypatch.setattr(parser, "_DATASET_ROOT", tmp_path)
    assert parser.resolve_dataset_path("OpenAssistant/oasst2") == path.parent
    assert load_dataset("oasst2", categories="en") == list(iter_dataset(path))
    assert load_dataset("oasst2", categories="zh") == []
    selected = sample_dataset("oasst2", 2, seed=7, min_turns=2, categories="en")
    assert selected == sample_dataset("oasst2", 2, seed=7, min_turns=2, categories="en")
    assert len({record.uid for record in selected}) == 2
    with pytest.raises(ValueError, match="at least 3 user turns"):
        sample_dataset("oasst2", 1, seed=7, min_turns=3)
    with pytest.raises(ValueError, match="refusing to repeat prompts"):
        sample_dataset("oasst2", 4, seed=7)
    assert sample_dataset("oasst2", 0, seed=7) == []


def test_oasst2_parquet_reassembles_nodes_across_shards(tmp_path):
    parquet = pytest.importorskip("pyarrow.parquet")
    pyarrow = pytest.importorskip("pyarrow")
    tree = _oasst2_tree()
    stack = [tree["prompt"]]
    rows = []
    while stack:
        node = stack.pop()
        rows.append({k: v for k, v in node.items() if k != "replies"} | {
            "message_tree_id": "root", "tree_state": "ready_for_export",
        })
        stack.extend(reversed(node["replies"]))
    path = tmp_path / "oasst2" / "data"
    path.mkdir(parents=True)
    parquet.write_table(pyarrow.Table.from_pylist(rows[1:]), path / "train-00000.parquet")
    parquet.write_table(pyarrow.Table.from_pylist(rows[:1]), path / "validation-00000.parquet")
    records = list(iter_dataset(path.parent))
    assert {record.uid for record in records} == {"leaf-a", "leaf-b", "leaf-c"}
    assert all(record.metadata["user_branch_ids"] == ("answer",) for record in records)
    selected = sample_dataset(path.parent, 2, seed=7, min_turns=2)
    assert len({record.uid for record in selected}) == 2


@pytest.mark.parametrize("kind", ["parent", "role", "duplicate", "root"])
def test_oasst2_rejects_invalid_tree_links(tmp_path, kind):
    tree = _oasst2_tree()
    answer = tree["prompt"]["replies"][0]
    if kind == "parent":
        answer["parent_id"] = "missing"
    elif kind == "role":
        answer["role"] = "prompter"
    elif kind == "duplicate":
        answer["replies"][1]["replies"][0]["message_id"] = "leaf-a"
    else:
        tree["message_tree_id"] = "other-root"
    path = tmp_path / "oasst2" / "sample.trees.jsonl.gz"
    _write_oasst2_tree(path, tree)
    with pytest.raises(ValueError, match="OASST2"):
        list(iter_dataset(path))


def test_oasst2_flat_messages_reject_missing_parent(tmp_path):
    path = tmp_path / "oasst2" / "sample.messages.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps({
        "message_tree_id": "root", "message_id": "child", "parent_id": "missing",
        "role": "assistant", "text": "answer", "lang": "en",
    }) + "\n")
    with pytest.raises(ValueError, match="Missing OASST2 parent"):
        list(iter_dataset(path))
