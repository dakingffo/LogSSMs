import json
from pathlib import Path

import pytest

from logssms.utils.platform import (
    configure_chat_tokenizer,
    render_chat_prompt,
    resolve_model_path,
)


@pytest.mark.parametrize(
    "source,module_name",
    [
        ("huggingface", "huggingface_hub"),
        ("modelscope", "modelscope"),
    ],
)
def test_resolve_remote_model(monkeypatch, source, module_name):
    calls = []

    def snapshot_download(model_path, **kwargs):
        calls.append((model_path, kwargs))
        return "/cache/model"

    module = __import__(module_name)
    monkeypatch.setattr(module, "snapshot_download", snapshot_download)

    result = resolve_model_path("org/model", source, download_weights=False)

    assert result == "/cache/model"
    assert calls[0][0] == "org/model"
    assert "*.safetensors" in calls[0][1]["ignore_patterns"]
    assert "*.bin" in calls[0][1]["ignore_patterns"]


def test_resolve_local_model_does_not_download(monkeypatch, tmp_path: Path):
    def fail(*args, **kwargs):
        raise AssertionError("snapshot_download should not be called for a local model")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fail)
    monkeypatch.setattr("modelscope.snapshot_download", fail)

    for source in ("huggingface", "modelscope"):
        assert resolve_model_path(str(tmp_path), source) == str(tmp_path)


def test_reject_unknown_model_source():
    with pytest.raises(ValueError, match="Unknown model source"):
        resolve_model_path("org/model", "unknown")


def test_qwen35_chat_template_preserves_non_thinking_history(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))

    class Tokenizer:
        chat_template = "{%- if loop.index0 > ns.last_query_index %}"

        def apply_chat_template(self, messages, **kwargs):
            self.last_kwargs = kwargs
            return "rendered"

    tokenizer = Tokenizer()
    assert configure_chat_tokenizer(tokenizer, str(tmp_path)) is tokenizer
    assert "preserve_thinking is defined" in tokenizer.chat_template
    assert render_chat_prompt(tokenizer, [{"role": "user", "content": "hi"}]) == "rendered"
    assert tokenizer.last_kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "preserve_thinking": True,
        "enable_thinking": False,
    }


def test_other_chat_templates_are_unchanged(tmp_path: Path):
    class Tokenizer:
        chat_template = "{%- if loop.index0 > ns.last_query_index %}"

        def apply_chat_template(self, messages, **kwargs):
            self.last_kwargs = kwargs
            return "rendered"

    tokenizer = Tokenizer()
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    configure_chat_tokenizer(tokenizer, str(tmp_path))
    assert tokenizer.chat_template == "{%- if loop.index0 > ns.last_query_index %}"
    render_chat_prompt(tokenizer, [])
    assert "preserve_thinking" not in tokenizer.last_kwargs


def test_qwen36_template_uses_existing_preserve_thinking_branch(tmp_path: Path):
    class Tokenizer:
        chat_template = (
            "{%- if (preserve_thinking is defined and preserve_thinking is true) "
            "or (loop.index0 > ns.last_query_index) %}"
            '{%- elif message.role == "assistant" %}'
            "assistant-body"
            '{%- elif message.role == "tool" %}'
        )

        def apply_chat_template(self, messages, **kwargs):
            self.last_kwargs = kwargs
            return "rendered"

    tokenizer = Tokenizer()
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    configure_chat_tokenizer(tokenizer, str(tmp_path))
    assert "message.raw_generated is string" in tokenizer.chat_template
    assert "assistant-body" in tokenizer.chat_template
    assert tokenizer._logssms_preserve_generated_reasoning
    assert render_chat_prompt(tokenizer, [{"role": "user", "content": "hi"}]) == "rendered"
    assert tokenizer.last_kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "preserve_thinking": True,
    }


def test_local_qwen35_chat_template_token_prefix():
    from transformers import AutoTokenizer

    from benchmarks.bench_common import _chat_prompt_len
    from logssms.tokenizer.tokenizer import TokenizeManager

    model_path = Path("/home/daking/models/huggingface/Qwen3.5-2B")
    if not model_path.is_dir():
        pytest.skip("optional local Qwen3.5-2B checkpoint is unavailable")

    tokenizer = configure_chat_tokenizer(
        AutoTokenizer.from_pretrained(model_path), str(model_path)
    )
    first = [{"role": "user", "content": "你好"}]
    second = first + [
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": "继续"},
    ]
    manager = TokenizeManager(tokenizer)
    first_ids, second_ids = manager.tokenize([
        type("Msg", (), {"text": first, "input_ids": None})(),
        type("Msg", (), {"text": second, "input_ids": None})(),
    ])
    assert second_ids[:len(first_ids)].tolist() == first_ids.tolist()
    assert _chat_prompt_len(tokenizer, second) == len(second_ids)
