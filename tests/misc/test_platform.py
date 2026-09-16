from pathlib import Path

import pytest

from logssms.utils.platform import resolve_model_path


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
