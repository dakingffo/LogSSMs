import json

import safetensors.torch
import torch

from picosgl.models.weight import iter_checkpoint_weights


def test_iter_checkpoint_weights_from_pytorch(tmp_path):
    expected = {
        "weight": torch.arange(6).reshape(2, 3),
        "ignored": torch.ones(1),
    }
    torch.save(expected, tmp_path / "pytorch_model.bin")

    loaded = dict(iter_checkpoint_weights(tmp_path, names={"weight"}))

    assert loaded.keys() == {"weight"}
    assert torch.equal(loaded["weight"], expected["weight"])


def test_iter_checkpoint_weights_prefers_safetensors(tmp_path):
    torch.save({"weight": torch.zeros(1)}, tmp_path / "pytorch_model.bin")
    safetensors.torch.save_file(
        {"weight": torch.ones(1)}, tmp_path / "model.safetensors"
    )

    loaded = dict(iter_checkpoint_weights(tmp_path))

    assert torch.equal(loaded["weight"], torch.ones(1))


def test_iter_checkpoint_weights_uses_shard_index(tmp_path):
    safetensors.torch.save_file(
        {"first": torch.tensor([1])}, tmp_path / "model-00001-of-00002.safetensors"
    )
    safetensors.torch.save_file(
        {"second": torch.tensor([2])}, tmp_path / "model-00002-of-00002.safetensors"
    )
    index = {
        "weight_map": {
            "second": "model-00002-of-00002.safetensors",
            "first": "model-00001-of-00002.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    loaded = list(iter_checkpoint_weights(tmp_path))

    assert [name for name, _ in loaded] == ["second", "first"]
