import json
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch
import torch.nn.functional as F

from logssms.core import Batch, Context, clear_global_ctx, set_global_ctx
from logssms.distributed import DistributedInfo, tp_override
from logssms.layers import GatedDeltaNet
from logssms.models.config import AffineLogCacheSpec
from logssms.models.weight import _get_merge_info, iter_checkpoint_weights, load_weight


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


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_gdn_projections_merge_after_tp_sharding(tmp_path, monkeypatch, tp_size):
    config = SimpleNamespace(
        is_hybrid=True, is_moe=False, num_kv_heads=2,
        make_linear_cache_spec=lambda tp: AffineLogCacheSpec(
            "gdn", 1, (8 // tp, 8, 4), (96 // tp, 3), num_groups=4 // tp,
        ),
        shard_weight=lambda *args: None,
    )
    monkeypatch.setattr("logssms.models.weight.load_model_config", lambda _: config)
    monkeypatch.setattr("logssms.models.register.make_model_config", lambda _: config)
    torch.manual_seed(91)
    hidden_size, key_dim, value_dim = 16, 32, 32
    prefix = "model.layers.0.linear_attn."
    shapes = {
        "in_proj_qkv.weight": (2 * key_dim + value_dim, hidden_size),
        "in_proj_z.weight": (value_dim, hidden_size),
        "in_proj_b.weight": (8, hidden_size),
        "in_proj_a.weight": (8, hidden_size),
        "out_proj.weight": (hidden_size, value_dim),
        "conv1d.weight": (2 * key_dim + value_dim, 1, 4),
        "norm.weight": (4,),
        "A_log": (8,),
        "dt_bias": (8,),
    }
    raw = {name: torch.randn(shape) * 0.1 for name, shape in shapes.items()}
    names = list(raw)
    # Both merge groups cross shard boundaries, with A appearing before B.
    first = {prefix + name: raw[name] for name in names if name not in (
        "in_proj_z.weight", "in_proj_b.weight",
    )}
    second = {prefix + name: raw[name] for name in (
        "in_proj_z.weight", "in_proj_b.weight",
    )}
    safetensors.torch.save_file(first, tmp_path / "model-00001-of-00002.safetensors")
    safetensors.torch.save_file(second, tmp_path / "model-00002-of-00002.safetensors")
    x = torch.randn(3, hidden_size)

    for rank in range(tp_size):
        with tp_override(DistributedInfo(rank=rank, size=tp_size)):
            loaded = dict(load_weight(str(tmp_path), "cpu"))
            layer = GatedDeltaNet(hidden_size, 4, 8, 8, 4, 4, 1e-6, 0)
        layer.load_state_dict(dict(loaded), prefix=prefix.removesuffix("."))
        q, k, v = raw["in_proj_qkv.weight"].split([key_dim, key_dim, value_dim])
        qkv_weight = torch.cat([part.chunk(tp_size)[rank] for part in (q, k, v)])
        z_weight = raw["in_proj_z.weight"].chunk(tp_size)[rank]
        b_weight = raw["in_proj_b.weight"].chunk(tp_size)[rank]
        a_weight = raw["in_proj_a.weight"].chunk(tp_size)[rank]
        torch.testing.assert_close(layer.in_proj_qkvz.weight, torch.cat([qkv_weight, z_weight]))
        torch.testing.assert_close(layer.in_proj_ba.weight, torch.cat([b_weight, a_weight]))
        assert len(loaded) == len(raw) - 2

        layer_config = layer._config
        expected_qkv = F.linear(x, qkv_weight)
        expected_z = F.linear(x, z_weight)
        expected_beta = F.linear(x, b_weight)
        expected_gate = F.linear(x, a_weight)
        expected_core = expected_qkv[:, -layer_config.value_dim:].clone()
        expected_core += expected_beta.repeat_interleave(layer_config.head_v_dim, dim=-1)
        expected_core += expected_gate.repeat_interleave(layer_config.head_v_dim, dim=-1)
        h = expected_core.view(3, layer_config.num_v_heads, layer_config.head_v_dim)
        h = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-6)
        h = h * raw["norm.weight"] * F.silu(expected_z.view(h.shape))
        expected = F.linear(h.flatten(1), raw["out_proj.weight"].chunk(tp_size, dim=1)[rank])

        class Backend:
            def forward(self, *, mixed_qkv, gate, beta, **kwargs):
                torch.testing.assert_close(mixed_qkv, expected_qkv)
                torch.testing.assert_close(gate, expected_gate)
                torch.testing.assert_close(beta, expected_beta)
                assert not mixed_qkv.is_contiguous()
                assert not gate.is_contiguous() and not beta.is_contiguous()
                return expected_core.clone()

        ctx = Context(page_size=64)
        ctx.linear_attn_backend = Backend()
        set_global_ctx(ctx)
        try:
            with ctx.forward_batch(Batch(reqs=[], phase="prefill")):
                actual = layer.forward(x)
        finally:
            clear_global_ctx()
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("name", ["in_proj_qkvz", "in_proj_ba"])
def test_gdn_fused_names_are_not_merged_again(name):
    assert _get_merge_info(f"model.layers.0.linear_attn.{name}.weight") is None
