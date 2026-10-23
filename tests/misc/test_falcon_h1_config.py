import pytest
import torch
from transformers import FalconH1Config as HFFalconH1Config

from logssms.models import make_model_config
from logssms.models.falcon_h1 import FalconH1Config
from logssms.layers.linear_attention_backend import make_linear_attention_backend, Mamba2Backend, Mamba2LogStateBackend
from logssms.cache.logstate.layout import LogStateSlotLayout


def raw_config():
    return HFFalconH1Config(
        architectures=["FalconH1ForCausalLM"], hidden_size=256,
        intermediate_size=512, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=64, vocab_size=128,
        mamba_d_ssm=256, mamba_n_heads=8, mamba_d_head=32,
        mamba_d_state=64, mamba_n_groups=1, mamba_d_conv=4,
        mamba_rms_norm=True, max_position_embeddings=512,
    )


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_falcon_parallel_hybrid_cache_spec(tp):
    config = make_model_config(raw_config())
    assert isinstance(config, FalconH1Config)
    spec = config.make_linear_cache_spec(tp)
    assert config.num_attention_layers == config.num_linear_layers == config.num_layers == 2
    assert spec.recurrent_type == "mamba2"
    assert spec.state_shape == (8 // tp, 64, 32)
    assert spec.num_groups == 1
    assert spec.conv_shape == (256 // tp + 128, 3)
    assert isinstance(make_linear_attention_backend("radix", recurrent_type="mamba2"), Mamba2Backend)
    assert isinstance(make_linear_attention_backend("logstate", recurrent_type="mamba2"), Mamba2LogStateBackend)


@pytest.mark.parametrize("kind,widths", [("in_proj.weight", [256, 256, 64, 64, 8]),
                                        ("conv1d.weight", [256, 64, 64]),
                                        ("conv1d.bias", [256, 64, 64])])
def test_falcon_tp_preserves_replicated_bc_and_shards_each_region(kind, widths):
    config = make_model_config(raw_config())
    shape = (sum(widths), 5) if kind.endswith("weight") else (sum(widths),)
    weight = torch.arange(torch.tensor(shape).prod()).reshape(shape)
    parts = weight.split(widths, 0)
    for rank in range(2):
        local = config.shard_weight("model.layers.0.mamba." + kind, weight, rank, 2)
        replicate = [False, False, True, True, False] if len(widths) == 5 else [False, True, True]
        expected = torch.cat([part if rep else part.chunk(2, 0)[rank]
                              for part, rep in zip(parts, replicate)], 0)
        torch.testing.assert_close(local, expected)


def test_mamba2_asymmetric_head_overlay_stays_inside_frame():
    layout = LogStateSlotLayout(64, 24, 1, 48, 256, 64, 3584, 3, torch.bfloat16, 32)
    frame_bytes = 2 * 24 * 64 * 2 * 128 * 2
    stride = layout.head_payload_bytes(frame_bytes)
    assert stride == 65536
    assert frame_bytes % stride == 0
    assert stride >= layout.head_log_key_bytes + layout.head_log_value_bytes
    assert stride >= layout.head_recurrent_state_bytes


def test_falcon_row_parallel_output_bias_is_added_once():
    config = make_model_config(raw_config())
    bias = torch.randn(config.hidden_size)
    shards = [config.shard_weight("model.layers.0.mamba.out_proj.bias", bias, rank, 2)
              for rank in range(2)]
    torch.testing.assert_close(sum(shards), bias)
