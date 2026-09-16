from types import SimpleNamespace

from logssms.models import make_model_config
from logssms.models.qwen3 import Qwen3Config, Qwen3MoeConfig
from logssms.models.qwen3_next import Qwen3_5Config, Qwen3_5MoeConfig, Qwen3NextConfig


def _config(**kwargs):
    values = {
        "architectures"           : ["Qwen3ForCausalLM"],
        "num_hidden_layers"       : 4,
        "num_attention_heads"     : 8,
        "num_key_value_heads"     : 2,
        "hidden_size"             : 512,
        "vocab_size"              : 32000,
        "intermediate_size"       : 1024,
        "hidden_act"              : "silu",
        "rms_norm_eps"            : 1e-6,
        "max_position_embeddings" : 4096,
        "rope_theta"              : 10000.0,
        "rope_scaling"            : None,
        "tie_word_embeddings"     : False,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def _hybrid_config(architecture: str, **kwargs):
    text = _config(
        architectures=None,
        rope_parameters={
            "rope_theta"           : 1000000.0,
            "partial_rotary_factor": 0.5,
        },
        layer_types=["linear_attention", "full_attention"] * 2,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        **kwargs,
    )
    return SimpleNamespace(architectures=[architecture], text_config=text), text


def test_make_dense_model_config():
    config = make_model_config(_config())

    assert type(config) is Qwen3Config
    assert not config.is_hybrid
    assert not config.is_moe
    assert config.num_attention_layers == config.num_layers


def test_make_qwen3_moe_config():
    raw_config = _config(
        architectures=["Qwen3MoeForCausalLM"],
        num_experts=128,
        num_experts_per_tok=8,
        moe_intermediate_size=768,
        norm_topk_prob=True,
    )
    config = make_model_config(raw_config)

    assert type(config) is Qwen3MoeConfig
    assert config.is_moe
    assert config.num_experts == 128
    assert config.num_experts_per_tok == 8
    assert config.moe_intermediate_size == 768


def test_make_qwen3_5_dense_config():
    top, _ = _hybrid_config(
        "Qwen3_5ForConditionalGeneration",
        mtp_num_hidden_layers=1,
    )
    config = make_model_config(top)

    assert type(config) is Qwen3_5Config
    assert not config.is_moe
    assert not config.use_moe(0)
    assert config.num_attention_layers == 2
    assert config.num_linear_layers == 2
    assert config.rotary_config.rotary_dim == 32


def test_make_qwen3_5_moe_config():
    top, text = _hybrid_config(
        "Qwen3_5MoeForConditionalGeneration",
        num_experts=16,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
        norm_topk_prob=False,
        mtp_num_hidden_layers=1,
    )
    del text.intermediate_size
    config = make_model_config(top)

    assert type(config) is Qwen3_5MoeConfig
    assert config.is_moe
    assert config.use_moe(0)
    assert config.num_experts == 16


def test_make_qwen3_next_config():
    config = _config(
        architectures=["Qwen3NextForCausalLM"],
        layer_types=["linear_attention", "full_attention"] * 2,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        num_experts=16,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
        norm_topk_prob=True,
        mlp_only_layers=[],
    )
    config = make_model_config(config)

    assert type(config) is Qwen3NextConfig
    assert config.is_moe
    assert config.use_moe(0)
    assert config.num_experts == 16
