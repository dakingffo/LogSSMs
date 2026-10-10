import pytest
import torch

from logssms.cache.linear_pool.state_pool import LinearStatePool
from logssms.core import Batch, Context, Request, clear_global_ctx, set_global_ctx
from logssms.distributed import DistributedInfo, tp_override
from logssms.kernel import store_state
from logssms.kernel.linear_attention.vendor.fla import (
    fused_recurrent_gated_delta_rule,
)
from logssms.layers import GatedDeltaNet
from logssms.models.config import RotaryConfig
from logssms.models.qwen3_next import Qwen3_5Config
from logssms.utils import torch_dtype


def _recurrent_gated_delta_rule_with_snapshots(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
):
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    query *= 1 / (query.shape[-1] ** 0.5)
    state = initial_state.to(torch.float32)
    outputs = []
    snapshots = []

    for i in range(query.shape[2]):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        state = state * g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta[:, :, i].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))
        snapshots.append(state)

    output = torch.stack(outputs, dim=2).transpose(1, 2).contiguous().to(initial_dtype)
    return output, torch.stack(snapshots, dim=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shape", [(3, 5, 4, 32, 32), (2, 5, 16, 128, 128)])
@pytest.mark.parametrize("fused_gates", [False, True])
def test_batched_kernel_matches_torch(shape, fused_gates):
    torch.manual_seed(0)
    B, S, H, K, V = shape
    device = torch.device("cuda")
    query = torch.randn(B, S, H, K, device=device, dtype=torch.bfloat16)
    key = torch.randn(B, S, H, K, device=device, dtype=torch.bfloat16)
    value = torch.randn(B, S, H, V, device=device, dtype=torch.bfloat16)
    gate = torch.randn(B, S, H, device=device, dtype=torch.bfloat16)
    beta = torch.randn(B, S, H, device=device, dtype=torch.bfloat16)
    if fused_gates:
        beta, gate = torch.cat([beta, gate], dim=-1).chunk(2, dim=-1)
    A_log = torch.zeros(H, device=device)
    dt_bias = torch.zeros(H, device=device)
    initial_state = torch.randn(B, H, K, V, device=device, dtype=torch.bfloat16)
    write_slots = torch.arange(B * S, device=device, dtype=torch.int32).view(B, S)
    state_pool = torch.empty(B * S, H, K, V, device=device, dtype=torch.bfloat16)

    expected_output, expected_states = _recurrent_gated_delta_rule_with_snapshots(
        (query.float() * torch.rsqrt(query.float().square().sum(-1, keepdim=True) + 1e-6)).to(query),
        (key.float() * torch.rsqrt(key.float().square().sum(-1, keepdim=True) + 1e-6)).to(key),
        value,
        -torch.nn.functional.softplus(gate.float()),
        beta.sigmoid(),
        initial_state,
    )
    output, states = fused_recurrent_gated_delta_rule(
        query, key, value, gate, beta, A_log, dt_bias, initial_state
    )
    store_state(state_pool, write_slots.flatten(), states.flatten(0, 1))
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected_output, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        state_pool.view(B, S, H, K, V),
        expected_states.to(torch.bfloat16),
        rtol=1e-3,
        atol=2e-2,
    )


def _tiny_config() -> Qwen3_5Config:
    return Qwen3_5Config(
        num_layers=1,
        num_qo_heads=2,
        num_kv_heads=2,
        head_dim=32,
        hidden_size=64,
        vocab_size=128,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        rotary_config=RotaryConfig(32, 128, 10000.0, None),
        hidden_act="silu",
        tie_word_embeddings=False,
        architectures=["Qwen3_5ForCausalLM"],
        layer_types=["linear_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        shared_expert_intermediate_size=0,
        norm_topk_prob=False,
        mlp_only_layers=[0],
        decoder_sparse_step=1,
        mtp_num_hidden_layers=0,
    )


def _set_tensor(module, name: str, value: torch.Tensor) -> None:
    parts = name.split(".")
    owner = module
    for part in parts[:-1]:
        owner = owner.__dict__[part]
    setattr(owner, parts[-1], value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_variable_length_layer_batch_matches_per_request():
    clear_global_ctx()
    config = _tiny_config()
    device = torch.device("cuda")
    with tp_override(DistributedInfo(rank=0, size=1)):
        with torch.device(device), torch_dtype(torch.bfloat16):
            layer = GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_key_heads=config.linear_num_key_heads,
                num_value_heads=config.linear_num_value_heads,
                head_k_dim=config.linear_key_head_dim,
                head_v_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_idx=0,
            )

    torch.manual_seed(1)
    for name, param in layer.state_dict().items():
        _set_tensor(layer, name, torch.randn_like(param) * 0.02)

    conv_dim = (
        config.linear_num_key_heads * config.linear_key_head_dim * 2
        + config.linear_num_value_heads * config.linear_value_head_dim
    )

    def make_pool() -> LinearStatePool:
        return LinearStatePool(
            num_slots=24,
            num_linear_layers=1,
            conv_dim=conv_dim,
            kernel_size=config.linear_conv_kernel_dim,
            num_v_heads=config.linear_num_value_heads,
            head_k_dim=config.linear_key_head_dim,
            head_v_dim=config.linear_value_head_dim,
            device=device,
            dtype=torch.bfloat16,
        )

    pool = make_pool()
    reference_pool = make_pool()
    torch.manual_seed(2)
    pool.conv_state.normal_()
    pool.recurrent_state.normal_(std=0.1)
    reference_pool.conv_state.copy_(pool.conv_state)
    reference_pool.recurrent_state.copy_(pool.recurrent_state)

    state_table = torch.full((4, 64), -1, dtype=torch.int32, device=device)
    work_state_table = torch.full((4, 5), -1, dtype=torch.int32, device=device)
    for table_idx in range(3):
        state_table[table_idx] = table_idx
        work_state_table[table_idx] = torch.arange(
            3 + table_idx * 5,
            3 + (table_idx + 1) * 5,
            dtype=torch.int32,
            device=device,
        )

    lengths = [5, 3, 5]
    reqs = []
    for table_idx, seq_len in enumerate(lengths):
        req = Request(
            input_ids=torch.zeros(10 + seq_len, dtype=torch.int32),
            table_idx=table_idx,
            cached_len=10,
            output_len=4,
            uid=table_idx,
            sampling_params=None,
            cache_handle=None,
            max_device_len=10 + seq_len + 4,
        )
        reqs.append(req)

    x = torch.randn(sum(lengths), config.hidden_size, dtype=torch.bfloat16, device=device)
    batch = Batch(reqs=reqs, phase="verify")
    batch.padded_reqs = reqs

    ctx = Context(page_size=64)
    ctx.linear_state = pool
    ctx.state_table = state_table.clone()
    ctx.work_state_table = work_state_table.clone()
    set_global_ctx(ctx)
    try:
        ctx.linear_attn_backend.prepare_metadata(batch)
        with ctx.forward_batch(batch):
            output = layer.forward(x)
    finally:
        clear_global_ctx()

    reference_ctx = Context(page_size=64)
    reference_ctx.linear_state = reference_pool
    reference_ctx.state_table = state_table.clone()
    reference_ctx.work_state_table = work_state_table.clone()
    reference_output = torch.empty_like(x)
    set_global_ctx(reference_ctx)
    try:
        offset = 0
        for req, seq_len in zip(reqs, lengths):
            one_batch = Batch(reqs=[req], phase="verify")
            reference_ctx.linear_attn_backend.prepare_metadata(one_batch)
            with reference_ctx.forward_batch(one_batch):
                reference_output[offset : offset + seq_len] = layer.forward(
                    x[offset : offset + seq_len]
                )
            offset += seq_len
    finally:
        clear_global_ctx()

    slots = torch.cat(
        [work_state_table[i, :length] for i, length in enumerate(lengths)]
    ).to(torch.int64)
    torch.testing.assert_close(output, reference_output, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(pool.conv_state[slots], reference_pool.conv_state[slots])
    torch.testing.assert_close(
        pool.recurrent_state[slots], reference_pool.recurrent_state[slots],
        rtol=1e-3, atol=1e-3,
    )
