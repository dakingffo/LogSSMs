import pytest
import torch
import torch.nn.functional as F

from logssms.cache.linear_pool.state_pool import LinearStatePool
from logssms.cache.logstate_pool.frame_pool import FramedMHALogStatePool
from logssms.core import Batch, Context, clear_global_ctx, set_global_ctx
from logssms.distributed import DistributedInfo, tp_override
from logssms.layers import GatedDeltaNet
from logssms.layers.linear_attention_backend import LogStateLinearAttentionBackend
from logssms.layers.linear_attention_backend.logstate import LogStateLinearAttentionMetadata
from logssms.layers.linear_attention_backend.native import LinearAttentionMetadata
from logssms.utils import torch_dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("cache_type", ["radix", "logstate"])
def test_fused_gdn_graph_matches_separate_projections(cache_type):
    torch.manual_seed(96)
    device, dtype = torch.device("cuda"), torch.bfloat16
    with tp_override(DistributedInfo(rank=0, size=1)):
        with torch.device(device), torch_dtype(dtype):
            layer = GatedDeltaNet(64, 2, 4, 32, 32, 4, 1e-6, 0)
    for tensor in layer.state_dict().values():
        tensor.normal_(std=0.1)

    ctx = Context(page_size=64)
    batch = Batch(reqs=[], phase="decode")
    work_slots = torch.tensor([2, 3], device=device, dtype=torch.int32)
    log_slots = torch.tensor([0, 1], device=device, dtype=torch.int32)
    zeros = torch.zeros(2, device=device, dtype=torch.int32)
    offsets = torch.tensor([0, 15], device=device, dtype=torch.int32)
    config = layer._config
    if cache_type == "logstate":
        pool = FramedMHALogStatePool(
            num_frames=64, page_size=64, granularity=32,
            num_attention_layers=1, num_kv_heads=1, head_dim=64,
            num_linear_layers=1, num_key_heads=2, num_value_heads=4,
            head_k_dim=32, head_v_dim=32, conv_dim=config.conv_dim, state_len=3,
            device=device, dtype=dtype,
        )
        assert pool.reserve(logs=2, works=2) is not None
        ctx.linear_attn_backend = LogStateLinearAttentionBackend()
        batch.linear_attn_metadata = LogStateLinearAttentionMetadata(
            phase="decode", work_slots=work_slots, log_slots=log_slots,
            token_offsets=offsets, conv_read_slots=work_slots,
            conv_read_granules=zeros, replay_counts=zeros,
        )
    else:
        pool = LinearStatePool(4, 1, config.conv_dim, 4, 4, 32, 32, device, dtype)
        batch.linear_attn_metadata = LinearAttentionMetadata(
            phase="decode", read_slots=work_slots, write_slots=log_slots,
        )
    ctx.linear_state = pool
    initial_state = torch.randn(2, 4, 32, 32, device=device, dtype=dtype) * 0.1
    initial_conv = torch.randn(2, config.conv_dim, 3, device=device, dtype=dtype) * 0.1
    x = torch.randn(2, 64, device=device, dtype=dtype)

    def reset():
        if cache_type == "logstate":
            pool.write_states(work_slots, 0, initial_state)
            pool.write_conv(work_slots, zeros, 0, initial_conv)
        else:
            pool.recurrent_state.zero_()
            pool.conv_state.zero_()
            pool.recurrent_state[work_slots, 0] = initial_state
            pool.conv_state[work_slots, 0] = initial_conv

    def state():
        if cache_type == "logstate":
            return (
                pool.read_states(work_slots, 0),
                pool.read_conv(log_slots, offsets // 32, 0),
                pool.read_conv(work_slots, zeros, 0),
            )
        return pool.recurrent_state.clone(), pool.conv_state.clone()

    def separate_forward():
        qkv_weight, z_weight = layer.in_proj_qkvz.weight.split(
            [config.conv_dim, config.value_dim]
        )
        b_weight, a_weight = layer.in_proj_ba.weight.chunk(2)
        core = ctx.linear_attn_backend.forward(
            mixed_qkv=F.linear(x, qkv_weight), gate=F.linear(x, a_weight),
            beta=F.linear(x, b_weight), conv_weight=layer.conv1d.weight,
            A_log=layer.A_log, dt_bias=layer.dt_bias, config=config, batch=batch,
        )
        hidden = core.reshape(-1, config.head_v_dim).float()
        hidden = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + layer.norm.eps)
        gate = F.linear(x, z_weight).reshape(hidden.shape).float()
        hidden = (hidden * layer.norm.weight * F.silu(gate)).to(dtype)
        return layer.out_proj.forward(hidden.reshape(2, -1))

    clear_global_ctx()
    set_global_ctx(ctx)
    try:
        with ctx.forward_batch(batch):
            reset()
            layer.forward(x)  # Warm the fused projection consumers before capture.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = layer.forward(x)
            for _ in range(2):
                x.copy_(torch.randn_like(x))
                reset()
                expected = separate_forward()
                expected_state = state()
                reset()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(captured, expected, rtol=2e-2, atol=2e-3)
                for actual, reference in zip(state(), expected_state):
                    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-3)
    finally:
        clear_global_ctx()
