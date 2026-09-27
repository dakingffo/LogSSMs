import pytest
import torch
import torch.nn.functional as F

from logssms.cache.logstate_pool.frame_pool import FramedMHALogStatePool
from logssms.kernel.linear_attention.conv_decode import conv_decode_qkv
from logssms.layers.gated_delta import GatedDeltaConfig


def _config() -> GatedDeltaConfig:
    return GatedDeltaConfig(
        num_k_heads=2, num_v_heads=2, head_k_dim=32, head_v_dim=32,
        conv_dim=192, state_len=3, layer_idx=1,
    )


def _reference(x, weight, history):
    return F.silu(F.conv1d(
        torch.cat([history, x.unsqueeze(-1)], dim=-1),
        weight, groups=x.shape[-1],
    )).squeeze(-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("read_slots,write_slots", [([0, 2], [0, 2]), ([0, 2], [1, 2])])
def test_standard_decode_conv_matches_reference(read_slots, write_slots):
    torch.manual_seed(93)
    config = _config()
    x = torch.randn(2, config.conv_dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(config.conv_dim, 1, 4, device="cuda", dtype=x.dtype) * 0.1
    state = torch.randn(3, 2, config.conv_dim, 3, device="cuda", dtype=x.dtype)
    before = state.clone()
    reads = torch.tensor(read_slots, device="cuda", dtype=torch.int64)
    writes = torch.tensor(write_slots, device="cuda", dtype=torch.int64)
    expected = _reference(x, weight, before[reads, 1])
    q, k, v = conv_decode_qkv(
        x, weight, config, read_slots=reads, write_slots=writes,
        standard_state=state,
    )
    actual = torch.cat([q.flatten(2), k.flatten(2), v.flatten(2)], dim=-1)[:, 0]
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    for row, slot in enumerate(write_slots):
        expected_state = torch.cat([before[read_slots[row], 1, :, 1:], x[row, :, None]], -1)
        torch.testing.assert_close(state[slot, 1], expected_state, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_framed_decode_conv_writes_log_and_work_and_replays_graph():
    config = _config()
    pool = FramedMHALogStatePool(
        num_frames=64, page_size=64, granularity=16,
        num_attention_layers=1, num_kv_heads=1, head_dim=64,
        num_linear_layers=2, num_key_heads=2, num_value_heads=2,
        head_k_dim=32, head_v_dim=32, conv_dim=192, state_len=3,
        device=torch.device("cuda"), dtype=torch.bfloat16,
    )
    assert pool.reserve(logs=2, works=2) is not None
    slots = torch.tensor([2, 3], device="cuda", dtype=torch.int32)
    logs = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    zeros = torch.zeros(2, device="cuda", dtype=torch.int32)
    granules = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    torch.manual_seed(94)
    x = torch.randn(2, 192, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(192, 1, 4, device="cuda", dtype=x.dtype) * 0.1
    history = torch.randn(2, 192, 3, device="cuda", dtype=x.dtype)
    pool.write_conv(slots, zeros, 1, history)
    expected = _reference(x, weight, history)

    def run():
        return conv_decode_qkv(
            x, weight, config, read_slots=slots, read_granules=zeros,
            write_slots=logs, write_granules=granules, work_slots=slots,
            pool=pool,
        )

    q, k, v = run()
    actual = torch.cat([q.flatten(2), k.flatten(2), v.flatten(2)], dim=-1)[:, 0]
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    shifted = torch.cat([history[:, :, 1:], x[:, :, None]], -1)
    torch.testing.assert_close(pool.read_conv(logs, granules, 1), shifted, rtol=0, atol=0)
    torch.testing.assert_close(pool.read_conv(slots, zeros, 1), shifted, rtol=0, atol=0)

    # Use a changed input on replay to detect stale captured QKV or frame addresses.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = run()
    x.copy_(torch.randn_like(x))
    pool.write_conv(slots, zeros, 1, history)
    graph.replay()
    torch.cuda.synchronize()
    expected_replay = _reference(x, weight, history)
    actual_replay = torch.cat(
        [tensor.flatten(2) for tensor in graph_output], dim=-1
    )[:, 0]
    torch.testing.assert_close(actual_replay, expected_replay, rtol=2e-2, atol=2e-2)
