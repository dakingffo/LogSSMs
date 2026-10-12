import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from logssms.cache.logstate_pool.mha_pool import MHALogStatePool
from logssms.kernel.linear_attention.gdn_conv_decode import (
    _frame_element_offset,
    conv_decode_qkv,
)
from logssms.layers.gated_delta import GatedDeltaConfig


@triton.jit
def _large_frame_offset_kernel(frame, result):
    value = tl.load(frame)
    tl.store(result, _frame_element_offset(value, 1 << 20))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_logstate_decode_conv_uses_64_bit_frame_offsets():
    frame = torch.tensor([2048], device="cuda", dtype=torch.int32)
    result = torch.empty(1, device="cuda", dtype=torch.int64)
    _large_frame_offset_kernel[(1,)](frame, result)
    assert result.item() == 1 << 31


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
@pytest.mark.parametrize("fused_input", [False, True])
def test_standard_decode_conv_matches_reference(read_slots, write_slots, fused_input):
    torch.manual_seed(93)
    config = _config()
    x = torch.randn(2, config.conv_dim, device="cuda", dtype=torch.bfloat16)
    if fused_input:
        projected = torch.randn(2, config.conv_dim + config.value_dim, device="cuda", dtype=x.dtype)
        projected[:, :config.conv_dim].copy_(x)
        x = projected[:, :config.conv_dim]
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
@pytest.mark.parametrize("fused_input", [False, True])
def test_logstate_decode_conv_writes_log_and_work_and_replays_graph(fused_input):
    config = _config()
    pool = MHALogStatePool(
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
    if fused_input:
        projected = torch.randn(2, 256, device="cuda", dtype=x.dtype)
        projected[:, :192].copy_(x)
        x = projected[:, :192]
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_packed_conv_and_snapshots_read_fused_qkv_views():
    from logssms.kernel.linear_attention.packed_conv import (
        packed_causal_conv1d, packed_conv_snapshots,
    )

    torch.manual_seed(95)
    channels, state_len = 192, 3
    x = torch.randn(8, channels + 64, device="cuda", dtype=torch.bfloat16)[:, :channels]
    history = torch.randn(2, channels, state_len, device="cuda", dtype=x.dtype)
    weight = torch.randn(channels, 1, state_len + 1, device="cuda", dtype=x.dtype) * 0.1
    offsets = torch.tensor([0, 3, 8], device="cuda", dtype=torch.int32)
    chunks = torch.tensor([[0, 0], [1, 0]], device="cuda", dtype=torch.int32)
    rows = torch.tensor([0, 2, 3, 7], device="cuda", dtype=torch.int32)
    requests = torch.tensor([0, 0, 1, 1], device="cuda", dtype=torch.int32)
    actual = packed_causal_conv1d(x, weight, history, offsets, chunks)
    expected = torch.cat([
        F.silu(F.conv1d(
            torch.cat([history[i:i + 1], x[begin:end].T.unsqueeze(0)], dim=-1),
            weight, groups=channels,
        )).squeeze(0).T for i, (begin, end) in enumerate([(0, 3), (3, 8)])
    ])
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    snapshots = packed_conv_snapshots(x, history, offsets, rows, requests)
    expected_snapshots = torch.stack([
        torch.cat([history[i], x[begin:row + 1].T], dim=-1)[:, -state_len:]
        for i, begin, row in [(0, 0, 0), (0, 0, 2), (1, 3, 3), (1, 3, 7)]
    ])
    torch.testing.assert_close(snapshots, expected_snapshots, rtol=0, atol=0)
