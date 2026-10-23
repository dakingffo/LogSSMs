import pytest
import torch

from logssms.cache.logstate.frame_pool import MHALogStatePool
from logssms.kernel.linear_attention.logstate import store_log_states
from logssms.kernel.linear_attention.packed_conv import (
    PackedConvStateIO, packed_causal_conv1d, packed_conv_snapshots,
)
from logssms.kernel.linear_attention.vendor.fla import (
    ChunkLogIO, ChunkStateIO, chunk_gated_delta_rule_to_cache,
)
from logssms.layers.linear_attention_backend.standard import PackedLinearIndices


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _pool(granularity, dim, dtype, head_ratio=1):
    return MHALogStatePool(
        num_frames=128, page_size=64, granularity=granularity,
        num_attention_layers=1, num_kv_heads=2, head_dim=256,
        num_linear_layers=2, num_key_heads=2, num_value_heads=2 * head_ratio,
        head_k_dim=dim, head_v_dim=dim, conv_dim=(4 + 2 * head_ratio) * dim, state_len=3,
        device=torch.device("cuda"), dtype=dtype,
    )


@pytest.mark.parametrize("granularity", [16, 32, 64])
@pytest.mark.parametrize("dim", [32, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_ratio", [1, 2, 3, 4])
@pytest.mark.parametrize("expand_key_heads", [False, True])
def test_prefill_log_direct_matches_separate_store(granularity, dim, dtype, head_ratio, expand_key_heads):
    torch.manual_seed(31)
    pool = _pool(granularity, dim, dtype, head_ratio)
    allocated = pool.reserve(logs=4, works=6)
    assert allocated is not None
    logs, works = allocated[1:]
    lengths = [1, 17, 65]
    indices = PackedLinearIndices.from_lengths(lengths, pool.device, granularity)
    total = sum(lengths)
    slots = torch.tensor([logs[0]] + [logs[1]] * 17 + [logs[2]] * 64 + [logs[3]],
                         dtype=torch.int32, device=pool.device)
    positions = torch.cat([torch.arange(n, device=pool.device) % 64 for n in lengths]).int()
    # A memorized Checkpoint interval must not be overwritten as a Log.
    slots[1:9] = -1
    query = torch.nn.functional.normalize(torch.randn(1, total, 2, dim, device=pool.device), dim=-1).to(dtype)
    key = torch.nn.functional.normalize(torch.randn_like(query).float(), dim=-1).to(dtype)
    log_key = key
    if expand_key_heads:
        query = query.repeat_interleave(head_ratio, dim=2)
        key = key.repeat_interleave(head_ratio, dim=2)
    value = torch.randn(1, total, 2 * head_ratio, dim, device=pool.device, dtype=dtype) * .1
    gate = -torch.rand(1, total, 2 * head_ratio, device=pool.device) * .1
    beta = torch.rand_like(gate)
    work_slots = torch.tensor(works[:3], dtype=torch.int32, device=pool.device)
    writes = torch.full((len(indices.chunk_indices),), -1, dtype=torch.int32, device=pool.device)
    writes.index_copy_(0, indices.endings, torch.tensor(works[3:], dtype=torch.int32, device=pool.device))
    io = ChunkStateIO(
        storage=pool.storage.view(dtype), read_slots=work_slots,
        chunk_write_slots=writes, final_write_slots=work_slots,
        payload_frames=pool.payload_frame_table, frame_bytes=pool.frame_bytes,
        head_payload_bytes=pool.head_payload_bytes, layer=1,
        num_layers=pool.layout.num_linear_layers, granularity=pool.layout.granularity,
        page_size=pool.page_size,
    )
    pool.storage.fill_(0)
    pool.write_states(work_slots, 1, torch.randn(3, 2 * head_ratio, dim, dim, device=pool.device, dtype=dtype) * .01)
    initial = pool.storage.clone()
    expected, residual = chunk_gated_delta_rule_to_cache(
        query, key, value, gate, beta, io, indices.cu_seqlens,
        chunk_size=granularity, **indices.fla_kwargs(),
    )
    active = slots >= 0
    store_log_states(pool, slots[active], positions[active],
                            log_key[0, active], residual[0, active], gate[0, active], 1)
    expected_storage = pool.storage.clone()
    pool.storage.copy_(initial)
    actual, actual_residual = chunk_gated_delta_rule_to_cache(
        query, key, value, gate, beta, io, indices.cu_seqlens,
        chunk_size=granularity, **indices.fla_kwargs(),
        log_io=ChunkLogIO(slots, positions, pool.decay_frame_table, gate,
                         pool.page_size, pool.layout.log_decay_bytes, pool.layout.num_key_heads),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_residual, residual, rtol=0, atol=0)
    # Includes unwritten granules, other layers, and shared decay records.
    torch.testing.assert_close(pool.storage, expected_storage, rtol=0, atol=0)


@pytest.mark.parametrize("granularity", [16, 32, 64])
@pytest.mark.parametrize("logstate", [False, True])
def test_conv_prefill_direct_snapshots_and_mutable_graph_destinations(granularity, logstate):
    torch.manual_seed(13)
    pool = _pool(granularity, 32, torch.bfloat16)
    allocated = pool.reserve(logs=5, works=3)
    assert allocated is not None
    logs, works = allocated[1:]
    lengths = [1, 17, 65]
    indices = PackedLinearIndices.from_lengths(lengths, pool.device, granularity)
    total = sum(lengths)
    channels, state_len = pool.layout.conv_dim, pool.layout.state_len
    # Strided projection view and nonzero input conv history, including short sequences.
    x = torch.randn(total, channels * 2, device=pool.device, dtype=pool.dtype)[:, ::2]
    history = torch.randn(3, channels, state_len, device=pool.device, dtype=pool.dtype)
    weight = torch.randn(channels, 1, state_len + 1, device=pool.device, dtype=pool.dtype) * .02
    bias = torch.randn(channels, device=pool.device, dtype=pool.dtype) * .01
    rows, requests = [], []
    begin = 0
    for req, length in enumerate(lengths):
        selected = list(range(granularity - 1, length, granularity))
        if not selected or selected[-1] != length - 1:
            selected.append(length - 1)
        rows.extend(begin + row for row in selected)
        requests.extend([req] * len(selected))
        begin += length
    rows = torch.tensor(rows, device=pool.device, dtype=torch.int64)
    requests = torch.tensor(requests, device=pool.device, dtype=torch.int64)
    snapshots = packed_conv_snapshots(x, history, indices.cu_seqlens, rows, requests)
    expected = packed_causal_conv1d(x, weight, history, indices.cu_seqlens, indices.conv_chunks, bias)
    slots = torch.full((total,), -1, device=pool.device, dtype=torch.int32)
    granules = torch.zeros_like(slots)
    # One physical page per request plus a second page for the last request.
    token_slots = torch.tensor([logs[0]] + [logs[1]] * 17 + [logs[2]] * 64 + [logs[3]],
                               device=pool.device, dtype=torch.int32)
    token_offsets = torch.cat([torch.arange(n, device=pool.device) % 64 for n in lengths])
    slots[rows] = token_slots[rows]
    granules[rows] = (token_offsets[rows] // granularity).int()
    final_slots = torch.tensor(works, device=pool.device, dtype=torch.int32)
    if logstate:
        storage = pool.storage.view(pool.dtype)
        io = PackedConvStateIO(storage, slots, granules, final_slots,
                               pool.conv_frame_table, pool.frame_bytes, 1)
    else:
        storage = torch.empty(128, 2, channels, state_len, device=pool.device, dtype=pool.dtype)
        # Standard slots refer to one snapshot each, not multiple granules per slot.
        slots[rows] = torch.arange(len(rows), device=pool.device, dtype=torch.int32)
        final_slots.fill_(-1)
        io = PackedConvStateIO(storage, slots, layer=1)

    def check():
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if logstate:
            got = pool.read_conv(slots[rows], granules[rows], 1)
            ends = indices.cu_seqlens[1:] - 1
            end_indices = torch.searchsorted(rows, ends)
            torch.testing.assert_close(pool.read_conv(final_slots, torch.zeros_like(final_slots), 1),
                                       snapshots[end_indices], rtol=0, atol=0)
        else:
            got = storage[slots[rows].long(), 1]
        torch.testing.assert_close(got, snapshots, rtol=0, atol=0)

    actual = packed_causal_conv1d(x, weight, history, indices.cu_seqlens,
                                 indices.conv_chunks, bias, state_io=io)
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = packed_causal_conv1d(x, weight, history, indices.cu_seqlens,
                                     indices.conv_chunks, bias, state_io=io)
    if logstate:
        # Mutate descriptors without recapturing; the graph must resolve them at replay.
        pool.conv_frame_table[logs[0], 0], pool.conv_frame_table[logs[4], 0] = (
            pool.conv_frame_table[logs[4], 0].clone(), pool.conv_frame_table[logs[0], 0].clone()
        )
    else:
        slots[rows] += len(rows)
    graph.replay()
    check()
