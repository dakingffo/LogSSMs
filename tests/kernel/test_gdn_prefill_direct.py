import pytest
import torch
from dataclasses import replace
from itertools import product

from logssms.cache.logstate.frame_pool import MHALogStatePool
from logssms.kernel.linear_attention.logstate import store_log_states
from logssms.kernel.linear_attention.packed_conv import (
    PackedConvStateIO, packed_causal_conv1d,
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


@pytest.mark.parametrize("algorithm", ["gdn", "mamba2"])
def test_prefill_metadata_alignment_reuses_kernel(algorithm):
    from logssms.kernel.linear_attention.mamba2 import mamba2_prefill, _mamba2_prefill_state_kernel
    from logssms.kernel.linear_attention.vendor.fla.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    )

    torch.manual_seed(73)
    pool = _pool(32, 128, torch.bfloat16, head_ratio=3)
    allocation = pool.reserve(logs=5, works=8)
    assert allocation is not None
    logs, works = allocation[1:]
    lengths = [1, 17, 65, 13]
    indices = PackedLinearIndices.from_lengths(lengths, pool.device, 32)
    total = sum(lengths)
    slots = torch.tensor([logs[0]] + [logs[1]] * 17 + [logs[2]] * 64
                         + [logs[3]] + [logs[4]] * 13, device=pool.device, dtype=torch.int32)
    positions = torch.cat([torch.arange(n, device=pool.device) % 64 for n in lengths]).int()
    reads = torch.tensor(works[:4], device=pool.device, dtype=torch.int32)
    writes = torch.full((len(indices.chunk_indices),), -1, device=pool.device, dtype=torch.int32)
    writes[indices.endings] = torch.tensor(works[4:], device=pool.device, dtype=torch.int32)
    io = ChunkStateIO(
        pool.storage.view(pool.dtype), reads, writes, reads,
        pool.payload_frame_table, pool.frame_bytes, pool.head_payload_bytes,
        layer=1, num_layers=2, granularity=32, page_size=64,
    )
    query = torch.nn.functional.normalize(
        torch.randn(1, total, 2, 128, device=pool.device), dim=-1,
    ).to(pool.dtype)
    key = torch.nn.functional.normalize(torch.randn_like(query).float(), dim=-1).to(pool.dtype)
    value = torch.randn(1, total, 6, 128, device=pool.device, dtype=pool.dtype) * .1
    gate = -torch.rand(1, total, 6, device=pool.device) * .1
    beta = torch.rand_like(gate)
    log_io = ChunkLogIO(slots, positions, pool.decay_frame_table, gate, 64,
                        pool.layout.log_decay_bytes, 2)
    xbc = torch.randn(total, 6 * 128, device=pool.device, dtype=pool.dtype)
    skip = torch.randn(6, device=pool.device)
    pool.storage.zero_()
    pool.write_states(reads, 1, torch.randn(4, 6, 128, 128, device=pool.device, dtype=pool.dtype) * .01)
    initial = pool.storage.clone()

    def shifted(tensor, offset):
        if not offset:
            return tensor
        backing = tensor.new_empty(tensor.numel() + offset)
        result = backing[offset:].view(tensor.shape)
        result.copy_(tensor)
        assert result.data_ptr() % 16 != 0
        return result

    kernel = (chunk_gated_delta_rule_fwd_kernel_h_blockdim64 if algorithm == "gdn"
              else _mamba2_prefill_state_kernel)
    expected = expected_storage = compiled = None
    for write_offset, index_offset in product((0, 1), repeat=2):
        current = replace(
            io, read_slots=shifted(reads, index_offset),
            chunk_write_slots=shifted(writes, write_offset),
            final_write_slots=shifted(reads, write_offset),
            payload_frames=shifted(io.payload_frames, index_offset),
        )
        packed = replace(
            indices, cu_seqlens=shifted(indices.cu_seqlens, write_offset),
            chunk_indices=shifted(indices.chunk_indices, index_offset),
            chunk_offsets=shifted(indices.chunk_offsets, index_offset),
        )
        current_log = replace(
            log_io, slots=shifted(slots, write_offset),
            token_offsets=shifted(positions, index_offset),
            decay_frames=shifted(log_io.decay_frames, write_offset),
        )
        pool.storage.copy_(initial)
        if algorithm == "gdn":
            output, _ = chunk_gated_delta_rule_to_cache(
                query, key, value, gate, beta, current, packed.cu_seqlens,
                chunk_size=32, **packed.fla_kwargs(), log_io=current_log,
            )
        else:
            output = mamba2_prefill(
                xbc, key[0], query[0], value[0], gate[0], skip,
                current, packed, 32, log_io=current_log,
            )
        torch.cuda.synchronize()
        count = len(kernel.device_caches[torch.cuda.current_device()][0])
        if expected is None:
            expected, expected_storage, compiled = output.clone(), pool.storage.clone(), count
        else:
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(pool.storage, expected_storage, rtol=0, atol=0)
            assert count == compiled


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
    rows, snapshots = [], []
    begin = 0
    for req, length in enumerate(lengths):
        selected = list(range(granularity - 1, length, granularity))
        if not selected or selected[-1] != length - 1:
            selected.append(length - 1)
        rows.extend(begin + row for row in selected)
        snapshots.extend(torch.cat((history[req], x[begin:begin + row + 1].T), -1)[:, -state_len:]
                         for row in selected)
        begin += length
    rows = torch.tensor(rows, device=pool.device, dtype=torch.int64)
    snapshots = torch.stack(snapshots)
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
