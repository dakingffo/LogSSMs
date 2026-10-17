import pytest
import torch

from logssms.core import Batch, Request
from logssms.kernel.metadata import make_batch_mapping, gather_paged_indices, gather_linear_slots
from logssms.utils.batch import BatchIndices


def test_hybrid_paged_metadata_follows_decode_backend():
    from types import SimpleNamespace
    from logssms.layers.attention_backend.base import HybridBackend

    paged = SimpleNamespace(uses_paged_indices=True)
    flat = SimpleNamespace(uses_paged_indices=False)
    assert HybridBackend(flat, paged).uses_paged_indices
    assert not HybridBackend(paged, flat).uses_paged_indices


def make_batch(phase):
    reqs = []
    for table, cached in ((2, 0), (0, 127), (3, 64), (3, 64)):
        length = 1 if phase == "decode" else (table + 1) * 17
        reqs.append(Request(torch.zeros(cached + length, dtype=torch.int32),
                            table, cached, 1, table, None, None, cached + length + 16))
    batch = Batch(reqs[:2], phase)
    batch.padded_reqs = reqs
    batch.logstate_replay_counts = [0, 15, 0, 0]
    return batch


@pytest.mark.parametrize("phase", ["prefill", "decode", "verify"])
def test_packet_cpu_views_and_write_semantics(phase):
    batch = make_batch(phase)
    batch.reqs[0].aborted = True
    packet = BatchIndices.make(batch, torch.device("cpu"), 32)
    assert packet.table_ids.tolist() == [2, 0, 3, 3]
    assert packet.replay_counts.tolist() == [0, 15, 0, 0]
    assert packet.cu_seqlens_q_cpu[-1] == packet.num_tokens
    assert packet.cu_seqlens_k_cpu[-1] == packet.num_pages
    assert packet.seq_lens_cpu.dtype == torch.int32
    assert packet.write_tuple[0].dtype == torch.int64
    assert packet.write_tuple[0].tolist() == ([] if phase == "verify" else [2, 0])
    assert packet.write_tuple[1].tolist() == ([] if phase == "verify" else [-1, batch.reqs[1].device_len])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("phase", ["prefill", "decode", "verify"])
@pytest.mark.parametrize("page_size", [1, 16, 32, 64, 128, 256])
def test_batched_gpu_indices_match_ragged_torch_reference(phase, page_size):
    batch = make_batch(phase)
    # Padded storage tests independent row stride, and unique mappings test that
    # the implementation never infers physical addresses from logical positions.
    storage = torch.arange(4 * 512, dtype=torch.int32, device="cuda").view(4, 512)
    table = storage[:, :256]
    packet = BatchIndices.make(batch, torch.device("cuda"), page_size)
    positions, mapping, out_loc = make_batch_mapping(packet, table, decode=phase == "decode")
    expected_positions = torch.tensor([p for r in batch.padded_reqs
                                      for p in range(r.cached_len, r.device_len)], device="cuda", dtype=torch.int32)
    expected_rows = torch.tensor([r.table_idx for r in batch.padded_reqs
                                 for _ in range(r.extend_len)], device="cuda", dtype=torch.int64)
    torch.testing.assert_close(positions, expected_positions)
    torch.testing.assert_close(mapping[0], expected_rows)
    torch.testing.assert_close(mapping[1], expected_positions.long())
    torch.testing.assert_close(out_loc, table[expected_rows, expected_positions])
    pages = gather_paged_indices(packet, table)
    expected_pages = torch.cat([table[r.table_idx, :r.device_len:page_size] // page_size
                                for r in batch.padded_reqs])
    torch.testing.assert_close(pages, expected_pages)
    if phase == "decode":
        native = gather_linear_slots(packet, table)
        expected = torch.tensor([[table[r.table_idx, r.cached_len - 1], table[r.table_idx, r.cached_len]]
                                 for r in batch.padded_reqs], device="cuda", dtype=torch.int64)
        torch.testing.assert_close(native, expected)
        works = torch.tensor([[7], [5], [11], [13]], device="cuda", dtype=torch.int32)
        log = gather_linear_slots(packet, table, work_table=works, out_loc=out_loc, page_size=64)
        torch.testing.assert_close(log[0], works[expected_rows, 0])
        torch.testing.assert_close(log[1], table[expected_rows, expected_positions])
        torch.testing.assert_close(log[2], out_loc % 64)
        assert not log[3].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("page_size", [1, 16, 32, 64])
def test_decode_paged_mapping_spans_multiple_blocks(page_size):
    req = Request(torch.zeros(8193, dtype=torch.int32), 1, 8192, 1, 0, None, None, 8194)
    batch = Batch([req], "decode")
    batch.padded_reqs = batch.reqs
    table = torch.arange(2 * 16384, dtype=torch.int32, device="cuda").reshape(2, 16384)[:, :8194]
    packet = BatchIndices.make(batch, torch.device("cuda"), page_size)
    positions, mapping, locations = make_batch_mapping(packet, table, decode=True)
    torch.testing.assert_close(positions, torch.tensor([8192], dtype=torch.int32, device="cuda"))
    torch.testing.assert_close(locations, table[1, 8192:8193])
    torch.testing.assert_close(gather_paged_indices(packet, table), table[1, :8193:page_size] // page_size)
    assert packet.page_indices is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_graph_replay_and_overlapped_packets_do_not_keep_old_mappings():
    stream = torch.cuda.Stream()
    batch = make_batch("decode")
    table = torch.arange(4 * 256, dtype=torch.int32, device="cuda").reshape(4, 256)
    stable = torch.zeros(4, dtype=torch.int32, device="cuda")
    result = torch.empty_like(stable)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result.copy_(stable + 1)
    stream.wait_stream(torch.cuda.current_stream())
    snapshots = []
    with torch.cuda.stream(stream):
        for turn in range(24):
            for req in batch.padded_reqs:
                req.cached_len = 16 + turn
                req.device_len = req.cached_len + 1
            packet = BatchIndices.make(batch, torch.device("cuda"), 32)
            positions, mapping, out_loc = make_batch_mapping(packet, table)
            stable.copy_(out_loc)
            graph.replay()
            snapshots.append(result.clone())
    stream.synchronize()
    for turn, snapshot in enumerate(snapshots):
        expected = table[torch.tensor([2, 0, 3, 3], device="cuda"), 16 + turn] + 1
        torch.testing.assert_close(snapshot, expected)
