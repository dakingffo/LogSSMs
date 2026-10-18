import pytest
import torch

from logssms.core import Batch, ChunkedRequest, Request
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


@pytest.mark.parametrize("lengths", [[1], [31, 1], [32, 33], [7, 17, 65]])
@pytest.mark.parametrize("decode_count", [1, 2, 3, 7])
def test_mixed_packet_matches_independent_packets(lengths, decode_count):
    reqs = [Request(torch.zeros(length + i * 32, dtype=torch.int32), i, i * 32,
                    8, i, None, None, 1024)
            for i, length in enumerate(lengths + [1] * decode_count)]
    batch = Batch(reqs, "mixed", num_prefill_reqs=len(lengths))
    batch.padded_reqs = reqs
    batch.logstate_replay_counts = [i % 16 for i in range(len(reqs))]
    combined = BatchIndices.make_mixed(batch, torch.device("cpu"), 16)
    for packet, start, end, phase in (
        (combined[0], 0, len(reqs), "mixed"),
        (combined[1], 0, len(lengths), "prefill"),
        (combined[2], len(lengths), len(reqs), "decode"),
    ):
        part = Batch(reqs[start:end], phase, num_prefill_reqs=len(lengths) if phase == "mixed" else 0)
        part.padded_reqs = part.reqs
        part.logstate_replay_counts = batch.logstate_replay_counts[start:end]
        expected = BatchIndices.make(part, torch.device("cpu"), 16)
        for name, value in vars(expected).items():
            actual = getattr(packet, name)
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual, value)
                assert actual.data_ptr() % 16 == 0
            elif name in ("write_tuple", "sample_tuple"):
                for a, b in zip(actual, value):
                    torch.testing.assert_close(a, b)
            else:
                assert actual == value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_linear_slot_metadata_reuses_variants_across_decode_counts():
    from logssms.engine.warmup import record_kernel_variants

    device = torch.device("cuda")
    table = torch.arange(32 * 128, device=device, dtype=torch.int32).reshape(32, 128)
    work = torch.arange(32, device=device, dtype=torch.int32)[:, None]
    packets = []
    for count in range(1, 33):
        reqs = [Request(torch.zeros(18, dtype=torch.int32), i, 17, 1, i, None, None, 64)
                for i in range(count)]
        batch = Batch(reqs, "decode")
        batch.padded_reqs = reqs
        packet = BatchIndices.make(batch, device, 16)
        out_loc = torch.arange(count + 1, device=device, dtype=torch.int32)[1:]
        packets.append((packet, out_loc))
    for packet, out_loc in (packets[0], packets[1], packets[-1]):
        gather_linear_slots(packet, table, work_table=work, out_loc=out_loc, page_size=64)
    with record_kernel_variants(strict=True):
        for packet, out_loc in packets:
            slots = gather_linear_slots(packet, table, work_table=work, out_loc=out_loc, page_size=64)
            torch.testing.assert_close(slots[0], work[:len(packet.table_ids), 0])


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


@pytest.mark.parametrize("phase", ["prefill", "mixed", "decode", "verify"])
def test_compact_chunk_writes_match_aligned_outputs(phase):
    from types import SimpleNamespace
    from logssms.scheduler.prefill import PrefillManager

    chunk = ChunkedRequest(torch.arange(4), 2, 0, 8, 20, None, None, 12)
    sampled = Request(torch.arange(3), 0, 2, 8, 21, None, None, 11)
    padding = Request(torch.arange(1), 3, 0, 8, 22, None, None, 9)
    batch = Batch([chunk, sampled], phase)
    batch.padded_reqs = [chunk, sampled, padding]
    packet = BatchIndices.make(batch, torch.device("cpu"), 32)
    expected = [] if phase == "verify" else [0]
    assert packet.write_tuple[0].tolist() == expected
    assert packet.write_tuple[1].tolist() == ([] if phase == "verify" else [3])
    # Input metadata stays aligned to all padded rows, including chunks.
    assert packet.table_ids.tolist() == [2, 0, 3]
    assert packet.seq_lens_cpu.tolist() == [4, 3, 1]
    pool = torch.full((4, 16), 99, dtype=torch.int32)
    manager = PrefillManager(pool, 32)
    output = SimpleNamespace(next_tokens_gpu=torch.tensor([-1, 7], dtype=torch.int32))
    manager.advance_for_next_schedule(None, (batch, None, None, packet.write_tuple), output)
    assert pool[0, 3].item() == (99 if phase == "verify" else 7)
    assert (pool[2] == 99).all() and (pool[3] == 99).all()


def test_all_chunk_prefill_has_empty_compact_writes():
    from types import SimpleNamespace
    from logssms.scheduler.prefill import PrefillManager

    chunk = ChunkedRequest(torch.arange(4), 0, 0, 8, 1, None, None, 12)
    batch = Batch([chunk], "prefill")
    batch.padded_reqs = [chunk]
    packet = BatchIndices.make(batch, torch.device("cpu"), 32)
    assert all(vector.numel() == 0 for vector in packet.write_tuple)
    pool = torch.full((1, 16), 99, dtype=torch.int32)
    PrefillManager(pool, 32).advance_for_next_schedule(
        None, (batch, None, None, packet.write_tuple),
        SimpleNamespace(next_tokens_gpu=torch.tensor([-1], dtype=torch.int32)),
    )
    assert (pool == 99).all()
    assert chunk.cached_len == chunk.device_len == 4


def test_mixed_subbatch_compact_writes_concatenate_in_real_request_order():
    chunk = ChunkedRequest(torch.arange(4), 2, 0, 8, 20, None, None, 12)
    decode = Request(torch.arange(3), 0, 2, 8, 21, None, None, 11)
    mixed = Batch([chunk, decode], "mixed", num_prefill_reqs=1)
    prefill = Batch([chunk], "prefill")
    ar = Batch([decode], "decode")
    for batch in (mixed, prefill, ar):
        batch.padded_reqs = batch.reqs
    packets = [BatchIndices.make(batch, torch.device("cpu"), 32)
               for batch in (mixed, prefill, ar)]
    for i in (0, 1):
        torch.testing.assert_close(
            packets[0].write_tuple[i],
            torch.cat([packets[1].write_tuple[i], packets[2].write_tuple[i]]),
        )


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
