import torch
import pytest

from logssms.cache.logstate_pool.mha_pool import MHALogStatePool
from tests.logstate_helpers import head_view, write_head, _head_indices
from tests.logstate_helpers import decay_view


def make_pool(device: torch.device | None = None) -> MHALogStatePool:
    return MHALogStatePool(
        num_frames=64,
        page_size=64,
        granularity=16,
        num_attention_layers=2,
        num_kv_heads=1,
        head_dim=32,
        num_linear_layers=2,
        num_key_heads=2,
        num_value_heads=2,
        head_k_dim=32,
        head_v_dim=32,
        conv_dim=192,
        state_len=3,
        device=torch.device("cpu") if device is None else device,
        dtype=torch.float16,
    )


def test_grouped_value_heads_use_value_head_payloads():
    pool = MHALogStatePool(
        num_frames=64,
        page_size=64,
        granularity=32,
        num_attention_layers=2,
        num_kv_heads=1,
        head_dim=32,
        num_linear_layers=2,
        num_key_heads=2,
        num_value_heads=6,
        head_k_dim=32,
        head_v_dim=32,
        conv_dim=320,
        state_len=3,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    assert pool.layout.num_value_heads == 3 * pool.layout.num_key_heads
    allocated = pool.reserve(works=1)
    assert allocated is not None
    slots = torch.tensor(allocated[2], dtype=torch.int32)
    values = torch.arange(6 * 32 * 32, dtype=pool.dtype).view(1, 6, 32, 32)
    pool.write_states(slots, 0, values)
    torch.testing.assert_close(pool.read_states(slots, 0), values)


@pytest.mark.parametrize("granularity", [16, 32, 64])
@pytest.mark.parametrize("key_heads", [1, 2])
def test_log_payload_uses_granule_layer_head_token_order(granularity, key_heads):
    pool = MHALogStatePool(
        num_frames=64, page_size=64, granularity=granularity,
        num_attention_layers=2, num_kv_heads=1, head_dim=32,
        num_linear_layers=2, num_key_heads=key_heads, num_value_heads=2,
        head_k_dim=32, head_v_dim=32, conv_dim=192, state_len=3,
        device=torch.device("cpu"), dtype=torch.float16,
    )
    # Force the logical payload to span non-adjacent physical frames.
    kv, logs, _works = pool.reserve(kv=20, logs=0)
    pool.release_kv(kv[::2])
    _kv, logs, _works = pool.reserve(logs=1)
    slot = logs[0]
    pool.storage.fill_(0)
    heads = []
    for layer in range(2):
        for head in range(2):
            values = torch.arange(4096, dtype=torch.float16) % 127 + layer * 128 + head * 256
            values[:2048] = torch.arange(2048, dtype=torch.float16) % 127 + layer * 128 + (head // (2 // key_heads)) * 256
            write_head(pool, slot, layer, head, values, state=False)
            heads.append(values)
    expected = []
    for begin in range(0, 64, granularity):
        for layer in range(2):
            for plane in range(2):
                for index_head in range(key_heads if plane == 0 else 2):
                    for token in range(begin, begin + granularity):
                        head = index_head * (2 // key_heads) if plane == 0 else index_head
                        index = plane * 64 * 32 + token * 32
                        expected.append(heads[layer * 2 + head][index:index + 32])
    frames = pool.allocator.slot(slot).payload
    actual = pool.storage[list(frames)].view(pool.dtype).flatten()
    expected = torch.cat(expected)
    torch.testing.assert_close(actual[:len(expected)], expected)
    occupied = set()
    for layer in range(2):
        for head in range(2):
            log = set(_head_indices(pool, slot, layer, head, state=False).tolist())
            if not pool.shared_keys:
                assert not occupied & log
            occupied.update(log)


@pytest.mark.parametrize("granularity", [16, 32, 64])
@pytest.mark.parametrize("key_heads", [1, 2])
def test_state_heads_are_contiguous_independent_of_log_layout(granularity, key_heads):
    pool = MHALogStatePool(
        num_frames=64, page_size=64, granularity=granularity,
        num_attention_layers=2, num_kv_heads=1, head_dim=32,
        num_linear_layers=2, num_key_heads=key_heads, num_value_heads=2,
        head_k_dim=32, head_v_dim=32, conv_dim=192, state_len=3,
        device=torch.device("cpu"), dtype=torch.float16,
    )
    kv, _, _ = pool.reserve(kv=20)
    pool.release_kv(kv[::2])
    _, _, works = pool.reserve(works=2)
    occupied = set()
    for slot in works:
        for layer in range(2):
            for head in range(2):
                indices = _head_indices(pool, slot, layer, head, state=True)
                physical = indices
                assert torch.all(physical.diff() == 1)
                assert not occupied.intersection(indices.tolist())
                occupied.update(indices.tolist())
                values = torch.arange(1024, dtype=pool.dtype).view(32, 32) + 100 * (layer + head)
                write_head(pool, slot, layer, head, values, state=True)
                torch.testing.assert_close(head_view(pool, slot, layer, head, state=True), values)


def test_reserve_profile_separates_allocator_and_descriptor_work():
    pool = make_pool()
    timings = {}
    assert pool.reserve(logs=1, works=1, timings=timings) is not None
    assert timings["descriptor_count"] == 2
    assert timings["poll_reclamation_ms"] >= 0
    assert timings["allocator_reserve_ms"] >= 0
    assert timings["descriptor_write_ms"] >= 0

    failed = {}
    assert pool.reserve(kv=1000, timings=failed) is None
    assert failed["failed"] == 1


def test_logstate_kv_preserves_sixteen_token_page_view():
    pool = make_pool()
    allocated = pool.reserve(kv=2)
    assert allocated is not None
    first, second = allocated[0]
    assert pool.k_cache(0).shape == (64 * 4, 16, 1, 32)
    pool.k_cache(0)[first * 4:first * 4 + 4].fill_(1)
    pool.v_cache(1)[second * 4:second * 4 + 4].fill_(2)
    assert torch.all(pool.k_cache(0)[first * 4 + 3] == 1)
    assert torch.all(pool.v_cache(1)[second * 4] == 2)
    assert pool.k_cache(0)[first * 4].data_ptr() != pool.v_cache(1)[second * 4].data_ptr()


def test_logstate_work_state_and_conv_roundtrip():
    pool = make_pool()
    allocated = pool.reserve(works=2)
    assert allocated is not None
    slots = torch.tensor(allocated[2], dtype=torch.int32)
    granules = torch.zeros(2, dtype=torch.int32)
    state = torch.arange(
        2 * pool.layout.num_value_heads * pool.layout.head_k_dim * pool.layout.head_v_dim,
        dtype=pool.dtype,
    ).view(2, pool.layout.num_value_heads, pool.layout.head_k_dim, pool.layout.head_v_dim)
    conv = torch.arange(
        2 * pool.layout.conv_dim * pool.layout.state_len,
        dtype=pool.dtype,
    ).view(2, pool.layout.conv_dim, pool.layout.state_len)
    pool.write_states(slots, 1, state)
    pool.write_conv(slots, granules, 1, conv)
    torch.testing.assert_close(pool.read_states(slots, 1), state)
    torch.testing.assert_close(pool.read_conv(slots, granules, 1), conv)


def test_state_payload_stores_value_dimension_contiguously():
    pool = MHALogStatePool(
        num_frames=64, page_size=64, granularity=32,
        num_attention_layers=1, num_kv_heads=1, head_dim=32,
        num_linear_layers=2, num_key_heads=1, num_value_heads=2,
        head_k_dim=32, head_v_dim=16, conv_dim=64, state_len=3,
        device=torch.device("cpu"), dtype=torch.float16,
    )
    _, _, works = pool.reserve(works=1)
    slots = torch.tensor(works, dtype=torch.int32)
    state = torch.arange(2 * 32 * 16, dtype=pool.dtype).view(1, 2, 32, 16)
    pool.write_states(slots, 1, state)
    torch.testing.assert_close(pool.read_states(slots, 1), state)
    record = pool.allocator.slot(works[0])
    physical = pool.storage[list(record.payload)].view(pool.dtype).flatten()
    for head in range(2):
        begin = (2 + head) * pool.head_payload_bytes // 2
        torch.testing.assert_close(
            physical[begin:begin + 32 * 16], state[0, head].flatten(),
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_copy_conv_snapshot_copies_all_layers(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    pool = make_pool(torch.device(device))
    allocated = pool.reserve(logs=1, works=1)
    assert allocated is not None
    source, destination = allocated[1][0], allocated[2][0]
    expected = torch.arange(
        pool.layout.num_linear_layers * pool.layout.conv_dim * pool.layout.state_len,
        dtype=pool.dtype, device=device,
    ).view(pool.layout.num_linear_layers, pool.layout.conv_dim, pool.layout.state_len)
    pool.conv_view(source, 2).copy_(expected)
    pool.conv_view(destination, 0).fill_(-1)

    pool.copy_conv_snapshot(source, 2, destination)

    torch.testing.assert_close(pool.conv_view(destination, 0), expected)
    torch.testing.assert_close(pool.conv_view(source, 2), expected)


def test_batch_zero_work_slots_does_not_clear_log_or_descriptors():
    pool = make_pool()
    allocated = pool.reserve(logs=1, works=2)
    assert allocated is not None
    log = allocated[1][0]
    works = allocated[2]
    pool.storage.fill_(7)
    payload_before = pool.payload_frame_table[works].clone()
    pool.zero_work_slots(works)
    for slot in works:
        record = pool.allocator.slot(slot)
        for frame in (*record.payload, *(f for group in record.conv for f in group)):
            assert torch.all(pool.storage[frame] == 0)
    assert torch.all(pool.storage[pool.allocator.slot(log).payload[0]] == 7)
    torch.testing.assert_close(pool.payload_frame_table[works], payload_before)


def test_batch_release_clears_only_selected_descriptors():
    pool = make_pool()
    allocated = pool.reserve(logs=2, works=2)
    assert allocated is not None
    selected = [allocated[1][0], allocated[2][0], allocated[2][1]]
    survivor = allocated[1][1]
    payload = pool.payload_frame_table[survivor].clone()
    conv = pool.conv_frame_table[survivor].clone()
    decay = pool.decay_frame_table[survivor].clone()
    pool.release_slots(selected)
    for table in (pool.payload_frame_table, pool.conv_frame_table, pool.decay_frame_table):
        assert torch.all(table[selected] == -1)
    torch.testing.assert_close(pool.payload_frame_table[survivor], payload)
    torch.testing.assert_close(pool.conv_frame_table[survivor], conv)
    torch.testing.assert_close(pool.decay_frame_table[survivor], decay)
    pool.allocator.check_integrity()
    assert pool.reserve(logs=1, works=1) is not None


def test_logstate_kv_store_uses_physical_token_offsets():
    pool = make_pool()
    allocated = pool.reserve(kv=2)
    assert allocated is not None
    frame = allocated[0][1]
    positions = torch.tensor([frame * 64, frame * 64 + 17, frame * 64 + 63])
    keys = torch.arange(3 * 32, dtype=pool.dtype).view(3, 1, 32)
    values = keys + 100
    pool.store_kv(keys, values, positions, 1)
    for row, position in enumerate(positions.tolist()):
        page, offset = divmod(position, pool.kv_page_size)
        torch.testing.assert_close(pool.k_cache(1)[page, offset], keys[row])
        torch.testing.assert_close(pool.v_cache(1)[page, offset], values[row])


def test_linear_descriptors_change_when_log_becomes_checkpoint():
    pool = make_pool()
    allocated = pool.reserve(logs=1, works=1)
    assert allocated is not None
    log, work = allocated[1][0], allocated[2][0]
    assert torch.all(pool.conv_frame_table[log] >= 0)
    assert torch.all(pool.conv_frame_table[work, 0] >= 0)
    assert torch.all(pool.conv_frame_table[work, 1:] == -1)
    payload = pool.payload_frame_table[log].clone()
    pool.materialize(log)
    torch.testing.assert_close(pool.payload_frame_table[log], payload)
    assert torch.all(pool.conv_frame_table[log, :-1] == -1)
    assert torch.all(pool.conv_frame_table[log, -1] >= 0)
    assert torch.all(pool.decay_frame_table[log] == -1)
    pool.allocator.check_integrity()
    pool.release_slots([log])
    assert torch.all(pool.payload_frame_table[log] == -1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batch_materialization_and_partial_release(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    pool = make_pool(torch.device(device))
    allocated = pool.reserve(logs=3)
    assert allocated is not None
    logs = allocated[1]
    payload = pool.payload_frame_table[logs].clone()
    pool.materialize_slots(logs)
    if device == "cuda":
        assert len(pool._retired_materializations) == 1
        assert all(pool.allocator.slot(slot).kind == "log" for slot in logs)
    pool.release_slots(logs[:1])
    assert not pool._retired_materializations
    assert all(pool.allocator.slot(slot).kind == "checkpoint" for slot in logs[1:])
    torch.testing.assert_close(pool.payload_frame_table[logs[1:]], payload[1:])
    assert torch.all(pool.conv_frame_table[logs[1:], :-1] == -1)
    assert torch.all(pool.decay_frame_table[logs] == -1)
    assert torch.all(pool.payload_frame_table[logs[0]] == -1)
    pool.allocator.check_integrity()
    pool.release_slots(logs[1:])
    allocated = pool.reserve(logs=3)
    assert allocated is not None
    assert torch.all(pool.conv_frame_table[allocated[1]] >= 0)
    pool.allocator.check_integrity()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_merged_slots_retire_without_reusing_inflight_frames(monkeypatch):
    pool = make_pool(torch.device("cuda"))
    actual_event = torch.cuda.Event
    events = []

    class Event:
        def __init__(self):
            self.event = actual_event()
            self.ready = False
            self.waits = 0
            events.append(self)

        def record(self, stream): self.event.record(stream)
        def query(self): return self.ready and self.event.query()
        def synchronize(self):
            self.waits += 1
            self.event.synchronize()
            self.ready = True

    monkeypatch.setattr(torch.cuda, "Event", Event)
    logs = pool.reserve(logs=3)[1]
    payload = pool.payload_frame_table[logs].clone()
    pool.materialize_slots(logs)
    pool.retire_slots(logs[:2])
    pool.poll_reclamation()
    assert not any(event.waits for event in events)
    assert pool.retiring_slots == frozenset(logs[:2])
    assert all(slot in pool.allocator.allocated_slots for slot in logs)
    torch.testing.assert_close(pool.payload_frame_table[logs], payload)
    # A capacity retry, not the merge commit, is allowed to wait. Consume all
    # remaining raw frames to force that retry deterministically.
    kv = pool.reserve(kv=len(pool.allocator._free))[0]
    replacement = pool.reserve(logs=1)
    assert replacement is not None
    assert any(event.waits for event in events)
    assert not pool.retiring_slots
    assert pool.allocator.slot(logs[2]).kind == "checkpoint"
    torch.testing.assert_close(pool.payload_frame_table[logs[2]], payload[2])
    pool.allocator.check_integrity()
    pool.release_kv(kv)
    pool.release_slots(replacement[1] + logs[2:])
    torch.cuda.synchronize()
    assert not pool.allocator.allocated_slots
    pool.allocator.check_integrity()


def test_head_overlay_and_side_views_survive_noncontiguous_allocation():
    pool = make_pool()
    allocated = pool.reserve(kv=20)
    assert allocated is not None
    pool.release_kv(allocated[0][::2])
    allocated = pool.reserve(logs=1)
    assert allocated is not None
    slot = allocated[1][0]
    write_head(pool, slot, 1, 0, 7, state=False, index=0)
    assert head_view(pool, slot, 1, 0, state=False)[0] == 7
    pool.conv_view(slot, 3).fill_(5)
    assert torch.all(pool.conv_view(slot, 3) == 5)
    decay_view(pool, slot).fill_(0.75)
    assert torch.all(decay_view(pool, slot) == 0.75)
    pool.materialize(slot)
    assert torch.all(pool.conv_view(slot, 3) == 5)
    pool.allocator.check_integrity()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_frame_descriptors_and_kv_views():
    pool = make_pool(torch.device("cuda"))
    allocated = pool.reserve(kv=1, logs=1)
    assert allocated is not None
    frame = allocated[0][0]
    slot = allocated[1][0]
    pool.k_cache(0)[frame * 4 + 2].fill_(3)
    write_head(pool, slot, 0, 0, 4, state=False)
    assert torch.all(pool.k_cache(0)[frame * 4 + 2] == 3)
    assert torch.all(head_view(pool, slot, 0, 0, state=False) == 4)
    torch.testing.assert_close(
        pool.payload_frame_table[slot],
        torch.tensor(pool.allocator.slot(slot).payload, dtype=torch.int32, device="cuda"),
    )
    pool.materialize(slot)
    torch.cuda.synchronize()
    pool.poll_reclamation()
    assert torch.all(pool.conv_frame_table[slot, :-1] == -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_graph_conv_indices_follow_updated_slots_and_granules():
    pool = make_pool(torch.device("cuda"))
    allocated = pool.reserve(logs=2, works=1)
    assert allocated is not None
    first, second = allocated[1]
    work = allocated[2][0]
    pool.conv_view(first, 0).fill_(1)
    pool.conv_view(second, 3).fill_(7)
    slots = torch.tensor([first], dtype=torch.int32, device="cuda")
    granules = torch.tensor([0], dtype=torch.int32, device="cuda")
    destination = torch.tensor([work], dtype=torch.int32, device="cuda")
    zeros = torch.zeros_like(destination)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pool.write_conv(
            destination, zeros, 0,
            pool.read_conv(slots, granules, 0),
        )
    slots.fill_(second)
    granules.fill_(3)
    graph.replay()
    torch.testing.assert_close(
        pool.read_conv(destination, zeros, 0),
        torch.full((1, pool.layout.conv_dim, pool.layout.state_len), 7,
                   dtype=pool.dtype, device="cuda"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_store_logstate_log_updates_payload_and_decay():
    from logssms.kernel.linear_attention.logstate import store_log_states

    pool = make_pool(torch.device("cuda"))
    allocated = pool.reserve(logs=1)
    assert allocated is not None
    slot = allocated[1][0]
    keys = torch.full((1, 2, 32), 3, dtype=torch.float16, device="cuda")
    values = torch.full((1, 2, 32), 5, dtype=torch.float16, device="cuda")
    decays = torch.full((1, 2), 0.75, dtype=torch.float32, device="cuda")
    store_log_states(
        pool,
        torch.tensor([slot], dtype=torch.int32, device="cuda"),
        torch.tensor([17], dtype=torch.int32, device="cuda"),
        keys, values, decays, 1,
    )
    head = head_view(pool, slot, 1, 0, state=False)
    key_size = pool.layout.head_log_key_bytes // pool.dtype.itemsize
    assert torch.all(head[17 * 32:18 * 32] == 3)
    assert torch.all(head[key_size + 17 * 32:key_size + 18 * 32] == 5)
    assert torch.all(decay_view(pool, slot)[1, 17] == 0.75)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_store_qwen35_2b_layout_across_payload_frames():
    from logssms.kernel.linear_attention.logstate import store_log_states

    pool = MHALogStatePool(
        num_frames=32,
        page_size=64,
        granularity=16,
        num_attention_layers=6,
        num_kv_heads=2,
        head_dim=256,
        num_linear_layers=18,
        num_key_heads=16,
        num_value_heads=16,
        head_k_dim=128,
        head_v_dim=128,
        conv_dim=6144,
        state_len=3,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    assert pool.allocator.payload_frames == 12
    assert pool.allocator.conv_frames == 1
    allocated = pool.reserve(logs=1)
    assert allocated is not None
    slot = allocated[1][0]
    keys = torch.full((1, 16, 128), 3, dtype=torch.bfloat16, device="cuda")
    values = torch.full((1, 16, 128), 5, dtype=torch.bfloat16, device="cuda")
    decays = torch.full((1, 16), 0.75, dtype=torch.float32, device="cuda")
    store_log_states(
        pool,
        torch.tensor([slot], dtype=torch.int32, device="cuda"),
        torch.tensor([63], dtype=torch.int32, device="cuda"),
        keys, values, decays, 17,
    )
    head = head_view(pool, slot, 17, 15, state=False)
    assert torch.all(head[63 * 128:64 * 128] == 3)
    assert torch.all(head[64 * 128 + 63 * 128:] == 5)
    assert torch.all(decay_view(pool, slot)[17, 63] == 0.75)
