import torch

from logssms.speculator.drafters.pool import DraftKVPool


def _make_pool() -> DraftKVPool:
    return DraftKVPool(
        max_running_req=3,
        window_size=4,
        max_batch_size=2,
        num_spec_tokens=3,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def test_persistent_slots_form_a_chronological_ring() -> None:
    pool = _make_pool()
    row = pool.persistent_table[1]

    assert torch.equal(pool.append_persistent_batch([1], [3]), row[:3])
    indices, valid = pool.batch_indices([1], torch.tensor([0]), scratch_depth=0)
    assert torch.equal(indices[valid], row[:3])

    assert torch.equal(pool.append_persistent_batch([1], [2]), row[torch.tensor([3, 0])])
    indices, valid = pool.batch_indices([1], torch.tensor([0]), scratch_depth=0)
    assert torch.equal(
        indices[valid],
        row[torch.tensor([1, 2, 3, 0])],
    )


def test_batch_indices_append_only_the_current_rows_scratch() -> None:
    pool = _make_pool()
    pool.append_persistent_batch([0, 2], [2, 4])
    batch_rows = torch.tensor([1, 0], dtype=torch.int64)

    indices, valid = pool.batch_indices([0, 2], batch_rows, scratch_depth=2)
    expected_0 = torch.cat([pool.persistent_table[0, :2], pool.scratch_table[1, :2]])
    expected_1 = torch.cat([pool.persistent_table[2], pool.scratch_table[0, :2]])

    assert torch.equal(indices[0, :4], expected_0)
    assert torch.equal(indices[1], expected_1)
    assert torch.equal(valid[0], torch.tensor([True, True, True, True, False, False]))
    assert valid[1].all()
    assert not torch.isin(pool.scratch_table, pool.persistent_table).any()


def test_store_preserves_native_kv_heads() -> None:
    pool = _make_pool()
    slots = pool.append_persistent_batch([0], [2])
    key = torch.arange(16, dtype=torch.float32).view(2, 2, 4)
    value = key + 100

    pool.store(slots, key, value)

    assert torch.equal(pool.k[slots.to(torch.int64)], key)
    assert torch.equal(pool.v[slots.to(torch.int64)], value)


def test_batched_append_matches_request_major_order() -> None:
    pool = _make_pool()
    slots = pool.append_persistent_batch([2, 0], [3, 2])

    assert torch.equal(
        slots,
        torch.cat([pool.persistent_table[2, :3], pool.persistent_table[0, :2]]),
    )
    indices, valid = pool.batch_indices([2, 0], torch.tensor([0, 1]), scratch_depth=0)
    assert torch.equal(indices[0, valid[0]], pool.persistent_table[2, :3])
    assert torch.equal(indices[1, valid[1]], pool.persistent_table[0, :2])
