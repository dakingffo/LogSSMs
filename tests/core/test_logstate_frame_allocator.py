from logssms.cache.logstate_pool.frame_allocator import LogStateFrameAllocator
from logssms.cache.logstate_pool.layout import LogStateSlotLayout
import random
import pytest
import torch


def make_allocator(num_frames: int = 80) -> LogStateFrameAllocator:
    return LogStateFrameAllocator(
        num_frames=num_frames,
        frame_bytes=100,
        payload_bytes=300,
        conv_snapshot_bytes=80,
        decay_bytes=20,
        num_granules=4,
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lock_frame_cost_matches_bitmap_projection_without_mutation(seed):
    rng = random.Random(seed)
    allocator = make_allocator(200)
    _, logs, states = allocator.reserve(logs=12, states=3)
    for slot in logs[:3]:
        allocator.materialize(slot)
    slots = logs + states
    for _ in range(100):
        for slot in slots:
            allocator.set_reclaimable(slot, bool(rng.randrange(2)))
        selected = rng.sample(slots, rng.randrange(len(slots) + 1))
        reclaimable = allocator._reclaimable_slots.copy()
        before = allocator.projected_free_capacity()[0]
        cost = allocator.lock_frame_cost(selected)
        assert allocator._reclaimable_slots == reclaimable
        assert allocator.projected_free_capacity()[0] == before
        for slot in selected:
            allocator.set_reclaimable(slot, False)
        assert before - allocator.projected_free_capacity()[0] == cost
        allocator.check_integrity()


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("records_per_frame", [1, 2, 5])
def test_incremental_capacity_matches_full_projection_through_lifecycle(seed, records_per_frame):
    rng = random.Random(seed)
    allocator = LogStateFrameAllocator(
        num_frames=80, frame_bytes=100, payload_bytes=300,
        conv_snapshot_bytes=80, decay_bytes=100 // records_per_frame,
        num_granules=4,
    )
    for _ in range(500):
        slots = list(allocator.allocated_slots)
        action = rng.randrange(6)
        if action == 0:
            allocator.reserve(logs=rng.randrange(1, 4))
        elif action == 1:
            allocator.reserve(states=1, kv=1)
        elif action == 2 and slots:
            slot = rng.choice(slots)
            allocator.set_reclaimable(slot, slot not in allocator._reclaimable_slots)
        elif action == 3 and slots:
            slot = rng.choice(slots)
            if allocator.slot(slot).kind == "log":
                allocator.materialize(slot)
        elif action == 4 and slots:
            allocator.release_slot(rng.choice(slots))
        elif action == 5 and allocator.allocated_kv:
            allocator.release_kv([rng.choice(tuple(allocator.allocated_kv))])
        projected = allocator.projected_free_capacity(allocator._reclaimable_slots)
        assert allocator.projected_free_capacity() == projected
        assert allocator.admissible_pages(evictable_kv_frames=0) == allocator.admissible_pages(
            evictable_kv_frames=0, evictable_slots=allocator._reclaimable_slots,
        )
        allocator.check_integrity()


def test_one_frame_pool_with_independent_log_frames():
    allocator = make_allocator()
    allocated = allocator.reserve(kv=2, logs=2, states=1)
    assert allocated is not None
    kv, logs, states = allocated
    assert len(kv) == 2
    assert len(allocator.slot(logs[0]).payload) == 3
    assert len(allocator.slot(logs[0]).conv) == 4
    assert allocator.slot(states[0]).decay is None
    assert allocator.slot(logs[0]).decay.frame == allocator.slot(logs[1]).decay.frame
    usage = allocator.profile_usage()
    assert usage["log_slots"] == 2
    assert usage["work_slots"] == 1
    assert usage["kv_frames"] == 2
    assert sum(usage[key] for key in (
        "free_frames", "kv_frames", "payload_frames", "conv_frames", "decay_frames"
    )) == usage["total_frames"]
    allocator.check_integrity()


def test_materialize_releases_three_snapshots_and_decay():
    allocator = make_allocator()
    allocated = allocator.reserve(logs=1)
    assert allocated is not None
    slot = allocated[1][0]
    record = allocator.slot(slot)
    payload = record.payload
    terminal_conv = record.conv[-1]
    before = allocator.free_frames
    released = allocator.materialize(slot)
    assert len(released) == 3
    assert allocator.free_frames == before + 4  # three conv frames and one decay frame
    assert allocator.slot(slot).payload == payload
    assert allocator.slot(slot).conv == (terminal_conv,)
    allocator.check_integrity()
    allocator.release_slot(slot)
    assert allocator.free_frames == allocator.num_frames


def test_shared_decay_frame_is_freed_after_last_record():
    allocator = make_allocator()
    allocated = allocator.reserve(logs=2)
    assert allocated is not None
    first, second = allocated[1]
    before = allocator.free_frames
    allocator.materialize(first)
    assert allocator.free_frames == before + 3
    allocator.materialize(second)
    assert allocator.free_frames == before + 7
    allocator.check_integrity()


def test_projected_reclamation_matches_checkpoint_and_shared_decay_release():
    allocator = make_allocator()
    allocated = allocator.reserve(logs=3)
    assert allocated is not None
    first, second, third = allocated[1]
    allocator.materialize(first)

    projected = allocator.projected_free_capacity((first, second))
    allocator.release_slot(first)
    allocator.release_slot(second)
    assert projected == (allocator.free_frames, allocator._free_decay_records)
    assert allocator.slot(third).decay is not None

    projected = allocator.projected_free_capacity((third,))
    allocator.release_slot(third)
    assert projected == (allocator.free_frames, allocator._free_decay_records)
    assert allocator.free_frames == allocator.num_frames
    allocator.check_integrity()


def test_admission_uses_free_decay_bitmap_records_before_new_frames():
    allocator = LogStateFrameAllocator(
        num_frames=8,
        frame_bytes=100,
        payload_bytes=100,
        conv_snapshot_bytes=80,
        decay_bytes=20,
        num_granules=1,
    )
    allocated = allocator.reserve(logs=1)
    assert allocated is not None
    assert allocator.free_frames == 5
    assert allocator.admissible_pages(
        evictable_kv_frames=0, evictable_slots=()
    ) == 1
    assert allocator.reserve(kv=1, logs=1, states=1) is not None
    allocator.check_integrity()


def test_failed_reservation_is_atomic_and_reused_frames_are_not_contiguous():
    allocator = make_allocator(num_frames=25)
    allocated = allocator.reserve(kv=10)
    assert allocated is not None
    kv = allocated[0]
    allocator.release_kv(kv[::2])
    before = allocator.free_frames
    assert allocator.reserve(logs=4) is None
    assert allocator.free_frames == before
    allocated = allocator.reserve(logs=1)
    assert allocated is not None
    assert len(allocator.slot(allocated[1][0]).payload) == 3
    allocator.check_integrity()


def test_slot_ids_reused_without_requiring_same_physical_frames():
    allocator = make_allocator()
    allocated = allocator.reserve(logs=1)
    assert allocated is not None
    slot = allocated[1][0]
    payload = allocator.slot(slot).payload
    allocator.release_slot(slot)
    allocated = allocator.reserve(kv=1, states=1)
    assert allocated is not None
    assert allocated[2] == [slot]
    assert allocator.slot(slot).payload != payload
    allocator.check_integrity()


def test_qwen35_2b_frame_layout_and_released_side_data():
    layout = LogStateSlotLayout(
        page_size=64,
        num_linear_layers=18,
        num_key_heads=16,
        num_value_heads=16,
        head_k_dim=128,
        head_v_dim=128,
        conv_dim=6144,
        state_len=3,
        dtype=torch.bfloat16,
        granularity=16,
    )
    frame_bytes = 2 * 6 * 64 * 2 * 256 * 2
    payload_bytes = layout.num_linear_layers * layout.num_value_heads * max(
        layout.head_log_key_bytes + layout.head_log_value_bytes,
        layout.head_recurrent_state_bytes,
    )
    conv_snapshot_bytes = layout.conv_state_bytes // layout.num_granules
    allocator = LogStateFrameAllocator(
        num_frames=256,
        frame_bytes=frame_bytes,
        payload_bytes=payload_bytes,
        conv_snapshot_bytes=conv_snapshot_bytes,
        decay_bytes=layout.log_decay_bytes,
        num_granules=layout.num_granules,
    )
    assert frame_bytes == 786432
    assert allocator.payload_frames == 12
    assert allocator.conv_frames == 1
    assert allocator.decay_records_per_frame == 10
    allocated = allocator.reserve(kv=1, logs=1)
    assert allocated is not None
    assert allocator.free_frames == 256 - 1 - 12 - 4 - 1
    slot = allocated[1][0]
    allocator.materialize(slot)
    assert allocator.free_frames == 256 - 1 - 12 - 1
    allocator.check_integrity()


def test_fragmentation_cannot_prevent_a_frame_reservation():
    allocator = make_allocator(num_frames=60)
    allocated = allocator.reserve(kv=60)
    assert allocated is not None
    kv = allocated[0]
    allocator.release_kv(kv[::2])
    allocated = allocator.reserve(logs=4)
    assert allocated is not None
    assert allocator.free_frames == 60 - 30 - (4 * 7 + 1)
    allocator.check_integrity()
