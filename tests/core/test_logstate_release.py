"""CPU coverage of release IDs, deferred ownership, and pool event ordering."""

from contextlib import nullcontext
from unittest.mock import patch

import pytest
from tests.core.test_logstate_cache_manager import make_manager


@pytest.mark.parametrize("lazy", [False, True])
def test_return_slots_releases_owned_frames(lazy):
    manager = make_manager()
    pool = manager.logstate_pool
    frames, logs, works = pool.reserve(kv=2, logs=2, works=1)
    pages, states = [frame * manager.page_size for frame in frames], logs + works
    with manager.lazy_free_region() if lazy else nullcontext():
        manager._return_slots(pages, states)
        if lazy:
            assert set(frames) <= pool.allocator.allocated_kv
            assert set(logs + works) <= pool.allocator.allocated_slots
    assert not set(frames) & pool.allocator.allocated_kv
    assert not set(logs + works) & pool.allocator.allocated_slots
    manager.check_integrity()


@pytest.mark.parametrize("exceptional", [False, True])
def test_lazy_release_snapshots_lists_and_flushes_once(exceptional):
    manager = make_manager()
    pool = manager.logstate_pool
    frames, logs, _ = pool.reserve(kv=2, logs=2)
    pages = [frame * manager.page_size for frame in frames]
    states = logs.copy()
    with patch.object(pool, "release_kv", wraps=pool.release_kv) as release_kv, patch.object(
        pool, "release_slots", wraps=pool.release_slots,
    ) as release_slots:
        with pytest.raises(ValueError, match="body") if exceptional else nullcontext():
            with manager.lazy_free_region():
                manager._return_slots(pages, [])
                manager._return_slots([], states)
                pages.clear()
                states.clear()
                release_kv.assert_not_called()
                release_slots.assert_not_called()
                if exceptional:
                    raise ValueError("body")
        release_kv.assert_called_once_with(frames)
        release_slots.assert_called_once_with(logs)
    assert manager._lazy_free_pages is None
    assert manager._lazy_free_states is None
    manager.check_integrity()


def test_cpu_ids_need_no_tensor_construction_or_concatenation():
    manager = make_manager()
    with patch("torch.tensor", side_effect=AssertionError("tensor construction")), patch(
        "torch.cat", side_effect=AssertionError("tensor concatenation"),
    ), patch.object(manager.logstate_pool, "release_kv") as release_kv, patch.object(
        manager.logstate_pool, "release_slots",
    ) as release_slots:
        with manager.lazy_free_region():
            manager._return_slots([64], [7])
            manager._return_slots([128], [8])
        release_kv.assert_called_once_with([1, 2])
        release_slots.assert_called_once_with([7, 8])
        release_kv.reset_mock()
        release_slots.reset_mock()
        with manager.lazy_free_region():
            manager._return_slots([], [])
        release_kv.assert_not_called()
        release_slots.assert_not_called()


@pytest.mark.parametrize("lazy", [False, True])
def test_release_waits_for_pending_materialization(lazy):
    manager = make_manager()
    pool = manager.logstate_pool
    _, logs, _ = pool.reserve(logs=1)

    class PendingEvent:
        ready = False
        waits = 0

        def synchronize(self):
            assert logs[0] in pool.allocator.allocated_slots
            self.waits += 1
            self.ready = True

        def query(self):
            return self.ready

    event = PendingEvent()
    pool._retired_materializations.append((event, tuple(logs)))
    with manager.lazy_free_region() if lazy else nullcontext():
        manager._return_slots([], logs)
        if lazy:
            assert event.waits == 0
            assert logs[0] in pool.allocator.allocated_slots
    assert event.waits == 1
    assert not pool._retired_materializations
    assert logs[0] not in pool.allocator.allocated_slots
    manager.check_integrity()


class _PendingEvent:
    def __init__(self):
        self.ready = False
        self.waits = 0
        self.queries = 0

    def synchronize(self):
        self.waits += 1
        self.ready = True

    def query(self):
        self.queries += 1
        return self.ready


def test_reserve_waits_only_for_enough_retired_capacity():
    pool = make_manager().logstate_pool
    _, _, works = pool.reserve(works=2)
    pool.reserve(kv=pool.allocator.free_frames)
    first, second = _PendingEvent(), _PendingEvent()
    pool._retiring_slots.update(works)
    pool._retired_releases.extend(((first, (works[0],)), (second, (works[1],))))
    assert pool.reserve(kv=1) is not None
    assert first.waits == 1 and second.waits == 0
    assert works[1] in pool.allocator.allocated_slots
    assert pool._retired_releases == [(second, (works[1],))]
    second.synchronize()
    pool.poll_reclamation()
    pool.allocator.check_integrity()


def test_rebinding_batch_queries_one_event_and_releases_all_records():
    pool = make_manager().logstate_pool
    event = _PendingEvent()
    records = tuple(object() for _ in range(128))
    pool._retired_rebindings.append((event, records))
    with patch.object(pool.allocator, "release_retirement") as release:
        pool.poll_reclamation()
        assert event.queries == 1
        release.assert_not_called()
        event.synchronize()
        pool.poll_reclamation()
        assert event.queries == 2
        assert release.call_count == 128
        assert not pool._retired_rebindings
