"""CPU coverage of release IDs, deferred ownership, and pool event ordering."""

from contextlib import contextmanager, ExitStack, nullcontext
from unittest.mock import patch

import pytest
import torch
from logssms.core import SamplingParams
from logssms.message import UserMsg
from logssms.scheduler.prefill import PrefillAdder, PrefillManager
from logssms.scheduler.table import TableManager
from tests.core.test_logstate_cache_manager import cache_checkpoint_chain, make_manager


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

    def record(self, _stream):
        pass


@contextmanager
def _enqueue_cuda_events(pool, *events):
    with ExitStack() as stack:
        stack.enter_context(patch.object(
            type(pool), "device", property(lambda _: torch.device("cuda")),
        ))
        stack.enter_context(patch("torch.cuda.Event", side_effect=events))
        stack.enter_context(patch("torch.cuda.current_stream", return_value=None))
        stack.enter_context(patch.object(pool, "_write_descriptors"))
        yield


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


def test_tp_reclamation_allocates_identical_frames_with_different_event_progress():
    def setup(ready):
        pool = make_manager().logstate_pool
        pool.deterministic_reclamation = True
        _, _, works = pool.reserve(works=3)
        pool.reserve(kv=pool.allocator.free_frames)
        events = [_PendingEvent() for _ in works]
        for slot, event, complete in zip(works, events, ready, strict=True):
            event.ready = complete
            pool.allocator.set_reclaimable(slot, True)
            pool._retiring_slots.add(slot)
            pool._retired_releases.append((event, (slot,)))
        return pool, events

    fast, fast_events = setup((True, True, True))
    slow, slow_events = setup((False, False, False))
    assert fast.reserve(kv=1) == slow.reserve(kv=1)
    assert fast.retiring_slots == slow.retiring_slots
    assert [e.waits for e in fast_events] == [e.waits for e in slow_events] == [1, 0, 0]
    assert fast.reserve(logs=1) == slow.reserve(logs=1)
    fast.allocator.check_integrity()
    slow.allocator.check_integrity()


@pytest.mark.parametrize("deterministic, expected", [
    (False, ((True, 0), (False, 0))),
    (True, ((True, 1), (True, 1))),
])
def test_tp_materialization_credits_admission_independent_of_event_progress(
    deterministic, expected,
):
    def admit(ready):
        manager = make_manager(granularity=32)
        pool = manager.logstate_pool
        pool.deterministic_reclamation = deterministic
        allocator = pool.allocator
        # The dummy log already occupies the first decay frame. The last log
        # is alone in the second frame so its conversion frees one frame.
        _, logs, _ = pool.reserve(logs=allocator.decay_records_per_frame)
        event = _PendingEvent()
        event.ready = ready
        handle = manager.tree.match_prefix(torch.arange(24, dtype=torch.int32)).cuda_handle
        demand = manager._reservation_size(24 + 512, handle=handle)
        assert 0 < demand <= allocator.free_frames
        pool.reserve(kv=allocator.free_frames - demand + 1)
        assert manager.admission_capacity == demand - 1
        with (
            patch.object(type(pool), "device", property(lambda _: torch.device("cuda"))),
            patch("torch.cuda.Event", return_value=event),
            patch("torch.cuda.current_stream", return_value=None),
            patch.object(pool, "_write_descriptors"),
        ):
            pool.materialize(logs[-1])
        result = manager._try_pin_prefill(demand, handle)
        allocator.check_integrity()
        return result, event.waits

    assert (admit(True), admit(False)) == expected


@pytest.mark.parametrize("free_frames,expected_uids,expected_waits", [
    (20, [687, 688], (0, 0)),
    (1, [687], (1, 0)),
    (0, [687], (1, 1)),
])
def test_tp_mixed_prefill_agrees_with_rebinding_and_release_pending(
    free_frames, expected_uids, expected_waits,
):
    def schedule(binding_ready, release_ready):
        manager = make_manager(granularity=32)
        pool = manager.logstate_pool
        pool.deterministic_reclamation = True
        _, logs, works = pool.reserve(logs=2, works=3)
        binding, release = _PendingEvent(), _PendingEvent()
        binding.ready, release.ready = binding_ready, release_ready
        pool.reserve(kv=pool.allocator.free_frames - free_frames)
        with (
            _enqueue_cuda_events(pool, binding, release),
            patch("logssms.kernel.linear_attention.logstate.copy_states"),
        ):
            pool.materialize_groups([tuple(logs)], works[:1])
            pool.retire_slots(works[1:])

        table = TableManager(3, manager.page_table)
        prefill = PrefillManager(table.token_pool)
        for uid in (687, 688):
            prefill.add_one_req(UserMsg(
                uid, torch.arange(24, dtype=torch.int32) + uid * 32,
                SamplingParams(max_tokens=1),
            ))
        adder = PrefillAdder(
            token_budget=8192, cache_manager=manager, table_manager=table,
            admission=manager.begin_prefill_admission((), ()),
            mixed_token_budget=128, mixed_partitions=((32, 32), (128,)),
        )
        with patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor):
            batch = prefill.schedule_next_batch(adder, mixed_max_requests=2)
        assert batch is not None and adder.use_mixed
        assert [req.uid for req in batch.reqs] == expected_uids
        if free_frames == 0:
            manager.allocate(batch.reqs, phase="prefill")
        assert (binding.waits, release.waits) == expected_waits
        assert binding.queries == release.queries == 0
        pool.allocator.check_integrity()
        return (
            [(req.uid, req.cached_len, req.extend_len, req.table_idx) for req in batch.reqs],
            [req.uid for req in prefill.pending_list],
            manager.admission_capacity,
            pool.allocator._slots,
            pool.allocator.decay_counts,
            pool.allocator._free,
        )

    assert schedule(True, False) == schedule(False, True) == schedule(False, False)


@pytest.mark.parametrize("online", [False, True])
def test_tp_eviction_agrees_after_draining_pending_releases(online):
    def evict(ready):
        manager = make_manager(enable_merge_adjustment=online)
        tokens, leaf, checkpoints = cache_checkpoint_chain(manager, (4, 2))
        if online:
            manager.analyzer.record_hit(leaf.analysis_owner, len(tokens) - 64)
        pool = manager.logstate_pool
        pool.deterministic_reclamation = True
        _, _, works = pool.reserve(works=2)
        retirement, eviction = _PendingEvent(), _PendingEvent()
        retirement.ready, eviction.ready = ready, not ready
        with _enqueue_cuda_events(pool, retirement):
            pool.retire_slots(works)

        demand = pool.allocator.free_frames + pool.allocator.required_frames(0, 0, 2) + 1
        # More than free plus already queued capacity, but one dropped
        # Checkpoint is enough. Let the actual eviction path retire it.
        retire = pool.retire_slots

        def retire_eviction(slots):
            with _enqueue_cuda_events(pool, eviction):
                retire(slots)

        with (
            patch.object(pool, "retire_slots", side_effect=retire_eviction),
            patch.object(manager.evictor, "plan", wraps=manager.evictor.plan) as plan,
        ):
            allocation = manager._allocate(demand, 0)
        plan.assert_called_once()
        assert retirement.waits == eviction.waits == 1
        remaining = tuple(manager.tree.iter_entries())
        assert len(remaining) == 1
        assert remaining[0].slot == checkpoints[-1].slot
        assert manager.tree.match_prefix(tokens).cuda_handle.cached_len == len(tokens)
        manager._return_slots(allocation[0], allocation[1])
        manager.check_integrity()
        return allocation, remaining

    assert evict(True) == evict(False)
