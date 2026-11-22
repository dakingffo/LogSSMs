import pytest

from logssms.kernel.logstate import EvictionPlanner, plan_eviction


def test_active_reclamation_precedes_checkpoint_leaf_eviction():
    capacity = (0, 0, 3, 0, 1, 1, 1, 0, 0, 3)
    models = [(1, 0, 32, 2., -16., 0.01, 32, 0, end, 1) for end in (0, 32)]
    planner = EvictionPlanner(
        [(0, -1, 1, 0), (1, 0, 0, 0)],
        [[], [9, 32, 32]], [[], [1]], [(1, 3, -1, 0, 32)], [], capacity,
        [(1, -1, 0, 1)], models, [(32, 1, 0)] * 2,
    )
    result = planner.plan(capacity, [], [(1., ())])
    assert result == [[], [], [], [], [0]]
    assert planner.plan(capacity, []) == [[1], [9, 32], [1], []]


def test_active_reclamation_uses_completed_logs_and_publishes_merged_checkpoint():
    import torch
    from experiment.ablation_online_analysis.cpu_common import NumericalKernels, numerical_boundaries
    from tests.core.test_logstate_cache_manager import make_manager, new_req

    manager = make_manager(num_pages=24, num_states=24, granularity=32)
    manager.logstate_pool.kernels = NumericalKernels()
    count = manager._allocator.materialize_log_count
    end = 2 * count * manager.granularity
    tokens = torch.arange(end + 1, dtype=torch.int32)
    with numerical_boundaries():
        req = new_req(manager, tokens)
        manager.allocate([req])
        req.cached_len = end
        req.input_ids = tokens[:-1]
        kv = manager.page_table[0, :end].clone()
        [action] = manager._active_candidates([req])
        assert action.end == end // 2
        assert manager._apply_active(action)["kind"] == "materialize"
        assert manager._active_candidates([req]) == []
        req.input_ids = tokens
        [action] = manager._active_candidates([req])
        assert action.end == end
        assert manager._apply_active(action)["kind"] == "materialize"
        [action] = manager._active_candidates([req])
        assert manager._apply_active(action)["kind"] == "merge"
        torch.testing.assert_close(manager.page_table[0, :end], kv)
        manager.cache_req(req, finished=True)
        [entry] = manager.tree.iter_entries()
        assert (entry.kind.value, entry.begin, entry.end) == ("checkpoint", 0, end)
        assert manager.tree.match_prefix(tokens).recoverable_len == end
        manager.check_integrity()


def test_decode_reclaims_completed_logs_when_allocation_is_short():
    import torch
    from experiment.ablation_online_analysis.cpu_common import NumericalKernels, numerical_boundaries
    from tests.core.test_logstate_cache_manager import make_manager, new_req

    manager = make_manager(num_pages=24, num_states=24, granularity=32)
    pool = manager.logstate_pool
    pool.kernels = NumericalKernels()
    end = pool.allocator.materialize_log_count * manager.granularity
    tokens = torch.arange(end + 1, dtype=torch.int32)
    with numerical_boundaries():
        req = new_req(manager, tokens[:-1])
        manager.allocate([req])
        req.input_ids = tokens
        req.cached_len, req.device_len = end, end + 1
        required = pool.allocator.required_frames(1, 1, 0)
        filled, _, _ = pool.reserve(kv=pool.allocator.free_frames - required + 1)
        manager.allocate([req], phase="decode")
        pool.release_kv(filled)
        manager.cache_req(req, finished=True)
        [entry] = manager.tree.iter_entries()
        assert (entry.kind.value, entry.begin, entry.end) == ("checkpoint", 0, end)
        manager.check_integrity()


@pytest.mark.parametrize("completed_group", [False, True])
def test_prefill_admission_uses_rolling_active_reservation(completed_group):
    import torch
    from experiment.ablation_online_analysis.cpu_common import NumericalKernels, numerical_boundaries
    from logssms.core import SamplingParams
    from logssms.scheduler.prefill import PendingRequest
    from tests.core.test_logstate_cache_manager import make_manager, new_req

    cache = make_manager(num_pages=32, num_states=64, granularity=32, max_seq_len=2048)
    pool = cache.logstate_pool
    pool.kernels = NumericalKernels()
    group = pool.allocator.materialize_log_count
    end = group * cache.granularity
    tokens = torch.arange(end + 1, dtype=torch.int32)
    with numerical_boundaries():
        active = new_req(cache, tokens)
        active.max_device_len = end + 1024
        cache.allocate([active])
        active.cached_len = end
        if not completed_group:
            active.input_ids = active.input_ids[:-1]
        state = cache._request_states[active.table_idx]
        tail_slot = state.live_granules[-1].state_slot
        kv = cache.page_table[active.table_idx, :end + 1].clone()

        pending = PendingRequest(1, torch.arange(24, dtype=torch.int32) + 1000,
                                 SamplingParams(max_tokens=1))
        handle = cache.match_req(pending).cuda_handle
        request_cost = cache._reservation_size(pending.input_len + pending.output_len, handle=handle)
        admission = cache.begin_prefill_admission(((active, active.remain_len),), ())
        full = admission.reserved
        filled, _, _ = pool.reserve(kv=pool.allocator.free_frames - full - request_cost + 1)
        result = cache.admit_prefill(pending, admission, lambda _: 32, lambda: 1)

        assert isinstance(result, str) is not completed_group
        assert state.compacted_end == (end if completed_group else 0)
        assert state.live_granules[-1].state_slot == tail_slot
        assert int(cache.state_table[active.table_idx, end]) == tail_slot
        torch.testing.assert_close(cache.page_table[active.table_idx, :end + 1], kv)
        if completed_group:
            future_kv = (active.max_device_len - end) // cache.granularity - 1
            assert cache._active_reservation(active, active.remain_len) == pool.allocator.required_frames(
                future_kv, group, 0, free_decay_records=0,
            )
            assert admission.reserved < full
        pool.release_kv(filled)
        pool.allocator.check_integrity()


def run(need, *, locked=False):
    # One chain with two Checkpoints and a tail Log. Each state has 3 payload
    # frames; the Log leaf also owns one KV frame.
    nodes = [(0, -1, 1, 0), (1, 0, 0, 0), (2, 1, int(locked), 2)]
    kv = [[], [10, 32, 32, 11, 32, 32], [12, 32, 32]]
    refs = [[], [1, 2], [3]]
    slots = [(1, 3, -1, 0, 32), (2, 3, -1, 32, 64), (3, 3, 20, 64, 96)]
    actions = [(1, 2, 0, 1), (2, -1, 1, 1)]
    models = [
        (1, 0, 96, 2.0, -12.0, 0.9, 32, b, e, 0)
        for b, e in ((0, 0), (0, 64), (64, 96))
    ]
    return plan_eviction(
        nodes, kv, refs, slots, [(20, 1)], (0, 0, 3, 0, 1, 1, need, 0, 0, 3), actions, models
    )


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("adjustment", ["disabled", "no_internal_hits", "fitted"])
def test_pressure_materialization_preserves_kv_and_respects_pins(pinned, adjustment):
    import torch
    from experiment.ablation_online_analysis.cpu_common import NumericalKernels, numerical_boundaries
    from logssms.cache.logstate.prefix_cache import LogStateEntry, LogStateKind
    from tests.core.test_logstate_cache_manager import make_manager
    from logssms.cache.logstate.evictor import LogStateCapacityError

    manager = make_manager(
        granularity=32,
        enable_merge_adjustment=adjustment != "disabled",
    )
    pool, tree = manager.logstate_pool, manager.tree
    pool.kernels = NumericalKernels()
    group = pool.allocator.materialize_log_count
    kv, slots, _ = pool.reserve(kv=group + 1, logs=group + 1)
    pool.materialize(slots[0])
    tokens = torch.arange(32 * len(slots), dtype=torch.int32)
    indices = torch.cat([torch.arange(frame * 32, (frame + 1) * 32, dtype=torch.int32)
                         for frame in kv])
    entries = tuple(LogStateEntry(
        slot, LogStateKind.CHECKPOINT if not i else LogStateKind.LOG,
        i * 32, (i + 1) * 32, span_granules=None if i else 1,
    ) for i, slot in enumerate(slots))
    node = tree.insert(tokens, indices, entries).node
    log_timestamp = 1
    node.timestamp = log_timestamp
    base_checkpoint_timestamp = tree.checkpoint_timestamp(slots[0])
    if adjustment != "disabled":
        analyzer = manager.evictor.analyzer
        stats = analyzer.make_hit_stats(0)
        node.analysis_owner = stats
        analyzer.advance_hit_stats(stats, len(tokens))
        analyzer.record_hit(stats, len(tokens))
        if adjustment == "fitted":
            analyzer.record_hit(stats, 32)
            for _ in range(32):
                analyzer.record_hit(stats, len(tokens))
        assert analyzer.has_internal_observations == (adjustment == "fitted")
    manager.evictor.cached_page_tokens.update({frame * 32: 32 for frame in kv})
    if pinned:
        tree.pin_recovery((slots[1],))
        with numerical_boundaries(), pytest.raises(LogStateCapacityError):
            manager._allocate(pool.allocator.free_frames + 1, 0)
        assert all(tree.has_slot(slot) for slot in slots)
        tree.pin_recovery((slots[1],), release=True)
        manager.check_integrity()
        return
    with numerical_boundaries():
        pages, _, _ = manager._allocate(pool.allocator.free_frames + 1, 0)
    assert tree.entry_for_slot(slots[-1]) == LogStateEntry(
        slots[-1], LogStateKind.CHECKPOINT, 32, len(tokens), span_granules=group,
    )
    assert tree.checkpoint_timestamp(slots[-1]) == log_timestamp
    assert tree.checkpoint_timestamp(slots[0]) == base_checkpoint_timestamp
    assert node.timestamp == log_timestamp
    assert tree.match_prefix(tokens).recoverable_len == len(tokens)
    assert tree.checkpoint_timestamp(slots[-1]) > log_timestamp
    assert tree.match_prefix(tokens[:64]).recoverable_len == 32
    assert not len(manager.recycle_bin)
    assert set(kv) <= pool.allocator.allocated_kv
    manager._return_slots(pages, [])
    manager.check_integrity()


def test_online_never_prunes_an_interior_log_or_a_locked_leaf():
    assert run(100) == []
    selected, _, _, drops = run(3, locked=True)
    assert selected == [] and drops == [1, 2]


def test_newly_enlarged_checkpoint_has_correct_successor_for_later_eviction():
    selected, _, released, drops = run(10)
    assert drops == [1, 2]
    assert set(released) == {1, 2, 3}
    assert selected == [2, 1]


def test_binary_checkpoint_thinning_removes_adjacent_equal_scales_first():
    nodes = [(0, -1, 1, 0), (1, 0, 0, 0)]
    slots = [(i, 3, -1, (i - 1) * 32, i * 32) for i in range(1, 5)]
    selected, kv, released, drops = plan_eviction(
        nodes,
        [[], []],
        [[], [1, 2, 3, 4]],
        slots,
        [],
        (0, 0, 3, 0, 1, 1, 6, 0, 0, 3),
        [(i, i + 1 if i < 4 else -1, i, 1) for i in range(1, 5)],
    )
    assert selected == [] and released == [1, 3] and drops == [1, 2, 3, 4]


def test_density_orders_leaves_subject_to_tree_dependencies():
    nodes = [(0, -1, 1, 0), (1, 0, 0, 0), (2, 1, 0, 0), (3, 0, 0, 10)]
    slots = [(i, 2, 20 + i, (i - 1) * 32, i * 32) for i in range(1, 4)]
    # A cold-looking interior Log is unavailable until its hot suffix is gone.
    models = [
        (i, 0, 96, 2.0, -4.0 if i < 3 else -12.0, 0.8 if i < 3 else 0.01, 32, b, e, 0)
        for i, b, e in ((0, 0, 0), (1, 0, 32), (1, 32, 64), (2, 64, 96))
    ]
    result = plan_eviction(
        nodes,
        [[]] * 4,
        [[], [1], [2], [3]],
        slots,
        [(21, 1), (22, 1), (23, 1)],
        (0, 0, 2, 0, 1, 1, 2, 0, 0, 2),
        [],
        models,
    )
    assert result[0] == [3]


@pytest.mark.parametrize("online", [False, True])
def test_checkpoint_leaf_is_pruned_whole_only_after_internal_thinning(online):
    models = [(1, 0, 96, 2., -12., .9, 32, 0, end, 0)
              for end in (0, 96)] if online else None
    result = plan_eviction(
        [(0, -1, 1, 0), (1, 0, 0, 0)],
        [[], [10, 96, 96]], [[], [1, 2, 3]],
        [(i, 3, -1, (i - 1) * 32, i * 32) for i in range(1, 4)],
        [], (0, 0, 3, 0, 1, 1, 10, 0, 0, 3),
        [(1, 2, 0, 1), (2, 3, 0, 1), (3, -1, 0, 1)], models,
    )
    selected, kv, released, merges = result
    assert len(merges) == 4
    assert selected == [1] and kv == [10, 96] and released == [1, 2, 3]


def test_terminal_checkpoint_cost_uses_actual_recovery_frontier():
    selected, _, _, merges = plan_eviction(
        [(0, -1, 1, 0), (1, 0, 0, 0), (2, 0, 0, 0)],
        [[], [], []], [[], [1], [2]],
        [(1, 3, -1, 480, 512), (2, 3, -1, 0, 64)], [],
        (0, 0, 3, 0, 1, 1, 3, 0, 0, 3),
        [(1, -1, 0, 1), (2, -1, 0, 2)],
        [(0, 0, 512, 2., -100., 1., 32, 0, 0, 0),
         (1, 0, 512, 2., -100., 1., 32, 416, 512, 0),
         (2, 0, 64, 2., -100., 1., 32, 0, 64, 0)],
    )
    assert selected == [2] and merges == []


def test_online_thins_a_node_before_pruning_a_cheaper_checkpoint_leaf():
    selected, kv, released, merges = plan_eviction(
        [(0, -1, 1, 0), (1, 0, 0, 0), (2, 0, 0, 0)],
        [[], [], [10, 32, 32]], [[], [1, 2], [3]],
        [(1, 3, -1, 0, 32), (2, 3, -1, 32, 64), (3, 3, -1, 0, 32)], [],
        (0, 0, 3, 0, 1, 1, 3, 0, 0, 3),
        [(1, 2, 0, 1), (2, -1, 0, 1), (3, -1, 0, 2)],
        [(0, 0, 64, 2., -12., .9, 32, 0, 0, 0),
         (1, 0, 64, 2., -2., .9, 32, 0, 64, 0),
         (2, 0, 32, 2., -30., 0., 32, 0, 32, 0)],
    )
    assert selected == [] and kv == [] and released == [1] and merges == [1, 2]


def test_density_amplitude_affects_leaf_selection_independently_of_heat_ties():
    selected, _, _, _ = plan_eviction(
        [(0, -1, 1, 0), (1, 0, 0, 0), (2, 0, 0, 100)],
        [[], [], []], [[], [1], [2]],
        [(1, 2, 21, 64, 96), (2, 2, 22, 64, 96)], [(21, 1), (22, 1)],
        (0, 0, 2, 0, 1, 1, 2, 0, 0, 2), [],
        [(0, 0, 128, 2., -12., 0., 32, 0, 0, 0),
         (1, 0, 128, 2., -6., 0., 32, 64, 96, 0),
         (2, 0, 128, 2., -12., 0., 32, 64, 96, 0)],
    )
    assert selected == [2]


def test_checkpoint_merge_cost_includes_missing_parent_state():
    selected, _, released, merges = plan_eviction(
        [(0, -1, 1, 0), (1, 0, 0, 0), (2, 0, 0, 0)],
        [[], [], []], [[], [1, 2], [3, 4]],
        [(1, 3, -1, 96, 128), (2, 3, -1, 128, 160),
         (3, 3, -1, 80, 128), (4, 3, -1, 128, 160)], [],
        (0, 0, 3, 0, 1, 1, 3, 0, 0, 3),
        [(1, 2, 0, 1), (2, -1, 0, 1), (3, 4, 0, 2), (4, -1, 0, 2)],
        [(0, 0, 192, 0., -10., 0., 16, 0, 0, 0),
         (1, 0, 192, 0., -10., 0., 16, 64, 160, 0),
         (2, 0, 192, 0., -10., 0., 16, 80, 160, 0)],
    )
    assert selected == [] and released == [3] and merges == [3, 4]


def test_native_planner_rejects_cross_node_checkpoint_merges():
    with pytest.raises(RuntimeError, match="inside one node"):
        plan_eviction(
            [(0, -1, 1, 0), (1, 0, 0, 0), (2, 1, 0, 0)],
            [[], [], []], [[], [1], [2]],
            [(1, 3, -1, 0, 32), (2, 3, -1, 32, 64)], [],
            (0, 0, 3, 0, 1, 1, 1, 0, 0, 3), [(1, 2, 0, 1)],
        )


@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("seed", range(8))
def test_actual_allocator_accepts_the_projected_reservation(online, seed):
    from random import Random
    from tests.core.test_logstate_eviction_planning import make_pressure_tree
    from experiment.ablation_online_analysis.cpu_common import NumericalKernels, numerical_boundaries

    rng = Random(seed)
    manager = make_pressure_tree(chains=8, pages=6, checkpoints=seed % 4)
    manager.analyzer.enable_merge_adjustment = online
    manager.logstate_pool.kernels = NumericalKernels()
    with numerical_boundaries():
        pages, logs, works = manager.evictor.allocate(
            rng.randrange(1, 6), rng.randrange(1, 6), 1, recycle_bin=manager.recycle_bin
        )
    manager.tree.check_integrity()
    manager._allocator.check_integrity()
    manager._return_slots(pages, logs)
    for slot in works:
        manager.logstate_pool.release_slots([slot])
    manager.tree.check_integrity()
    manager._allocator.check_integrity()


def test_model_refresh_promotes_a_candidate_below_the_old_heap_top():
    from logssms.kernel.logstate import EvictionPlanner

    nodes = [(0, -1, 1, 0), (1, 0, 0, 1), (2, 0, 0, 2)]
    kv = [[], [10, 32, 32], [11, 32, 32]]
    refs = [[], [1], [2]]
    slots = [(1, 2, 20, 0, 32), (2, 2, 21, 0, 32)]
    occupied = [(20, 1), (21, 1)]
    config = (0, 0, 3, 0, 1, 1, 1, 0, 0, 2)
    models = [(0, 0, 32, 2., -12., .5, 32, 0, 0, 0),
              (1, 0, 32, 2., -12., .01, 32, 0, 32, 0),
              (2, 0, 32, 2., -12., .9, 32, 0, 32, 0)]
    assert plan_eviction(nodes, kv, refs, slots, occupied, config, [], models)[0] == [1]
    planner = EvictionPlanner(nodes, kv, refs, slots, occupied, config, [], models,
                              [(0, 0, 0), (32, 1, 0), (32, 2, 0)])
    updated = [models[0], (*models[1][:5], .9, *models[1][6:]),
               (*models[2][:5], .01, *models[2][6:])]
    planner.refresh_models([0, 1, 2], updated)
    expected = plan_eviction(nodes, kv, refs, slots, occupied, config, [], updated)
    assert expected[0] == [2]
    assert planner.plan(config, occupied) == expected


def test_merge_cost_preserves_a_previously_observed_tail():
    from logssms.kernel.logstate import EvictionPlanner

    nodes = [(0, -1, 1, 0), (1, 0, 0, 0)]
    kv = [[], []]
    refs = [[], [1, 2, 3]]
    slots = [(1, 38, -1, 0, 576), (2, 38, -1, 576, 768), (3, 38, -1, 768, 960)]
    actions = [(1, 2, 0, 1), (2, 3, 0, 1), (3, -1, 0, 1)]
    config = (0, 0, 37, 1, 1, 7, 1, 0, 0, 13)
    models = [(0, 0, 32, 2., -7.4, .008, 32, 0, 0, 0),
              (1, 0, 960, 2., -7.4, .008, 32, 0, 960, 0)]
    args = (nodes, kv, refs, slots, [], config, actions, models)

    assert plan_eviction(*args)[3] == [1, 2]
    assert plan_eviction(*args, past_tails=[(1, 576, .008)])[3] == [2, 3]
    assert plan_eviction(*args, past_tails=[(1, 768, .008)])[3] == [1, 2]

    planner = EvictionPlanner(*args, [(0, 0, 0), (960, 1, 0)])
    planner.refresh_models([0, 1], models, past_tails=[(1, 576, .008)])
    assert planner.plan(config, [])[3] == [2, 3]


@pytest.mark.parametrize("granularity", [16, 32, 64])
def test_pressure_group_advances_after_materialization(granularity):
    import torch
    from experiment.ablation_online_analysis.cpu_common import NumericalKernels, numerical_boundaries
    from logssms.cache.logstate.prefix_cache import LogStateEntry, LogStateKind
    from tests.core.test_logstate_cache_manager import make_manager

    manager = make_manager(num_states=128, granularity=granularity)
    pool, tree = manager.logstate_pool, manager.tree
    pool.kernels = NumericalKernels()
    count = pool.allocator.materialize_log_count
    size = 2 * count + 2
    kv, slots, _ = pool.reserve(kv=size, logs=size)
    pool.materialize(slots[0])
    tokens = torch.arange(granularity * size, dtype=torch.int32)
    indices = torch.cat([torch.arange(frame * granularity, (frame + 1) * granularity,
                                      dtype=torch.int32) for frame in kv])
    entries = tuple(LogStateEntry(
        slot, LogStateKind.CHECKPOINT if i == 0 else LogStateKind.LOG,
        i * granularity, (i + 1) * granularity, span_granules=None if i else 1,
    ) for i, slot in enumerate(slots))
    tree.insert(tokens, indices, entries)
    first, second = slots[1:count + 1], slots[count + 1:2 * count + 1]
    assert tree.pressure_materializations(count) == [(slots[0], second[-1], *first)]
    with numerical_boundaries():
        manager.evictor._materialize((slots[0], second[-1], *first))
    assert tree.pressure_materializations(count) == [(first[-1], second[-1], *second)]


@pytest.mark.parametrize("logs", [1, 2])
@pytest.mark.parametrize("with_checkpoint", [False, True])
def test_pressure_treats_short_log_tails_as_one_unit(logs, with_checkpoint, monkeypatch):
    import torch
    from experiment.ablation_online_analysis.cpu_common import NumericalKernels, numerical_boundaries
    from logssms.cache.logstate.prefix_cache import LogStateEntry, LogStateKind
    from tests.core.test_logstate_cache_manager import make_manager

    manager = make_manager(granularity=32)
    pool, tree = manager.logstate_pool, manager.tree
    monkeypatch.setattr(type(pool.allocator), "materialize_log_count", property(lambda _: 3))
    pool.kernels = NumericalKernels()
    size = logs + int(with_checkpoint)
    kv, slots, _ = pool.reserve(kv=size, logs=size)
    if with_checkpoint:
        pool.materialize(slots[0])
    tokens = torch.arange(32 * size, dtype=torch.int32)
    indices = torch.cat([torch.arange(frame * 32, (frame + 1) * 32,
                                      dtype=torch.int32) for frame in kv])
    tree.insert(tokens, indices, tuple(
        LogStateEntry(slot, LogStateKind.CHECKPOINT if with_checkpoint and i == 0 else LogStateKind.LOG,
                      i * 32, (i + 1) * 32, span_granules=1 if with_checkpoint and i == 0 else None)
        for i, slot in enumerate(slots)
    ))
    manager.evictor.cached_page_tokens.update({frame * 32: 32 for frame in kv})
    with numerical_boundaries():
        pages, _, _ = manager._allocate(pool.allocator.free_frames + 1, 0)
    if with_checkpoint:
        assert tree.entry_for_slot(slots[0]).end == len(tokens)
        assert tree.match_prefix(tokens).recoverable_len == len(tokens)
        assert set(kv) <= pool.allocator.allocated_kv
    else:
        assert tree.trace(tokens).matched_len == 0
        assert len(manager.recycle_bin) == 1
    manager._return_slots(pages, [])
    manager.check_integrity()


@pytest.mark.parametrize("with_checkpoint", [False, True])
def test_pressure_materializes_leaf_using_ancestor_log(with_checkpoint):
    import torch
    from unittest.mock import patch
    from experiment.ablation_online_analysis.cpu_common import NumericalKernels, numerical_boundaries
    from logssms.cache.logstate.prefix_cache import LogStateEntry, LogStateKind
    from tests.core.test_logstate_cache_manager import make_manager

    manager = make_manager(granularity=32,
                           enable_merge_adjustment=True)
    pool, tree = manager.logstate_pool, manager.tree
    pool.kernels = NumericalKernels()
    ancestor = 1 + int(with_checkpoint)
    size = ancestor + pool.allocator.materialize_log_count
    kv, slots, _ = pool.reserve(kv=size, logs=size)
    if with_checkpoint:
        pool.materialize(slots[0])
    tokens = torch.arange(32 * size, dtype=torch.int32)
    indices = torch.cat([torch.arange(frame * 32, (frame + 1) * 32,
                                      dtype=torch.int32) for frame in kv])
    leaf = tree.insert(tokens, indices, tuple(
        LogStateEntry(slot, LogStateKind.CHECKPOINT if with_checkpoint and i == 0 else LogStateKind.LOG,
                      i * 32, (i + 1) * 32, span_granules=1 if with_checkpoint and i == 0 else None)
        for i, slot in enumerate(slots)
    )).node
    parent = tree.split_at(leaf, ancestor * 32)
    tree.protect(parent)
    tree.pin_recovery((slots[ancestor - 1],))
    analyzer = manager.evictor.analyzer
    stats = leaf.analysis_owner = analyzer.make_hit_stats(0)
    analyzer.advance_hit_stats(stats, len(tokens))
    analyzer.record_hit(stats, 32)
    assert analyzer.has_internal_observations
    manager.evictor.cached_page_tokens.update({frame * 32: 32 for frame in kv})
    with numerical_boundaries(), patch.object(pool.kernels, "restore", wraps=pool.kernels.restore) as restore:
        pages, _, _ = manager._allocate(pool.allocator.free_frames + 1, 0)
    restore.assert_called_once_with(pool, [(
        slots[0] if with_checkpoint else None,
        [(slot, 0, 32) for slot in slots[ancestor - 1:]],
        manager.evictor.pressure_work_slot,
    )])
    assert tree.entry_for_slot(slots[ancestor - 1]).kind is LogStateKind.LOG
    assert tree.entry_for_slot(slots[-1]).begin == parent.depth
    assert tree.match_prefix(tokens).recoverable_len == len(tokens)
    tree.pin_recovery((slots[ancestor - 1],), release=True)
    manager._return_slots(pages, [])
    manager.check_integrity()
