import pytest

from logssms.kernel.logstate import plan_eviction


def run(need, *, online=False, atom=0.9, log_a=-12.0, exponent=2.0, locked=False, fallback=False):
    # One chain with two Checkpoints and a tail Log. Each state has 3 payload
    # frames; each tail segment also owns one independent KV frame.
    nodes = [(0, -1, 1, 0, 0), (1, 0, 0, 1, 0), (2, 1, int(locked), 0, 2)]
    kv = [[], [10, 32, 32, 11, 32, 32], [12, 32, 32]]
    refs = [[], [1, 2], [3]]
    slots = [(1, 3, -1, 0, 32), (2, 3, -1, 32, 64), (3, 3, 20, 64, 96)]
    actions = [(1, 2, 0, 1), (2, -1, 1, 1)]
    models = (
        [
            (1, 0, 96, exponent, log_a, atom, 32, b, e, int(fallback))
            for b, e in ((0, 0), (0, 64), (64, 96))
        ]
        if online
        else None
    )
    return plan_eviction(
        nodes, kv, refs, slots, [(20, 1)], (0, 0, 3, 0, 1, 1, need, 0, 0, 3), actions, models
    )


def test_default_evicts_log_before_checkpoint_and_stops_when_satisfied():
    selected, kv, slots, drops = run(4)
    assert selected == [2] and not drops and slots == [3]


def test_online_can_retain_a_hot_tail_and_drop_a_middle_checkpoint():
    selected, kv, slots, drops = run(3, online=True)
    assert selected == [] and kv == [] and slots == [1] and drops == [1, 2]


def test_online_never_prunes_an_interior_log_or_a_locked_leaf():
    assert run(100, online=True) == []
    selected, _, _, drops = run(3, online=True, locked=True)
    assert selected == [] and drops == [1, 2]


def test_newly_enlarged_checkpoint_has_correct_successor_for_later_eviction():
    selected, _, released, drops = run(10, online=True)
    assert drops == [1, 2]
    assert set(released) == {1, 2, 3}
    assert selected == [2, 1]


def test_binary_checkpoint_thinning_removes_adjacent_equal_scales_first():
    nodes = [(0, -1, 1, 0, 0), (1, 0, 0, 1, 0)]
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


def test_density_changes_order_across_chains_without_changing_tail_only_rule():
    nodes = [(0, -1, 1, 0, 0), (1, 0, 0, 0, 0), (2, 1, 0, 0, 0), (3, 0, 0, 0, 10)]
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


def test_scoring_does_not_hide_released_kv_frames_from_the_execution_plan():
    selected, kv, released, drops = run(5, online=True, atom=0.0, exponent=0.0)
    assert selected == [2] and kv == [12, 32] and released == [3] and drops == []


@pytest.mark.parametrize("online", [False, True])
def test_checkpoint_leaf_is_pruned_whole_only_after_internal_thinning(online):
    models = [(1, 0, 96, 2., -12., .9, 32, 0, end, 0)
              for end in (0, 96)] if online else None
    result = plan_eviction(
        [(0, -1, 1, 0, 0), (1, 0, 0, 1, 0)],
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
        [(0, -1, 1, 0, 0), (1, 0, 0, 1, 0), (2, 0, 0, 1, 0)],
        [[], [], []], [[], [1], [2]],
        [(1, 3, -1, 480, 512), (2, 3, -1, 0, 64)], [],
        (0, 0, 3, 0, 1, 1, 3, 0, 0, 3),
        [(1, -1, 0, 1), (2, -1, 0, 2)],
        [(0, 0, 512, 2., -100., 1., 32, 0, 0, 0),
         (1, 0, 512, 2., -100., 1., 32, 416, 512, 0),
         (2, 0, 64, 2., -100., 1., 32, 0, 64, 0)],
    )
    assert selected == [2] and merges == []


def test_density_amplitude_affects_tail_selection_independently_of_heat_ties():
    selected, _, _, _ = plan_eviction(
        [(0, -1, 1, 0, 0), (1, 0, 0, 0, 0), (2, 0, 0, 0, 100)],
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
        [(0, -1, 1, 0, 0), (1, 0, 0, 1, 0), (2, 0, 0, 1, 0)],
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
            [(0, -1, 1, 0, 0), (1, 0, 0, 1, 0), (2, 1, 0, 1, 0)],
            [[], [], []], [[], [1], [2]],
            [(1, 3, -1, 0, 32), (2, 3, -1, 32, 64)], [],
            (0, 0, 3, 0, 1, 1, 1, 0, 0, 3), [(1, 2, 0, 1)],
        )


@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("seed", range(8))
def test_actual_allocator_accepts_the_projected_reservation(online, seed):
    from random import Random
    from tests.core.test_logstate_eviction_planning import make_pressure_tree

    rng = Random(seed)
    manager = make_pressure_tree(chains=8, pages=6, checkpoints=seed % 4)
    manager.analyzer.enable_merge_adjustment = online
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


def test_zero_internal_fallback_defers_checkpoint_until_its_log_tail_is_reclaimed():
    selected, _, released, merges = run(3, online=True, fallback=True)
    assert selected == [2] and released == [3] and merges == []
    selected, _, released, merges = run(7, online=True, fallback=True)
    assert selected == [2] and merges == [1, 2]
    assert set(released) == {1, 3}


def test_zero_internal_pool_prefers_another_owners_log_to_middle_checkpoint():
    selected, _, released, merges = plan_eviction(
        [(0, -1, 1, 0, 0), (1, 0, 1, 1, 0), (2, 0, 0, 0, 1)],
        [[], [], []], [[], [1, 2], [3]],
        [(1, 3, -1, 0, 32), (2, 3, -1, 32, 64), (3, 3, 20, 0, 32)],
        [(20, 1)], (0, 0, 3, 0, 1, 1, 3, 0, 0, 3),
        [(1, 2, 0, 1), (2, -1, 0, 1)],
        [(0, 0, 64, 2., -12., .9, 32, 0, 0, 1),
         (1, 0, 64, 2., -12., .9, 32, 0, 64, 1),
         (2, 0, 32, 2., -12., .9, 32, 0, 32, 1)],
    )
    assert selected == [2] and released == [3] and merges == []


@pytest.mark.parametrize('need', [1, 3, 4, 5, 7, 10, 13, 100])
@pytest.mark.parametrize('locked', [False, True])
def test_all_zero_internal_owners_match_default_plan_exactly(need, locked):
    assert run(need, online=True, fallback=True, locked=locked) == run(need, locked=locked)


def test_zero_internal_logs_use_timestamp_even_when_cost_prefers_newer_chain():
    result = plan_eviction(
        [(0, -1, 1, 0, 0), (1, 0, 0, 0, 0), (2, 0, 0, 0, 100)],
        [[], [], []], [[], [1], [2]],
        [(1, 2, 21, 0, 32), (2, 2, 22, 0, 32)], [(21, 1), (22, 1)],
        (0, 0, 2, 0, 1, 1, 2, 0, 0, 2), [],
        [(0, 0, 32, 2., -12., .9, 32, 0, 0, 1),
         (1, 0, 32, 2., -6., .9, 32, 0, 32, 1),
         (2, 0, 32, 2., -12., .01, 32, 0, 32, 1)],
    )
    assert result[0] == [1]


@pytest.mark.parametrize('checkpoints', [0, 2, 4])
@pytest.mark.parametrize('needed', [1, 15, 75])
def test_all_zero_internal_manager_matches_default_enumeration(checkpoints, needed):
    from unittest.mock import patch
    from tests.core.test_logstate_eviction_planning import make_pressure_tree, signature
    manager = make_pressure_tree(chains=8, pages=4, checkpoints=checkpoints)
    manager.analyzer.enable_merge_adjustment = False
    expected = signature(manager.evictor.plan(needed, 0, 0))
    manager.analyzer.enable_merge_adjustment = True
    assert not manager.analyzer.has_internal_observations
    with patch.object(manager.tree, 'log_eviction_snapshot', wraps=manager.tree.log_eviction_snapshot) as window:
        assert signature(manager.evictor.plan(needed, 0, 0)) == expected
        assert window.call_count > 0


def test_model_refresh_promotes_a_candidate_below_the_old_heap_top():
    from logssms.kernel.logstate import EvictionPlanner

    nodes = [(0, -1, 1, 0, 0), (1, 0, 0, 0, 1), (2, 0, 0, 0, 2)]
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
