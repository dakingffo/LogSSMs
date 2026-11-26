from tests.logstate_helpers import materialize_pool_log, seed_checkpoint_slot
from random import Random
from unittest.mock import patch

import pytest
import torch

from logssms.cache.logstate.prefix_cache import LogStateEntry, LogStateKind
from tests.core.test_logstate_cache_manager import make_manager


def make_pressure_tree(chains=16, pages=4, checkpoints=1):
    count = chains * pages
    manager = make_manager(num_pages=count, num_states=count,
                           max_seq_len=pages * 32 + 128, granularity=32)
    kv, logs, states = manager.logstate_pool.reserve(
        kv=count, logs=chains * (pages - checkpoints), works=chains * checkpoints,
    )
    for slot in states:
        seed_checkpoint_slot(manager.logstate_pool, slot)
    logs, states = iter(logs), iter(states)
    for chain in range(chains):
        frames = kv[chain * pages:(chain + 1) * pages]
        indices = (torch.tensor(frames, dtype=torch.int32)[:, None] * 32
                   + torch.arange(32, dtype=torch.int32)).flatten()
        entries = tuple(LogStateEntry(
            next(states) if page < checkpoints else next(logs),
            LogStateKind.CHECKPOINT if page < checkpoints else LogStateKind.LOG,
            page * 32, (page + 1) * 32, span_granules=1 if page < checkpoints else None,
        ) for page in range(pages))
        node = manager.tree.insert(
            torch.arange(pages * 32, dtype=torch.int32) + chain * 4096,
            indices, entries,
        ).node
        for part in manager.tree.path(node):
            part.timestamp = chain
        manager.evictor.cached_page_tokens.update({frame * 32: 32 for frame in frames})
    manager.logstate_pool.reserve(kv=manager._allocator.free_frames)
    return manager


def signature(plan):
    return ([(part.node.uuid, part.begin, part.end) for part in plan.leaves],
            plan.kv_references, plan.retired_slots, plan.checkpoint_merges)


@pytest.mark.parametrize("internal", [False, True])
def test_materialized_node_refresh_matches_rebuilt_plan(internal):
    from contextlib import nullcontext
    from experiment.utils.cpu_replay.scheduler import NumericalKernels, numerical_boundaries

    def run(rebuild):
        manager = make_pressure_tree(chains=6, pages=12, checkpoints=1)
        tree, evictor = manager.tree, manager.evictor
        manager.logstate_pool.kernels = NumericalKernels()
        if internal:
            for leaf in tuple(tree._leaves.values()):
                tree.split_at(leaf, 5 * tree.granularity)
        trace = []
        original = evictor.plan

        def record(*args, **kwargs):
            plan = original(*args, **kwargs)
            trace.append((
                [part.node.key[0].item() for part in plan.leaves],
                plan.kv_references, plan.retired_slots, plan.checkpoint_merges,
                plan.materialization, plan.active_action,
            ))
            return plan

        def rebuild_after_materialization(owner):
            evictor._planner = None
            return False

        force_rebuild = (
            patch.object(evictor, "_update_materialized_node", side_effect=rebuild_after_materialization)
            if rebuild else nullcontext()
        )
        with (force_rebuild, numerical_boundaries(),
              patch.object(evictor, "plan", side_effect=record),
              patch.object(tree, "eviction_snapshot", wraps=tree.eviction_snapshot) as snapshots):
            for _ in range(8):
                manager._allocate(1, 1)
        tree.check_integrity()
        manager._allocator.check_integrity()
        return trace, snapshots.call_count

    reference, full_snapshots = run(rebuild=True)
    actual, incremental_snapshots = run(rebuild=False)
    assert any(action[4] for action in actual)
    assert actual == reference
    assert incremental_snapshots < full_snapshots


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("checkpoints", [5, 12])
def test_online_planner_reuses_candidates_without_changing_eviction(seed, checkpoints):
    from logssms.cache.logstate.evictor import LogStateEvictor

    manager = make_pressure_tree(chains=12, pages=checkpoints, checkpoints=checkpoints)
    tree, analyzer = manager.tree, manager.analyzer
    analyzer.enable_merge_adjustment = True
    leaves = tuple(tree._leaves.values())
    for leaf in leaves:
        stats = analyzer.make_hit_stats(0)
        analyzer.advance_hit_stats(stats, leaf.depth)
        analyzer.record_hit(stats, 64)
        leaf.analysis_owner = stats
        leaf.timestamp = 1
    rng = Random(seed)
    reference = LogStateEvictor(
        tree, manager.logstate_pool, analyzer=analyzer, profile_enabled=False,
        profile_stats={}, allocation_samples=[], record_lifecycle_event=lambda _: None,
        pressure_work_slot=manager._pressure_work_slot,
    )
    reference.cached_page_tokens = manager.evictor.cached_page_tokens
    original = manager.evictor.plan
    reused = 0
    refreshed = 0

    def compare(*args, **kwargs):
        nonlocal reused, refreshed
        before = manager.evictor._planner
        model_changed = (
            before is not None
            and manager.evictor._planner_epoch == manager.evictor._eviction_epoch()
            and manager.evictor._planner_model_epoch != manager.evictor._model_epoch()
        )
        expected = reference.plan(*args)
        actual = original(*args, **kwargs)
        reused += before is not None and manager.evictor._planner is before
        if model_changed:
            assert manager.evictor._planner is before
            refreshed += 1
        assert signature(actual) == signature(expected)
        return actual

    with patch.object(manager.evictor, "plan", side_effect=compare):
        for step in range(40):
            if step == 8:
                locked_leaf = next(iter(tree._leaves.values()))
                tree.lock(locked_leaf)
            if step == 10:
                tree.lock(locked_leaf, unlock=True)
            if step == 12:
                # A miss still advances the global query clock.
                analyzer.record_hit(None, 0)
            if step == 16:
                leaf = next(iter(tree._leaves.values()))
                analyzer.record_hit(leaf.analysis_owner, leaf.depth)
            if step == 20:
                leaf = next(iter(tree._leaves.values()))
                leaf.timestamp += 1
            if step == 24:
                leaf = next(iter(tree._leaves.values()))
                pinned = (leaf.entries[-1].slot,)
                tree.pin_recovery(pinned)
            if step == 28:
                tree.pin_recovery(pinned, release=True)
            pages, slots, works = manager.evictor.allocate(1, 1)
            assert len(pages) == len(slots) == 1 and not works
            if rng.randrange(2):
                manager.logstate_pool.retire_slots(slots)
            tree.check_integrity()
            manager._allocator.check_integrity()
    assert reused > 10
    assert refreshed >= 2


@pytest.mark.parametrize("analysis", ["disabled", "prior", "fitted"])
def test_eviction_profile_records_rebuild_causes_and_stage_timings(analysis):
    manager = make_pressure_tree(chains=8, pages=12, checkpoints=12)
    tree, analyzer, evictor = manager.tree, manager.analyzer, manager.evictor
    analyzer.enable_merge_adjustment = analysis != "disabled"
    if analyzer.enable_merge_adjustment:
        for leaf in tuple(tree._leaves.values()):
            stats = analyzer.make_hit_stats(0)
            analyzer.advance_hit_stats(stats, leaf.depth)
            if analysis == "fitted":
                analyzer.record_hit(stats, 64)
            leaf.analysis_owner = stats
    evictor._profile_enabled = True

    def allocate():
        evictor.allocate(1, 1)
        return evictor._profile_allocation_samples[-1]["planning"]

    first = allocate()
    assert first["causes"] == ["cold_start"]
    assert first["mode"] == "rebuild" and first["full_snapshots"] == 1
    assert allocate()["mode"] == "reuse"
    leaf = next(iter(tree._leaves.values()))
    leaf.timestamp += 1
    assert allocate()["causes"] == ["access"]
    pinned = (leaf.entries[-1].slot,)
    tree.pin_recovery(pinned)
    locked = allocate()
    assert locked["causes"] == ["locks"]
    assert locked["total_ms"] > 0
    assert locked["mode"] == "reuse" and locked["full_snapshots"] == 0
    assert "eviction_locks_ms" in locked["timing_ms"]
    assert "eviction_snapshot_ms" not in locked["timing_ms"]
    assert "eviction_build_ms" not in locked["timing_ms"]
    tree.pin_recovery(pinned, release=True)


def test_eviction_frame_counts_follow_token_order_after_append():
    from collections import Counter
    from logssms.cache.logstate.prefix_cache import LogRadixTreeNode

    manager = make_pressure_tree(chains=1)
    node = LogRadixTreeNode(manager.tree.key_fn, 32)
    frames = torch.tensor([3] * 16 + [5] * 32 + [3] * 16, dtype=torch.int32)
    node.initialize(torch.arange(64, dtype=torch.int32), frames * 32, frames)
    for append in (False, True):
        if append:
            extra = torch.tensor([7] * 16 + [5] * 16, dtype=torch.int32)
            node.append(torch.arange(64, 96, dtype=torch.int32), extra * 32, extra)
        assert list(node.eviction_frame_counts.items()) == list(Counter(node._kv_frames.tolist()).items())


def test_rebase_refreshes_checkpoint_pins_when_node_stays_blocked():
    from logssms.cache.logstate.evictor import LogStateEvictor

    manager = make_pressure_tree(chains=2, pages=4, checkpoints=4)
    tree, evictor = manager.tree, manager.evictor
    leaf = min(tree._leaves.values(), key=lambda node: node.timestamp)
    leaf.timestamp = 0
    tree.lock(leaf)
    pins = (leaf.entries[1].slot,)
    tree.pin_recovery(pins)
    plan = evictor.plan(0, 0, 0)
    assert not plan.retired_slots

    tree.pin_recovery(pins, release=True)
    assert leaf.is_locked
    other = next(node for node in tree._leaves.values() if node is not leaf)
    split = tree.split_at(other, 32)
    with tree.match_transaction():
        tree.split_at(other, 32)
    tree.protect(split)

    reference = LogStateEvictor(
        tree, manager.logstate_pool, analyzer=manager.analyzer, profile_enabled=False,
        profile_stats={}, allocation_samples=[], record_lifecycle_event=lambda _: None,
        pressure_work_slot=manager._pressure_work_slot,
    )
    reference.cached_page_tokens = evictor.cached_page_tokens
    planner = evictor._planner
    expected = reference.plan(1, 0, 0)
    assert leaf.entries[0].slot in expected.retired_slots
    actual = evictor.plan(1, 0, 0)
    assert evictor._planner is planner
    assert signature(actual) == signature(expected)


def test_exhausted_plan_does_not_keep_tentative_releases():
    from logssms.cache.logstate.evictor import LogStateCapacityError

    manager = make_pressure_tree(chains=2, pages=4, checkpoints=4)
    tree, evictor = manager.tree, manager.evictor
    evictor.plan(0, 0, 0)
    evictor._planner_epoch = evictor._eviction_epoch()
    expected = tuple(tree.iter_entries())
    with pytest.raises(LogStateCapacityError):
        evictor.plan(manager._allocator.num_frames + 1, 0, 0)
    assert tuple(tree.iter_entries()) == expected
    manager._allocate(1, 1)
    tree.check_integrity()
    manager._allocator.check_integrity()


def test_internal_materialization_keeps_committed_descendant_geometry_dirty():
    from experiment.utils.cpu_replay.scheduler import NumericalKernels, numerical_boundaries
    from logssms.cache.logstate.evictor import LogStateEvictor

    manager = make_pressure_tree(chains=2, pages=12)
    tree, pool, evictor = manager.tree, manager.logstate_pool, manager.evictor
    pool.kernels = NumericalKernels()
    child = min(tree._leaves.values(), key=lambda node: node.timestamp)
    parent = tree.split_at(child, 5 * tree.granularity)
    checkpoints = child.entries[:2]
    for entry in checkpoints:
        materialize_pool_log(pool, entry.slot)
        tree.detach_entry(entry.slot)
        tree.attach_entry(child, LogStateEntry(
            entry.slot, LogStateKind.CHECKPOINT, entry.begin, entry.end, span_granules=1,
        ))
    evictor.plan(0, 0, 0)
    evictor._planner_epoch = evictor._eviction_epoch()
    left, right = checkpoints
    tree.drop_checkpoint(left.slot, right.slot)
    pool.retire_slots((left.slot,))

    conversion = tree.pressure_materializations(pool.allocator.materialize_log_count, node=parent)[0]
    with numerical_boundaries():
        evictor._materialize(conversion)
    assert evictor._planner_needs_rebase

    reference = LogStateEvictor(
        tree, pool, analyzer=manager.analyzer, profile_enabled=False,
        profile_stats={}, allocation_samples=[], record_lifecycle_event=lambda _: None,
        pressure_work_slot=manager._pressure_work_slot,
    )
    reference.cached_page_tokens = evictor.cached_page_tokens
    assert evictor.plan(12, 0, 0) == reference.plan(12, 0, 0)
