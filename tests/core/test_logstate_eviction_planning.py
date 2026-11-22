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
        manager.logstate_pool.commit_checkpoint(slot)
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
    nodes, pages, slots, compressed = plan
    return [(part.node.uuid, part.begin, part.end) for part in nodes], pages, slots, compressed


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


def test_eviction_profile_records_rebuild_causes_and_stage_timings():
    manager = make_pressure_tree(chains=8, pages=12, checkpoints=12)
    tree, analyzer, evictor = manager.tree, manager.analyzer, manager.evictor
    analyzer.enable_merge_adjustment = True
    for leaf in tuple(tree._leaves.values()):
        stats = analyzer.make_hit_stats(0)
        analyzer.advance_hit_stats(stats, leaf.depth)
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
