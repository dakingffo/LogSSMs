from tests.logstate_helpers import eviction_range
from random import Random
from unittest.mock import patch

import pytest
import torch

from logssms.cache.logstate.evictor import LogStateCapacityError
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


def assert_full_plan_matches(manager, pages, logs=0, works=0):
    with patch.object(manager.tree, "log_eviction_snapshot", return_value=([], [])):
        try:
            expected = manager.evictor.plan(pages, logs, works)
        except LogStateCapacityError:
            expected = None
    if expected is None:
        with pytest.raises(LogStateCapacityError):
            manager.evictor.plan(pages, logs, works)
    else:
        assert signature(manager.evictor.plan(pages, logs, works)) == signature(expected)


def test_log_eviction_does_not_build_full_tree_metadata():
    manager = make_pressure_tree(chains=32)
    with patch.object(manager.tree, "eviction_snapshot", side_effect=AssertionError):
        nodes, _pages, _slots, compressed = manager.evictor.plan(1, 1, 0)
    assert len(nodes) == 1 and nodes[0].timestamp == 0
    assert not compressed


@pytest.mark.parametrize("checkpoints", [0, 1, 3, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_log_window_matches_global_order_and_capacity(checkpoints, seed):
    manager = make_pressure_tree(checkpoints=checkpoints)
    tree = manager.tree
    rng = Random(seed)
    for node in tree._iter_nodes():
        node.timestamp = rng.randrange(20)
    leaves = tuple(tree._leaves.values())
    tree.lock(leaves[0])
    pinned = tuple(entry.slot for entry in leaves[1].entries)
    tree.pin_recovery(pinned)
    tree.split_at(leaves[2], 32)
    tree.check_integrity()
    revision, free = tree.revision, manager._allocator.free_frames
    for pages, logs, works in ((1, 0, 0), (1, 1, 0), (4, 4, 0), (9, 9, 1), (5000, 0, 0)):
        assert_full_plan_matches(manager, pages, logs, works)
    assert (tree.revision, manager._allocator.free_frames) == (revision, free)
    tree.pin_recovery(pinned, release=True)
    tree.lock(leaves[0], unlock=True)
    tree.check_integrity()


def test_log_window_respects_hot_ancestor_and_excluded_cold_leaf():
    manager = make_pressure_tree(checkpoints=0)
    tree = manager.tree
    leaves = sorted(tuple(tree._leaves.values()), key=lambda node: node.timestamp)
    for leaf in leaves:
        for node in tree.path(leaf)[:-1]:
            node.timestamp = 1000
    assert_full_plan_matches(manager, 9, 9)
    with patch.object(tree, "eviction_snapshot", wraps=tree.eviction_snapshot) as full:
        nodes, _pages, _slots, _compressed = manager.evictor.plan(9, 9, 0)
    assert full.call_count == 0
    assert list(dict.fromkeys(part.node for part in nodes)) == leaves[:3]


def test_log_window_expands_before_full_tree_planning():
    manager = make_pressure_tree(chains=128, pages=4, checkpoints=3)
    with patch.object(manager.tree, "eviction_snapshot", side_effect=AssertionError), patch.object(
        manager.tree, "log_eviction_snapshot", wraps=manager.tree.log_eviction_snapshot,
    ) as window:
        nodes, _pages, _slots, compressed = manager.evictor.plan(16, 16, 0)
    assert len(nodes) > 8 and not compressed
    assert [call.args[0] for call in window.call_args_list] == [1, 4, 16]


def test_leaf_index_tracks_split_branch_prune_and_reset():
    manager = make_pressure_tree(chains=2, checkpoints=0)
    tree = manager.tree
    leaf = tuple(tree._leaves.values())[0]
    prefix = tree.split_at(leaf, 32)
    tokens = torch.cat([*(node.key for node in tree.path(prefix)),
                        torch.arange(10000, 10032, dtype=torch.int32)])
    indices = torch.cat([*(node.kv for node in tree.path(prefix)),
                         torch.arange(10000, 10032, dtype=torch.int32)])
    branch = tree.insert(tokens, indices, cache_len=len(tokens)).node
    assert branch.parent is prefix and leaf.parent is prefix
    tree.check_integrity()
    assert leaf.uuid in tree._leaves and branch.uuid in tree._leaves
    tree.evict([eviction_range(tree, branch), eviction_range(tree, leaf)])
    tree.check_integrity()
    assert prefix.uuid in tree._leaves
    tree.reset()
    tree.check_integrity()
    assert not tree._leaves


def test_log_window_keeps_an_unselected_shared_branch_attached():
    manager = make_pressure_tree(checkpoints=0)
    tree = manager.tree
    leaves = sorted(tuple(tree._leaves.values()), key=lambda node: node.timestamp)
    for leaf in leaves:
        for node in tree.path(leaf)[:-1]:
            node.timestamp = 2000
    source = leaves[-1]
    branch_kv = source.kv[:32].clone()
    branch_entry = LogStateEntry(source.entries[0].slot, LogStateKind.LOG, 32, 64)
    removed = tree.evict([eviction_range(tree, source)])
    for index in removed.kv_indices[::32]:
        manager.evictor.cached_page_tokens.pop(int(index) // 32 * 32)
    prefix = tree.split_at(leaves[0], 32)
    prefix.timestamp = 0
    tokens = torch.cat([*(node.key for node in tree.path(prefix)),
                        torch.arange(100000, 100032, dtype=torch.int32)])
    indices = torch.cat([*(node.kv for node in tree.path(prefix)), branch_kv])
    branch = tree.insert(tokens, indices, (branch_entry,), cache_len=len(tokens)).node
    branch.timestamp = 1000
    manager.evictor.cached_page_tokens[int(branch_kv[0]) // 32 * 32] = 32
    tree.check_integrity()
    assert_full_plan_matches(manager, 4, 4)
    nodes, _pages, _slots, _compressed = manager.evictor.plan(4, 4, 0)
    assert leaves[0] in [part.node for part in nodes]
    assert prefix not in [part.node for part in nodes] and branch not in [part.node for part in nodes]


def assert_log_index_matches_scan(tree, limit=8):
    expected = sorted((
        leaf for leaf in tree._leaves.values()
        if not leaf.is_locked and (not leaf.entries or (
            leaf.entries[-1].kind is LogStateKind.LOG
            and leaf.entries[-1].slot not in tree._recovery_slot_refs
        ))
    ), key=lambda node: (node.timestamp, node.depth, node.uuid))
    nodes, _rows = tree.log_eviction_snapshot(limit)
    assert [part.node for part in nodes if part.node.is_leaf() and part.end == part.node.depth] == sorted(
        expected[:limit], key=lambda node: (node.depth, node.uuid))
    tree.check_integrity()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_log_index_tracks_access_locks_pins_and_splits(seed):
    manager = make_pressure_tree(chains=16, checkpoints=1)
    tree = manager.tree
    rng = Random(seed)
    for step in range(80):
        leaf = rng.choice(tuple(tree._leaves.values()))
        action = rng.randrange(5)
        if action == 0:
            leaf.timestamp = rng.randrange(10)
        elif action == 1:
            tree.lock(leaf)
            assert_log_index_matches_scan(tree)
            tree.lock(leaf)
            tree.lock(leaf, unlock=True)
            assert_log_index_matches_scan(tree)
            tree.lock(leaf, unlock=True)
        elif action == 2:
            slots = tuple(entry.slot for entry in leaf.entries)
            tree.pin_recovery(slots)
            tree.pin_recovery(slots)
            assert_log_index_matches_scan(tree)
            tree.pin_recovery(slots, release=True)
            assert_log_index_matches_scan(tree)
            tree.pin_recovery(slots, release=True)
        elif action == 3 and leaf.length > 32:
            tree.split_at(leaf, 32)
        else:
            tree.protect(leaf, not leaf.is_protected)
        assert_log_index_matches_scan(tree, 1 + step % 12)
        if step % 8 == 0:
            assert_full_plan_matches(manager, 1, 1)


def test_log_snapshot_reads_index_without_scanning_leaves():
    manager = make_pressure_tree(chains=64, checkpoints=1)
    tree = manager.tree
    expected = signature(manager.evictor.plan(1, 1, 0))

    class IndexedLeaves(dict):
        def values(self):
            raise AssertionError("eviction scanned every leaf")

    with patch.object(tree, "_leaves", IndexedLeaves(tree._leaves)):
        assert signature(manager.evictor.plan(1, 1, 0)) == expected


def test_log_index_removes_checkpoint_and_retired_leaf_records():
    manager = make_pressure_tree(chains=2, checkpoints=0)
    tree = manager.tree
    leaf = tuple(tree._leaves.values())[0]
    entry = leaf.entries[-1]
    prefix_slots = leaf.eviction_slots[:-1]
    assert leaf.eviction_slots == (*prefix_slots, entry.slot)
    tree.detach_entry(entry.slot)
    assert leaf.eviction_slots == prefix_slots
    checkpoint = LogStateEntry(entry.slot, LogStateKind.CHECKPOINT,
                               entry.begin, entry.end, span_granules=1)
    tree.attach_entry(leaf, checkpoint)
    assert leaf.eviction_slots == (*prefix_slots, entry.slot)
    assert leaf._log_leaf_index == -1
    assert_log_index_matches_scan(tree)
    tree.detach_entry(entry.slot)
    tree.attach_entry(leaf, entry)
    assert_log_index_matches_scan(tree)
    for step in range(1000):
        leaf.timestamp = step
    assert len(tree._log_leaf_heap) == 2
    tree.evict([eviction_range(tree, leaf)])
    assert leaf._log_leaf_index == -1 and leaf._log_leaf_tree is None
    leaf.timestamp = 0
    assert_log_index_matches_scan(tree)
    previous = tuple(tree._leaves.values())
    tree.reset()
    for node in previous:
        node.timestamp = 0
        assert node._log_leaf_index == -1 and node._log_leaf_tree is None
    assert_log_index_matches_scan(tree)


@pytest.mark.parametrize("checkpoints", [0, 1, 64])
@pytest.mark.parametrize("same_heat", [False, True])
def test_bounded_tails_match_complete_planning_across_long_chains(checkpoints, same_heat):
    manager = make_pressure_tree(chains=12, pages=96, checkpoints=checkpoints)
    if same_heat:
        for node in manager.tree._iter_nodes():
            node.timestamp = 0
    for needed in (8, 40, 96, 300):
        assert_full_plan_matches(manager, needed, needed)
    manager.tree.check_integrity()


def test_bounded_tail_expansion_respects_hidden_pins_and_hot_ancestors():
    manager = make_pressure_tree(chains=12, pages=96, checkpoints=0)
    tree = manager.tree
    cold, locked = sorted(tree._leaves.values(), key=lambda node: node.timestamp)[:2]
    tree.split_at(cold, 64).timestamp = 1000
    pinned = (cold.entries[-20].slot,)
    tree.pin_recovery(pinned)
    tree.lock(locked)
    revision, free = tree.revision, manager._allocator.free_frames
    for needed in (8, 40, 200):
        assert_full_plan_matches(manager, needed, needed)
    assert (tree.revision, manager._allocator.free_frames) == (revision, free)
    tree.pin_recovery(pinned, release=True)
    tree.lock(locked, unlock=True)
    tree.check_integrity()


def test_small_eviction_serializes_only_bounded_tail_frame_counts(monkeypatch):
    from logssms.cache.logstate.prefix_cache import LogRadixEvictionSegment

    manager = make_pressure_tree(chains=32, pages=128, checkpoints=0)
    descriptor = LogRadixEvictionSegment._kv_frame_counts
    counted = []

    def counts(segment):
        counted.append((segment.begin, segment.end))
        return descriptor.fget(segment)

    monkeypatch.setattr(LogRadixEvictionSegment, "_kv_frame_counts", property(counts))
    with patch.object(manager.tree, "eviction_snapshot", side_effect=AssertionError):
        selected, _pages, _slots, _pairs = manager.evictor.plan(1, 1, 0)
    assert len(selected) == 1
    assert len(counted) <= 16 + 1
    assert all(end - begin <= 32 for begin, end in counted)


@pytest.mark.parametrize("needed", [1, 40, 300])
def test_bounded_tail_plan_can_reclaim_and_reallocate_with_recycling(needed):
    manager = make_pressure_tree(chains=12, pages=96, checkpoints=1)
    ballast = manager._allocator.allocated_kv - {
        offset // 32 for offset in manager.evictor.cached_page_tokens
    } - {manager._dummy_kv_frame}
    expected, _pages, released, _compressed = manager.evictor.plan(needed, needed, 1)
    assert expected and released
    pages, logs, works = manager.evictor.allocate(
        needed, needed, 1, recycle_bin=manager.recycle_bin,
    )
    assert (len(pages), len(logs), len(works)) == (needed, needed, 1)
    assert all(not manager.tree.has_slot(slot) for slot in released)
    manager.logstate_pool.release_kv([*(page // 32 for page in pages), *ballast])
    manager.logstate_pool.retire_slots([*logs, *works])
    manager.check_integrity()


def test_equal_heat_candidate_order_matches_full_plan_for_different_chain_lengths():
    manager = make_pressure_tree(chains=12, pages=96, checkpoints=0)
    tree = manager.tree
    leaves = sorted(tree._leaves.values(), key=lambda node: node.uuid)
    for chain, leaf in enumerate(leaves):
        keep = 32 + chain
        ranges, _rows = tree.eviction_snapshot()
        suffix = [r for r in ranges if r.node is leaf and r.begin >= keep * 32]
        if suffix:
            eviction = tree.evict(reversed(suffix))
            manager.evictor._release_cached_kv({
                int(index) // 32: 32 for index in eviction.kv_indices[::32]
            })
            manager.logstate_pool.retire_slots([entry.slot for entry in eviction.entries])
    for node in tree._iter_nodes():
        node.timestamp = 0
    # Restore physical pressure after the explicit fixture trimming.
    manager.logstate_pool.reserve(kv=manager._allocator.free_frames)
    for needed in (4, 40, 80):
        assert_full_plan_matches(manager, needed, needed)
    tree.check_integrity()
