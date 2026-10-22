from random import Random
from unittest.mock import patch

import pytest
import torch

from logssms.cache.logstate import LogStateCapacityError
from logssms.cache.prefix_cache.logstate import LogStateEntry, LogStateKind
from tests.core.test_logstate_cache_manager import make_manager


def make_pressure_tree(chains=16, pages=4, checkpoints=1):
    count = chains * pages
    manager = make_manager(num_pages=count, num_states=count,
                           max_seq_len=pages * 64 + 128, granularity=32)
    kv, logs, states = manager.logstate_pool.reserve(
        kv=count, logs=chains * (pages - checkpoints), works=chains * checkpoints,
    )
    for slot in states:
        manager.logstate_pool.commit_checkpoint(slot)
    logs, states = iter(logs), iter(states)
    for chain in range(chains):
        frames = kv[chain * pages:(chain + 1) * pages]
        indices = (torch.tensor(frames, dtype=torch.int32)[:, None] * 64
                   + torch.arange(64, dtype=torch.int32)).flatten()
        entries = tuple(LogStateEntry(
            next(states) if page < checkpoints else next(logs),
            LogStateKind.CHECKPOINT if page < checkpoints else LogStateKind.LOG,
            page * 64, (page + 1) * 64, span_pages=1 if page < checkpoints else None,
        ) for page in range(pages))
        node = manager.tree.insert(
            torch.arange(pages * 64, dtype=torch.int32) + chain * 4096,
            indices, entries,
        ).node
        for part in manager.tree.path(node):
            part.timestamp = chain
        manager._cached_page_tokens.update({frame * 64: 64 for frame in frames})
    manager.logstate_pool.reserve(kv=manager._allocator.free_frames)
    return manager


def signature(plan):
    nodes, pages, slots, compressed = plan
    return [node.uuid for node in nodes], pages, slots, compressed


def assert_full_plan_matches(manager, pages, logs=0, works=0):
    with patch.object(manager.tree, "log_eviction_snapshot", return_value=([], [])):
        try:
            expected = manager._plan_eviction(pages, logs, works)
        except LogStateCapacityError:
            expected = None
    if expected is None:
        with pytest.raises(LogStateCapacityError):
            manager._plan_eviction(pages, logs, works)
    else:
        assert signature(manager._plan_eviction(pages, logs, works)) == signature(expected)


def test_log_eviction_does_not_build_full_tree_metadata():
    manager = make_pressure_tree(chains=32)
    with patch.object(manager.tree, "eviction_snapshot", side_effect=AssertionError):
        nodes, _pages, _slots, compressed = manager._plan_eviction(1, 1, 0)
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
    leaves = tree.leaves()
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
    leaves = sorted(tree.leaves(), key=lambda node: node.timestamp)
    for leaf in leaves:
        for node in tree.path(leaf)[:-1]:
            node.timestamp = 1000
    assert_full_plan_matches(manager, 9, 9)
    with patch.object(tree, "eviction_snapshot", wraps=tree.eviction_snapshot) as full:
        nodes, _pages, _slots, _compressed = manager._plan_eviction(9, 9, 0)
    assert full.call_count == 0
    assert nodes[:9] == leaves[:9]


def test_log_window_expands_before_full_tree_planning():
    manager = make_pressure_tree(chains=128, pages=4, checkpoints=3)
    with patch.object(manager.tree, "eviction_snapshot", side_effect=AssertionError), patch.object(
        manager.tree, "log_eviction_snapshot", wraps=manager.tree.log_eviction_snapshot,
    ) as window:
        nodes, _pages, _slots, compressed = manager._plan_eviction(16, 16, 0)
    assert len(nodes) > 8 and not compressed
    assert [call.args[0] for call in window.call_args_list] == [8, 32]


def test_leaf_index_tracks_split_branch_prune_and_reset():
    manager = make_pressure_tree(chains=2, checkpoints=0)
    tree = manager.tree
    leaf = tree.leaves()[0]
    prefix = tree.split_at(leaf, 32)
    tokens = torch.cat([*(node.key for node in tree.path(prefix)),
                        torch.arange(10000, 10032, dtype=torch.int32)])
    indices = torch.cat([*(node.kv for node in tree.path(prefix)),
                         torch.arange(10000, 10032, dtype=torch.int32)])
    branch = tree.insert(tokens, indices, cache_len=len(tokens)).node
    assert branch.parent is prefix and leaf.parent is prefix
    tree.check_integrity()
    assert leaf.uuid in tree._leaves and branch.uuid in tree._leaves
    tree.evict_nodes([branch, leaf])
    tree.check_integrity()
    assert prefix.uuid in tree._leaves
    tree.reset()
    tree.check_integrity()
    assert not tree._leaves


def test_log_window_keeps_an_unselected_shared_branch_attached():
    manager = make_pressure_tree(checkpoints=0)
    tree = manager.tree
    leaves = sorted(tree.leaves(), key=lambda node: node.timestamp)
    for leaf in leaves:
        for node in tree.path(leaf)[:-1]:
            node.timestamp = 2000
    source = leaves[-1]
    branch_kv = source.kv[:32].clone()
    branch_entry = source.entries[0].slice(224, 256)
    removed = tree.evict_nodes([source])
    manager._cached_page_tokens.pop(int(removed.kv_indices[0]) // 64 * 64)
    prefix = tree.split_at(leaves[0], 32)
    prefix.timestamp = 0
    tokens = torch.cat([*(node.key for node in tree.path(prefix)),
                        torch.arange(100000, 100032, dtype=torch.int32)])
    indices = torch.cat([*(node.kv for node in tree.path(prefix)), branch_kv])
    branch = tree.insert(tokens, indices, (branch_entry,), cache_len=len(tokens)).node
    branch.timestamp = 1000
    manager._cached_page_tokens[int(branch_kv[0]) // 64 * 64] = 32
    tree.check_integrity()
    assert_full_plan_matches(manager, 4, 4)
    nodes, _pages, _slots, _compressed = manager._plan_eviction(4, 4, 0)
    assert leaves[0] in nodes and prefix not in nodes and branch not in nodes


def assert_log_index_matches_scan(tree, limit=8):
    expected = sorted((
        leaf for leaf in tree._leaves.values()
        if not leaf.is_locked and not any(
            entry.kind is LogStateKind.CHECKPOINT
            or entry.slot in tree._recovery_slot_refs for entry in leaf.entries
        )
    ), key=lambda node: (node.timestamp, node.uuid))
    nodes, _rows = tree.log_eviction_snapshot(limit)
    assert [node for node in nodes if node.is_leaf()] == expected[:limit]
    tree.check_integrity()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_log_index_tracks_access_locks_pins_and_splits(seed):
    manager = make_pressure_tree(chains=16, checkpoints=1)
    tree = manager.tree
    rng = Random(seed)
    for step in range(80):
        leaf = rng.choice(tree.leaves())
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
    expected = signature(manager._plan_eviction(1, 1, 0))

    class IndexedLeaves(dict):
        def values(self):
            raise AssertionError("eviction scanned every leaf")

    with patch.object(tree, "_leaves", IndexedLeaves(tree._leaves)):
        assert signature(manager._plan_eviction(1, 1, 0)) == expected


def test_log_index_removes_checkpoint_and_retired_leaf_records():
    manager = make_pressure_tree(chains=2, checkpoints=0)
    tree = manager.tree
    leaf = tree.leaves()[0]
    entry, = leaf.entries
    assert leaf.eviction_slots == (entry.slot,)
    tree.detach_entry(entry.slot)
    assert leaf.eviction_slots == ()
    checkpoint = LogStateEntry(entry.slot, LogStateKind.CHECKPOINT,
                               entry.begin, entry.end, span_pages=1)
    tree.attach_entry(leaf, checkpoint)
    assert leaf.eviction_slots == (entry.slot,)
    assert leaf._log_leaf_index == -1
    assert_log_index_matches_scan(tree)
    tree.detach_entry(entry.slot)
    tree.attach_entry(leaf, entry)
    assert_log_index_matches_scan(tree)
    for step in range(1000):
        leaf.timestamp = step
    assert len(tree._log_leaf_heap) == 2
    tree.evict_nodes([leaf])
    assert leaf._log_leaf_index == -1 and leaf._log_leaf_tree is None
    leaf.timestamp = 0
    assert_log_index_matches_scan(tree)
    previous = tree.leaves()
    tree.reset()
    for node in previous:
        node.timestamp = 0
        assert node._log_leaf_index == -1 and node._log_leaf_tree is None
    assert_log_index_matches_scan(tree)
