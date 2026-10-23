import random
from collections import deque

import torch

from logssms.cache.logstate.recycler import (
    LogStateRecycleBin,
    RecycledBranch,
    RecycledLog,
)
from logssms.cache.logstate.prefix_cache import (
    LogRadixTree,
    LogStateEntry,
    LogStateKind,
    LogStateMergeStats,
)
from tests.logstate_helpers import remember_log_tail


def make_tree() -> LogRadixTree:
    return LogRadixTree(4, torch.device("cpu"), granularity=1)


def insert_logs(tree, tokens, slot=1):
    return tree.insert(tokens, tokens + 100, tuple(
        LogStateEntry(slot + offset // 4, LogStateKind.LOG, offset, offset + 4)
        for offset in range(0, len(tokens), 4)
    )).node


def branch_size(tokens):
    return RecycledBranch(tuple(tokens.tolist()), (RecycledLog(0, len(tokens)),), frozenset()).bytes


def test_recycle_remembers_only_the_evicted_node_logs():
    tree = make_tree()
    tokens = torch.arange(16, dtype=torch.int32)
    leaf = tree.insert(tokens, tokens + 100, (
        LogStateEntry(1, LogStateKind.CHECKPOINT, 0, 8, span_pages=2),
        LogStateEntry(2, LogStateKind.LOG, 8, 12),
        LogStateEntry(3, LogStateKind.LOG, 12, 16),
    )).node
    recycle = LogStateRecycleBin(4096)
    recycle.remember(tree, leaf)
    hit = recycle.match(tokens, start=12, granularity=1)
    assert hit.matched_len == 16
    assert hit.branch.logs == (RecycledLog(12, 16),)
    assert all(not hasattr(log, "slot") and not hasattr(log, "kind") for log in hit.branch.logs)
    assert recycle.match(tokens[:12], start=0, granularity=1) is None
    assert recycle._native["usage"]()[0] == hit.branch.bytes


def test_gpu_checkpoint_compression_does_not_rewrite_a_log_record():
    tree = make_tree()
    tokens = torch.arange(12, dtype=torch.int32)
    leaf = tree.insert(tokens, tokens + 100, (
        LogStateEntry(1, LogStateKind.CHECKPOINT, 0, 4, span_pages=1),
        LogStateEntry(2, LogStateKind.CHECKPOINT, 4, 8, span_pages=1),
        LogStateEntry(3, LogStateKind.LOG, 8, 12),
    )).node
    recycle = LogStateRecycleBin(4096)
    recycle.remember(tree, leaf)
    old = recycle.match(tokens, start=8, granularity=1)
    revision = recycle.revision
    tree.evict_nodes((leaf,))
    tree.drop_checkpoint(1, 2)
    new = recycle.match(tokens, start=8, granularity=1)
    assert new.branch is old.branch and recycle.revision == revision
    assert new.branch.logs == (RecycledLog(8, 12),)
    assert tree.matched_entries(tree.trace(tokens[:8]).node)[0].span_pages == 2


def test_checkpoint_and_kv_only_nodes_create_no_records():
    tree = make_tree()
    first = torch.arange(4, dtype=torch.int32)
    checkpoint = tree.insert(first, first + 100, (
        LogStateEntry(1, LogStateKind.CHECKPOINT, 0, 4, span_pages=1),
    )).node
    kv_only = tree.insert(first + 10, first + 200).node
    recycle = LogStateRecycleBin(4096)
    recycle.remember(tree, checkpoint)
    recycle.remember(tree, kv_only)
    assert len(recycle) == recycle._native["usage"]()[0] == recycle.revision == 0


def test_recycle_is_byte_bounded_lru():
    tree = make_tree()
    first = torch.arange(4, dtype=torch.int32)
    second = first + 10
    recycle = LogStateRecycleBin(branch_size(first))
    recycle.remember(tree, insert_logs(tree, first))
    recycle.remember(tree, insert_logs(tree, second, 2))
    assert len(recycle) == 1
    assert recycle.match(first, start=0, granularity=1) is None
    assert recycle.match(second, start=0, granularity=1) is not None
    assert recycle._native["usage"]()[0] == branch_size(second)


def test_stale_match_cannot_discard_a_newer_record():
    tree = make_tree()
    tokens = torch.arange(4, dtype=torch.int32)
    node = insert_logs(tree, tokens)
    recycle = LogStateRecycleBin(4096)
    recycle.remember(tree, node)
    old = recycle.match(tokens, start=0, granularity=1)
    recycle.remember(tree, node)
    recycle.discard(old.branch)
    new = recycle.match(tokens, start=0, granularity=1)
    assert new.branch is not old.branch
    assert new.branch.logs == (RecycledLog(0, 4),)
    assert recycle._native["usage"]()[0] == new.branch.bytes
    recycle.discard(new.branch)
    assert len(recycle) == recycle._native["usage"]()[0] == 0


def test_native_recycle_hit_updates_lru_and_zero_capacity_stays_empty():
    tree = make_tree()
    inputs = [torch.arange(4, dtype=torch.int32) + i * 10 for i in range(3)]
    size = branch_size(inputs[0])
    recycle = LogStateRecycleBin(2 * size)
    disabled = LogStateRecycleBin(0)
    for index, values in enumerate(inputs[:2]):
        node = insert_logs(tree, values, index + 1)
        recycle.remember(tree, node)
        disabled.remember(tree, node)
    assert recycle.match(inputs[0].to(torch.int64), start=0, granularity=1) is not None
    recycle.remember(tree, insert_logs(tree, inputs[2], 3))
    assert recycle.match(inputs[1], start=0, granularity=1) is None
    assert recycle.match(inputs[0], start=0, granularity=1) is not None
    assert recycle.match(inputs[2], start=0, granularity=1) is not None
    assert recycle._native["usage"]()[0] == 2 * size and len(recycle) == 2
    assert disabled._native["usage"]()[0] == 0 and len(disabled) == 0


def test_reusing_a_recycled_match_touches_native_lru_without_rematching():
    tree = make_tree()
    inputs = [torch.arange(4, dtype=torch.int32) + i * 10 for i in range(3)]
    recycle = LogStateRecycleBin(2 * branch_size(inputs[0]))
    for index, values in enumerate(inputs[:2]):
        recycle.remember(tree, insert_logs(tree, values, index + 1))
    first = recycle.match(inputs[0], start=0, granularity=1)
    recycle.match(inputs[1], start=0, granularity=1)
    recycle.touch(first.branch)
    recycle.remember(tree, insert_logs(tree, inputs[2], 3))
    assert recycle.match(inputs[0], start=0, granularity=1) is not None
    assert recycle.match(inputs[1], start=0, granularity=1) is None


def test_evicted_parent_adds_logs_to_its_older_ghost_child():
    tree = make_tree()
    tokens = torch.arange(8, dtype=torch.int32)
    leaf = insert_logs(tree, tokens)
    parent = leaf.parent
    recycle = LogStateRecycleBin(4096)
    recycle.remember(tree, leaf)
    old = recycle.match(tokens, start=4, granularity=1)
    tree.evict_nodes((leaf,))
    recycle.remember(tree, parent)
    recycle.discard(old.branch)
    hit = recycle.match(tokens, start=0, granularity=1)
    assert hit.matched_len == 8 and hit.branch is not old.branch
    assert hit.branch.logs == (RecycledLog(0, 4), RecycledLog(4, 8))
    records = [hit.branch, recycle.match(tokens[:4], start=0, granularity=1).branch]
    assert recycle._native["usage"]()[0] == sum(branch.bytes for branch in records)


def test_parent_adds_logs_only_to_its_recycled_descendants():
    tree = make_tree()
    prefix = torch.arange(4, dtype=torch.int32)
    first = torch.cat((prefix, prefix + 10))
    second = torch.cat((prefix, prefix + 20))
    other = prefix + 100
    first_node = insert_logs(tree, first)
    second_node = tree.insert(second, second + 200, (
        LogStateEntry(3, LogStateKind.LOG, 4, 8),
    )).node
    other_node = insert_logs(tree, other, 4)
    recycle = LogStateRecycleBin(16 * 1024)
    for node in (first_node, second_node, other_node):
        recycle.remember(tree, node)
    other_before = recycle.match(other, start=0, granularity=1)
    recycle.remember(tree, first_node.parent)
    for tokens in (first, second):
        hit = recycle.match(tokens, start=0, granularity=1)
        assert hit.matched_len == 8
        assert hit.branch.logs == (RecycledLog(0, 4), RecycledLog(4, 8))
    assert recycle.match(other, start=0, granularity=1).branch is other_before.branch


def test_half_page_log_remembers_only_the_pruned_suffix():
    tree = make_tree()
    tokens = torch.arange(4, dtype=torch.int32)
    leaf = insert_logs(tree, tokens)
    tree.split_at(leaf, 2)
    recycle = LogStateRecycleBin(4096)
    recycle.remember(tree, leaf)
    assert recycle.match(tokens[:2], start=0, granularity=1) is None
    hit = recycle.match(tokens, start=2, granularity=1)
    assert hit.branch.logs == (RecycledLog(2, 4),)
    tree.evict_nodes((leaf,))
    assert tree.match_prefix(tokens).recoverable_len == 2


def test_match_can_find_a_neighbour_when_the_nearest_record_has_no_matching_logs():
    recycle = LogStateRecycleBin(4096)
    tree = make_tree()
    first = torch.tensor([0, 1, 0, 0], dtype=torch.int32)
    recycle.remember(tree, insert_logs(tree, first))
    nearest = torch.tensor([0, 1, 1, 0, 2, 2, 2, 2], dtype=torch.int32)
    other_tree = make_tree()
    node = insert_logs(other_tree, nearest, 2)
    recycle.remember(other_tree, node)
    query = torch.tensor([0, 1, 1, 1], dtype=torch.int32)
    hit = recycle.match(query, start=0, granularity=1)
    assert hit.matched_len == 2 and hit.branch.tokens == tuple(first.tolist())


def test_sorted_recycle_index_finds_longest_prefix():
    rng = random.Random(17)
    recycle = LogStateRecycleBin(1_000_000)
    keys = []
    for _ in range(60):
        key = tuple(rng.randrange(5) for _ in range(rng.choice((4, 8, 12))))
        if key in keys:
            continue
        keys.append(key)
        tree = make_tree()
        tokens = torch.tensor(key, dtype=torch.int32)
        remember_log_tail(recycle, tree, insert_logs(tree, tokens))
    for _ in range(100):
        query = tuple(rng.randrange(5) for _ in range(12))
        hit = recycle.match(torch.tensor(query, dtype=torch.int32), start=0, granularity=1)
        expected = max(
            next((i for i, (left, right) in enumerate(zip(key, query)) if left != right),
                 min(len(key), len(query)))
            for key in keys
        )
        assert (0 if hit is None else hit.matched_len) == expected


def test_checkpoint_or_kv_only_snapshot_does_not_erase_remembered_logs():
    tree = make_tree()
    tokens = torch.arange(4, dtype=torch.int32)
    leaf = insert_logs(tree, tokens)
    recycle = LogStateRecycleBin(4096)
    recycle.remember(tree, leaf)
    old = recycle.match(tokens, start=0, granularity=1)
    tree.detach_entry(1)
    recycle.remember(tree, leaf)
    tree.attach_entry(leaf, LogStateEntry(1, LogStateKind.CHECKPOINT, 0, 4, span_pages=1))
    recycle.remember(tree, leaf)
    assert recycle.match(tokens, start=0, granularity=1).branch is old.branch


def test_recycled_owner_statistics_are_immutable_cpu_metadata():
    tree = make_tree()
    tokens = torch.arange(8, dtype=torch.int32)
    leaf = insert_logs(tree, tokens)
    leaf.merge_stats = LogStateMergeStats(0, 8, 6, deque([2, 4]), 12)
    recycle = LogStateRecycleBin(4096)
    recycle.remember(tree, leaf)
    leaf.merge_stats.hit_positions.clear()
    leaf.merge_stats.intensity = 2
    hit = recycle.match(tokens, start=4, granularity=1)
    assert hit.branch.merge_stats == (0, 8, 6, 12, 2, 4)
    assert recycle._native["usage"]()[0] == hit.branch.bytes
