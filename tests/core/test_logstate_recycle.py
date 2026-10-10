import torch
import random

from logssms.cache.logstate import LogStateRecycleBin
from logssms.cache.logstate import RecycledBranch
from logssms.cache.prefix_cache.logstate import LogRadixTree, LogStateEntry, LogStateKind


def make_tree() -> LogRadixTree:
    return LogRadixTree(4, torch.device("cpu"), granularity=1)


def test_recycle_remembers_shape_without_device_slots() -> None:
    tree = make_tree()
    tokens = torch.arange(12, dtype=torch.int32)
    node = tree.insert(
        tokens, tokens + 100,
        (LogStateEntry(15, LogStateKind.CHECKPOINT, 0, 8, span_pages=2),
         LogStateEntry(16, LogStateKind.LOG, 8, 12)),
    ).node
    recycle = LogStateRecycleBin(4096)
    recycle.remember(tree, node)
    tree.evict_nodes(tuple(tree.iter_evict_candidates()))

    hit = recycle.match(torch.cat([tokens, tokens[:2]]), start=0, granularity=1)
    assert hit is not None and hit.matched_len == 12
    assert [(entry.kind, entry.begin, entry.end) for entry in hit.branch.entries] == [
        (LogStateKind.CHECKPOINT, 0, 8),
        (LogStateKind.LOG, 8, 12),
    ]
    assert all(not hasattr(entry, "slot") for entry in hit.branch.entries)


def test_recycle_is_byte_bounded_lru() -> None:
    tree = make_tree()
    first = torch.arange(4, dtype=torch.int32)
    second = first + 10
    one_branch = RecycledBranch(tuple(first.tolist()), (), frozenset())
    recycle = LogStateRecycleBin(one_branch.bytes + 16)
    recycle.remember(tree, tree.insert(first, first + 100).node)
    recycle.remember(tree, tree.insert(second, second + 200).node)
    assert len(recycle) == 1
    assert recycle.match(first, start=0, granularity=1) is None
    assert recycle.match(second, start=0, granularity=1) is not None


def test_discard_does_not_remove_a_newer_shape_for_same_prefix() -> None:
    tree = make_tree()
    tokens = torch.arange(4, dtype=torch.int32)
    node = tree.insert(tokens, tokens + 100, (
        LogStateEntry(1, LogStateKind.LOG, 0, 4),
    )).node
    recycle = LogStateRecycleBin(4096)
    recycle.remember(tree, node)
    old = recycle.match(tokens, start=0, granularity=1)
    assert old is not None
    tree.detach_entry(1)
    tree.attach_entry(node, LogStateEntry(
        1, LogStateKind.CHECKPOINT, 0, 4, span_pages=1
    ))
    recycle.remember(tree, node)
    recycle.discard(old.branch)
    new = recycle.match(tokens, start=0, granularity=1)
    assert new is not None
    assert new.branch.entries[0].kind is LogStateKind.CHECKPOINT


def test_native_recycle_hit_updates_lru_and_zero_capacity_stays_empty() -> None:
    tree = make_tree()
    inputs = [torch.arange(4, dtype=torch.int32) + i * 10 for i in range(3)]
    size = RecycledBranch(tuple(inputs[0].tolist()), (), frozenset()).bytes
    recycle = LogStateRecycleBin(2 * size)
    disabled = LogStateRecycleBin(0)
    for values in inputs[:2]:
        node = tree.insert(values, values + 100).node
        recycle.remember(tree, node)
        disabled.remember(tree, node)
    assert recycle.match(inputs[0].to(torch.int64), start=0, granularity=1) is not None
    recycle.remember(tree, tree.insert(inputs[2], inputs[2] + 100).node)
    assert recycle.match(inputs[1], start=0, granularity=1) is None
    assert recycle.match(inputs[0], start=0, granularity=1) is not None
    assert recycle.match(inputs[2], start=0, granularity=1) is not None
    assert recycle.used_bytes == 2 * size and len(recycle) == 2
    assert disabled.used_bytes == 0 and len(disabled) == 0
    assert disabled.match(inputs[0], start=0, granularity=1) is None


def test_reusing_a_recycled_match_touches_native_lru_without_rematching():
    tree = make_tree()
    inputs = [torch.arange(4, dtype=torch.int32) + i * 10 for i in range(3)]
    size = RecycledBranch(tuple(inputs[0].tolist()), (), frozenset()).bytes
    recycle = LogStateRecycleBin(2 * size)
    for values in inputs[:2]:
        recycle.remember(tree, tree.insert(values, values + 100).node)
    first = recycle.match(inputs[0], start=0, granularity=1)
    recycle.match(inputs[1], start=0, granularity=1)
    recycle.touch(first.branch)
    recycle.remember(tree, tree.insert(inputs[2], inputs[2] + 100).node)
    assert recycle.match(inputs[0], start=0, granularity=1) is not None
    assert recycle.match(inputs[1], start=0, granularity=1) is None


def test_stale_child_handle_cannot_discard_a_refreshed_record() -> None:
    tree = make_tree()
    tokens = torch.arange(8, dtype=torch.int32)
    leaf = tree.insert(tokens, tokens + 100, (
        LogStateEntry(1, LogStateKind.LOG, 0, 4),
        LogStateEntry(2, LogStateKind.LOG, 4, 8),
    )).node
    recycle = LogStateRecycleBin(4096)
    recycle.remember(tree, leaf)
    old = recycle.match(tokens, start=0, granularity=1)
    parent = tree.entry_owner(1)
    tree.detach_entry(1)
    tree.attach_entry(parent, LogStateEntry(1, LogStateKind.CHECKPOINT, 0, 4, span_pages=1))
    recycle.remember(tree, parent)
    recycle.discard(old.branch)
    new = recycle.match(tokens, start=0, granularity=1)
    assert new.matched_len == 8 and new.branch is not old.branch
    assert new.branch.entries[0].kind is LogStateKind.CHECKPOINT
    assert len(recycle) == 2


def test_evicted_parent_refreshes_older_ghost_child_prefix() -> None:
    tree = make_tree()
    tokens = torch.arange(8, dtype=torch.int32)
    leaf = tree.insert(tokens, tokens + 100, (
        LogStateEntry(1, LogStateKind.LOG, 0, 4),
        LogStateEntry(2, LogStateKind.LOG, 4, 8),
    )).node
    parent = tree.entry_owner(1)
    recycle = LogStateRecycleBin(4096)
    recycle.remember(tree, leaf)
    tree.evict_nodes((leaf,))
    tree.detach_entry(1)
    tree.attach_entry(parent, LogStateEntry(
        1, LogStateKind.CHECKPOINT, 0, 4, span_pages=1
    ))
    recycle.remember(tree, parent)
    hit = recycle.match(tokens, start=0, granularity=1)
    assert hit is not None and hit.matched_len == 8
    assert [(entry.kind, entry.begin, entry.end) for entry in hit.branch.entries] == [
        (LogStateKind.CHECKPOINT, 0, 4),
        (LogStateKind.LOG, 4, 8),
    ]


def test_parent_refreshes_only_its_recycled_descendants() -> None:
    tree = make_tree()
    prefix = torch.arange(4, dtype=torch.int32)
    first = torch.cat((prefix, prefix + 10))
    second = torch.cat((prefix, prefix + 20))
    other = torch.arange(4, dtype=torch.int32) + 100
    first_node = tree.insert(first, first + 200, (
        LogStateEntry(1, LogStateKind.LOG, 0, 4),
        LogStateEntry(2, LogStateKind.LOG, 4, 8),
    )).node
    second_node = tree.insert(second, second + 200, (
        LogStateEntry(3, LogStateKind.LOG, 4, 8),
    )).node
    other_node = tree.insert(other, other + 200, (
        LogStateEntry(4, LogStateKind.LOG, 0, 4),
    )).node
    parent = tree.entry_owner(1)
    recycle = LogStateRecycleBin(16 * 1024)
    for node in (first_node, second_node, other_node):
        recycle.remember(tree, node)
    other_before = recycle.match(other, start=0, granularity=1)
    assert other_before is not None

    tree.detach_entry(1)
    tree.attach_entry(parent, LogStateEntry(
        1, LogStateKind.CHECKPOINT, 0, 4, span_pages=1
    ))
    recycle.remember(tree, parent)

    for tokens in (first, second):
        hit = recycle.match(tokens, start=0, granularity=1)
        assert hit is not None and hit.matched_len == 8
        assert [(entry.kind, entry.begin, entry.end) for entry in hit.branch.entries] == [
            (LogStateKind.CHECKPOINT, 0, 4),
            (LogStateKind.LOG, 4, 8),
        ]
    assert recycle.match(other, start=0, granularity=1).branch is other_before.branch
    records = [recycle.match(tokens, start=0, granularity=1).branch
               for tokens in (first, second, other, prefix)]
    assert recycle.used_bytes == sum(branch.bytes for branch in records)


def test_short_parent_snapshot_preserves_a_complete_checkpoint_branch() -> None:
    tokens = torch.arange(8, dtype=torch.int32)
    old_tree = make_tree()
    leaf = old_tree.insert(tokens, tokens + 100, (
        LogStateEntry(1, LogStateKind.CHECKPOINT, 0, 8, span_pages=2),
    )).node
    recycle = LogStateRecycleBin(4096)
    recycle.remember(old_tree, leaf)

    new_tree = make_tree()
    parent = new_tree.insert(tokens[:4], tokens[:4] + 200, (
        LogStateEntry(2, LogStateKind.LOG, 0, 4),
    )).node
    recycle.remember(new_tree, parent)

    hit = recycle.match(tokens, start=0, granularity=1)
    assert hit is not None and hit.matched_len == 8
    assert hit.branch.entries[0].end == 8
    assert len(recycle) == 2


def test_sorted_recycle_index_finds_longest_prefix() -> None:
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
        recycle.remember(tree, tree.insert(tokens, tokens + 100).node)
    for _ in range(100):
        query = tuple(rng.randrange(5) for _ in range(12))
        hit = recycle.match(torch.tensor(query, dtype=torch.int32), start=0, granularity=1)
        expected = max(
            next((i for i, (left, right) in enumerate(zip(key, query))
                  if left != right), min(len(key), len(query)))
            for key in keys
        )
        assert (0 if hit is None else hit.matched_len) == expected


def test_kv_only_snapshot_does_not_erase_a_remembered_layout():
    tokens = torch.arange(8, dtype=torch.int32)
    tree = make_tree()
    leaf = tree.insert(tokens, tokens + 100, (
        LogStateEntry(1, LogStateKind.CHECKPOINT, 0, 8, span_pages=2),
    )).node
    recycle = LogStateRecycleBin(4096)
    recycle.remember(tree, leaf)
    tree.detach_entry(1)
    recycle.remember(tree, leaf)
    hit = recycle.match(tokens, start=0, granularity=1)
    assert hit is not None and hit.branch.entries[0].span_pages == 2


def test_recycled_owner_statistics_are_immutable_cpu_metadata():
    from collections import deque
    from logssms.cache.prefix_cache.logstate import LogStateMergeStats

    tokens = torch.arange(8, dtype=torch.int32)
    tree = make_tree()
    leaf = tree.insert(tokens, tokens + 100).node
    leaf.merge_stats = LogStateMergeStats(0, 8, 6, deque([2, 4]), 12)
    recycle = LogStateRecycleBin(4096)
    recycle.remember(tree, leaf)
    leaf.merge_stats.hit_positions.clear()
    leaf.merge_stats.intensity = 2
    hit = recycle.match(tokens, start=0, granularity=1)
    assert hit.branch.merge_stats == (0, 8, 6, 12, 2, 4)
    assert recycle.used_bytes == hit.branch.bytes
