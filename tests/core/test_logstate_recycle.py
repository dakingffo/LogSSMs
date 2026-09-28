import torch
from types import SimpleNamespace
from unittest.mock import patch
import random

from logssms.cache.logstate_recycle import LogStateRecycleBin
from logssms.cache.logstate_recycle import RecycledBranch, RecycledEntry, RecycledMatch
from logssms.cache.logstate import LogStateCacheManager, OnlineAnalysisScheduler, _LivePage
from logssms.cache.logstate_pool.frame_pool import FramedMHALogStatePool
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
    assert recycle.used_bytes == sum(branch.bytes for branch in recycle._branches.values())


def test_parent_refresh_discards_descendant_spanning_its_boundary() -> None:
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

    assert tuple(tokens.tolist()) not in recycle._branches
    assert recycle.used_bytes == sum(branch.bytes for branch in recycle._branches.values())


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


def test_recycled_checkpoint_shape_is_restored_from_new_logs() -> None:
    tree = make_tree()
    tokens = torch.arange(8, dtype=torch.int32)
    original = tree.make_handle(tree.root, 0)
    leaf = tree.insert(tokens, tokens + 100, (
        LogStateEntry(11, LogStateKind.LOG, 0, 4),
        LogStateEntry(12, LogStateKind.LOG, 4, 8),
    )).node
    manager = LogStateCacheManager.__new__(LogStateCacheManager)
    manager.prefix_cache = tree
    manager.device = torch.device("cpu")
    manager.page_size = 4
    manager.granularity = 1
    manager.analyzer = OnlineAnalysisScheduler(4, 1)
    manager.resources = SimpleNamespace(
        linear_state=FramedMHALogStatePool.__new__(FramedMHALogStatePool)
    )
    manager._recycle_matches = {0: RecycledMatch(
        RecycledBranch(
            tuple(tokens.tolist()),
            (RecycledEntry(LogStateKind.CHECKPOINT, 0, 8, 2),),
            frozenset(),
        ),
        8,
    )}
    manager._recycle_starts = {0: 0}
    manager._recycle_rebuilt_slots = {0: set()}
    submitted = []
    released = []
    manager._construct = lambda restores, plans: submitted.extend(plans)
    manager._return_slots = lambda pages, states: released.extend(states.tolist())
    req = SimpleNamespace(table_idx=0, cache_handle=original, input_ids=tokens)
    pages = (_LivePage(0, 4, 100, 11), _LivePage(4, 8, 104, 12))

    with patch.object(FramedMHALogStatePool, "materialize") as materialize:
        manager._restore_recycled_layout(req, leaf, 0, 8, 0, pages)

    assert len(submitted) == 1
    assert submitted[0].replay_log_slots == (11, 12)
    assert released == [11]
    materialize.assert_called_once_with(12)
    assert tree.matched_entries(leaf) == (
        LogStateEntry(12, LogStateKind.CHECKPOINT, 0, 8, span_pages=2),
    )
    tree.check_integrity()


def test_recycled_checkpoint_waits_for_later_prefill_chunk() -> None:
    tree = make_tree()
    tokens = torch.arange(8, dtype=torch.int32)
    first = tree.insert(tokens[:4], tokens[:4] + 100, (
        LogStateEntry(11, LogStateKind.LOG, 0, 4),
    )).node
    manager = LogStateCacheManager.__new__(LogStateCacheManager)
    manager.prefix_cache = tree
    manager.device = torch.device("cpu")
    manager.page_size = 4
    manager.granularity = 1
    manager.analyzer = OnlineAnalysisScheduler(4, 1)
    manager.resources = SimpleNamespace(
        linear_state=FramedMHALogStatePool.__new__(FramedMHALogStatePool)
    )
    manager._recycle_matches = {0: RecycledMatch(
        RecycledBranch(
            tuple(tokens.tolist()),
            (RecycledEntry(LogStateKind.CHECKPOINT, 0, 8, 2),),
            frozenset(),
        ),
        8,
    )}
    manager._recycle_starts = {0: 0}
    manager._recycle_rebuilt_slots = {0: set()}
    submitted = []
    manager._construct = lambda restores, plans: submitted.extend(plans)
    manager._return_slots = lambda pages, states: None
    first_req = SimpleNamespace(
        table_idx=0, cache_handle=tree.make_handle(tree.root, 0),
        input_ids=tokens[:4],
    )
    with patch.object(FramedMHALogStatePool, "materialize") as materialize:
        manager._restore_recycled_layout(
            first_req, first, 0, 4, 0, (_LivePage(0, 4, 100, 11),)
        )
        assert not submitted
        assert not materialize.called
        assert manager._unmaterialized_recycle_slots() == frozenset({11})
        second_handle = tree.make_handle(first, 4)
        second = tree.insert(tokens, tokens + 100, (
            LogStateEntry(12, LogStateKind.LOG, 4, 8),
        ), cache_len=8).node
        second_req = SimpleNamespace(
            table_idx=0, cache_handle=second_handle, input_ids=tokens,
        )
        manager._restore_recycled_layout(
            second_req, second, 4, 8, 4, (_LivePage(4, 8, 104, 12),)
        )
    assert len(submitted) == 1
    assert submitted[0].replay_log_slots == (11, 12)
    assert tree.matched_entries(second) == (
        LogStateEntry(12, LogStateKind.CHECKPOINT, 0, 8, span_pages=2),
    )
    assert not manager._unmaterialized_recycle_slots()
