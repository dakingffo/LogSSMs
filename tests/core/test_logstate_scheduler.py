import random
from unittest.mock import patch

import torch

from logssms.cache.logstate.online_analysis_scheduler import (
    OnlineAnalysisScheduler,
)
from logssms.cache.logstate.prefix_cache import (
    LogRadixTree,
    LogRadixTreeNode,
    LogStateEntry,
    LogStateKind,
)


PAGE_SIZE = 64
GRANULARITY = 64


def test_batch_branch_points_match_granule_trie_oracle():
    rng = random.Random(711)
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, 1, enable_hit_materialization=True)
    tree = LogRadixTree(PAGE_SIZE, torch.device("cpu"))
    for _ in range(100):
        paths = []
        for _ in range(rng.randint(1, 24)):
            parent = rng.choice(paths) if paths else ()
            shared = parent[:rng.randrange(len(parent) + 1)]
            path = shared + tuple(rng.randrange(4) for _ in range(rng.randrange(8)))
            paths.append(path)
        inputs = [torch.tensor(
            [value for value in path for _ in range(GRANULARITY)]
            + [9] * rng.randrange(GRANULARITY), dtype=torch.int32,
        ) for path in paths]
        expected = {}
        prefixes = {path[:end] for path in paths for end in range(1, len(path) + 1)}
        for prefix in prefixes:
            indices = {i for i, path in enumerate(paths) if path[:len(prefix)] == prefix}
            terminals = sum(paths[i] == prefix for i in indices)
            children = {paths[i][len(prefix)] for i in indices if len(paths[i]) > len(prefix)}
            if len(children) + int(terminals > 0) >= 2 or terminals >= 2:
                tokens = tuple(value for value in prefix for _ in range(GRANULARITY))
                expected[tokens] = indices
        actual = {
            tuple(inputs[value.request_indices[0]][:value.position].tolist()): set(value.request_indices)
            for value in scheduler._find_protections(tree, inputs)
        }
        assert actual == expected


def ids(length: int, begin: int = 0) -> torch.Tensor:
    return torch.arange(begin, begin + length, dtype=torch.int32)


def make_history() -> tuple[LogRadixTree, torch.Tensor]:
    tree = LogRadixTree(
        PAGE_SIZE, torch.device("cpu"),
    )
    input_ids = ids(256)
    entries = (
        LogStateEntry(
            0, LogStateKind.CHECKPOINT, 0, 64, span_granules=1
        ),
        LogStateEntry(1, LogStateKind.LOG, 64, 128),
        LogStateEntry(2, LogStateKind.LOG, 128, 192),
        LogStateEntry(3, LogStateKind.LOG, 192, 256),
    )
    tree.insert(input_ids, ids(256, 1000), entries)
    return tree, input_ids


def test_attach_many_logs_does_not_recopy_mapping_suffixes():
    tree = LogRadixTree(PAGE_SIZE, torch.device("cpu"))
    tokens = ids(8192)
    leaf = tree.insert(tokens, tokens).node
    with patch.object(LogRadixTreeNode, "initialize", side_effect=AssertionError("mapping copied")):
        for i in range(128):
            tree.attach_entry(leaf, LogStateEntry(i, LogStateKind.LOG, i * 64, (i + 1) * 64))
    assert len({node.kv.untyped_storage().data_ptr() for node in tree.path(leaf)}) == 1
    tree.check_integrity()


def test_insert_accepts_cpu_page_spans_without_reading_mapping_back():
    tree = LogRadixTree(PAGE_SIZE, torch.device("cpu"))
    tokens = ids(128)
    mapping = torch.cat((ids(64, 640), ids(64, 1280)))
    with patch.object(torch.Tensor, "to", side_effect=AssertionError("mapping readback")):
        leaf = tree.insert(tokens, mapping, kv_frame_spans=[(0, 64, 10), (64, 128, 20)]).node
    assert leaf._kv_frame_counts == {10: 64, 20: 64}
    tree.check_integrity()


def test_edge_split_does_not_protect_a_node_without_a_branch():
    tree, _history = make_history()
    leaf = tuple(tree._leaves.values())[0]
    prefix = tree.split_at(leaf, GRANULARITY)
    assert len(prefix.children) == 1
    assert not prefix.is_protected
    assert not leaf.is_protected


def test_trace_prefix_matches_a_fresh_walk_at_all_recovery_boundaries():
    tree, history = make_history()
    trace = tree.trace(history)
    for position in range(0, len(history) + 1, GRANULARITY):
        assert tree.trace_prefix(trace, position) == tree.trace(history[:position])


def test_checkpoint_branch_remains_protected_in_later_batch():
    tree, _leaf = make_checkpoint_history(2)
    history = ids(2 * PAGE_SIZE)
    branch = torch.cat([history[:PAGE_SIZE], ids(PAGE_SIZE, 10_000)])
    tree.match_prefix(branch)
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, 1, enable_hit_materialization=True)
    first = scheduler.analyze(tree, [branch])
    assert len(first.protections) == 1
    assert first.protections[0].node is not None
    tree.protect(first.protections[0].node)
    tree.insert(branch, ids(2 * PAGE_SIZE, 20_000))

    scheduler.analyze(tree, [history])

    assert tree.trace(history[:PAGE_SIZE]).node.is_protected
    actions = tree.checkpoint_eviction_actions(tree.eviction_snapshot()[0])
    assert all(right < 0 for left, right, *_ in actions if left == 0)


def test_analysis_only_maintains_chains_touched_by_prefill_batch():
    tree = LogRadixTree(
        PAGE_SIZE, torch.device("cpu"),
    )
    first = ids(128)
    second = ids(128, 1000)
    tree.insert(
        first,
        ids(128, 2000),
        (
            LogStateEntry(0, LogStateKind.LOG, 0, 64),
            LogStateEntry(1, LogStateKind.LOG, 64, 128),
        ),
    )
    tree.insert(
        second,
        ids(128, 3000),
        (
            LogStateEntry(2, LogStateKind.LOG, 0, 64),
            LogStateEntry(3, LogStateKind.LOG, 64, 128),
        ),
    )
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, 1, enable_hit_materialization=True)

    plan = scheduler.analyze(tree, [first])

    assert {task.right_slot for task in plan.materialize} == {0, 1}
    assert all(
        slot not in {2, 3}
        for task in plan.materialize
        for slot in task.slots
    )


def test_empty_prefill_batch_does_not_trigger_tree_maintenance():
    tree, _history = make_history()

    plan = OnlineAnalysisScheduler(PAGE_SIZE, 1, enable_hit_materialization=True).analyze(tree, [])

    assert plan.materialize == ()
    assert plan.protections == ()


def test_prefill_snapshot_excludes_rebound_log_slots():
    tree, _history = make_history()
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, 1, enable_hit_materialization=True)
    snapshot = scheduler.snapshot_prefill(tree, [tree.trace(_history)])
    leaf = tuple(tree._leaves.values())[0]
    tree.detach_entry(3)
    tree.attach_entry(leaf, LogStateEntry(6, LogStateKind.LOG, 192, 256))

    plan = scheduler.analyze(tree, [_history], snapshot)

    assert {task.right_slot for task in plan.materialize} == {1, 2}


def make_checkpoint_history(
    count: int,
) -> tuple[LogRadixTree, LogRadixTreeNode]:
    tree = LogRadixTree(
        PAGE_SIZE, torch.device("cpu"),
    )
    length = count * PAGE_SIZE
    tree.insert(
        ids(length),
        ids(length, 1000),
        tuple(
            LogStateEntry(
                slot,
                LogStateKind.CHECKPOINT,
                slot * PAGE_SIZE,
                (slot + 1) * PAGE_SIZE,
                span_granules=1,
            )
            for slot in range(count)
        ),
    )
    return tree, tuple(tree._leaves.values())[0]


def test_analysis_matches_each_request_only_once():
    tree, history = make_history()
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, 1, enable_hit_materialization=True)
    request = torch.cat([history, ids(64, 10000)])
    with patch.object(tree, "trace", wraps=tree.trace) as trace:
        plan = scheduler.analyze(tree, [request])
    assert trace.call_count == 1
    assert len(plan.materialize) == 3


def test_prefill_snapshot_contains_only_matched_paths():
    tree, history = make_history()
    # Matching splits the radix edge without fragmenting granule-sized Logs.
    tree.match_prefix(torch.cat([history[:80], ids(32, 10000)]))
    expected = frozenset(entry.slot for entry in tree.iter_entries())
    assert tree.entry_for_slot(1).end - tree.entry_for_slot(1).begin == GRANULARITY
    tree.insert(ids(64, 20000), ids(64, 10000),
                (LogStateEntry(999, LogStateKind.LOG, 0, 64),))
    with patch.object(tree, "iter_entries", side_effect=AssertionError):
        actual = OnlineAnalysisScheduler(PAGE_SIZE, 1, enable_hit_materialization=True).snapshot_prefill(tree, [tree.trace(history)])
    assert actual.eligible_slots == expected
