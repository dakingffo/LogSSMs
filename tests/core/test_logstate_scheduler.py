from collections import deque

import torch

from logssms.cache.logstate import (
    OnlineAnalysisScheduler,
    _MergePathContext,
)
from logssms.cache.prefix_cache.logstate import (
    LogRadixTree,
    LogRadixTreeNode,
    LogStateEntry,
    LogStateKind,
    LogStateMergeStats,
)


PAGE_SIZE = 64
GRANULARITY = 16


def ids(length: int, begin: int = 0) -> torch.Tensor:
    return torch.arange(begin, begin + length, dtype=torch.int32)


def make_history() -> tuple[LogRadixTree, torch.Tensor]:
    tree = LogRadixTree(
        PAGE_SIZE, torch.device("cpu"), granularity=GRANULARITY
    )
    input_ids = ids(256)
    entries = (
        LogStateEntry(
            0, LogStateKind.CHECKPOINT, 0, 64, span_pages=1
        ),
        LogStateEntry(1, LogStateKind.LOG, 64, 128),
        LogStateEntry(2, LogStateKind.LOG, 128, 192),
        LogStateEntry(3, LogStateKind.LOG, 192, 256),
    )
    tree.insert(input_ids, ids(256, 1000), entries)
    return tree, input_ids


def test_branch_inside_log_page_restores_work_state_and_protects_page():
    tree, history = make_history()
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)
    round_info = scheduler.begin_round(tree)
    request = torch.cat([history[:80], ids(32, 1000)])

    plan = scheduler.analyze(tree, [request], round_info)

    assert len(plan.protections) == 1
    assert plan.protections[0].position == 80
    assert plan.protections[0].log_slot == 1
    recovery = tree.match_prefix(request).cuda_handle.recovery
    assert recovery.position == 80
    assert recovery.base_checkpoint_slot == 0
    assert recovery.log_slots == (1,)
    assert plan.materialize == ()


def test_branch_at_checkpoint_uses_checkpoint_and_blocks_crossing_coalesce():
    tree, history = make_history()
    request = torch.cat([history[:64], ids(32, 1000)])

    handle = tree.match_prefix(request).cuda_handle
    assert handle.cached_len == 64
    assert handle.recovery.position == 64
    assert handle.recovery.base_checkpoint_slot == 0
    assert handle.recovery.logs == ()
    assert handle.recovery.conv_slot == 0
    assert handle.recovery.conv_granule == 3

    plan = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY).analyze(
        tree, [request]
    )
    assert len(plan.protections) == 1
    assert plan.protections[0].node is not None
    assert plan.protections[0].node.depth == 64
    assert not any(
        task.begin < 64 < task.end for task in plan.coalesce
    )


def test_protected_log_page_is_not_materialized_but_later_pages_are():
    tree, _history = make_history()
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)
    tree.protect(tree.entry_owner(1))

    plan = scheduler.analyze(tree, [_history])

    assert {task.log_slot for task in plan.materialize} == {2, 3}


def test_adjacent_page_checkpoints_coalesce_one_level_per_round():
    tree, _history = make_history()
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)
    leaf = tree.leaves()[0]
    tree.detach_entry(2)
    tree.detach_entry(3)
    tree.attach_entry(
        leaf,
        LogStateEntry(
            4, LogStateKind.CHECKPOINT, 128, 192, span_pages=1
        ),
    )
    tree.attach_entry(
        leaf,
        LogStateEntry(
            5, LogStateKind.CHECKPOINT, 192, 256, span_pages=1
        ),
    )

    plan = scheduler.analyze(tree, [_history])

    task = next(
        task
        for task in plan.coalesce
        if (task.left_slot, task.right_slot) == (4, 5)
    )
    assert (task.left_slot, task.right_slot) == (4, 5)
    assert (task.begin, task.end, task.next_span) == (128, 256, 2)


def test_materialized_checkpoints_coalesce_in_the_same_round():
    tree, _history = make_history()
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)

    plan = scheduler.analyze(tree, [_history])

    assert {task.log_slot for task in plan.materialize} == {1, 2, 3}
    assert {
        (task.left_slot, task.right_slot, task.next_span)
        for task in plan.coalesce
    } == {(0, 1, 2), (2, 3, 2)}


def test_internal_match_endpoint_plans_merges_only_on_the_hit_path():
    tree = LogRadixTree(
        PAGE_SIZE, torch.device("cpu"), granularity=GRANULARITY
    )
    shared = ids(4 * PAGE_SIZE)
    first = torch.cat([shared, ids(PAGE_SIZE, 10_000)])
    second = torch.cat([shared, ids(PAGE_SIZE, 20_000)])
    entries = tuple(
        LogStateEntry(
            slot,
            LogStateKind.CHECKPOINT,
            slot * PAGE_SIZE,
            (slot + 1) * PAGE_SIZE,
            span_pages=1,
        )
        for slot in range(5)
    )
    tree.insert(first, ids(len(first), 1000), entries)
    tree.insert(second, ids(len(second), 2000))
    endpoint = tree.trace(shared)
    assert endpoint.matched_len == len(shared)
    assert not endpoint.node.is_leaf()

    plan = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY).analyze(
        tree, [shared], collect_diagnostics=True
    )

    assert tuple(task.slots for task in plan.coalesce) == ((0, 1), (2, 3))
    diagnostics = dict(plan.merge_diagnostics)
    assert diagnostics["internal_endpoints"] == 1
    assert diagnostics["merge_anchors"] == 1
    assert diagnostics["planned_groups"] == 2
    assert diagnostics["planned_slot_releases"] == 2
    assert diagnostics["entries_beyond_match_endpoint"] == 0
    assert 4 not in {
        slot for task in plan.coalesce for slot in task.slots
    }


def test_adaptive_internal_endpoint_merges_without_an_owner():
    tree, leaf = make_checkpoint_history(4)
    leaf.merge_stats = LogStateMergeStats(origin=0, tail=256, intensity=4)
    shared = ids(128)
    branch = torch.cat([shared, ids(128, 10_000)])
    tree.insert(branch, ids(256, 20_000))
    endpoint = tree.trace(shared)
    assert not endpoint.node.is_leaf()
    assert endpoint.node.merge_stats is None

    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE, GRANULARITY, enable_merge_adjustment=True
    )
    plan = scheduler.analyze(tree, [shared])

    assert tuple(task.slots for task in plan.coalesce) == ((0, 1),)


def test_checkpoint_branch_remains_protected_in_later_batch():
    tree, _leaf = make_checkpoint_history(2)
    history = ids(2 * PAGE_SIZE)
    branch = torch.cat([history[:PAGE_SIZE], ids(PAGE_SIZE, 10_000)])
    tree.match_prefix(branch)
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)
    first = scheduler.analyze(tree, [branch])
    assert len(first.protections) == 1
    assert first.protections[0].node is not None
    tree.protect(first.protections[0].node)
    tree.insert(branch, ids(2 * PAGE_SIZE, 20_000))

    later = scheduler.analyze(tree, [history])

    assert not any(task.begin < PAGE_SIZE < task.end for task in later.coalesce)


def test_analysis_only_maintains_chains_touched_by_prefill_batch():
    tree = LogRadixTree(
        PAGE_SIZE, torch.device("cpu"), granularity=GRANULARITY
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
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)

    plan = scheduler.analyze(tree, [first])

    assert {task.log_slot for task in plan.materialize} == {0, 1}
    assert all(
        slot not in {2, 3}
        for task in plan.coalesce
        for slot in task.slots
    )


def test_empty_prefill_batch_does_not_trigger_tree_maintenance():
    tree, _history = make_history()

    plan = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY).analyze(tree, [])

    assert plan.materialize == ()
    assert plan.coalesce == ()


def test_active_handle_boundary_blocks_checkpoint_coalesce():
    tree, _history = make_history()
    leaf = tree.leaves()[0]
    handle = tree.make_handle(leaf, PAGE_SIZE)
    tree.lock_handle(handle)

    plan = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY).analyze(
        tree, [_history]
    )

    assert not any(
        task.left_slot == 0 and task.right_slot == 1
        for task in plan.coalesce
    )
    assert all(task.begin >= PAGE_SIZE for task in plan.coalesce)


def test_branch_between_checkpoints_blocks_coalesce_without_covering_log():
    tree = LogRadixTree(
        PAGE_SIZE, torch.device("cpu"), granularity=GRANULARITY
    )
    history = ids(128)
    tree.insert(
        history,
        ids(128, 1000),
        (
            LogStateEntry(
                0, LogStateKind.CHECKPOINT, 0, 64, span_pages=1
            ),
            LogStateEntry(
                1, LogStateKind.CHECKPOINT, 64, 128, span_pages=1
            ),
        ),
    )
    request = torch.cat([history[:96], ids(32, 1000)])

    plan = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY).analyze(
        tree, [request], collect_diagnostics=True
    )

    assert len(plan.protections) == 0
    assert tree.match_prefix(request).cuda_handle.cached_len == 64
    assert len(plan.coalesce) == 0
    diagnostics = dict(plan.merge_diagnostics)
    assert diagnostics["entries_beyond_match_endpoint"] == 1
    assert diagnostics["segments_without_merge_pair"] == 1
    assert diagnostics["planned_groups"] == 0


def test_new_log_pages_are_not_touched_in_their_birth_round():
    tree, _history = make_history()
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)
    round_info = scheduler.begin_round(tree)
    leaf = tree.leaves()[0]
    tree.detach_entry(3)
    tree.attach_entry(leaf, LogStateEntry(6, LogStateKind.LOG, 192, 256))

    plan = scheduler.analyze(tree, [_history], round_info)

    assert {task.log_slot for task in plan.materialize} == {1, 2}


def test_partial_physical_log_page_is_replay_only():
    tree = LogRadixTree(
        PAGE_SIZE, torch.device("cpu"), granularity=GRANULARITY
    )
    tree.insert(
        ids(80),
        ids(80, 1000),
        (
            LogStateEntry(
                0, LogStateKind.CHECKPOINT, 0, 64, span_pages=1
            ),
            LogStateEntry(
                1,
                LogStateKind.LOG,
                64,
                80,
                storage_begin=64,
            ),
        ),
        cache_len=80,
    )

    plan = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY).analyze(
        tree, [ids(80)]
    )

    assert plan.materialize == ()


def make_checkpoint_history(
    count: int,
) -> tuple[LogRadixTree, LogRadixTreeNode]:
    tree = LogRadixTree(
        PAGE_SIZE, torch.device("cpu"), granularity=GRANULARITY
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
                span_pages=1,
            )
            for slot in range(count)
        ),
    )
    return tree, tree.leaves()[0]


def test_cached_merge_path_checks_match_tree_checks():
    tree, leaf = make_checkpoint_history(6)
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)
    tree.protect_entry(2)
    branch = torch.cat([ids(80), ids(32, 10_000)])
    pending = scheduler._find_protections(tree, [branch])
    assert pending

    path = tree.path(leaf)
    context = _MergePathContext.build(
        tree, path, scheduler._entries_for_node(tree, leaf), pending,
    )
    for begin in range(0, 6 * PAGE_SIZE, GRANULARITY):
        for end in range(begin + GRANULARITY, 6 * PAGE_SIZE + 1, GRANULARITY):
            assert context.is_protected(begin, end) == scheduler._interval_is_protected(
                tree, leaf, begin, end, pending,
            )
    for position in range(0, 6 * PAGE_SIZE + 1, GRANULARITY):
        assert (position in context.lock_boundaries) == tree.is_lock_boundary(
            leaf, position,
        )


def test_adaptive_merge_uses_intensity_as_a_maximum_group_size():
    expected = {
        7: ((0, 1, 2, 3, 4, 5),),
        15: (tuple(range(8)), tuple(range(8, 14))),
        16: (tuple(range(8)), tuple(range(8, 16))),
    }
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE,
        GRANULARITY,
        enable_merge_adjustment=True,
    )

    for count, groups in expected.items():
        tree, leaf = make_checkpoint_history(count)
        leaf.merge_stats = LogStateMergeStats(
            origin=0,
            tail=count * PAGE_SIZE,
            intensity=8,
        )

        plan = scheduler.analyze(tree, [ids(count * PAGE_SIZE)])

        assert tuple(task.slots for task in plan.coalesce) == groups


def test_adaptive_merge_keeps_different_physical_spans_distinct():
    tree, leaf = make_checkpoint_history(10)
    for slot in range(10):
        tree.detach_entry(slot)
    tree.attach_entry(
        leaf, LogStateEntry(
            10, LogStateKind.CHECKPOINT, 0, 8 * PAGE_SIZE, span_pages=8
        )
    )
    tree.attach_entry(
        leaf, LogStateEntry(
            11, LogStateKind.CHECKPOINT,
            8 * PAGE_SIZE, 10 * PAGE_SIZE, span_pages=2,
        )
    )
    leaf.merge_stats = LogStateMergeStats(0, 10 * PAGE_SIZE, intensity=8)
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE, GRANULARITY, enable_merge_adjustment=True
    )

    assert scheduler.analyze(tree, [ids(10 * PAGE_SIZE)]).coalesce == ()


def test_adaptive_merge_only_changes_its_owned_chain_suffix():
    tree, leaf = make_checkpoint_history(8)
    leaf.merge_stats = LogStateMergeStats(
        origin=4 * PAGE_SIZE,
        tail=8 * PAGE_SIZE,
        intensity=4,
    )
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE,
        GRANULARITY,
        enable_merge_adjustment=True,
    )

    plan = scheduler.analyze(tree, [ids(8 * PAGE_SIZE)])

    assert tuple(task.slots for task in plan.coalesce) == ((4, 5, 6, 7),)


def test_adaptive_merge_absorbs_older_lower_level_checkpoint():
    tree, leaf = make_checkpoint_history(6)
    for slot in range(6):
        tree.detach_entry(slot)
    tree.attach_entry(
        leaf,
        LogStateEntry(6, LogStateKind.CHECKPOINT, 0, 64, span_pages=1),
    )
    tree.attach_entry(
        leaf,
        LogStateEntry(7, LogStateKind.CHECKPOINT, 64, 192, span_pages=2),
    )
    tree.attach_entry(
        leaf,
        LogStateEntry(8, LogStateKind.CHECKPOINT, 192, 384, span_pages=3),
    )
    leaf.merge_stats = LogStateMergeStats(0, 384, intensity=8)
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE,
        GRANULARITY,
        enable_merge_adjustment=True,
    )

    plan = scheduler.analyze(tree, [ids(6 * PAGE_SIZE)])

    assert tuple(task.slots for task in plan.coalesce) == ((6, 7),)
    assert plan.coalesce[0].next_span == 3


def test_merge_sampling_retires_completed_interval_hits():
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE,
        GRANULARITY,
        enable_merge_adjustment=True,
        min_sampling_len=1024,
    )
    stats = LogStateMergeStats(
        origin=0,
        tail=1023,
        hit_positions=deque((900, 100, 700)),
    )

    scheduler.advance_merge_stats(stats, 1024)

    assert tuple(stats.hit_positions) == ()


def test_merge_sampling_fits_log_log_slope_and_floors_intensity():
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE,
        GRANULARITY,
        enable_merge_adjustment=True,
        min_sampling_len=1024,
    )
    tail = 1024
    hits = []
    for index in range(8):
        distance = int((index + 0.5) * PAGE_SIZE)
        count = round(1e8 / distance ** 3.1)
        hits.extend([tail - distance] * count)
    stats = LogStateMergeStats(
        origin=0,
        tail=tail - 1,
        hit_positions=deque(hits),
    )

    scheduler.advance_merge_stats(stats, tail)

    assert stats.intensity == 3


def test_merge_sampling_multiplies_once_per_completed_interval():
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE,
        GRANULARITY,
        enable_merge_adjustment=True,
        min_sampling_len=128,
        max_merge_intensity=5,
        default_merge_growth_rate=2,
    )
    stats = LogStateMergeStats(origin=0, tail=0)

    scheduler.advance_merge_stats(stats, 64)
    assert stats.intensity == 2
    scheduler.advance_merge_stats(stats, 128)
    assert stats.intensity == 4
    scheduler.advance_merge_stats(stats, 192)
    assert stats.intensity == 4
    scheduler.advance_merge_stats(stats, 256)
    assert stats.intensity == 5


def test_merge_sampling_uses_complete_nonoverlapping_token_intervals():
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE, GRANULARITY,
        enable_merge_adjustment=True,
        min_sampling_len=1024,
        default_merge_growth_rate=2,
    )
    stats = scheduler.make_merge_stats(0)
    assert stats is not None
    scheduler.advance_merge_stats(stats, 1023)
    assert stats.intensity == 2
    scheduler.advance_merge_stats(stats, 1024)
    assert stats.intensity == 4
    scheduler.advance_merge_stats(stats, 1536)
    assert stats.intensity == 4
    scheduler.advance_merge_stats(stats, 2048)
    assert stats.intensity == 8
