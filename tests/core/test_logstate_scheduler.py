from collections import deque

import torch

from logssms.cache.logstate import (
    OnlineAnalysisScheduler,
    WorkStateRestoreMode,
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
            0, LogStateKind.CHECKPOINT, 0, 64, merge_level=0
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
    assert len(plan.restores) == 1
    assert plan.restores[0].mode is WorkStateRestoreMode.LOG_REPLAY
    assert plan.restores[0].replay_end == 80
    assert plan.restores[0].base_checkpoint_slot == 0
    assert plan.restores[0].log_slots == (1,)
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
    assert plan.protections == ()
    assert plan.restores == ()
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
            4, LogStateKind.CHECKPOINT, 128, 192, merge_level=0
        ),
    )
    tree.attach_entry(
        leaf,
        LogStateEntry(
            5, LogStateKind.CHECKPOINT, 192, 256, merge_level=0
        ),
    )

    plan = scheduler.analyze(tree, [_history])

    task = next(
        task
        for task in plan.coalesce
        if (task.left_slot, task.right_slot) == (4, 5)
    )
    assert (task.left_slot, task.right_slot) == (4, 5)
    assert (task.begin, task.end, task.next_level) == (128, 256, 1)


def test_materialized_checkpoints_coalesce_in_the_same_round():
    tree, _history = make_history()
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)

    plan = scheduler.analyze(tree, [_history])

    assert {task.log_slot for task in plan.materialize} == {1, 2, 3}
    assert {
        (task.left_slot, task.right_slot, task.next_level)
        for task in plan.coalesce
    } == {(0, 1, 1), (2, 3, 1)}


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
                0, LogStateKind.CHECKPOINT, 0, 64, merge_level=0
            ),
            LogStateEntry(
                1, LogStateKind.CHECKPOINT, 64, 128, merge_level=0
            ),
        ),
    )
    request = torch.cat([history[:96], ids(32, 1000)])

    plan = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY).analyze(
        tree, [request]
    )

    assert len(plan.protections) == 0
    assert len(plan.restores) == 1
    assert plan.restores[0].position == 96
    assert plan.restores[0].replay_end == 64
    assert plan.restores[0].mode is WorkStateRestoreMode.PREFILL
    assert len(plan.coalesce) == 0


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
                0, LogStateKind.CHECKPOINT, 0, 64, merge_level=0
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
                merge_level=0,
            )
            for slot in range(count)
        ),
    )
    return tree, tree.leaves()[0]


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
    tree, leaf = make_checkpoint_history(3)
    for slot in range(3):
        tree.detach_entry(slot)
    tree.attach_entry(
        leaf,
        LogStateEntry(3, LogStateKind.CHECKPOINT, 0, 64, merge_level=1),
    )
    tree.attach_entry(
        leaf,
        LogStateEntry(4, LogStateKind.CHECKPOINT, 64, 128, merge_level=2),
    )
    tree.attach_entry(
        leaf,
        LogStateEntry(5, LogStateKind.CHECKPOINT, 128, 192, merge_level=3),
    )
    leaf.merge_stats = LogStateMergeStats(0, 192, intensity=8)
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE,
        GRANULARITY,
        enable_merge_adjustment=True,
    )

    plan = scheduler.analyze(tree, [ids(3 * PAGE_SIZE)])

    assert tuple(task.slots for task in plan.coalesce) == ((3, 4),)
    assert plan.coalesce[0].next_level == 2


def test_merge_sampling_evicts_old_hits_regardless_of_queue_order():
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE,
        GRANULARITY,
        enable_merge_adjustment=True,
        min_sampling_len=1024,
    )
    stats = LogStateMergeStats(
        origin=0,
        tail=1024,
        hit_positions=deque((900, 100, 700)),
    )

    scheduler.advance_merge_stats(stats, 1200)

    assert tuple(stats.hit_positions) == (900, 700)


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
        tail=tail,
        hit_positions=deque(hits),
    )

    scheduler.record_hit(stats, tail)

    assert stats.intensity == 3


def test_merge_sampling_increases_intensity_when_no_fit_is_available():
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE,
        GRANULARITY,
        enable_merge_adjustment=True,
        min_sampling_len=128,
        max_merge_intensity=5,
        base_intensity_increment=2,
    )
    stats = LogStateMergeStats(origin=0, tail=0)

    scheduler.advance_merge_stats(stats, 64)
    assert stats.intensity == 2
    scheduler.advance_merge_stats(stats, 128)
    assert stats.intensity == 4
    scheduler.advance_merge_stats(stats, 192)
    assert stats.intensity == 5
