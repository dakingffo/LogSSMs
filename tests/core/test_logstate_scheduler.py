import torch

from logssms.cache.logstate import (
    LogStateCacheManager,
    OnlineAnalysisScheduler,
    WorkStateRestoreMode,
)
from logssms.cache.prefix_cache.logstate import (
    LogRadixTree,
    LogStateEntry,
    LogStateKind,
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
    assert {task.log_slot for task in plan.materialize} == {2, 3}

    chains = LogStateCacheManager._make_materialize_chains(plan.materialize)
    assert len(chains) == 1
    assert tuple(entry.slot for entry in chains[0].logs) == (1, 2, 3)
    assert chains[0].checkpoint_slots == {2, 3}


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

    plan = scheduler.analyze(tree, [])

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

    plan = scheduler.analyze(tree, [])

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

    plan = scheduler.analyze(tree, [])

    assert {task.log_slot for task in plan.materialize} == {1, 2, 3}
    assert {
        (task.left_slot, task.right_slot, task.next_level)
        for task in plan.coalesce
    } == {(0, 1, 1), (2, 3, 1)}


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

    plan = scheduler.analyze(tree, [], round_info)

    assert {task.log_slot for task in plan.materialize} == {1, 2}
