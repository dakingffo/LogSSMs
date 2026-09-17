import torch

from logssms.cache.logstate import (
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
    assert plan.output_slots == 2


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

    assert len(plan.coalesce) == 1
    task = plan.coalesce[0]
    assert (task.left_slot, task.right_slot) == (4, 5)
    assert (task.begin, task.end, task.next_level) == (128, 256, 1)


def test_new_log_pages_are_not_touched_in_their_birth_round():
    tree, _history = make_history()
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)
    round_info = scheduler.begin_round(tree)
    leaf = tree.leaves()[0]
    tree.detach_entry(3)
    tree.attach_entry(leaf, LogStateEntry(6, LogStateKind.LOG, 192, 256))

    plan = scheduler.analyze(tree, [], round_info)

    assert {task.log_slot for task in plan.materialize} == {1, 2}
