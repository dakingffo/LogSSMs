from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from logssms.cache.logstate.manager import LogStateConstruction, RestoreTask
from logssms.cache.logstate.prefix_cache import LogStateEntry, LogStateKind


def make_pool():
    return SimpleNamespace(
        device=torch.device("cpu"),
        copy_states=Mock(),
        wait_slots_on_current_stream=Mock(),
        kernels=SimpleNamespace(restore=Mock(), copy_conv=Mock()),
    )


@pytest.mark.parametrize("mode", ["empty", "direct", "replay", "mixed"])
def test_restore_submits_independent_tasks_and_waits_on_dependencies(mode):
    pool = make_pool()
    construction = LogStateConstruction(pool)
    direct = RestoreTask(20, 10, (), 10, 2, 0)
    logs = (
        LogStateEntry(1, LogStateKind.LOG, 64, 96),
        LogStateEntry(2, LogStateKind.LOG, 96, 128),
    )
    replay = [RestoreTask(21 + i, None if i == 0 else 10, logs, 2, 1, 0) for i in range(5)]
    tasks = {
        "empty": [],
        "direct": [direct],
        "replay": replay,
        "mixed": [replay[0], direct, *replay[1:]],
    }[mode]
    expected = [
        (task.base_slot, [(1, 0, 32), (2, 0, 32)], task.output_slot)
        for task in tasks
        if task.logs
    ]
    with (
        patch.object(pool, "copy_states") as copy,
        patch.object(pool.kernels, "restore") as restore,
        patch.object(pool.kernels, "copy_conv") as conv,
    ):
        construction.restore(tasks)

    if tasks:
        pool.wait_slots_on_current_stream.assert_called_once()
        deps = set(pool.wait_slots_on_current_stream.call_args.args[0])
        for t in tasks:
            assert t.source_conv_slot in deps
            if t.base_slot is not None:
                assert t.base_slot in deps
            for e in t.logs:
                assert e.slot in deps
    else:
        pool.wait_slots_on_current_stream.assert_not_called()

    if expected:
        restore.assert_called_once_with(pool, expected)
    else:
        restore.assert_not_called()
    if direct in tasks:
        copy.assert_called_once_with([10], [20])
    else:
        copy.assert_not_called()
    if tasks:
        conv.assert_called_once()
        assert [vector.tolist() for vector in conv.call_args.args[1:]] == [
            [task.source_conv_slot for task in tasks],
            [task.source_conv_granule for task in tasks],
            [task.output_slot for task in tasks],
            [task.output_conv_granule for task in tasks],
        ]
    else:
        conv.assert_not_called()


def test_restore_rejects_checkpoint_copy_without_base():
    pool = make_pool()
    construction = LogStateConstruction(pool)
    with pytest.raises(AssertionError):
        construction.restore([RestoreTask(20, None, (), 10, 0, 0)])
    pool.copy_states.assert_not_called()
