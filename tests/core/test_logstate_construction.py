from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from logssms.cache.logstate_construction import LogStateConstruction, RestoreTask
from logssms.cache.prefix_cache.logstate import LogStateEntry, LogStateKind


def make_pool():
    # No manager, tree, allocator, or device allocation is needed for submission.
    return SimpleNamespace(
        device=torch.device("cpu"),
        copy_states=Mock(),
        kernels=SimpleNamespace(restore=Mock(), copy_conv=Mock()),
    )


@pytest.mark.parametrize("mode", ["empty", "direct", "replay", "mixed"])
@pytest.mark.parametrize("profile", [False, True])
def test_restore_submits_independent_tasks_without_planner(mode, profile):
    pool = make_pool()
    construction = LogStateConstruction(pool)
    direct = RestoreTask(20, 10, (), 10, 2, 0)
    # Sliced entries must retain their physical storage offsets. Shared sources
    # still produce independent work outputs, including batches larger than four.
    logs = (
        LogStateEntry(1, LogStateKind.LOG, 80, 128, storage_begin=64),
        LogStateEntry(2, LogStateKind.LOG, 128, 144),
    )
    replay = [RestoreTask(21 + i, None if i == 0 else 10, logs, 2, 1, 0)
              for i in range(5)]
    tasks = {"empty": [], "direct": [direct], "replay": replay,
             "mixed": [replay[0], direct, *replay[1:]]}[mode]
    expected = [(task.base_slot, [(1, 16, 64), (2, 0, 16)], task.output_slot)
                for task in tasks if task.logs]
    with (
        patch("logssms.kernel.logstate._load_logstate_module",
              side_effect=AssertionError("restore submission must not invoke planning")),
        patch.object(pool, "copy_states") as copy,
        patch.object(pool.kernels, "restore") as restore,
        patch.object(pool.kernels, "copy_conv") as conv,
    ):
        stats = construction.restore(
            tasks, profile_enabled=profile, pending_cuda_events=[], record_cuda=False,
        )
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
    assert stats["chains"] == stats["outputs"] == len(expected)
    assert stats["segments"] == 2 * len(expected)
    assert stats["waves"] == bool(expected)
    assert stats["direct_checkpoint_copies"] == (direct in tasks)
    assert stats["cuda_ms"] == (None if expected else 0.0)
    assert ("submit_timing_ms" in stats) == profile


@pytest.mark.parametrize("profile,record_cuda,has_tasks,expected_events", [
    (True, True, True, 2),
    (True, False, True, 0),
    (False, True, True, 0),
    (True, True, False, 0),
])
def test_restore_records_events_only_when_requested(profile, record_cuda, has_tasks, expected_events):
    pool = make_pool()
    construction = LogStateConstruction(pool)
    pending = []
    tasks = [RestoreTask(20, 10, (), 10, 2, 0)] if has_tasks else []
    start, end = Mock(), Mock()
    order = []
    start.record.side_effect = lambda: order.append("start")
    end.record.side_effect = lambda: order.append("end")
    pool.copy_states.side_effect = lambda *_: order.append("state")
    pool.kernels.copy_conv.side_effect = lambda *_: order.append("conv")
    with patch("logssms.cache.logstate_construction.torch.cuda.Event", side_effect=[start, end]) as event:
        stats = construction.restore(
            tasks, profile_enabled=profile, pending_cuda_events=pending,
            record_cuda=record_cuda,
        )
    assert event.call_count == expected_events
    if expected_events:
        assert len(pending) == 1
        assert pending[0][0] is stats
        assert pending[0][1:] == (start, end)
        assert order == ["start", "state", "conv", "end"]
    else:
        assert pending == []
    start.query.assert_not_called()
    start.elapsed_time.assert_not_called()
    end.query.assert_not_called()
    end.synchronize.assert_not_called()


def test_restore_uses_current_event_list_without_retaining_old_list():
    construction = LogStateConstruction(make_pool())
    task = RestoreTask(20, 10, (), 10, 2, 0)
    previous, current = [], []
    with patch("logssms.cache.logstate_construction.torch.cuda.Event", side_effect=lambda **_: Mock()):
        first = construction.restore([task], profile_enabled=True, pending_cuda_events=previous)
        second = construction.restore([task], profile_enabled=True, pending_cuda_events=current)
    assert len(previous) == len(current) == 1
    assert previous[0][0] is first
    assert current[0][0] is second
    assert vars(construction) == {"pool": construction.pool}


def test_restore_rejects_checkpoint_copy_without_base():
    pool = make_pool()
    construction = LogStateConstruction(pool)
    with pytest.raises(AssertionError):
        construction.restore(
            [RestoreTask(20, None, (), 10, 0, 0)],
            profile_enabled=False, pending_cuda_events=[],
        )
    pool.copy_states.assert_not_called()
