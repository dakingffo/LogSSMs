from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from logssms.message import AbortBackendMsg, BatchBackendMsg
from logssms.scheduler.scheduler import Scheduler


class _Event:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def synchronize(self) -> None:
        self.calls.append("synchronize")


@pytest.mark.parametrize("tracked", [True, False])
@pytest.mark.parametrize("phase", ["decode", "mixed"])
def test_abort_waits_for_submitted_forward_before_reusing_resources(
    tracked: bool, phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "logssms.scheduler.scheduler.logger.debug_rank0", lambda *_args: None
    )
    calls = []
    req = SimpleNamespace(uid=7, table_idx=3, aborted=False)
    event = _Event(calls)
    last_data = (
        SimpleNamespace(batch=SimpleNamespace(reqs=[req], phase=phase)),
        SimpleNamespace(copy_done_event=event),
    )
    scheduler = object.__new__(Scheduler)
    scheduler._deferred_req_frees = []
    scheduler.prefill_manager = SimpleNamespace(abort_req=lambda _uid: (None, False))

    def abort(_uid):
        if not tracked:
            return None, False
        req.aborted = True
        return req, True

    scheduler.ar_manager = SimpleNamespace(
        abort_req=abort,
        process=lambda *_args: ([], []),
    )
    scheduler.profiler = SimpleNamespace(
        record_wait=lambda _phase: nullcontext(),
        record_process=lambda *_args: nullcontext(),
        record_process_step=lambda *_args: nullcontext(),
    )
    scheduler.engine = SimpleNamespace(ctx=None)
    scheduler.send_result = lambda _reply: None
    scheduler._free_req_resources = lambda _req, inflight=False: calls.append(
        f"free:{inflight}"
    )

    scheduler._process_one_msg(
        BatchBackendMsg([AbortBackendMsg(uid=req.uid)]), last_data
    )
    assert req.aborted
    assert calls == []

    scheduler._process_last_data(last_data)
    assert calls == ["synchronize", "free:True"]
    assert scheduler._deferred_req_frees == []


def test_abort_without_submitted_forward_releases_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "logssms.scheduler.scheduler.logger.debug_rank0", lambda *_args: None
    )
    calls = []
    req = SimpleNamespace(uid=7, table_idx=3, aborted=False)
    scheduler = object.__new__(Scheduler)
    scheduler._deferred_req_frees = []
    scheduler.prefill_manager = SimpleNamespace(abort_req=lambda _uid: (req, False))
    scheduler.ar_manager = SimpleNamespace(abort_req=lambda _uid: (None, False))
    scheduler._free_req_resources = lambda _req, inflight=False: calls.append(inflight)

    scheduler._process_one_msg(AbortBackendMsg(uid=req.uid))
    assert calls == [False]
    assert scheduler._deferred_req_frees == []


def test_finished_request_waits_for_overlapping_forward() -> None:
    calls = []
    req = SimpleNamespace(uid=7, table_idx=3)
    previous = (
        SimpleNamespace(batch=SimpleNamespace(reqs=[req], phase="decode")),
        SimpleNamespace(copy_done_event=_Event(calls)),
    )
    ongoing = (
        SimpleNamespace(batch=SimpleNamespace(reqs=[req], phase="decode")),
        SimpleNamespace(copy_done_event=_Event(calls)),
    )
    scheduler = object.__new__(Scheduler)
    scheduler._deferred_req_frees = []
    scheduler.ar_manager = SimpleNamespace(process=lambda *_args: ([], [req]))
    scheduler.profiler = SimpleNamespace(
        record_wait=lambda _phase: nullcontext(),
        record_process=lambda *_args: nullcontext(),
        record_process_step=lambda *_args: nullcontext(),
    )
    scheduler.engine = SimpleNamespace(ctx=None)
    scheduler.send_result = lambda _reply: None
    scheduler._free_req_resources = lambda _req, inflight=False: calls.append("free")

    scheduler._process_last_data(previous, ongoing)
    assert calls == ["synchronize"]
    assert len(scheduler._deferred_req_frees) == 1

    scheduler.ar_manager.process = lambda *_args: ([], [])
    scheduler._process_last_data(ongoing)
    assert calls == ["synchronize", "synchronize", "free"]
    assert scheduler._deferred_req_frees == []


def test_finished_requests_release_in_uid_order() -> None:
    class Finished:
        def __init__(self, uid: int) -> None:
            self.uid = uid

    reqs = [Finished(9), Finished(2), Finished(5)]
    calls = []
    previous = (
        SimpleNamespace(batch=SimpleNamespace(reqs=reqs, phase="decode")),
        SimpleNamespace(copy_done_event=_Event(calls)),
    )
    scheduler = object.__new__(Scheduler)
    scheduler._debug_tp_schedule = False
    scheduler._deferred_req_frees = []
    scheduler.ar_manager = SimpleNamespace(
        process=lambda *_args: ([], set(reqs))
    )
    scheduler.profiler = SimpleNamespace(
        record_wait=lambda _phase: nullcontext(),
        record_process=lambda *_args: nullcontext(),
        record_process_step=lambda *_args: nullcontext(),
    )
    scheduler.engine = SimpleNamespace(ctx=None)
    scheduler.send_result = lambda _reply: None
    scheduler._free_req_resources = lambda req, inflight=False: calls.append(req.uid)

    scheduler._process_last_data(previous)

    assert calls == ["synchronize", 2, 5, 9]


@pytest.mark.parametrize("enabled", [False, True])
def test_scheduler_cache_profile_sampling_is_opt_in(enabled: bool) -> None:
    calls = []
    scheduler = object.__new__(Scheduler)
    scheduler.config = SimpleNamespace(scheduling_policy="prefill-first")
    scheduler.profiler = SimpleNamespace(
        enabled=enabled,
        record_schedule=lambda: nullcontext(),
        record_schedule_step=lambda _name: nullcontext(),
    )
    scheduler.prefill_manager = SimpleNamespace(
        pending_list=[],
        runnable=False,
        need_tokens=0,
        need_capacity=0,
        schedule_next_batch=lambda _adder: None,
    )
    scheduler.ar_manager = SimpleNamespace(
        running_reqs={},
        runnable=False,
        need_tokens=0,
        need_capacity=0,
        schedule_next_batch=lambda: None,
    )
    scheduler.cache_manager = SimpleNamespace(
        record_prefill_queue=lambda *_args: calls.append("queue"),
        record_active_requests=lambda *_args: calls.append("active"),
    )
    scheduler.table_manager = SimpleNamespace()
    scheduler.prefill_budget = 64
    scheduler._report_stall = lambda _adder: None

    assert scheduler._schedule_next_batch() is None
    assert calls == (["queue", "active"] if enabled else [])
