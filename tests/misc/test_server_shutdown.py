import logging
import multiprocessing as mp
import signal
from types import SimpleNamespace

import pytest

from logssms.server import launch


def _scheduler_worker(shutdown, ready):
    ready.set()
    shutdown.wait()


def _tokenizer_worker(ready):
    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)
    ready.set()
    try:
        mp.Event().wait()
    except KeyboardInterrupt:
        pass


def test_shutdown_reaps_real_children_without_group_signal():
    ctx = mp.get_context("spawn")
    shutdown = ctx.Event()
    scheduler_ready, tokenizer_ready = ctx.Event(), ctx.Event()
    workers = [
        ctx.Process(target=_scheduler_worker, args=(shutdown, scheduler_ready),
                    name="logssms-TP0-scheduler"),
        ctx.Process(target=_tokenizer_worker, args=(tokenizer_ready,),
                    name="logssms-tokenizer-0"),
    ]
    try:
        for worker in workers:
            worker.start()
        assert scheduler_ready.wait(30)
        assert tokenizer_ready.wait(30)
        launch._stop_workers(workers, shutdown, logging.getLogger(__name__), timeout=10)
        assert shutdown.is_set()
        assert [worker.exitcode for worker in workers] == [0, 0]
        assert all(not worker.is_alive() for worker in workers)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
            worker.join(timeout=5)


def test_shutdown_escalates_and_records_stuck_worker(monkeypatch, caplog):
    calls = []

    class Worker:
        name = "logssms-TP1-scheduler"
        pid = 123
        exitcode = None

        def is_alive(self):
            return self.exitcode is None

        def terminate(self):
            calls.append("terminate")

        def kill(self):
            calls.append("kill")
            self.exitcode = -signal.SIGKILL

    worker = Worker()
    monkeypatch.setattr(launch, "_join_workers", lambda workers, timeout: calls.append(timeout))
    shutdown = SimpleNamespace(set=lambda: calls.append("shutdown"))
    launch._stop_workers([worker, SimpleNamespace(pid=None)], shutdown,
                         logging.getLogger(__name__), timeout=0.25)
    assert calls == ["shutdown", 0.25, 5.0, "terminate", 5.0, "kill", 5.0]
    assert worker.name in caplog.text


@pytest.mark.parametrize("failed_start", [False, True])
def test_launcher_always_cleans_up(monkeypatch, failed_start):
    from logssms.server import api_server, args

    monkeypatch.setattr(args, "parse_args", lambda _: SimpleNamespace())
    monkeypatch.setattr(launch, "_stop_workers", lambda *a: calls.append("cleanup"))
    calls = []

    def run_api_server(*_):
        calls.append("api")
        if failed_start:
            raise RuntimeError("startup failed")

    monkeypatch.setattr(api_server, "run_api_server", run_api_server)
    previous_handler = signal.getsignal(signal.SIGINT)
    if failed_start:
        with pytest.raises(RuntimeError, match="startup failed"):
            launch.launch_server()
    else:
        launch.launch_server()
    assert calls == ["api", "cleanup"]
    assert signal.getsignal(signal.SIGINT) == previous_handler


def test_join_timeout_is_shared_by_all_workers(monkeypatch):
    times = iter([10.0, 10.5, 13.0])
    monkeypatch.setattr(launch.time, "monotonic", lambda: next(times))
    timeouts = []
    workers = [SimpleNamespace(join=lambda **kw: timeouts.append(kw["timeout"])) for _ in range(2)]
    launch._join_workers(workers, 2.0)
    assert timeouts == [1.5, 0.0]
