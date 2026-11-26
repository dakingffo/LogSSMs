from __future__ import annotations

import faulthandler
import json
import sys
from threading import Event, Thread
from time import perf_counter_ns


class ReplayDiagnostics:
    def __init__(self, heartbeat_seconds, timing_path):
        self.heartbeat_seconds = heartbeat_seconds
        self.timing_path = timing_path
        self.iteration = 0
        self._active = None
        self._sequence = 0
        self._stop = Event()
        self._thread = None

    def __enter__(self):
        if self.timing_path is not None:
            self.timing_path.parent.mkdir(parents=True, exist_ok=True)
            self._timings = self.timing_path.open("w", encoding="utf-8", buffering=1)
        else:
            self._timings = None
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            faulthandler.cancel_dump_traceback_later()
        if self._timings is not None:
            self._timings.close()

    def attach(self, scheduler, manager):
        admit = manager.admit_prefill
        plan = manager.evictor.plan

        def admit_prefill(req, *args, **kwargs):
            return self._measure("admit_prefill", admit, (req, *args), kwargs, {
                "uid": req.uid, "tag": req.profile_tag, "input_tokens": req.input_len,
                "output_cap": req.output_len,
            })

        def plan_eviction(needed_pages, needed_states, needed_works, active_candidates=()):
            return self._measure("eviction_plan", plan,
                                 (needed_pages, needed_states, needed_works, active_candidates), {}, {
                "needed_pages": needed_pages, "needed_states": needed_states, "needed_works": needed_works,
                "active_candidates": len(active_candidates),
            })

        manager.admit_prefill = admit_prefill
        manager.evictor.plan = plan_eviction
        if self.heartbeat_seconds:
            self._arm_watchdog()
            self._thread = Thread(target=self._heartbeat, args=(scheduler,), daemon=True)
            self._thread.start()

    def _measure(self, operation, fn, args, kwargs, details):
        self._sequence += 1
        record = dict(call=self._sequence, iteration=self.iteration, operation=operation, **details)
        if self._timings is not None:
            self._timings.write(json.dumps(dict(record, event="start")) + "\n")
        previous = self._active
        start = perf_counter_ns()
        self._active = record, start
        outcome = "error"
        try:
            result = fn(*args, **kwargs)
            outcome = result if isinstance(result, str) else "ok"
            return result
        finally:
            elapsed_ms = (perf_counter_ns() - start) / 1e6
            self._active = previous
            if self._timings is not None:
                self._timings.write(json.dumps(dict(record, event="finish", elapsed_ms=elapsed_ms,
                                                    outcome=outcome)) + "\n")

    def _arm_watchdog(self):
        faulthandler.dump_traceback_later(self.heartbeat_seconds * 2, repeat=True, file=sys.stderr)

    def _heartbeat(self, scheduler):
        while not self._stop.wait(self.heartbeat_seconds):
            self._arm_watchdog()
            pending = scheduler.prefill_manager.pending_list
            queue_head = next(iter(pending), None)
            head = None if queue_head is None else dict(
                uid=queue_head.uid, tag=queue_head.profile_tag,
                input_tokens=queue_head.input_len, output_cap=queue_head.output_len,
            )
            active = self._active
            active_details = None if active is None else dict(
                active[0], elapsed_ms=round((perf_counter_ns() - active[1]) / 1e6, 1),
            )
            print("Heartbeat " + json.dumps(dict(
                iteration=self.iteration, completed=len(scheduler.finished), total=len(scheduler.records),
                pending=len(pending), running=len(scheduler.ar_manager.running_reqs),
                head=head, active_operation=active_details,
                virtual_time_s=None if scheduler.clock is None else scheduler.clock.now / 1e9,
            )), file=sys.stderr, flush=True)
