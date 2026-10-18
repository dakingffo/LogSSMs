from __future__ import annotations

import time
from typing import TYPE_CHECKING, NoReturn

import torch

from logssms.core import Batch, Request
from logssms.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    ExitMsg,
    UserMsg,
)
from logssms.speculator import make_speculator_client
from logssms.utils import init_logger, load_tokenizer
from logssms.engine import Engine, ForwardOutput, ForwardData

from .ar import ForwardInput
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import PrefillAdder, PrefillManager
from .profile import SchedulerProfiler
from .table import TableManager
from .verify import VerifyManager

logger = init_logger(__name__)

if TYPE_CHECKING:
    import multiprocessing as mp
    from logssms.speculator.data_plane import DataPlane


class Scheduler(SchedulerIOMixin):
    def __init__(
        self,
        config                : SchedulerConfig,
        speculator_start_event: mp.synchronize.Event | None = None,
        speculator_ready_event: mp.synchronize.Event | None = None,
        speculator_data_plane : DataPlane | None = None,
        shutdown_event         : mp.synchronize.Event | None = None,
    ):
        self.engine = Engine(config, speculator_start_event, speculator_ready_event)
        super().__init__(config, self.engine.tp_cpu_group, shutdown_event)

        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)


        self.config = config
        self.speculator_data_plane = speculator_data_plane
        self.cache_resources = self.engine.cache_resources
        self.table_manager = TableManager(
            config.max_running_req, self.cache_resources.page_table
        )
        self.cache_manager = self.engine.cache_manager
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.speculator_client = (
            make_speculator_client(self, config)
            if config.enable_specualtive_decoding else None
        )

        if config.enable_specualtive_decoding:
            self.ar_manager = VerifyManager(
                config, self.device,
                self.cache_manager, self.table_manager,
                self.eos_token_id, self.speculator_client,
                config.model_config.vocab_size,
            )
        else:
            self.ar_manager = DecodeManager(
                config, self.device,
                self.cache_manager, self.table_manager,
                self.eos_token_id
            )

        self.decode_manager = self.ar_manager
        self.verify_manager = self.ar_manager
        self.prefill_manager = PrefillManager(
            self.token_pool, config.page_size, self.cache_manager,
        )
        self.prefill_budget = config.max_prefill_tokens
        self.profiler = SchedulerProfiler(
            self.device, self.cache_manager,
            getattr(self.engine, "linear_attn_backend", None),
            self.engine,
        )
        self._last_stall_report = 0.0
        self._deferred_req_frees: list[tuple[ForwardOutput, Request, bool]] = []

    def run_when_idle(self) -> None:
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        self.profiler.poll()
        blocking = not (
            last_data is not None
            or self.prefill_manager.runnable
            or self.ar_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg, last_data)

        forward_input = self._schedule_next_batch(
            has_inflight=last_data is not None
        )
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                batch = forward_input.batch
                with self.profiler.record_submit(batch):
                    ongoing_data = (forward_input, self._forward(forward_input))
                    if not batch.is_prefill:
                        self.ar_manager.advance_for_overlap(batch)

        self._process_last_data(last_data, ongoing_data)
        return ongoing_data

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        assert torch.cuda.current_stream() == self.stream
        data = None
        while True:
            data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.profiler.dump()
        if self.speculator_client is not None:
            self.speculator_client.destroy()
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(
        self,
        last_data   : ForwardData | None,
        ongoing_data: ForwardData | None = None,
    ) -> None:
        if last_data is None:
            return
        last_input, last_output = last_data
        with self.profiler.record_wait(last_input.batch.phase):
            last_output.copy_done_event.synchronize()

        phase = last_input.batch.phase
        with self.profiler.record_process(phase, last_input.batch):
            with self.profiler.record_process_step(phase, "ar_process"):
                reply, finished_reqs = self.ar_manager.process(
                    self.engine.ctx, last_input, last_output
                )
            with self.profiler.record_process_step(phase, "send_result"):
                self.send_result(reply)
            with self.profiler.record_process_step(phase, "release"):


                for req in sorted(finished_reqs, key=lambda req: req.uid):
                    if ongoing_data is not None and any(
                        candidate is req for candidate in ongoing_data[0].batch.reqs
                    ):


                        self._deferred_req_frees.append((ongoing_data[1], req, False))
                    else:
                        self._free_req_resources(req)

                remaining = []
                for output, req, inflight in self._deferred_req_frees:
                    if output is last_output:
                        self._free_req_resources(req, inflight)
                    else:
                        remaining.append((output, req, inflight))
                self._deferred_req_frees = remaining

    def _process_one_msg(
        self,
        msg      : BaseBackendMsg,
        last_data: ForwardData | None = None,
    ) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg, last_data)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.cache_manager.record_arrival_profile(msg)
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req: Request
            inflight: bool
            submitted_req = None
            if last_data is not None:
                submitted_req = next(
                    (
                        candidate for candidate in last_data[0].batch.reqs
                        if candidate.uid == msg.uid and not candidate.aborted
                    ),
                    None,
                )
            req, inflight = self.prefill_manager.abort_req(msg.uid)
            if req is None:
                req, inflight = self.ar_manager.abort_req(msg.uid)
            if req is None and submitted_req is not None:
                req, inflight = submitted_req, True
            if req is not None:
                if submitted_req is not None:
                    assert last_data is not None
                    assert submitted_req.table_idx == req.table_idx
                    submitted_req.aborted = True


                    self._deferred_req_frees.append((last_data[1], req, True))
                else:
                    self._free_req_resources(req, inflight)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Request, inflight: bool = False) -> None:
        if inflight:
            self.cache_manager.free_to_cached(req)
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _schedule_next_batch(
        self,
        *,
        has_inflight: bool = False,
    ) -> ForwardInput | None:
        if self.profiler.enabled:
            pending = self.prefill_manager.pending_list
            now_ns = time.perf_counter_ns()
            self.cache_manager.record_prefill_queue(
                len(pending),
                max(
                    ((now_ns - req.queued_at_ns) / 1e6 for req in pending),
                    default=0.0,
                ),
            )
            self.cache_manager.record_active_requests(
                tuple(self.ar_manager.running_reqs.values())
            )
        mixed = self.config.scheduling_policy == "mixed"
        mixed_decode_cap = 0
        if mixed and self.config.max_running_req >= 2:
            mixed_decode_cap = min(
                self.config.max_running_req - 1,
                self.ar_manager.candidate_count,
                self.ar_manager.token_budget,
            )
        adder = PrefillAdder(
            token_budget=self.prefill_budget,
            reserved_size=(
                (self.ar_manager.need_capacity + self.prefill_manager.need_capacity)
                if self.prefill_manager.runnable else 0
            ),
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
            profiler=self.profiler if self.profiler.enabled else None,
            mixed_token_budget=(
                self.config.mixed_prefill_tokens if mixed_decode_cap > 0 else None
            ),
            mixed_partitions=(
                tuple(p for p in self.engine.mixed_graph_runner.partitions.values()
                      if len(p) < self.config.max_running_req)
                if mixed_decode_cap > 0 else ()
            ),
        )
        with self.profiler.record_schedule() as schedule:
            with self.profiler.record_schedule_step("prefill_select"):
                if mixed_decode_cap > 0:
                    batch = self.prefill_manager.schedule_next_batch(
                        adder, mixed_max_requests=self.config.max_running_req - 1,
                    )
                else:
                    batch = self.prefill_manager.schedule_next_batch(adder)
            if not adder.use_mixed:
                mixed_decode_cap = 0
            if batch is not None and mixed_decode_cap > 0:
                mixed_decode_cap = min(mixed_decode_cap, self.config.max_running_req - batch.size)
                with self.profiler.record_schedule_step("ar_select"):
                    decode = self.ar_manager.schedule_next_batch(max_requests=mixed_decode_cap)
                if decode is not None:
                    batch = Batch(
                        reqs=batch.reqs + decode.reqs, phase="mixed",
                        num_prefill_reqs=batch.size,
                    )
            elif batch is not None:
                self.ar_manager.on_prefill_scheduled()
            else:
                with self.profiler.record_schedule_step("ar_select"):
                    if mixed:
                        batch = self.ar_manager.schedule_next_batch(
                            max_requests=self.config.max_running_req,
                        )
                    else:
                        batch = self.ar_manager.schedule_next_batch()
            if schedule is not None and batch is not None:
                schedule["phase"] = batch.phase
                if batch.prefill_count:
                    schedule["prefill_budget"] = (
                        self.config.mixed_prefill_tokens if mixed_decode_cap else self.prefill_budget
                    )
                    schedule["prefill_tokens"] = batch.prefill_tokens
                    schedule["prefill_budget_unused"] = adder.token_budget
                    schedule["prefill_selection_stop"] = (
                        "token_budget" if adder.token_budget <= 0 else
                        adder.rejection_reason or (
                            "request_budget" if mixed_decode_cap and batch.prefill_count
                            >= self.config.max_running_req - 1 else "pending_exhausted"
                        )
                    )
        if batch is None:
            if self.ar_manager.runnable and not self.prefill_manager.runnable:
                self.profiler.record_ar_no_batch()
            if not has_inflight:
                self._report_stall(adder)
            return None

        graph_used = (
            self.engine.piecewise_graph_runner is not None
            and self.engine.piecewise_graph_runner.can_use_cuda_graph(batch)
        ) or (
            batch.is_decode and self.engine.decode_graph_runner.can_use_cuda_graph(batch)
        ) or (
            batch.is_mixed and self.engine.mixed_graph_runner.can_use_cuda_graph(batch)
        )
        with self.profiler.record_prepare(batch, graph_used=graph_used):
            forward_input = self.engine.prepare_batch(batch, self.cache_manager)
        return forward_input

    def _report_stall(self, adder: PrefillAdder) -> None:
        if not self.prefill_manager.runnable and not self.ar_manager.runnable:
            return
        now = time.monotonic()
        if now - self._last_stall_report < 10.0:
            return
        self._last_stall_report = now
        pending = self.prefill_manager.pending_list
        head = pending[0] if pending else None
        running = tuple(
            (req.uid, req.cached_len, req.device_len, req.max_device_len)
            for req in self.ar_manager.running_reqs.values()
        )
        logger.warning_rank0(
            "Scheduler made no progress: pending=%d, running=%s, "
            "inflight=%s/%s, table_free=%d, cache_available=%d, "
            "head=%s, reason=%s",
            len(pending),
            running,
            sorted(self.ar_manager.inflight_uids[0]),
            sorted(self.ar_manager.inflight_uids[1]),
            self.table_manager.available_size,
            self.cache_manager.available_size,
            (
                None
                if head is None else
                (head.uid, head.input_len, head.output_len)
            ),
            adder.rejection_reason,
        )

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, _ = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.prefill_manager.advance_for_next_schedule(
            self.engine.ctx, forward_input, forward_output
        )
        return forward_output
