from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import torch

from picosgl.core import Request
from picosgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    ExitMsg,
    UserMsg,
)
from picosgl.speculator import make_speculator_client
from picosgl.utils import init_logger, load_tokenizer
from picosgl.engine import Engine, ForwardOutput, ForwardData

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
    from picosgl.speculator.data_plane import DataPlane


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
        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
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
        self.prefill_manager = PrefillManager(self.token_pool, config.page_size)
        self.prefill_budget = config.max_prefill_tokens
        self.profiler = SchedulerProfiler(self.device)

    def run_when_idle(self) -> None:
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        self.profiler.poll()
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.ar_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                batch = forward_input.batch
                with self.profiler.record_submit(batch.phase):
                    ongoing_data = (forward_input, self._forward(forward_input))
                    if not batch.is_prefill:
                        self.ar_manager.advance_for_overlap(batch)

        self._process_last_data(last_data)
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

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return
        last_input, last_output = last_data
        with self.profiler.record_wait(last_input.batch.phase):
            last_output.copy_done_event.synchronize()

        with self.profiler.record_process(last_input.batch.phase):
            reply, finished_reqs = self.ar_manager.process(
                self.engine.ctx, last_input, last_output
            )
            self.send_result(reply)
            for req in finished_reqs:
                self._free_req_resources(req)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
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
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req: Request
            inflight: bool
            req, inflight = self.prefill_manager.abort_req(msg.uid)
            if req is None:
                req, inflight = self.ar_manager.abort_req(msg.uid)
            if req is not None:
                self._free_req_resources(req, inflight)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Request, inflight: bool = False) -> None:
        if inflight: # The inflight aborted req has been allocated at [cached_len, device_len)
            self.cache_manager.free_to_cached(req)
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _schedule_next_batch(self) -> ForwardInput | None:
        with self.profiler.record_schedule():
            batch = (
                self.prefill_manager.schedule_next_batch(
                    PrefillAdder(
                        token_budget=self.prefill_budget,
                        reserved_size=self.ar_manager.need_tokens + self.prefill_manager.need_tokens,
                        cache_manager=self.cache_manager,
                        table_manager=self.table_manager,
                    )
                ) or
                self.ar_manager.schedule_next_batch()
            )
        if batch is None:
            if self.ar_manager.runnable and not self.prefill_manager.runnable:
                self.profiler.record_ar_no_batch()
            return None

        graph_used = batch.is_decode and self.engine.graph_runner.can_use_cuda_graph(batch)
        with self.profiler.record_prepare(batch, graph_used=graph_used):
            forward_input = self.engine.prepare_batch(batch, self.cache_manager)
        return forward_input

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, _ = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.prefill_manager.advance_for_next_schedule(
            self.engine.ctx, forward_input, forward_output
        )
        return forward_output
