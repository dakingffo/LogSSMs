from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import os
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

import torch

from logssms.layers.attention_backend import make_attention_backend
from logssms.layers.linear_attention_backend import make_linear_attention_backend
from logssms.layers.moe_backend import make_moe_backend
from logssms.core import Batch, Context, Request, set_global_ctx
from logssms.distributed import (
    destroy_distributed,
    enable_pynccl_distributed,
    set_tp_info,
)
from logssms.cache import get_cache_factory
from logssms.layers import set_rope_device
from logssms.models import make_model, load_target_weight
from logssms.utils import (
    init_logger, is_sm90_supported, is_sm100_supported, torch_dtype
)

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

if TYPE_CHECKING:
    import multiprocessing as mp

logger = init_logger(__name__)

Indice2D : TypeAlias = tuple[torch.Tensor, torch.Tensor]


class ForwardInput(NamedTuple):
    batch      : Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


ForwardData: TypeAlias = tuple[ForwardInput, ForwardOutput]


class Engine:
    def __init__(
        self,
        config                : EngineConfig,
        speculator_start_event: mp.synchronize.Event | None = None,
        speculator_ready_event: mp.synchronize.Event | None = None,
    ):
        assert not torch.cuda.is_initialized()
        set_tp_info(config.tp_info)
        self._adjust_config(config)
        self.config = config
        self._profile_prepare_enabled = os.getenv(
            "LOGSSMS_PROFILE_SCHEDULER", "0"
        ).lower() in ("1", "true", "yes")
        self._prepare_profile_totals: dict[str, dict[str, float]] = {}
        self._prepare_profile_counts: dict[str, int] = {}
        self._prepare_profile_prefill_samples: list[dict] = []

        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[0]  # min across ranks (tightest rank anchors budget)
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        if (
            config.enable_specualtive_decoding
            and not config.dt_separation
            and config.tp_info.is_primary()
        ):
            assert speculator_start_event is not None
            speculator_start_event.set()

        # ======================= Model initialization ========================
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = make_model(config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict(config))

        if (
            config.enable_specualtive_decoding
            and not config.dt_separation
            and config.tp_info.is_primary()
        ):
            assert speculator_ready_event is not None
            speculator_ready_event.wait()

        if config.speculator_config is not None:
            self.speculator_reserve = config.speculator_config.make_reserve(
                config.max_running_req
            )
        else:
            self.speculator_reserve = None

        current_free_memory = self._sync_get_memory()[0]
        model_memory = init_free_memory - current_free_memory
        available_memory = (
            int(config.memory_ratio * init_free_memory) - model_memory
        )
        cache_factory = get_cache_factory(config.cache_type)
        self.cache_plan = cache_factory.plan(
            config=config,
            available_memory=available_memory,
            speculator_reserve=self.speculator_reserve,
        )
        self.cache_resources = cache_factory.allocate(
            config=config,
            plan=self.cache_plan,
            device=self.device,
            dtype=self.dtype,
        )
        self.cache_manager = cache_factory.make_manager(
            config=config,
            plan=self.cache_plan,
            resources=self.cache_resources,
        )

        self.num_pages = self.cache_plan.num_pages
        self.max_seq_len = self.cache_plan.max_seq_len
        self.kv_cache = self.cache_resources.kv_pool
        self.page_table = self.cache_resources.page_table
        self.linear_state = self.cache_resources.linear_state
        self.state_table = self.cache_resources.state_table
        self.work_state_table = self.cache_resources.work_state_table
        self.ctx.kv_cache = self.kv_cache
        self.ctx.page_table = self.page_table
        self.ctx.linear_state = self.linear_state
        self.ctx.state_table = self.state_table
        self.ctx.work_state_table = self.work_state_table

        # ======================= Attention & MoE backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = make_attention_backend(
            config.attention_backend, config.model_config
        )
        if config.model_config.is_hybrid:
            self.ctx.linear_attn_backend = self.linear_attn_backend = (
                make_linear_attention_backend(config.cache_type)
            )
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = make_moe_backend(config.moe_backend)

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        self.dummy_req = Request(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
            max_device_len=2,
        )

        speculative_skip_graphs = config.enable_specualtive_decoding
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            linear_attn_backend=(
                self.linear_attn_backend if config.model_config.is_hybrid else None
            ),
            cuda_graph_bs=[] if speculative_skip_graphs else config.cuda_graph_bs,
            cuda_graph_max_bs=0 if speculative_skip_graphs else config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=self.cache_plan.aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        torch.distributed.init_process_group(
            backend="gloo",
            rank=config.tp_info.rank,
            world_size=config.tp_info.size,
            timeout=timedelta(seconds=config.distributed_timeout),
            init_method=config.distributed_addr,
        )
        tp_cpu_group = torch.distributed.group.WORLD
        assert tp_cpu_group is not None
        if config.tp_info.size > 1:
            max_bytes=config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            enable_pynccl_distributed(
                config.tp_info,
                tp_cpu_group,
                max_bytes=max_bytes
            )
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            # Cast each tensor to its parameter's dtype (not blanket self.dtype) so that
            # fp32 params (e.g. GatedDeltaNet A_log / gated norm weight) stay fp32.
            param_dtypes = {k: v.dtype for k, v in self.model.state_dict().items()}
            return {
                k: v.to(param_dtypes.get(k, self.dtype))
                for k, v in load_target_weight(config.model_path, self.device)
            }

    def _sync_get_memory(self) -> tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        return min_free_memory, max_free_memory

    def prepare_batch(self, batch: Batch, cache_manager) -> ForwardInput:
        timings: dict[str, float] | None = (
            {} if self._profile_prepare_enabled else None
        )
        start = perf_counter_ns() if timings is not None else 0
        stage = start
        self.graph_runner.pad_batch(batch)
        if timings is not None:
            now = perf_counter_ns()
            timings["pad_batch_ms"] = (now - stage) / 1e6
            stage = now
        cache_manager.allocate(batch.reqs, phase=batch.phase)
        if timings is not None:
            now = perf_counter_ns()
            timings["allocate_ms"] = (now - stage) / 1e6
            stage = now
        batch.positions = self._make_positions(batch, self.device)
        input_mapping = self._make_input_tuple(batch, self.device)
        write_mapping = self._make_write_tuple(batch, self.device)
        batch.out_loc = self.page_table[input_mapping]
        if timings is not None:
            now = perf_counter_ns()
            timings["positions_mappings_ms"] = (now - stage) / 1e6
            stage = now
        cache_manager.prepare_batch(batch)
        if timings is not None:
            now = perf_counter_ns()
            timings["cache_prepare_ms"] = (now - stage) / 1e6
            stage = now
        self.attn_backend.prepare_metadata(batch)
        if timings is not None:
            now = perf_counter_ns()
            timings["attention_metadata_ms"] = (now - stage) / 1e6
            stage = now
        if self.config.model_config.is_hybrid:
            self.linear_attn_backend.prepare_metadata(batch)
        if timings is not None:
            now = perf_counter_ns()
            timings["linear_metadata_ms"] = (now - stage) / 1e6
            stage = now
        sample_args = self.sampler.prepare(batch)
        if timings is not None:
            now = perf_counter_ns()
            timings["sampler_prepare_ms"] = (now - stage) / 1e6
            timings["total_ms"] = (now - start) / 1e6
            phase = batch.phase
            totals = self._prepare_profile_totals.setdefault(phase, {})
            for key, value in timings.items():
                totals[key] = totals.get(key, 0.0) + value
            self._prepare_profile_counts[phase] = (
                self._prepare_profile_counts.get(phase, 0) + 1
            )
            if batch.is_prefill:
                self._prepare_profile_prefill_samples.append({
                    "batch_size": batch.size,
                    "logical_tokens": sum(req.extend_len for req in batch.reqs),
                    **timings,
                })
        return ForwardInput(
            batch=batch,
            sample_args=sample_args,
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def prepare_profile_result(self) -> dict | None:
        if not self._profile_prepare_enabled:
            return None
        return {
            "phases": {
                phase: {"batches": count, **self._prepare_profile_totals[phase]}
                for phase, count in self._prepare_profile_counts.items()
            },
            "prefill_batches": self._prepare_profile_prefill_samples,
        }

    def forward_batch(
        self, batch: Batch, args: BatchSamplingArgs
    ) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        if self.config.enable_specualtive_decoding and (
            batch.is_prefill or batch.is_verify
        ):
            from logssms.speculator.drafters import make_hidden_captor

            algorithm = self.config.speculative_algorithm
            speculator_config = self.config.speculator_config
            assert algorithm is not None and speculator_config is not None
            batch.hidden_captor = make_hidden_captor(algorithm, speculator_config)

        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()
        if batch.is_verify:
            next_tokens_gpu = self.sampler.reject_sample(logits, batch).to(torch.int32)
        else: # prefill / decode
            next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)

        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()

    @staticmethod
    def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
        needed_size = sum(r.extend_len for r in batch.padded_reqs)
        indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
        offset = 0
        for req in batch.padded_reqs:
            length = req.extend_len
            torch.arange(
                req.cached_len,
                req.device_len,
                dtype=torch.int32,
                out=indices_host[offset: offset + length],
            )
            offset += length
        return indices_host.to(device, non_blocking=True)

    @staticmethod
    def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
        mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
        offset = 0
        for req in batch.padded_reqs:
            length = req.extend_len
            mapping_host[offset: offset + length].fill_(req.table_idx)
            offset += length
        return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)

    @staticmethod
    def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
        # verify writes committed tokens itself (settle/process); the write tuple is unused.
        if batch.is_verify:
            return (
                torch.tensor([], dtype=torch.int64, device=device),
                torch.tensor([], dtype=torch.int64, device=device),
            )
        mapping_list = [req.table_idx for req in batch.reqs]
        mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
        write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
        write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
        return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)

    @staticmethod
    def _adjust_config(config: EngineConfig) -> None:
        def override(attr: str, value: Any):  # this is dangerous, use with caution
            object.__setattr__(config, attr, value)

        def override_page_size(value: int) -> None:
            from logssms.cache import StandardCacheConfig

            assert isinstance(config.cache_config, StandardCacheConfig)
            override("cache_config", replace(config.cache_config, page_size=value))

        if config.cache_type == "logstate":
            assert config.attention_backend in ("auto", "fi"), (
                "LogState shared KV storage currently requires FlashInfer attention."
            )
            override("attention_backend", "fi")
        elif config.attention_backend == "auto":
            backend = "trtllm" if is_sm100_supported() else "fa,fi" if is_sm90_supported() else "fi"
            override("attention_backend", backend)
            logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

        if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64, 128]:
            override_page_size(64)
            logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")

        if config.model_config.is_moe and config.moe_backend == "auto":
            override("moe_backend", "fused")
            logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")

        if config.model_config.is_hybrid:
            if config.page_size % 64 != 0:
                original_page_size = config.page_size
                override_page_size(64)
                logger.warning_rank0(
                    f"Page size overridden to 64 for hybrid model (must be a multiple of 64, got {original_page_size})."
                )
        if config.enable_specualtive_decoding:
            speculator_config = config.speculator_config
            assert speculator_config is not None, (
                "speculator_config is required when speculative decoding is enabled"
            )
            assert speculator_config.algorithm == config.speculative_algorithm, (
                f"Speculator config {type(speculator_config).__name__} belongs to "
                f"{speculator_config.algorithm}, not {config.speculative_algorithm}."
            )
            speculator_config.validate(config)
            if config.dt_separation:
                import torch as _torch

                assert _torch.cuda.device_count() > config.tp_info.size, (
                    "--enable-dt-separation requires at least tp_size + 1 GPUs "
                    f"(tp_size={config.tp_info.size})."
                )
        elif config.speculator_config is not None:
            raise ValueError("speculator_config requires --speculative-algorithm.")
        elif config.dt_separation:
            raise ValueError(
                "--enable-dt-separation requires --speculative-algorithm."
            )
