from __future__ import annotations

from dataclasses import replace
from contextlib import nullcontext
from datetime import timedelta
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

import torch

from logssms.layers.attention_backend import make_attention_backend
from logssms.layers.linear_attention_backend import make_linear_attention_backend
from logssms.layers.moe_backend import make_moe_backend
from logssms.core import Batch, ChunkedRequest, Context, Request, set_global_ctx
from logssms.distributed import (
    destroy_distributed,
    enable_pynccl_distributed,
    set_tp_info,
)
from logssms.cache import LogStateCacheConfig, get_cache_factory
from logssms.layers import set_rope_device
from logssms.models import make_model, load_target_weight
from logssms.utils import (
    init_logger, is_sm90_supported, is_sm100_supported, torch_dtype
)

from .config import EngineConfig
from .decode_graph import DecodeGraphRunner, get_free_memory, mem_GB
from .piecewise_graph import PiecewiseGraphRunner
from .mixed_graph import MixedGraphRunner
from .profile import EngineProfiler
from .sample import BatchSamplingArgs, Sampler

if TYPE_CHECKING:
    import multiprocessing as mp

logger = init_logger(__name__)

Indice2D : TypeAlias = tuple[torch.Tensor, torch.Tensor]


class ForwardInput(NamedTuple):
    batch      : Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D
    write_tuple: Indice2D


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
        self.profiler = EngineProfiler()

        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        if (
            config.enable_specualtive_decoding
            and not config.dt_separation
            and config.tp_info.is_primary()
        ):
            assert speculator_start_event is not None
            speculator_start_event.set()


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


        self.ctx.attn_backend = self.attn_backend = make_attention_backend(
            config.attention_backend, config.model_config
        )
        if config.model_config.is_hybrid:
            eager_state = (
                isinstance(config.cache_config, LogStateCacheConfig)
                and config.cache_config.enable_eager_state
            )
            self.ctx.linear_attn_backend = self.linear_attn_backend = (
                make_linear_attention_backend(
                    config.cache_type,
                    recurrent_type=config.model_config.make_linear_cache_spec(config.tp_info.size).recurrent_type,
                    enable_eager_state=eager_state,
                    tile_min_batch_size=(
                        max(1, config.max_running_req
                            - config.mixed_prefill_tokens // config.cache_config.granularity)
                        if config.scheduling_policy == "mixed"
                        and isinstance(config.cache_config, LogStateCacheConfig)
                        else 32
                    ),
                )
            )
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = make_moe_backend(config.moe_backend)


        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")


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
        if config.scheduling_policy == "mixed" and config.max_running_req > 1:
            self.attn_backend.warmup_mixed(self.dummy_req)
        self.decode_graph_runner = DecodeGraphRunner(
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
        piecewise_tokens = config.max_prefill_tokens
        self.piecewise_graph_runner = self.ctx.piecewise_graph_runner = (
            PiecewiseGraphRunner(
                self.model, piecewise_tokens, self.stream,
            )
            if self.decode_graph_runner.max_graph_bs > 0 else None
        )
        self.mixed_graph_runner = MixedGraphRunner(self)
        if config.scheduling_policy == "mixed":
            from .warmup import warmup_mixed_linear

            warmup_mixed_linear(self)

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


            param_dtypes = {k: v.dtype for k, v in self.model.state_dict().items()}
            return {
                k: v.to(param_dtypes.get(k, self.dtype))
                for k, v in load_target_weight(config.model_path, self.device)
            }

    def _sync_get_memory(self) -> tuple[int, int]:
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
        if batch.is_mixed:
            if self.profiler.enabled:
                with self.profiler.record_batch(
                    batch, "prepare", graph_used=(
                        self.mixed_graph_runner.can_use_cuda_graph(batch)
                    ),
                ):
                    return self._prepare_mixed_batch(batch, cache_manager)
            return self._prepare_mixed_batch(batch, cache_manager)
        if self.profiler.enabled:
            return self._prepare_batch_profiled(batch, cache_manager)
        self.decode_graph_runner.pad_batch(batch)
        cache_manager.allocate(batch.reqs, phase=batch.phase)
        cache_manager.prepare_batch(batch)
        input_mapping, write_mapping = self._prepare_indices(batch)
        self.attn_backend.prepare_metadata(batch)
        if self.config.model_config.is_hybrid:
            self.linear_attn_backend.prepare_metadata(batch)
        sample_args = self.sampler.prepare(batch)
        return ForwardInput(
            batch=batch,
            sample_args=sample_args,
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _prepare_mixed_batch(self, batch: Batch, cache_manager) -> ForwardInput:
        from logssms.layers.linear_attention_backend.mixed import prepare_mixed_linear_metadata

        assert batch.prefill_count > 0 and batch.decode_reqs
        assert batch.prefill_tokens <= self.config.mixed_prefill_tokens
        prefill = Batch(batch.prefill_reqs, "prefill")
        decode = Batch(batch.decode_reqs, "decode")
        prefill.padded_reqs = prefill.reqs
        size = self.mixed_graph_runner.padded_size(batch.size)
        decode.padded_reqs = decode.reqs + [self.dummy_req] * (size - batch.size)
        batch.padded_reqs = prefill.reqs + decode.padded_reqs
        batch.prefill_token_capacity = self.config.mixed_prefill_tokens
        batch.prefill_partition = self.mixed_graph_runner.partition(batch.prefill_count)
        assert all(req.extend_len <= capacity for req, capacity in zip(prefill.reqs, batch.prefill_partition))
        batch.mixed_parts = prefill, decode
        stage = self.profiler.record_stage if self.profiler.enabled else lambda _: nullcontext()
        with stage("allocate"):
            cache_manager.allocate(prefill.reqs, phase="prefill")
            cache_manager.allocate(decode.reqs, phase="decode")
        with stage("cache_prepare"):
            cache_manager.prepare_batch(decode)
            cache_manager.prepare_batch(prefill)
            if decode.logstate_replay_counts is not None:
                batch.logstate_replay_counts = [0] * prefill.size + decode.logstate_replay_counts
        with stage("positions_mappings"):
            from logssms.kernel.metadata import make_batch_mapping
            from logssms.utils.batch import BatchIndices

            page_size = getattr(self.ctx.kv_cache, "kv_page_size", 1)
            packet, p_packet, d_packet = BatchIndices.make_mixed(batch, self.device, page_size)
            batch.prepared_indices = packet
            batch.positions, mapping, batch.out_loc = make_batch_mapping(
                packet, self.page_table, decode=self.attn_backend.uses_paged_indices,
                dummy_row=self.dummy_req.table_idx,
            )
            packet.token_table_ids = mapping[0]
            token_begin = page_begin = 0
            for part, indices in ((prefill, p_packet), (decode, d_packet)):
                token_end = token_begin + indices.num_tokens
                page_end = page_begin + indices.num_pages
                part.prepared_indices = indices
                part.positions = batch.positions[token_begin:token_end]
                part.out_loc = batch.out_loc[token_begin:token_end]
                indices.token_table_ids = mapping[0][token_begin:token_end]
                if packet.page_indices is not None:
                    indices.page_indices = packet.page_indices[page_begin:page_end]
                token_begin = batch.prefill_token_capacity if part is prefill else token_end
                page_begin = page_end
            writes = packet.write_tuple
        with stage("attention_metadata"):
            self.attn_backend.prepare_mixed_metadata(batch)
        with stage("linear_metadata"):
            if self.config.model_config.is_hybrid:
                for part in (prefill, decode):
                    self.linear_attn_backend.prepare_metadata(part)
                prepare_mixed_linear_metadata(
                    self.linear_attn_backend, prefill, decode, batch.prefill_token_capacity,
                )
        with stage("sampler_prepare"):
            params = [req.sampling_params for req in batch.reqs if not isinstance(req, ChunkedRequest)]
            batch.sample_indices, batch.sample_req_indices = packet.sample_tuple
            sample_args = self.sampler.prepare_params(params)
        return ForwardInput(batch, sample_args, mapping, writes)

    def _prepare_batch_profiled(self, batch: Batch, cache_manager) -> ForwardInput:
        profiler = self.profiler
        graph_used = self.decode_graph_runner.can_use_cuda_graph(batch)
        prefill_runner = self.piecewise_graph_runner
        graph_used = graph_used or (
            prefill_runner is not None and prefill_runner.can_use_cuda_graph(batch)
        )
        with profiler.record_batch(batch, "prepare", graph_used=graph_used):
            with profiler.record_stage("pad_batch"):
                self.decode_graph_runner.pad_batch(batch)
            with profiler.record_stage("allocate"):
                cache_manager.allocate(batch.reqs, phase=batch.phase)
            with profiler.record_stage("cache_prepare"):
                cache_manager.prepare_batch(batch)
            with profiler.record_stage("positions_mappings"):
                input_mapping, write_mapping = self._prepare_indices(batch)
            with profiler.record_stage("attention_metadata"):
                self.attn_backend.prepare_metadata(batch)
            with profiler.record_stage("linear_metadata"):
                if self.config.model_config.is_hybrid:
                    self.linear_attn_backend.prepare_metadata(batch)
            with profiler.record_stage("sampler_prepare"):
                sample_args = self.sampler.prepare(batch)
        return ForwardInput(batch, sample_args, input_mapping, write_mapping)

    def prepare_profile_result(self) -> dict | None:
        return self.profiler.result("prepare")

    def forward_profile_result(self) -> dict | None:
        return self.profiler.result("forward")

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

        if self.profiler.enabled:
            return self._forward_batch_profiled(batch, args)
        with self.ctx.forward_batch(batch):
            if batch.is_mixed and self.mixed_graph_runner.can_use_cuda_graph(batch):
                logits = self.mixed_graph_runner.replay(batch)
            elif self.decode_graph_runner.can_use_cuda_graph(batch):
                logits = self.decode_graph_runner.replay(batch)
            else:
                logits = self.model.forward()
        next_tokens_gpu = self._sample_batch(logits, batch, args)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)

        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def _sample_batch(self, logits: torch.Tensor, batch: Batch, args: BatchSamplingArgs) -> torch.Tensor:
        if batch.is_verify:
            return self.sampler.reject_sample(logits, batch).to(torch.int32)
        tokens = self.sampler.sample(logits[:batch.size], args).to(torch.int32)
        if batch.is_mixed:
            assert batch.sample_req_indices is not None
            if len(batch.sample_req_indices) != batch.size:
                output = tokens.new_full((batch.size,), -1)
                output.scatter_(0, batch.sample_req_indices, tokens)
                return output
        return tokens

    def _forward_batch_profiled(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        profiler = self.profiler
        decode_graph = self.decode_graph_runner.can_use_cuda_graph(batch)
        mixed_graph = batch.is_mixed and self.mixed_graph_runner.can_use_cuda_graph(batch)
        prefill_runner = self.piecewise_graph_runner
        piecewise_graph = prefill_runner is not None and prefill_runner.can_use_cuda_graph(batch)
        graph_used = decode_graph or piecewise_graph or mixed_graph
        with profiler.record_batch(batch, "forward", graph_used=graph_used) as sample:
            sample["graph_kind"] = "mixed" if mixed_graph else "decode" if decode_graph else "piecewise" if piecewise_graph else None
            if piecewise_graph:
                sample["piece_bucket_tokens"] = prefill_runner.token_bucket(len(batch.input_ids))
            if self.config.model_config.is_hybrid:
                sample["linear_backend"] = type(self.linear_attn_backend).__name__
                if hasattr(self.linear_attn_backend, "enable_eager_state"):
                    sample["eager_state"] = self.linear_attn_backend.enable_eager_state
            counts = batch.logstate_replay_counts
            if counts is not None:

                sample["replay_counts"] = list(counts[:batch.size])
                sample["effective_replay_counts"] = (
                    [0] * batch.size if sample.get("eager_state", False)
                    else list(counts[:batch.size])
                )
            with self.ctx.forward_batch(batch):
                if mixed_graph:
                    with profiler.record_stage("mixed_graph_replay"):
                        logits = self.mixed_graph_runner.replay(batch)
                elif decode_graph:
                    logits = self.decode_graph_runner.replay(batch, profiler=profiler)
                else:
                    replays = prefill_runner.replays if piecewise_graph else 0
                    padded = prefill_runner.padded_tokens if piecewise_graph else 0
                    with profiler.record_stage("model"):
                        logits = self.model.forward()
                    if piecewise_graph:
                        sample["piece_replays"] = prefill_runner.replays - replays
                        sample["piece_padding"] = prefill_runner.padded_tokens - padded
                    plan_cpu_ms = getattr(batch.attn_metadata, "plan_cpu_ms", None)
                    if plan_cpu_ms is not None:
                        sample["attention_plan_cpu_ms"] = plan_cpu_ms
            with profiler.record_stage("sampling"):
                next_tokens_gpu = self._sample_batch(logits, batch, args)
            with profiler.record_stage("output_copy"):
                next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
                copy_done_event = torch.cuda.Event()
                copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        self.mixed_graph_runner.destroy_cuda_graphs()
        if self.piecewise_graph_runner is not None:
            self.piecewise_graph_runner.destroy_cuda_graphs()
        self.decode_graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()

    def _prepare_indices(self, batch: Batch) -> tuple[Indice2D, Indice2D]:
        from logssms.kernel.metadata import make_batch_mapping
        from logssms.utils.batch import BatchIndices

        page_size = getattr(self.ctx.kv_cache, "kv_page_size", 1)
        indices = BatchIndices.make(batch, self.device, page_size)
        batch.prepared_indices = indices
        batch.positions, mapping, batch.out_loc = make_batch_mapping(
            indices, self.page_table,
            decode=batch.is_decode and self.attn_backend.uses_paged_indices,
        )
        indices.token_table_ids = mapping[0]
        return mapping, indices.write_tuple

    @staticmethod
    def _adjust_config(config: EngineConfig) -> None:
        def override(attr: str, value: Any):
            object.__setattr__(config, attr, value)

        def override_page_size(value: int) -> None:
            from logssms.cache import StandardCacheConfig

            assert isinstance(config.cache_config, StandardCacheConfig)
            override("cache_config", replace(config.cache_config, page_size=value))

        if config.scheduling_policy == "mixed":
            if config.attention_backend not in ("auto", "fi"):
                raise ValueError("Mixed scheduling currently requires FlashInfer attention")
            override("attention_backend", "fi")

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
                    "--dt-separation requires at least tp_size + 1 GPUs "
                    f"(tp_size={config.tp_info.size})."
                )
        elif config.speculator_config is not None:
            raise ValueError("speculator_config requires --speculative-algorithm.")
        elif config.dt_separation:
            raise ValueError(
                "--dt-separation requires --speculative-algorithm."
            )
