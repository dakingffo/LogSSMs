from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

import torch

from picosgl.layers.attention_backend import make_attention_backend
from picosgl.layers.linear_attention_backend import LinearAttentionBackend
from picosgl.layers.moe_backend import make_moe_backend
from picosgl.core import Batch, Context, Request, set_global_ctx
from picosgl.distributed import (
    destroy_distributed,
    enable_pynccl_distributed,
    set_tp_info,
)
from picosgl.cache import (
    make_kvcache_pool,
    make_linear_state_pool,
    linear_state_slot_bytes_for_config,
)
from picosgl.layers import set_rope_device
from picosgl.models import make_model, load_target_weight
from picosgl.utils import (
    align_ceil, div_even, init_logger, is_sm90_supported, is_sm100_supported, torch_dtype
)

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

if TYPE_CHECKING:
    import multiprocessing as mp

logger = init_logger(__name__)

# Smallest KV cache we will ever build; the MTP draft-state pool must fit alongside it.
_MIN_KV_PAGES = 2


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

        self.num_pages, num_tokens, self.num_draft_states = self._determine_num_pages(init_free_memory, config)
        # ======================= KV cache initialization ========================
        self.ctx.kv_cache = self.kv_cache = make_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
        )

        # ======================= Linear attention state pool ========================
        self.num_states = 0
        self.linear_state = None
        self.state_table = None
        self.work_state_table = None
        if config.model_config.is_hybrid:
            self.num_states = self.num_pages
            self.ctx.linear_state = self.linear_state = make_linear_state_pool(
                model_config=config.model_config,
                num_slots=self.num_states + self.num_draft_states + 1,
                device=self.device,
                dtype=self.dtype,
            )


        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = align_ceil(self.max_seq_len, 32)
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )
        # ======================= Linear state table ========================
        if config.model_config.is_hybrid:
            reserve_cols = (
                self.speculator_reserve.draft_slots_per_request
                if self.speculator_reserve is not None else 0
            )
            self.ctx.state_table = self.state_table = torch.full(
                (config.max_running_req + 1, aligned_max_seq_len),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            if reserve_cols != 0:
                reserve = torch.arange(
                    self.num_states,
                    self.num_states + self.num_draft_states,
                    dtype=torch.int32,
                    device=self.device,
                ).view(config.max_running_req, reserve_cols)
                self.ctx.work_state_table = self.work_state_table = reserve
            self.dummy_state_slot = self.num_states + self.num_draft_states
            self.state_table[-1].fill_(self.dummy_state_slot)

        # ======================= Attention & MoE backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = make_attention_backend(
            config.attention_backend, config.model_config
        )
        if config.model_config.is_hybrid:
            self.ctx.linear_attn_backend = self.linear_attn_backend = LinearAttentionBackend()
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
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page

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
            max_seq_len=aligned_max_seq_len,
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

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> tuple[int, int, int | None]:
        # min free across ranks: the non-DT speculator worker shares rank0's device, so
        # its bytes must shrink the budget (max would ignore the tightest rank).
        new_free_memory = self._sync_get_memory()[0]
        cache_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.page_size
            * self.dtype.itemsize
            * config.model_config.num_attention_layers
        )
        # For hybrid models every KV page also owns one linear-state slot. Target-side
        # verify states are reserved according to the selected speculator configuration.
        cache_per_state = linear_state_slot_bytes_for_config(config.model_config, self.dtype)
        num_draft_states = (
            self.speculator_reserve.num_draft_slots
            if self.speculator_reserve is not None else 0
        )
        num_dummy_states = 1 if config.model_config.is_hybrid else 0
        extra_state_slots = num_draft_states + num_dummy_states
        extra_state_bytes = extra_state_slots * cache_per_state

        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = (
                int(config.memory_ratio * old_free_memory)
                - model_memory
            )
            if config.model_config.is_hybrid:
                min_pages_bytes = _MIN_KV_PAGES * (cache_per_page + cache_per_state)
                available_states = (available_memory - min_pages_bytes) // cache_per_state
                assert extra_state_slots <= available_states, (
                    f"Not enough memory for {extra_state_slots} reserved linear states: "
                    f"only {available_states} fit alongside {_MIN_KV_PAGES} minimum KV pages. "
                    f"Raise --memory-ratio / Reduce --max-running-req, or pin --num-pages."
                )
                available_memory -= extra_state_bytes
                num_pages = max(
                    _MIN_KV_PAGES,
                    available_memory // (cache_per_page + cache_per_state),
                )
            else:
                num_pages = available_memory // cache_per_page

        assert num_pages > 1, (
            "Not enough memory for KV cache, try reducing --num-pages. "
            f"(hybrid: cache_per_page={mem_GB(cache_per_page)}, cache_per_state="
            f"{mem_GB(cache_per_state)}, reserve_bytes={mem_GB(extra_state_bytes)})"
        )
        num_tokens = num_pages * config.page_size
        logger.info(f"Allocating {num_tokens} tokens for KV cache,\
                     K + V = {mem_GB(num_pages * cache_per_page)}")
        if config.model_config.is_hybrid:
            logger.info(
                f"Allocating {num_pages + extra_state_slots}"
                f" linear state slots = "
                f"{mem_GB(num_pages * cache_per_state + extra_state_bytes)}"
            )
            return num_pages, num_tokens, num_draft_states
        else:
            return num_pages, num_tokens, None

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
        self.graph_runner.pad_batch(batch)
        cache_manager.allocate(batch.reqs)
        batch.positions = self._make_positions(batch, self.device)
        input_mapping = self._make_input_tuple(batch, self.device)
        write_mapping = self._make_write_tuple(batch, self.device)
        batch.out_loc = self.page_table[input_mapping]
        self.attn_backend.prepare_metadata(batch)
        if self.config.model_config.is_hybrid:
            self.linear_attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def forward_batch(
        self, batch: Batch, args: BatchSamplingArgs
    ) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        if self.config.enable_specualtive_decoding and (
            batch.is_prefill or batch.is_verify
        ):
            from picosgl.speculator.drafters import make_hidden_captor

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

        if config.attention_backend == "auto":
            backend = "trtllm" if is_sm100_supported() else "fa,fi" if is_sm90_supported() else "fi"
            override("attention_backend", backend)
            logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

        if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64, 128]:
            override("page_size", 64)
            logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")

        if config.model_config.is_moe and config.moe_backend == "auto":
            override("moe_backend", "fused")
            logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")

        if config.model_config.is_hybrid:
            if getattr(config, "cache_type", "radix") != "hybrid_radix":
                override("cache_type", "hybrid_radix")
                logger.info_rank0("Prefix cache set to hybrid_radix for hybrid (linear attention) model.")
            if config.page_size % 64 != 0:
                original_page_size = config.page_size
                override("page_size", 64)
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
