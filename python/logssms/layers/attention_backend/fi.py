from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import os
from time import perf_counter_ns
from typing import TYPE_CHECKING, Dict, List

import torch
from logssms.core import Batch, get_global_ctx
from logssms.distributed import get_tp_info
from logssms.env import ENV
from logssms.utils import div_even, init_logger, nvtx_annotate

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPODWithPagedKVCacheWrapper,
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    )
    from logssms.models import ModelConfig
    from logssms.core import Request


logger = init_logger(__name__)


@dataclass
class FICaptureData(BaseCaptureData):
    @property
    def one_tensor(self) -> torch.Tensor:
        return self.seq_lens

    @property
    def indices(self) -> torch.Tensor:
        return self.page_table


@dataclass
class FIMetadata(BaseAttnMetadata):
    # fmt: off
    cu_seqlens_q_cpu:   torch.Tensor
    cu_seqlens_k_cpu:   torch.Tensor
    cu_seqlens_q_gpu:   torch.Tensor
    indices:            torch.Tensor
    last_page_len_cpu:  torch.Tensor
    num_qo_heads:       int
    num_kv_heads:       int
    head_dim:           int
    page_size:          int
    pos_encoding_mode:  str
    seq_lens_cpu:       torch.Tensor
    dtype:              torch.dtype
    wrapper:            BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper
    initialized:        bool = False
    # fmt: on

    def __post_init__(self) -> None:
        assert self.page_size > 0
        assert (
            self.cu_seqlens_k_cpu.is_cpu
            and self.cu_seqlens_q_cpu.is_cpu
            and self.cu_seqlens_q_gpu.is_cuda
            and self.indices.is_cuda
            and self.last_page_len_cpu.is_cpu
            and self.seq_lens_cpu.is_cpu
        )

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


@dataclass
class FIPODMetadata(BaseAttnMetadata):
    prefill: FIMetadata
    decode: FIMetadata
    wrapper: BatchPODWithPagedKVCacheWrapper
    initialized: bool = False
    plan_cpu_ms: dict | None = None

    def get_last_indices(self, bs: int) -> torch.Tensor:
        prefill = self.prefill.cu_seqlens_q_gpu[1:] - 1
        decode = self.decode.cu_seqlens_q_gpu[1:] - 1
        return torch.cat((prefill, decode + int(self.prefill.cu_seqlens_q_cpu[-1])))[:bs]


class FlashInferBackend(BaseAttnBackend):
    uses_paged_indices = True

    def __init__(self, config: ModelConfig) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.config = config
        self.profile_plan = os.getenv("LOGSSMS_PROFILE_SCHEDULER", "0").lower() in ("1", "true", "yes")
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",
        )
        self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            use_tensor_cores=self.use_tensor_cores,
            kv_layout="NHD",
            backend="fa2",
        )


        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer


        tp_size = get_tp_info().size
        self.qo_head_local = div_even(self.config.num_qo_heads, tp_size)
        self.kv_head_local = div_even(self.config.num_kv_heads, tp_size, allow_replicate=True)


        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.graph_wrappers: Dict[int, CUDAGraphBatchDecodeWithPagedKVCacheWrapper] = {}
        self.capture: FICaptureData | None = None
        self.last_event = torch.cuda.Event()
        self.last_event.record()

    @nvtx_annotate("FlashInfer::plan")
    def _initialize_metadata_once(self, metadata: FIMetadata | FIPODMetadata) -> None:
        if metadata.initialized:
            return
        if isinstance(metadata, FIPODMetadata):
            self._initialize_pod(metadata)
            return

        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        metadata.initialized = True


        self.last_event.synchronize()
        if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                data_type=metadata.dtype,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
            )
        else:
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu,
                paged_kv_indptr=metadata.cu_seqlens_k_cpu,
                paged_kv_indices=metadata.indices,
                paged_kv_last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim_qk=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
                causal=True,
            )
        self.last_event.record()

    def forward(
        self, q: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, (FIMetadata, FIPODMetadata))
        self._initialize_metadata_once(metadata)
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
        page_size = metadata.prefill.page_size if isinstance(metadata, FIPODMetadata) else metadata.page_size
        if page_size == 1:
            kv_cache = tuple(
                cache.view(-1, 1, cache.shape[2], cache.shape[3])
                for cache in kv_cache
            )
        if isinstance(metadata, FIPODMetadata):
            from logssms.layers.piecewise import attention_output_buffer
            from .pod import run_pod

            split = batch.prefill_tokens
            out = attention_output_buffer(q, q.shape)
            return run_pod(metadata.wrapper, q, kv_cache, split, out)
        return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)

    @nvtx_annotate("FlashInfer::metadata")
    def prepare_metadata(self, batch: Batch) -> None:
        from logssms.kernel.metadata import gather_paged_indices
        from logssms.utils.batch import BatchIndices

        page_size = getattr(self.kvcache, "kv_page_size", 1)
        packet = batch.prepared_indices
        if packet is None:
            packet = BatchIndices.make(batch, self.device, page_size)
        assert packet.page_size == page_size
        indices = gather_paged_indices(packet, get_global_ctx().page_table)
        batch.attn_metadata = FIMetadata(
            cu_seqlens_q_cpu=packet.cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=packet.cu_seqlens_k_cpu,
            cu_seqlens_q_gpu=packet.cu_seqlens_q_gpu,
            indices=indices,
            last_page_len_cpu=packet.last_page_len_cpu,
            num_qo_heads=self.qo_head_local,
            num_kv_heads=self.kv_head_local,
            head_dim=self.config.head_dim,
            page_size=page_size,
            pos_encoding_mode="NONE",
            seq_lens_cpu=packet.seq_lens_cpu,
            dtype=self.kvcache.dtype,
            wrapper=self.decode_wrappers if batch.is_decode else self.prefill_wrapper,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FICaptureData.make(max_bs, max_seq_len, self.kvcache.device)
        capture.page_table = capture.page_table.view(-1)
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    @cached_property
    def use_tensor_cores(self) -> bool:
        if (overriden_value := ENV.FLASHINFER_USE_TENSOR_CORES.value) is not None:
            logger.warning(f"Overriding FlashInfer tensor core usage to {overriden_value}")
            return overriden_value
        GQA = self.config.num_qo_heads // self.config.num_kv_heads
        return GQA >= 4

    def prepare_for_capture(self, batch: Batch) -> None:
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        bs = batch.size
        assert bs in self.capture_bs and bs not in self.graph_wrappers and self.capture
        capture = self.capture
        self.graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=self.use_tensor_cores,
            indptr_buffer=capture.cu_seqlens_k[: bs + 1],
            indices_buffer=capture.indices,
            last_page_len_buffer=capture.one_tensor[:bs],
        )
        self.graph_wrappers[bs]._backend = "fa2"
        self.graph_wrappers[bs]._int_workspace_buffer = self.int_workspace_buffer
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)

    @nvtx_annotate("FlashInfer::prepare_replay")
    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FIMetadata) and not metadata.initialized
        assert self.capture is not None and bs in self.capture_bs
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)

    @cached_property
    def _mixed_wrapper(self):
        from flashinfer import BatchPODWithPagedKVCacheWrapper

        return BatchPODWithPagedKVCacheWrapper(self.float_workspace_buffer, kv_layout="NHD")

    def prepare_mixed_metadata(self, batch: Batch) -> None:
        assert batch.is_mixed and batch.mixed_parts is not None
        prefill, decode = batch.mixed_parts
        for part in (prefill, decode):
            self.prepare_metadata(part)
        batch.attn_metadata = FIPODMetadata(
            prefill.attn_metadata, decode.attn_metadata, self._mixed_wrapper,
        )

    def warmup_mixed(self, dummy_req: Request) -> None:
        logger.info_rank0("Preparing mixed attention kernels")
        parts = Batch([dummy_req], "prefill"), Batch([dummy_req], "decode")
        for part in parts:
            part.padded_reqs = part.reqs
        batch = Batch([dummy_req, dummy_req], "mixed", num_prefill_reqs=1)
        batch.mixed_parts = parts
        self.prepare_mixed_metadata(batch)
        q = torch.zeros(
            (2, self.qo_head_local, self.config.head_dim),
            dtype=self.kvcache.dtype, device=self.device,
        )
        self.forward(q, 0, batch)
        torch.cuda.current_stream(self.device).synchronize()

    @nvtx_annotate("FlashInfer::POD_plan")
    def _initialize_pod(self, metadata: FIPODMetadata) -> None:
        p, d = metadata.prefill, metadata.decode
        assert (p.page_size, p.dtype, p.head_dim) == (d.page_size, d.dtype, d.head_dim)
        started = perf_counter_ns() if self.profile_plan else 0
        self.last_event.synchronize()
        ready = perf_counter_ns() if self.profile_plan else 0
        metadata.wrapper.plan(
            qo_indptr_p=p.cu_seqlens_q_cpu,
            kv_indptr_p=p.cu_seqlens_k_cpu,
            kv_indices_p=p.indices,
            last_page_len_p=p.last_page_len_cpu,
            qo_indptr_d=d.cu_seqlens_q_cpu,
            kv_indptr_d=d.cu_seqlens_k_cpu,
            kv_indices_d=d.indices,
            last_page_len_d=d.last_page_len_cpu,
            num_qo_heads=p.num_qo_heads, num_kv_heads=p.num_kv_heads,
            head_dim=p.head_dim, page_size=p.page_size,
            pos_encoding_mode="NONE", q_data_type=p.dtype, kv_data_type=p.dtype,
            non_blocking=True,
        )
        self.last_event.record()
        metadata.initialized = True
        if self.profile_plan:
            metadata.plan_cpu_ms = {
                "workspace_wait": (ready - started) / 1e6,
                "plan": (perf_counter_ns() - ready) / 1e6,
            }
