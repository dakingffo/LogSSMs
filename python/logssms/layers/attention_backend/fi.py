from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
import os
from time import perf_counter_ns
from typing import TYPE_CHECKING, Dict, List

import torch
from logssms.core import Batch, get_global_ctx
from logssms.distributed import get_tp_info
from logssms.env import ENV
from logssms.utils import div_ceil, div_even, init_logger, nvtx_annotate

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
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
    plan_cpu_ms:        dict | None = None
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
class FIMixedMetadata(BaseAttnMetadata):
    prefill: FIMetadata
    decode: FIMetadata
    prefill_capacity: int

    def get_last_indices(self, bs: int) -> torch.Tensor:
        count = len(self.prefill.seq_lens_cpu) - 1
        indices = self.prefill.get_last_indices(min(count, bs))
        if bs <= count:
            return indices
        decode_indices = torch.arange(bs - count, device=indices.device, dtype=indices.dtype)
        return torch.cat((indices, decode_indices + self.prefill_capacity))

    @property
    def plan_cpu_ms(self) -> dict | None:
        timings = [m.plan_cpu_ms for m in (self.prefill, self.decode) if m.plan_cpu_ms]
        return {key: sum(t.get(key, 0) for t in timings)
                for key in {k for t in timings for k in t}} if timings else None


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

    @cached_property
    def _plan_events(self):
        return {}

    @nvtx_annotate("FlashInfer::plan")
    def _initialize_metadata_once(self, metadata: FIMetadata) -> None:
        if metadata.initialized:
            return

        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        metadata.initialized = True
        started = perf_counter_ns() if self.profile_plan else 0
        event = self._plan_events.get(metadata.wrapper)
        if event is not None:
            event.synchronize()
        else:
            event = self._plan_events[metadata.wrapper] = torch.cuda.Event()
        ready = perf_counter_ns() if self.profile_plan else 0
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
        event.record()
        if self.profile_plan:
            metadata.plan_cpu_ms = {
                "workspace_wait": (ready - started) / 1e6,
                "plan": (perf_counter_ns() - ready) / 1e6,
            }

    def forward(
        self, q: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        if isinstance(metadata, FIMixedMetadata):
            from logssms.layers.piecewise import attention_output_buffer

            out = attention_output_buffer(q, q.shape)
            boundary = metadata.prefill_capacity
            for part, begin, end in ((metadata.prefill, 0, boundary),
                                     (metadata.decode, boundary, q.shape[0])):
                self._initialize_metadata_once(part)
                kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
                if part.page_size == 1:
                    kv_cache = tuple(cache.view(-1, 1, cache.shape[2], cache.shape[3])
                                     for cache in kv_cache)
                part.wrapper.run(q=q[begin:end], paged_kv_cache=kv_cache, out=out[begin:end])
            return out
        assert isinstance(metadata, FIMetadata)
        self._initialize_metadata_once(metadata)
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
        page_size = metadata.page_size
        if page_size == 1:
            kv_cache = tuple(
                cache.view(-1, 1, cache.shape[2], cache.shape[3])
                for cache in kv_cache
            )
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
    def _mixed_float_workspaces(self):
        return (torch.empty_like(self.float_workspace_buffer),
                torch.empty_like(self.float_workspace_buffer))

    @cached_property
    def _mixed_wrappers(self):
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper

        return (
            BatchPrefillWithPagedKVCacheWrapper(
                self._mixed_float_workspaces[0], kv_layout="NHD", backend="fa2"),
            BatchDecodeWithPagedKVCacheWrapper(
                self._mixed_float_workspaces[1], kv_layout="NHD", backend="fa2",
                use_tensor_cores=self.use_tensor_cores),
        )

    def prepare_mixed_metadata(self, batch: Batch) -> None:
        from logssms.kernel.metadata import gather_paged_indices
        from logssms.utils.batch import BatchIndices

        assert batch.is_mixed and batch.prefill_count > 0
        page_size = getattr(self.kvcache, "kv_page_size", 1)
        packet = batch.prepared_indices
        if packet is None:
            packet = BatchIndices.make(batch, self.device, page_size)
        pages = gather_paged_indices(packet, get_global_ctx().page_table)
        q = packet.cu_seqlens_q_cpu
        k = packet.cu_seqlens_k_cpu
        count = batch.prefill_count
        first_pages = int(k[count])
        padding = packet.prefill_padding
        qo = torch.cat((q[:count + 1], q[count:count + 1] + padding))
        kv = torch.cat((k[:count + 1], k[count:count + 1] + 1))
        indices = torch.cat((pages[:first_pages], pages[:1]))
        seq = torch.cat((packet.seq_lens_cpu[:count], packet.seq_lens_cpu.new_ones(1)))
        last = torch.cat((packet.last_page_len_cpu[:count], packet.last_page_len_cpu.new_ones(1)))
        prefill = FIMetadata(
            qo, kv, qo.to(self.device, non_blocking=True), indices, last,
            self.qo_head_local, self.kv_head_local, self.config.head_dim,
            page_size, "NONE", seq, self.kvcache.dtype, self._mixed_wrappers[0],
        )
        decode_q = q[count:] - q[count]
        decode = FIMetadata(
            decode_q, k[count:] - first_pages, packet.cu_seqlens_q_gpu[count:],
            pages[first_pages:], packet.last_page_len_cpu[count:],
            self.qo_head_local, self.kv_head_local, self.config.head_dim,
            page_size, "NONE", packet.seq_lens_cpu[count:], self.kvcache.dtype,
            self._mixed_wrappers[1],
        )
        batch.attn_metadata = FIMixedMetadata(prefill, decode, int(qo[-1]))

    def make_mixed_capture(self, batch: Batch, max_seq_len: int) -> None:
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper

        metadata = batch.attn_metadata
        for index, field in enumerate(("prefill", "decode")):
            source = getattr(metadata, field)
            size = len(source.seq_lens_cpu)
            kwargs = dict(dtype=torch.int32, device=self.device)
            qo = torch.empty(size + 1, **kwargs)
            kv = torch.empty(size + 1, **kwargs)
            pages = torch.empty(size * div_ceil(max_seq_len, source.page_size), **kwargs)
            last = torch.empty(size, **kwargs)
            common = dict(kv_layout="NHD", backend="fa2", use_cuda_graph=True)
            workspace = self._mixed_float_workspaces[index]
            if field == "prefill":
                wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    workspace, **common, qo_indptr_buf=qo, paged_kv_indptr_buf=kv,
                    paged_kv_indices_buf=pages, paged_kv_last_page_len_buf=last)
            else:
                qo.copy_(source.cu_seqlens_q_cpu)
                wrapper = BatchDecodeWithPagedKVCacheWrapper(
                    workspace, **common, use_tensor_cores=self.use_tensor_cores,
                    paged_kv_indptr_buffer=kv, paged_kv_indices_buffer=pages,
                    paged_kv_last_page_len_buffer=last)
            planned = replace(source, wrapper=wrapper, cu_seqlens_q_gpu=qo, initialized=False)
            setattr(metadata, field, planned)
            self._initialize_metadata_once(planned)

    def prepare_mixed_replay(self, captured: Batch, runtime: Batch) -> None:
        for field in ("prefill", "decode"):
            source = getattr(runtime.attn_metadata, field)
            target = getattr(captured.attn_metadata, field)
            planned = replace(source, wrapper=target.wrapper, initialized=False)
            self._initialize_metadata_once(planned)
            source.plan_cpu_ms = planned.plan_cpu_ms

    def warmup_mixed(self, dummy_req: Request) -> None:
        logger.info_rank0("Preparing mixed attention kernels")
        batch = Batch([dummy_req, dummy_req], "mixed", num_prefill_reqs=1)
        batch.padded_reqs = batch.reqs
        self.prepare_mixed_metadata(batch)
        q = torch.zeros(
            (2, self.qo_head_local, self.config.head_dim),
            dtype=self.kvcache.dtype, device=self.device,
        )
        self.forward(q, 0, batch)
        torch.cuda.current_stream(self.device).synchronize()
