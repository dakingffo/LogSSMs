from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
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
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    )
    from logssms.models import ModelConfig


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


class FlashInferBackend(BaseAttnBackend):
    uses_paged_indices = True

    def __init__(self, config: ModelConfig) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.config = config
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
    def _initialize_metadata_once(self, metadata: FIMetadata) -> None:
        if metadata.initialized:
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
        assert isinstance(metadata, FIMetadata)
        self._initialize_metadata_once(metadata)
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
        if metadata.page_size == 1:
            kv_cache = tuple(
                cache.view(-1, 1, cache.shape[2], cache.shape[3])
                for cache in kv_cache
            )
        if (batch.is_prefill
                and metadata.wrapper is getattr(self, "_mixed_eager_wrappers", (None,))[0]
                and q.shape[0] > int(metadata.cu_seqlens_q_cpu[-1])):
            live = int(metadata.cu_seqlens_q_cpu[-1])
            output = metadata.wrapper.run(q=q[:live], paged_kv_cache=kv_cache)
            return torch.cat((output, output.new_zeros(
                (q.shape[0] - live,) + output.shape[1:])), dim=0)
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
    def _mixed_float_workspaces(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Eager and graph execution share scratch on the serialized model stream.
        return (torch.zeros_like(self.float_workspace_buffer),
                torch.zeros_like(self.float_workspace_buffer))

    def prepare_mixed_metadata(self, parts: tuple[Batch, Batch]) -> None:
        """Assign eager wrappers after both parts' metadata is prepared.

        Planning is deferred to forward on the model stream. The engine must
        order that stream after scheduler-side metadata preparation.
        """
        assert parts[0].is_prefill and parts[1].is_decode
        if not hasattr(self, "_mixed_eager_wrappers"):
            from flashinfer import (
                BatchDecodeWithPagedKVCacheWrapper,
                BatchPrefillWithPagedKVCacheWrapper,
            )
            self._mixed_eager_wrappers = (
                BatchPrefillWithPagedKVCacheWrapper(
                    self._mixed_float_workspaces[0],
                    kv_layout="NHD", backend="fa2",
                ),
                BatchDecodeWithPagedKVCacheWrapper(
                    self._mixed_float_workspaces[1],
                    kv_layout="NHD", backend="fa2",
                    use_tensor_cores=self.use_tensor_cores,
                ),
            )
        for part, wrapper in zip(parts, self._mixed_eager_wrappers):
            assert isinstance(part.attn_metadata, FIMetadata)
            if part.attn_metadata.wrapper is not wrapper:
                part.attn_metadata = replace(part.attn_metadata, wrapper=wrapper,
                                             initialized=False)

    def make_mixed_capture(
        self, prefill: Batch, decode: Batch, max_seq_len: int,
    ) -> tuple[Batch, Batch]:
        """Plan capture parts at P prefill rows and D decode rows on the model stream.

        Wrappers own integer planning buffers and share mixed float scratch.
        Retain these parts for the graph's lifetime. After planning L < P,
        replay the graph directly; Python forward requires matching live rows.
        """
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        assert prefill.is_prefill and decode.is_decode and max_seq_len > 0
        for part in (prefill, decode):
            if not isinstance(getattr(part, "attn_metadata", None), FIMetadata):
                self.prepare_metadata(part)
        pm, dm = prefill.attn_metadata, decode.attn_metadata
        assert isinstance(pm, FIMetadata) and isinstance(dm, FIMetadata)
        assert len(pm.cu_seqlens_q_cpu) == 2, "One prefill request per mixed graph"
        assert int(pm.cu_seqlens_q_cpu[-1]) == prefill.input_ids.shape[0]
        assert 0 < prefill.input_ids.shape[0] <= max_seq_len
        assert len(dm.seq_lens_cpu) == decode.input_ids.shape[0] > 0
        assert torch.all(dm.cu_seqlens_q_cpu.diff() == 1)

        for part, source, is_prefill in ((prefill, pm, True), (decode, dm, False)):
            bs = len(source.seq_lens_cpu)
            max_pages = (max_seq_len + source.page_size - 1) // source.page_size
            index_args = dict(dtype=source.indices.dtype, device=self.device)
            kv_indptr = torch.empty(bs + 1, **index_args)
            kv_indices = torch.empty(bs * max_pages, **index_args)
            last_len = torch.empty(bs, **index_args)
            workspace = self._mixed_float_workspaces[0 if is_prefill else 1]
            if is_prefill:
                qo_indptr = torch.empty_like(source.cu_seqlens_q_gpu)
                wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    workspace, kv_layout="NHD", backend="fa2", use_cuda_graph=True,
                    qo_indptr_buf=qo_indptr,
                    paged_kv_indptr_buf=kv_indptr,
                    paged_kv_indices_buf=kv_indices,
                    paged_kv_last_page_len_buf=last_len,
                )
            else:
                qo_indptr = source.cu_seqlens_q_gpu.clone()
                wrapper = BatchDecodeWithPagedKVCacheWrapper(
                    workspace, kv_layout="NHD", backend="fa2", use_cuda_graph=True,
                    use_tensor_cores=self.use_tensor_cores,
                    paged_kv_indptr_buffer=kv_indptr,
                    paged_kv_indices_buffer=kv_indices,
                    paged_kv_last_page_len_buffer=last_len,
                )
            metadata = replace(source, wrapper=wrapper,
                               cu_seqlens_q_gpu=qo_indptr, initialized=False)
            self._initialize_metadata_once(metadata)
            part.attn_metadata = metadata
        return prefill, decode

    def prepare_mixed_replay(
        self, captured_parts: tuple[Batch, Batch], runtime_parts: tuple[Batch, Batch],
    ) -> None:
        """Plan actual lengths outside capture on the graph's execution stream.

        Runtime metadata may contain d <= D decode requests. Missing attention
        lanes read the first request's KV pages (no writes); the runner must
        independently mask KV stores and provide safe dummy recurrent slots.
        """
        for i, (captured, runtime) in enumerate(zip(captured_parts, runtime_parts)):
            target, source = captured.attn_metadata, runtime.attn_metadata
            assert isinstance(target, FIMetadata) and isinstance(source, FIMetadata)
            assert (source.page_size, source.dtype, source.num_qo_heads,
                    source.num_kv_heads, source.head_dim) == (
                        target.page_size, target.dtype, target.num_qo_heads,
                        target.num_kv_heads, target.head_dim)
            capacity = int(target.cu_seqlens_q_cpu[-1])
            if i == 0:
                assert len(source.cu_seqlens_q_cpu) == 2
                assert 0 < int(source.cu_seqlens_q_cpu[-1]) <= capacity
            else:
                bs = len(target.seq_lens_cpu)
                live = len(source.seq_lens_cpu)
                assert 0 < live <= bs
                assert torch.all(source.cu_seqlens_q_cpu.diff() == 1)
                if live < bs:
                    # Repeat an actual valid KV segment; no empty-page edge
                    # cases or dependence on the scheduler's dummy KV table.
                    begin, end = map(int, source.cu_seqlens_k_cpu[:2])
                    pages = end - begin
                    assert pages > 0
                    extra = bs - live
                    tail = (torch.arange(1, extra + 1,
                                         dtype=source.cu_seqlens_k_cpu.dtype)
                            * pages + source.cu_seqlens_k_cpu[-1])
                    source = replace(
                        source,
                        cu_seqlens_k_cpu=torch.cat((source.cu_seqlens_k_cpu, tail)),
                        indices=torch.cat((source.indices,
                                           source.indices[begin:end].repeat(extra))),
                        last_page_len_cpu=torch.cat((source.last_page_len_cpu,
                                                     source.last_page_len_cpu[:1].repeat(extra))),
                        seq_lens_cpu=torch.cat((source.seq_lens_cpu,
                                               source.seq_lens_cpu[:1].repeat(extra))),
                        cu_seqlens_q_cpu=target.cu_seqlens_q_cpu,
                    )
            # Retain capture metadata (especially its capacity) and all graph
            # pointers. Only wrapper-owned planning buffers are updated.
            planned = replace(source, wrapper=target.wrapper, initialized=False)
            self._initialize_metadata_once(planned)
