from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from logssms.core import Batch, Request, get_global_ctx
from logssms.distributed import get_tp_info
from logssms.utils import init_logger, nvtx_annotate

if TYPE_CHECKING:
    from .profile import EngineProfiler
    from logssms.layers.attention_backend import BaseAttnBackend
    from logssms.layers.linear_attention_backend import BaseLinearAttentionBackend
    from logssms.models import BaseLLMModel

logger = init_logger(__name__)


@dataclass(init=False)
class DecodeGraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc  : torch.Tensor
    positions: torch.Tensor
    logits   : torch.Tensor

    def __init__(self, bs: int, vocab_size: int, device: torch.device):
        self.input_ids=torch.zeros(bs, dtype=torch.int32, device=device)
        self.out_loc=torch.zeros(bs, dtype=torch.int32, device=device)
        self.positions=torch.zeros(bs, dtype=torch.int32, device=device)
        self.logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device)

    def set_batch(self, batch: Batch) -> None:
        batch.input_ids = self.input_ids[0:batch.padded_size]
        batch.out_loc = self.out_loc[0:batch.padded_size]
        batch.positions = self.positions[0:batch.padded_size]

    @nvtx_annotate("DecodeGraph::copy_inputs")
    def copy_from(self, batch: Batch) -> None:
        from logssms.kernel.metadata import copy_metadata

        copy_metadata(
            (batch.input_ids, batch.out_loc, batch.positions),
            (self.input_ids[:batch.padded_size], self.out_loc[:batch.padded_size],
             self.positions[:batch.padded_size]),
        )


def _determine_cuda_graph_bs(
    cuda_graph_bs    : list[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory      : int,
) -> list[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    if cuda_graph_max_bs is None:
        free_memory_gb = free_memory / (1024**3)
        if free_memory_gb > 80:
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    bs = [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
    return [b for b in bs if b <= cuda_graph_max_bs]


def mem_GB(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class DecodeGraphRunner:
    def __init__(
        self,
        stream             : torch.cuda.Stream,
        device             : torch.device,
        model              : BaseLLMModel,
        attn_backend       : BaseAttnBackend,
        linear_attn_backend: BaseLinearAttentionBackend | None,
        cuda_graph_bs      : list[int] | None,
        cuda_graph_max_bs  : int | None,
        free_memory        : int,
        max_seq_len        : int,
        vocab_size         : int,
        dummy_req          : Request,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.linear_attn_backend = linear_attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.stream = stream
        self.device = device
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("Decode CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)
        if self.linear_attn_backend is not None:
            self.linear_attn_backend.init_capture_graph(
                max_seq_len=max_seq_len, bs_list=self.graph_bs_list
            )

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = DecodeGraphCaptureBuffer(self.max_graph_bs, vocab_size, self.device)

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.buffer.set_batch(batch)
            self.attn_backend.prepare_for_capture(batch)
            if self.linear_attn_backend is not None:
                self.linear_attn_backend.prepare_for_capture(batch)
            with get_global_ctx().forward_batch(batch):
                self.buffer.logits[:bs] = model.forward()
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
            if pool is None:
                pool = graph.pool()
            self.graphs[bs] = graph

        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.size <= self.max_graph_bs

    @nvtx_annotate("DecodeGraph::replay")
    def replay(self, batch: Batch, *, profiler: EngineProfiler | None = None) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        if profiler is not None:
            return self._replay_profiled(batch, profiler)
        self.buffer.copy_from(batch)
        self.attn_backend.prepare_for_replay(batch)
        if self.linear_attn_backend is not None:
            self.linear_attn_backend.prepare_for_replay(batch)
        self.graphs[batch.padded_size].replay()
        return self.buffer.logits[:batch.size]

    def _replay_profiled(self, batch: Batch, profiler: EngineProfiler) -> torch.Tensor:
        with profiler.record_stage("graph_copy_inputs"):
            self.buffer.copy_from(batch)
        with profiler.record_stage("graph_attention_metadata"):
            self.attn_backend.prepare_for_replay(batch)
        with profiler.record_stage("graph_linear_metadata"):
            if self.linear_attn_backend is not None:
                self.linear_attn_backend.prepare_for_replay(batch)
        with profiler.record_stage("graph_replay"):
            self.graphs[batch.padded_size].replay()
        return self.buffer.logits[:batch.size]

    def pad_batch(self, batch: Batch) -> None:
        if batch.is_verify:
            batch.padded_reqs = batch.reqs
        else:
            padded_size = (
                next(bs for bs in self.graph_bs_list if bs >= batch.size)
                if self.can_use_cuda_graph(batch) else batch.size
            )
            batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)


    def destroy_cuda_graphs(self) -> None:
        del self.graphs
        gc.collect()
