from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from logssms.core import Batch, get_global_ctx
from logssms.kernel.metadata import copy_metadata
from logssms.layers.linear_attention_backend.mixed import (
    copy_mixed_linear_metadata_, prepare_mixed_linear_metadata,
)
from logssms.utils import init_logger

if TYPE_CHECKING:
    from .engine import Engine

logger = init_logger(__name__)


@dataclass
class MixedGraph:
    graph : torch.cuda.CUDAGraph
    batch : Batch
    logits: torch.Tensor


class MixedGraphRunner:
    def __init__(self, engine: Engine):
        self.graphs: dict[int, MixedGraph] = {}
        self.backend = engine.attn_backend
        self.linear_backend = getattr(engine, "linear_attn_backend", None)
        self.capacity = engine.config.mixed_prefill_tokens
        maximum = min(engine.config.max_running_req, engine.decode_graph_runner.max_graph_bs)
        if engine.config.scheduling_policy != "mixed" or maximum < 2:
            return
        buckets = [1 << n for n in range(1, maximum.bit_length())]
        pool = None
        for size in reversed(buckets):
            batch = self._capture_batch(engine, size)
            self.backend.make_mixed_capture(*batch.mixed_parts, engine.max_seq_len)
            with get_global_ctx().forward_batch(batch):
                logits = engine.model.forward()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=engine.stream):
                    logits = engine.model.forward()
            if pool is None:
                pool = graph.pool()
            self.graphs[size] = MixedGraph(graph, batch, logits)
        logger.info_rank0(f"Captured mixed CUDA graphs: prefill={self.capacity}, requests={buckets}")

    def _capture_batch(self, engine: Engine, size: int) -> Batch:
        prefill_req = replace(
            engine.dummy_req,
            input_ids=torch.zeros(self.capacity, dtype=torch.int32, device="cpu"),
            max_device_len=self.capacity + 1,
        )
        prefill = Batch([prefill_req], "prefill")
        decode = Batch([engine.dummy_req] * (size - 1), "decode")
        prefill.padded_reqs, decode.padded_reqs = prefill.reqs, decode.reqs
        prefill.logstate_first_offsets = [0]
        decode.logstate_replay_counts = [0] * (size - 1)
        for part in (prefill, decode):
            engine._prepare_indices(part)
            part.input_ids = torch.zeros(len(part.positions), dtype=torch.int32, device=engine.device)
            self.backend.prepare_metadata(part)
            if self.linear_backend is not None:
                self.linear_backend.prepare_metadata(part)
        if self.linear_backend is not None:
            prepare_mixed_linear_metadata(self.linear_backend, prefill, decode, self.capacity)
        batch = Batch(prefill.reqs + decode.reqs, "mixed")
        batch.padded_reqs = batch.reqs
        batch.mixed_parts = prefill, decode
        batch.prefill_token_capacity = self.capacity
        tokens = self.capacity + size - 1
        batch.input_ids = torch.zeros(tokens, dtype=torch.int32, device=engine.device)
        batch.positions = torch.cat((prefill.positions, decode.positions))
        batch.out_loc = torch.cat((prefill.out_loc, decode.out_loc))
        batch.sample_indices = torch.tensor(
            [self.capacity - 1, *range(self.capacity, tokens)],
            dtype=torch.int64, device=engine.device,
        )
        return batch

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_mixed and (1 << (batch.size - 1).bit_length()) in self.graphs

    def replay(self, batch: Batch) -> torch.Tensor:
        captured = self.graphs[batch.padded_size]
        target = captured.batch
        copy_metadata(
            (batch.input_ids, batch.positions, batch.out_loc, batch.sample_indices),
            (target.input_ids, target.positions, target.out_loc, target.sample_indices),
        )
        self.backend.prepare_mixed_replay(target.mixed_parts, batch.mixed_parts)
        if self.linear_backend is not None:
            copy_mixed_linear_metadata_(
                tuple(part.linear_attn_metadata for part in target.mixed_parts),
                tuple(part.linear_attn_metadata for part in batch.mixed_parts),
            )
        captured.graph.replay()
        return captured.logits[:batch.size]

    def destroy_cuda_graphs(self) -> None:
        self.graphs.clear()
