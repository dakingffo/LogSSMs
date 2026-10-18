from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace

import torch

from logssms.core import Batch
from logssms.kernel.metadata import copy_metadata, make_batch_mapping
from logssms.layers.linear_attention_backend.mixed import (
    copy_mixed_linear_metadata_, prepare_mixed_linear_metadata,
)
from logssms.utils import init_logger
from logssms.utils.batch import BatchIndices
from logssms.utils.mixed import prefill_partitions


logger = init_logger(__name__)


@dataclass
class MixedGraph:
    graph : torch.cuda.CUDAGraph
    batch : Batch
    logits: torch.Tensor


class MixedGraphRunner:
    def __init__(self, engine):
        self.graphs = {}
        self.sizes = []
        self.capacity = engine.config.mixed_prefill_tokens
        self.partitions = {len(p): p for p in prefill_partitions(
            self.capacity, engine.config.cache_config.prefill_alignment,
        )}
        self.backend = engine.attn_backend
        self.linear_backend = getattr(engine, "linear_attn_backend", None)
        maximum = min(engine.config.max_running_req, engine.decode_graph_runner.max_graph_bs)
        if engine.config.scheduling_policy != "mixed" or maximum < 2:
            return
        self.sizes = sorted({maximum, *(1 << n for n in range(1, maximum.bit_length()))})
        pool = None
        for size in reversed(self.sizes):
            for count, partition in self.partitions.items():
                if count >= size:
                    continue
                batch = self._capture_batch(engine, size, partition)
                self.backend.make_mixed_capture(batch, engine.max_seq_len)
                with engine.ctx.forward_batch(batch):
                    engine.model.forward()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool, stream=engine.stream):
                        logits = engine.model.forward()
                if pool is None:
                    pool = graph.pool()
                self.graphs[size, count] = MixedGraph(graph, batch, logits)
        logger.info_rank0(f"Captured {len(self.graphs)} mixed CUDA graphs: prefill={self.capacity}, requests={self.sizes}")

    def _capture_batch(self, engine, size, partition):
        count = len(partition)
        prefill = Batch([
            replace(engine.dummy_req, input_ids=torch.zeros(capacity, dtype=torch.int32),
                    max_device_len=capacity + 1)
            for capacity in partition
        ], "prefill")
        decode = Batch([engine.dummy_req] * (size - count), "decode")
        prefill.padded_reqs, decode.padded_reqs = prefill.reqs, decode.reqs
        prefill.logstate_first_offsets = [0] * count
        decode.logstate_replay_counts = [0] * (size - count)
        batch = Batch(prefill.reqs + decode.reqs, "mixed", num_prefill_reqs=count)
        batch.padded_reqs = batch.reqs
        batch.mixed_parts = prefill, decode
        batch.prefill_token_capacity = self.capacity
        batch.prefill_partition = partition
        page_size = getattr(engine.kv_cache, "kv_page_size", 1)
        packet, pp, dp = BatchIndices.make_mixed(batch, engine.device, page_size)
        batch.prepared_indices = packet
        batch.positions, mapping, batch.out_loc = make_batch_mapping(packet, engine.page_table, decode=True)
        packet.token_table_ids = mapping[0]
        offset = 0
        for part, indices in ((prefill, pp), (decode, dp)):
            part.prepared_indices = indices
            end = offset + indices.num_tokens
            part.positions, part.out_loc = batch.positions[offset:end], batch.out_loc[offset:end]
            indices.token_table_ids = mapping[0][offset:end]
            if self.linear_backend is not None:
                self.linear_backend.prepare_metadata(part)
            offset = end
        if self.linear_backend is not None:
            prepare_mixed_linear_metadata(self.linear_backend, prefill, decode, self.capacity)
        self.backend.prepare_mixed_metadata(batch)
        batch.input_ids = torch.zeros(packet.num_tokens, dtype=torch.int32, device=engine.device)
        batch.sample_indices, batch.sample_req_indices = packet.sample_tuple
        return batch

    def padded_size(self, size):
        return self.sizes[bisect_left(self.sizes, size)] if self.sizes and size <= self.sizes[-1] else size

    def partition(self, count):
        return self.partitions[count]

    def can_use_cuda_graph(self, batch):
        return (batch.is_mixed and batch.hidden_captor is None
                and (self.padded_size(batch.size), batch.prefill_count) in self.graphs)

    def replay(self, batch):
        captured = self.graphs[batch.padded_size, batch.prefill_count]
        target = captured.batch
        count = len(batch.sample_indices)
        copy_metadata(
            (batch.input_ids, batch.positions, batch.out_loc, batch.sample_indices),
            (target.input_ids, target.positions, target.out_loc, target.sample_indices[:count]),
        )
        self.backend.prepare_mixed_replay(target, batch)
        if self.linear_backend is not None:
            copy_mixed_linear_metadata_(
                tuple(part.linear_attn_metadata for part in target.mixed_parts),
                tuple(part.linear_attn_metadata for part in batch.mixed_parts),
            )
        captured.graph.replay()
        return captured.logits[:count]

    def destroy_cuda_graphs(self):
        self.graphs.clear()
