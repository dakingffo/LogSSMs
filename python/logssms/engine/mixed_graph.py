from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace
from contextlib import nullcontext

import torch

from logssms.core import Batch
from logssms.cache.logstate.metadata import LogStateCacheMetadata
from logssms.kernel.metadata import copy_metadata, make_batch_mapping
from logssms.layers.linear_attention_backend.mixed import (
    MixedLinearCopyPlan, prepare_mixed_linear_metadata,
)
from logssms.utils import init_logger
from logssms.utils.batch import BatchIndices
from logssms.utils.mixed import prefill_buckets, prefill_partitions


logger = init_logger(__name__)


@dataclass
class MixedGraph:
    graph : torch.cuda.CUDAGraph
    batch : Batch
    logits: torch.Tensor
    linear_copy: MixedLinearCopyPlan | None = None


class MixedGraphRunner:
    def __init__(self, engine):
        self.graphs = {}
        self.sizes = []
        self.capacity = engine.config.mixed_prefill_tokens
        self.partitions = {len(p): p for p in prefill_partitions(
            self.capacity, engine.config.cache_config.prefill_alignment,
        )}
        self.buckets = prefill_buckets(self.capacity, engine.config.cache_config.prefill_alignment)
        self.buckets_by_count = {
            count: tuple(p for p in self.buckets if len(p) == count)
            for count in self.partitions
        }
        self.backend = engine.attn_backend
        self.linear_backend = getattr(engine, "linear_attn_backend", None)
        maximum = min(engine.config.max_running_req, engine.decode_graph_runner.max_graph_bs)
        if engine.config.scheduling_policy != "mixed" or maximum < 2:
            return
        self.sizes = sorted({maximum, *(1 << n for n in range(1, maximum.bit_length()))})
        if maximum >= 24 and 24 not in self.sizes:
            self.sizes.insert(bisect_left(self.sizes, 24), 24)
        pool = None
        for size in reversed(self.sizes):
            for partition in reversed(self.buckets):
                count = len(partition)
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
                linear_copy = (MixedLinearCopyPlan(tuple(part.linear_attn_metadata for part in batch.mixed_parts))
                               if self.linear_backend is not None else None)
                copy_metadata(
                    (batch.input_ids, batch.positions, batch.out_loc, batch.sample_indices),
                    (batch.input_ids, batch.positions, batch.out_loc, batch.sample_indices),
                )
                if linear_copy is not None:
                    linear_copy.copy(tuple(part.linear_attn_metadata for part in batch.mixed_parts))
                self.graphs[size, partition] = MixedGraph(graph, batch, logits, linear_copy)
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
        if engine.config.cache_type == "logstate":
            prefill.cache_metadata = LogStateCacheMetadata(first_offsets=[0] * count)
            decode.cache_metadata = LogStateCacheMetadata(replay_counts=[0] * (size - count))
        batch = Batch(prefill.reqs + decode.reqs, "mixed", num_prefill_reqs=count)
        batch.padded_reqs = batch.reqs
        batch.mixed_parts = prefill, decode
        batch.prefill_token_capacity = sum(partition)
        prefill.prefill_token_capacity = batch.prefill_token_capacity
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
            prepare_mixed_linear_metadata(self.linear_backend, prefill, decode, batch.prefill_token_capacity)
        self.backend.prepare_mixed_metadata(batch)
        batch.input_ids = torch.zeros(packet.num_tokens, dtype=torch.int32, device=engine.device)
        batch.sample_indices, batch.sample_req_indices = packet.sample_tuple
        return batch

    def padded_size(self, size):
        return self.sizes[bisect_left(self.sizes, size)] if self.sizes and size <= self.sizes[-1] else size

    def partition(self, lengths):
        return next(p for p in self.buckets_by_count[len(lengths)]
                    if all(n <= capacity for n, capacity in zip(lengths, p)))

    def _graph_key(self, batch):
        partition = batch.prefill_partition or self.partition(
            tuple(req.extend_len for req in batch.prefill_reqs)
        )
        return self.padded_size(batch.size), partition

    def can_use_cuda_graph(self, batch):
        return (batch.is_mixed and batch.hidden_captor is None
                and self._graph_key(batch) in self.graphs)

    def replay(self, batch, profiler=None):
        captured = self.graphs[self._graph_key(batch)]
        target = captured.batch
        count = len(batch.sample_indices)
        stage = profiler.record_stage if profiler is not None else lambda _: nullcontext()
        with stage("mixed_input_copy"):
            copy_metadata(
                (batch.input_ids, batch.positions, batch.out_loc, batch.sample_indices),
                (target.input_ids, target.positions, target.out_loc, target.sample_indices[:count]),
            )
        with stage("mixed_attention_plan"):
            self.backend.prepare_mixed_replay(target, batch)
        with stage("mixed_linear_copy"):
            if self.linear_backend is not None:
                captured.linear_copy.copy(tuple(part.linear_attn_metadata for part in batch.mixed_parts))
        with stage("mixed_graph_launch"):
            captured.graph.replay()
        return captured.logits[:count]

    def destroy_cuda_graphs(self):
        self.graphs.clear()
