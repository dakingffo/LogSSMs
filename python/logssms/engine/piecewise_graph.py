from __future__ import annotations

from bisect import bisect_left
from contextlib import contextmanager
from dataclasses import dataclass
import gc
from time import perf_counter
from typing import Callable

import torch

from logssms.cache.logstate.metadata import LogStateCacheMetadata
from logssms.core import Batch, Request
from logssms.kernel.metadata import make_batch_mapping
from logssms.utils import init_logger, nvtx_annotate
from logssms.utils.batch import BatchIndices


logger = init_logger(__name__)


def piecewise_token_buckets(max_tokens: int) -> list[int]:
    if max_tokens < 1:
        return []
    buckets = [1, 2, 4, 8, 16, 32]
    buckets += list(range(64, min(max_tokens, 1024) + 1, 32))
    buckets += list(range(1088, min(max_tokens, 4096) + 1, 64))
    buckets += list(range(4224, max_tokens + 1, 128))
    buckets = [size for size in buckets if size <= max_tokens]
    if buckets[-1] != max_tokens:
        buckets.append(max_tokens)
    return buckets


def _warmup_lengths(granularity, token_budget):
    from logssms.kernel.linear_attention.vendor.fla.config import PREFILL_CHUNK_BOUNDS

    chunks = {1, *(n - 1 for n in PREFILL_CHUNK_BOUNDS), *PREFILL_CHUNK_BOUNDS}
    return sorted({min(token_budget, max(1, n * granularity - offset))
                   for n in chunks for offset in (0, 1)})


def _warmup_cases(granularity, token_budget, max_requests):
    cases = set()
    for count in range(1, min(4, max_requests, token_budget) + 1):
        for total in _warmup_lengths(granularity, token_budget):
            total = max(total, count)
            length, remainder = divmod(total, count)
            cases.add(tuple(length + (i < remainder) for i in range(count)))
    return sorted(cases)


@contextmanager
def _warmup_storage(engine, lengths):
    from logssms.cache.logstate.frame_pool import MHALogStatePool
    from logssms.cache.marconi.block_pool import HybridBlockPool

    ctx, pool = engine.ctx, engine.ctx.linear_state
    original = ctx.page_table, ctx.state_table, ctx.work_state_table
    ctx.page_table = original[0].clone()
    ctx.state_table = original[1].clone() if original[1] is not None else None
    ctx.work_state_table = original[2].clone() if original[2] is not None else None
    kv_size = getattr(engine.kv_cache, "kv_page_size", engine.config.page_size)
    kv_counts = [(n + kv_size - 1) // kv_size for n in lengths]
    state_counts = [(n + engine.config.page_size - 1) // engine.config.page_size for n in lengths]
    allocation = None
    shared_kv, shared_states = (), []
    try:
        if isinstance(pool, HybridBlockPool):
            shared_kv = pool.allocator.allocate(sum(kv_counts))
            for _ in lengths:
                slot = pool.allocate_state()
                shared_states.append(slot)
                pool.zero_state(slot)
            kv = torch.tensor(shared_kv, device=engine.device) * kv_size
            states = torch.tensor([slot for slot, count in zip(shared_states, state_counts)
                                   for _ in range(count)], device=engine.device)
        elif isinstance(pool, MHALogStatePool):
            allocation = pool.reserve(kv=sum(kv_counts), logs=sum(state_counts), works=len(lengths))
            if allocation is None:
                raise RuntimeError("Insufficient cache capacity for piecewise graph warmup")
            kv, states, works = allocation
            kv = torch.tensor(kv, device=engine.device) * kv_size
            states = torch.tensor(states, device=engine.device)
            ctx.work_state_table[:len(lengths), 0] = torch.tensor(works, device=engine.device)
            frames = []
            for slot in (*allocation[1], *works):
                record = pool.allocator.slot(slot)
                frames.extend(record.payload)
                frames.extend(frame for group in record.conv for frame in group)
                if record.decay is not None:
                    start = record.decay.index * pool.layout.log_decay_bytes
                    pool.storage[record.decay.frame, start:start + pool.layout.log_decay_bytes].zero_()
            pool.storage.index_fill_(0, torch.tensor(frames, device=engine.device), 0)
        else:
            kv = engine.cache_manager.free_pages[:sum(kv_counts)]
            if len(kv) != sum(kv_counts):
                raise RuntimeError("Insufficient KV capacity for piecewise graph warmup")
            if pool is not None:
                states = engine.cache_manager.free_states[:sum(state_counts)]
                if len(states) != sum(state_counts):
                    raise RuntimeError("Insufficient state capacity for piecewise graph warmup")
                pool.conv_state.index_fill_(0, states, 0)
                pool.recurrent_state.index_fill_(0, states, 0)
        kv_offset = state_offset = 0
        for row, length in enumerate(lengths):
            pages = kv[kv_offset:kv_offset + kv_counts[row]]
            mapping = (pages[:, None] + torch.arange(kv_size, device=engine.device)).flatten()
            ctx.page_table[row, :length] = mapping[:length]
            if pool is not None:
                slots = states[state_offset:state_offset + state_counts[row]]
                ctx.state_table[row, :length] = slots.repeat_interleave(engine.config.page_size)[:length]
            kv_offset += kv_counts[row]
            state_offset += state_counts[row]
        yield
    finally:
        torch.cuda.current_stream(engine.device).synchronize()
        ctx.page_table, ctx.state_table, ctx.work_state_table = original
        if isinstance(pool, HybridBlockPool):
            pool.allocator.release(shared_kv)
            for slot in shared_states:
                pool.release_state(slot)
        if allocation is not None:
            pool.release_kv(allocation[0])
            pool.release_slots((*allocation[1], *allocation[2]))


@dataclass
class PiecewiseGraphPiece:
    graph : torch.cuda.CUDAGraph
    inputs: tuple[torch.Tensor | None, ...]
    output: tuple[torch.Tensor, ...]


class GraphBridge:
    def __init__(self, finish, prepare):
        self.finish = finish
        self.prepare = prepare

    def forward(self, *args):
        hidden, residual = self.finish.piecewise_function(self.finish.__self__, *args)
        return self.prepare.piecewise_function(self.prepare.__self__, hidden, residual)

    forward.piecewise_function = forward


class PiecewiseGraphRunner:
    def __init__(self, model, max_tokens: int, stream: torch.cuda.Stream):
        self.enabled = True
        self.stream = stream
        self.buckets = piecewise_token_buckets(max_tokens)
        self.graphs: dict[tuple[object, Callable, int], PiecewiseGraphPiece] = {}
        self.buffers: dict[tuple, torch.Tensor] = {}
        self.pieces: set[tuple[object, Callable]] = set()
        self.pool = None
        self.replays = 0
        self.padded_tokens = 0
        self.capture_seconds = 0.0
        self.capture_memory_bytes = 0
        layers = model.model.layers.op_list
        self.layers = layers
        self.plans = {}
        specifications = []
        layer_specs = []
        for layer in layers:
            if hasattr(layer, "piecewise_graph_pieces"):
                pieces = layer.piecewise_graph_pieces()
                layer_specs.append(pieces)
                specifications.extend(pieces)
        self.chained = (bool(layers) and len(layer_specs) == len(layers)
                        and all(len(pieces) == 2 and hasattr(layer, "attend_features")
                                for layer, pieces in zip(layers, layer_specs)))
        if self.chained:
            specifications = [layer_specs[0][0]]
            for previous, following in zip(layer_specs, layer_specs[1:]):
                finish, examples, names, _ = previous[1]
                prepare, _, _, output_names = following[0]
                bridge = GraphBridge(finish, prepare)
                specifications.append((bridge.forward, examples, names, output_names))
            specifications.append(layer_specs[-1][1])
        if not specifications or not self.buckets:
            logger.warning_rank0("No graph pieces registered for this model/budget; using eager execution.")
            return
        logger.info_rank0(
            f"Preparing piecewise graphs: {len(specifications)} regions, "
            f"{len(self.buckets)} buckets (max={self.buckets[-1]})."
        )
        start = perf_counter()
        allocated_before = torch.cuda.memory_allocated(stream.device)


        torch.cuda.synchronize(stream.device)
        gc.collect()
        torch.cuda.empty_cache()

        with torch.cuda.stream(stream), torch.no_grad():
            def get_buffer(name, example):
                key = (name, example.shape[1:], example.dtype, example.device)
                if key not in self.buffers:
                    self.buffers[key] = torch.zeros(
                        (self.buckets[-1],) + example.shape[1:],
                        device=example.device, dtype=example.dtype,
                    )
                return self.buffers[key]

            for piece_index, (method, examples, names, output_names) in enumerate(specifications):
                function = method.piecewise_function
                key = (method.__self__, function)
                self.pieces.add(key)
                buffers = []
                for example, name in zip(examples, names):
                    if example is None:
                        buffers.append(None)
                        continue
                    buffers.append(get_buffer(name, example))
                for bucket_index, tokens in enumerate(reversed(self.buckets)):
                    inputs = tuple(buffer[:tokens] if buffer is not None else None
                                   for buffer in buffers)
                    for _ in range(3 if bucket_index == 0 else 1):
                        sample = function(method.__self__, *inputs)
                    outputs = tuple(get_buffer(name, tensor)[:tokens]
                                    for name, tensor in zip(output_names, sample))
                    del sample
                    graph = torch.cuda.CUDAGraph()
                    graph.capture_begin(pool=self.pool)
                    try:
                        result = function(method.__self__, *inputs)
                        for source, target in zip(result, outputs):
                            if source.data_ptr() != target.data_ptr() or source.stride() != target.stride():
                                target.copy_(source)
                    finally:
                        graph.capture_end()
                    del result
                    if self.pool is None:
                        self.pool = graph.pool()
                    self.graphs[key + (tokens,)] = PiecewiseGraphPiece(graph, inputs, outputs)
                if (piece_index + 1) % 8 == 0:
                    logger.info_rank0(f"Piecewise graph regions captured: {piece_index + 1}/{len(specifications)}")
        torch.cuda.synchronize(stream.device)
        self.capture_seconds = perf_counter() - start
        if self.chained:
            for bucket in self.buckets:
                self.plans[bucket] = tuple(
                    self.graphs[(method.__self__, method.piecewise_function, bucket)]
                    for method, *_ in specifications
                )
        self.capture_memory_bytes = torch.cuda.memory_allocated(stream.device) - allocated_before
        logger.info_rank0(
            f"Captured {len(self.graphs)} pieces: "
            f"{len(self.pieces)} regions × {len(self.buckets)} token buckets "
            f"(max={self.buckets[-1]})."
        )

    def can_use_cuda_graph(self, batch) -> bool:
        if (not self.enabled or not self.graphs
                or not batch.is_prefill or batch.hidden_captor is not None):
            return False
        inputs = getattr(batch, "input_ids", None)
        tokens = len(inputs) if inputs is not None else sum(req.extend_len for req in batch.reqs)
        return 0 < tokens <= self.buckets[-1]

    @torch.inference_mode()
    def warmup(self, engine) -> None:
        if not self.enabled or not self.graphs:
            return
        limit = min(self.buckets[-1], engine.max_seq_len)
        cases = {(n,) for n in self.buckets if n <= limit}
        if engine.config.model_config.is_hybrid:
            cases.update(_warmup_cases(
                engine.config.cache_config.prefill_alignment, limit, engine.config.max_running_req,
            ))
        cases = sorted(cases, key=lambda lengths: (sum(lengths), lengths), reverse=True)
        logger.info_rank0(f"Preparing complete piecewise forwards: {len(cases)} cases")
        replays, padded = self.replays, self.padded_tokens
        try:
            with torch.cuda.stream(engine.stream):
                for index, lengths in enumerate(cases):
                    with _warmup_storage(engine, lengths):
                        reqs = [Request(torch.zeros(n, dtype=torch.int32), row, 0, 1,
                                        -row - 1, None, None, n + 1)
                                for row, n in enumerate(lengths)]
                        batch = Batch(reqs, "prefill")
                        batch.padded_reqs = reqs
                        if engine.config.cache_type == "logstate":
                            batch.cache_metadata = LogStateCacheMetadata(first_offsets=[0] * len(reqs))
                        page_size = getattr(engine.kv_cache, "kv_page_size", 1)
                        packet = batch.prepared_indices = BatchIndices.make(batch, engine.device, page_size)
                        batch.positions, mapping, batch.out_loc = make_batch_mapping(
                            packet, engine.ctx.page_table, decode=engine.attn_backend.uses_paged_indices,
                        )
                        packet.token_table_ids = mapping[0]
                        batch.sample_indices, batch.sample_req_indices = packet.sample_tuple
                        batch.input_ids = torch.zeros(sum(lengths), dtype=torch.int32, device=engine.device)
                        engine.attn_backend.prepare_metadata(batch)
                        if engine.config.model_config.is_hybrid:
                            engine.linear_attn_backend.prepare_metadata(batch)
                        with engine.ctx.forward_batch(batch):
                            engine.model.forward()
                    if (index + 1) % 8 == 0:
                        logger.info_rank0(f"Complete piecewise forwards: {index + 1}/{len(cases)}")
        finally:
            self.replays, self.padded_tokens = replays, padded
        logger.info_rank0("Complete piecewise forwards ready")

    def token_bucket(self, tokens: int) -> int:
        return self.buckets[bisect_left(self.buckets, tokens)]

    def can_run_layers(self, batch) -> bool:
        return self.chained and self.can_use_cuda_graph(batch)

    @nvtx_annotate("PiecewiseGraph::layers")
    def forward_layers(self, hidden, positions):
        tokens = len(hidden)
        bucket = self.token_bucket(tokens)
        plan = self.plans[bucket]
        output = self._replay(plan[0], (hidden, None), tokens)
        for layer, piece in zip(self.layers, plan[1:]):
            *features, residual = output
            args = (*layer.attend_features(features, positions), residual)
            output = self._replay(piece, args, tokens)
        self.replays += len(plan)
        self.padded_tokens += (bucket - tokens) * len(plan)
        return output

    @staticmethod
    def _replay(piece, args, tokens):
        for source, target in zip(args, piece.inputs):
            if target is None:
                assert source is None
                continue
            assert source is not None and source.shape[0] == tokens
            if source.data_ptr() != target.data_ptr() or source.stride() != target.stride():
                target[:tokens].copy_(source)
        piece.graph.replay()
        return tuple(output[:tokens] for output in piece.output)

    @nvtx_annotate("PiecewiseGraph::replay")
    def run(self, owner, function, args):
        key = (owner, function)
        if key not in self.pieces:
            return function(owner, *args)
        tokens = args[0].shape[0]
        bucket = self.token_bucket(tokens)
        piece = self.graphs[key + (bucket,)]
        output = self._replay(piece, args, tokens)
        self.replays += 1
        self.padded_tokens += bucket - tokens
        return output

    def destroy_cuda_graphs(self) -> None:
        self.enabled = False
        self.graphs.clear()
        self.buffers.clear()
        self.pieces.clear()
        self.plans.clear()
        self.pool = None
