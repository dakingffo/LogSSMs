from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import gc
from time import perf_counter
from typing import Callable

import torch

from logssms.utils import init_logger, nvtx_annotate


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

    def warmup(self, engine) -> None:
        from .prefill_warmup import warmup_prefill

        warmup_prefill(engine)

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
