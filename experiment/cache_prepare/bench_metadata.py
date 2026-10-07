"""Isolated metadata/reclamation microbench; not a serving or cache-hit benchmark.

--source-root can point to a saved python/ tree, including uncommitted sources.
No model, sampling, chat template or synthetic conversation is involved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-root", type=Path, default=Path("python"))
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--repeats", type=int, default=50)
args = parser.parse_args()
sys.path.insert(0, str(args.source_root.resolve()))

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from logssms.core import Batch, Context, Request, clear_global_ctx, set_global_ctx
from logssms.engine.engine import Engine
from logssms.cache.logstate_pool.frame_pool import FramedMHALogStatePool
from logssms.layers.attention_backend.fi import FlashInferBackend
from logssms.layers.linear_attention_backend import GatedDeltaLogStateBackend, GatedDeltaBackend

torch.set_num_threads(1)
device = torch.device("cuda")


class TransferCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.transfers = 0
        self.bytes = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        destination = kwargs.get("device")
        if func == torch.ops.aten._to_copy.default and args[0].is_cpu and destination is not None:
            if torch.device(destination).type == "cuda":
                self.transfers += 1
                self.bytes += args[0].numel() * args[0].element_size()
        return func(*args, **kwargs)


def make_pool():
    return FramedMHALogStatePool(
        num_frames=2048, page_size=64, granularity=32,
        num_attention_layers=2, num_kv_heads=1, head_dim=32,
        num_linear_layers=2, num_key_heads=2, num_value_heads=2,
        head_k_dim=32, head_v_dim=32, conv_dim=192, state_len=3,
        device=device, dtype=torch.float16,
    )


def metadata_case(bs, phase, page_size, linear):
    ctx = Context(64)
    ctx.page_table = torch.arange(8192, dtype=torch.int32, device=device).repeat(bs, 1)
    # Use different physical pages in every row, without assuming contiguous IDs.
    for row in range(bs):
        ctx.page_table[row] += row * 16384
    ctx.state_table = (ctx.page_table % 31).to(torch.int32)
    ctx.work_state_table = torch.arange(bs, dtype=torch.int32, device=device)[:, None]
    ctx.linear_state = make_pool()
    ctx.kv_cache = SimpleNamespace(device=device, dtype=torch.float16, kv_page_size=page_size)
    set_global_ctx(ctx)
    engine = object.__new__(Engine)
    engine.device = device
    engine.ctx = ctx
    engine.page_table = ctx.page_table
    engine.profiler = SimpleNamespace(enabled=False)
    engine.config = SimpleNamespace(model_config=SimpleNamespace(is_hybrid=True))
    engine.graph_runner = SimpleNamespace(pad_batch=lambda batch: setattr(batch, "padded_reqs", batch.reqs))
    engine.attn_backend = object.__new__(FlashInferBackend)
    for name, value in dict(device=device, kvcache=ctx.kv_cache, qo_head_local=4,
                            kv_head_local=1, config=SimpleNamespace(head_dim=32),
                            decode_wrappers=None, prefill_wrapper=None,
                            cached_ones_cpu=torch.empty(0, dtype=torch.int32, pin_memory=True)).items():
        setattr(engine.attn_backend, name, value)
    engine.linear_attn_backend = GatedDeltaLogStateBackend() if linear == "logstate" else GatedDeltaBackend()
    engine.sampler = SimpleNamespace(prepare=lambda batch: None)
    def cache_prepare(batch):
        batch.logstate_replay_counts = [i % 16 for i in range(bs)]
        batch.logstate_first_offsets = [req.cached_len % 64 for req in batch.reqs]
    cache = SimpleNamespace(allocate=lambda *a, **kw: None, prepare_batch=cache_prepare)
    reqs = []
    for row in range(bs):
        cached = 1024 + (row % 8) * 64
        length = 1 if phase == "decode" else 64 + (row % 4) * 64
        reqs.append(Request(torch.zeros(cached + length, dtype=torch.int32), row, cached,
                            1, row, None, None, 8192))
    batch = Batch(reqs, phase)
    try:
        result = engine.prepare_batch(batch, cache)
        positions = [p for req in reqs for p in range(req.cached_len, req.device_len)]
        rows = [req.table_idx for req in reqs for _ in range(req.extend_len)]
        torch.testing.assert_close(batch.positions.cpu(), torch.tensor(positions, dtype=torch.int32))
        torch.testing.assert_close(result.input_tuple[0].cpu(), torch.tensor(rows, dtype=torch.int64))
        expected_pages = torch.cat([ctx.page_table[r.table_idx, :r.device_len:page_size] // page_size for r in reqs])
        torch.testing.assert_close(batch.attn_metadata.indices, expected_pages)
        torch.testing.assert_close(batch.out_loc, ctx.page_table[result.input_tuple])
        if phase == "decode":
            if linear == "logstate":
                torch.testing.assert_close(batch.linear_attn_metadata.log_slots, ctx.state_table[result.input_tuple])
            else:
                expected = ctx.state_table[torch.arange(bs, device=device)[:, None],
                                           torch.tensor([[r.cached_len - 1, r.cached_len] for r in reqs], device=device)]
                torch.testing.assert_close(batch.linear_attn_metadata.slots, expected.long())
        for _ in range(10):
            engine.prepare_batch(batch, cache)
        torch.cuda.synchronize()
        with TransferCounter() as transfers:
            engine.prepare_batch(batch, cache)
        torch.cuda.synchronize()
        cpu, gpu = [], []
        for _ in range(args.repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            tic = time.perf_counter_ns()
            engine.prepare_batch(batch, cache)
            cpu.append((time.perf_counter_ns() - tic) / 1e6)
            end.record()
            end.synchronize()
            gpu.append(start.elapsed_time(end))
        return dict(bs=bs, phase=phase, page_size=page_size, linear=linear,
                    explicit_h2d_transfers=transfers.transfers, explicit_h2d_bytes=transfers.bytes,
                    cpu_submit_p50_ms=statistics.median(cpu),
                    stream_span_p50_ms=statistics.median(gpu))
    finally:
        clear_global_ctx()


def reclamation_case():
    pool = make_pool()
    samples = []
    for iteration in range(15):
        logs = pool.reserve(logs=8)[1]
        torch.cuda.synchronize()
        # Stand in for a still-running construction kernel. Measure submission,
        # not claim that the artificial delay is a real TTFT improvement.
        torch.cuda._sleep(5_000_000)
        pool.materialize_slots(logs)
        tic = time.perf_counter_ns()
        retire = getattr(pool, "retire_slots", pool.release_slots)
        retire(logs[:4])
        elapsed = (time.perf_counter_ns() - tic) / 1e6
        torch.cuda.synchronize()
        pool.poll_reclamation()
        pool.release_slots(logs[4:])
        torch.cuda.synchronize()
        pool.allocator.check_integrity()
        assert not pool.allocator.allocated_slots
        if iteration >= 5:
            samples.append(elapsed)
    return dict(cpu_submit_p50_ms=statistics.median(samples), pending_slots=4,
                artificial_gpu_delay_cycles=5_000_000)


results = [metadata_case(bs, "decode", page, linear)
           for linear, page in (("logstate", 32), ("native", 128)) for bs in (1, 32, 64)]
results += [metadata_case(4, "prefill", 32, "logstate")]
report = dict(source_root=str(args.source_root.resolve()), metadata=results,
              reclamation=reclamation_case(), serving_benchmark=False)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
