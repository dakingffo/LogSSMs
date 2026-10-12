"""Isolated logstate GDN decode microbenchmark; not a prefix-cache benchmark."""

import argparse
import hashlib
import importlib.util
import inspect
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from logssms.cache.logstate_pool.mha_pool import MHALogStatePool
from logssms.kernel.linear_attention.logstate import store_log_states
from logssms.kernel.linear_attention.vendor.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_to_logstate,
)


class KernelProbe:
    def __init__(self, kernel):
        self.kernel = kernel
        self.last = None

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            self.last = launch(*args, **kwargs)
            return self.last

        return run


def benchmark(fn, batch_size, num_key_heads, num_value_heads, count, eager, log_decay,
              warmup_ms):
    globals_ = inspect.unwrap(fn).__globals__
    name = "_fused_recurrent_gated_delta_kernel"
    if not isinstance(globals_[name], KernelProbe):
        globals_[name] = KernelProbe(globals_[name])
    probe = globals_[name]
    torch.manual_seed(511)
    device, dtype = torch.device("cuda"), torch.bfloat16
    pool = MHALogStatePool(
        num_frames=batch_size * 12 + 32, page_size=64, granularity=32,
        num_attention_layers=1, num_kv_heads=8, head_dim=128,
        num_linear_layers=1, num_key_heads=num_key_heads, num_value_heads=num_value_heads,
        head_k_dim=128, head_v_dim=128,
        conv_dim=(2 * num_key_heads + num_value_heads) * 128, state_len=3,
        device=device, dtype=dtype,
    )
    allocation = pool.reserve(logs=batch_size, works=batch_size)
    assert allocation is not None
    logs = torch.tensor(allocation[1], dtype=torch.int32, device=device)
    works = torch.tensor(allocation[2], dtype=torch.int32, device=device)
    state = torch.randn(batch_size, num_value_heads, 128, 128, device=device, dtype=dtype) * 0.01
    pool.write_states(works, 0, state)
    keys = torch.randn(batch_size, 64, num_value_heads, 128, device=device, dtype=dtype) * 0.01
    updates = torch.randn_like(keys) * 0.01
    alpha = torch.empty(batch_size, 64, num_value_heads, device=device).uniform_(0.95, 0.999)
    store_log_states(
        pool, logs.repeat_interleave(64), torch.arange(64, device=device, dtype=torch.int32).repeat(batch_size),
        keys.flatten(0, 1), updates.flatten(0, 1),
        (alpha.log() if log_decay else alpha).flatten(0, 1), 0,
    )
    query = torch.randn(batch_size, 1, num_key_heads, 128, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn(batch_size, 1, num_value_heads, 128, device=device, dtype=dtype)
    gate = torch.randn(batch_size, 1, num_value_heads, device=device, dtype=dtype) - 2
    beta = torch.randn_like(gate)
    A_log = torch.zeros(num_value_heads, device=device)
    dt_bias = torch.zeros_like(A_log)
    offsets = torch.full((batch_size,), count, dtype=torch.int32, device=device)
    counts = offsets.clone()

    def run():
        return fn(query, key, value, gate, beta, A_log, dt_bias, pool,
                  works, logs, offsets, 0, counts, enable_eager_state=eager)

    # Independent FP32 oracle, including the current token's delta-rule update.
    previous = state.float()
    if not eager:
        for token in range(count):
            previous = previous * alpha[:, token, :, None, None]
            previous += keys[:, token].float()[..., :, None] * updates[:, token].float()[..., None, :]
    q = torch.nn.functional.normalize(query[:, 0].float(), dim=-1, eps=1e-6)
    k = torch.nn.functional.normalize(key[:, 0].float(), dim=-1, eps=1e-6)
    q = q.repeat_interleave(num_value_heads // num_key_heads, dim=1) * 128**-0.5
    k = k.repeat_interleave(num_value_heads // num_key_heads, dim=1)
    decay = torch.exp(-torch.nn.functional.softplus(gate[:, 0].float()))
    previous *= decay[..., None, None]
    update = (value[:, 0].float() - torch.einsum("bhkv,bhk->bhv", previous, k)) * beta[:, 0].float().sigmoid()[..., None]
    expected_state = previous + k[..., :, None] * update[..., None, :]
    expected = torch.einsum("bhkv,bhk->bhv", expected_state, q)
    output = run()
    torch.testing.assert_close(output[:, 0].float(), expected, rtol=0.03, atol=0.005)
    observed = pool.read_states(works, 0)
    if eager or count == 15:
        torch.testing.assert_close(observed.float(), expected_state, rtol=0.03, atol=0.01)
    else:
        torch.testing.assert_close(observed, state, rtol=0, atol=0)
    for _ in range(5):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(32):
            run()
    warmup_end = time.perf_counter() + warmup_ms / 1000
    while time.perf_counter() < warmup_end:
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
    samples = []
    for _ in range(11):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 320)
    return dict(batch_size=batch_size, key_heads=num_key_heads, value_heads=num_value_heads,
                replay_count=count, eager_state=eager, median_us=statistics.median(samples),
                samples_us=samples, fp32_oracle_passed=True,
                registers=probe.last.n_regs, spills=probe.last.n_spills,
                shared_memory_bytes=probe.last.metadata.shared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Optional saved kernel source for baseline")
    parser.add_argument("--log-decay", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare-source", type=Path, help="Saved baseline kernel; run cases alternately")
    parser.add_argument("--compare-log-decay", action="store_true",
                        help="The saved baseline also stores logarithmic decay")
    parser.add_argument("--batches", default="1,32,64")
    parser.add_argument("--replay-counts", default="0,7,15")
    parser.add_argument("--warmup-ms", type=float, default=200,
                        help="Warm each captured variant before GPU event timing")
    args = parser.parse_args()
    fn = fused_recurrent_gated_delta_rule_to_logstate
    source = Path("python/logssms/kernel/linear_attention/vendor/fla/fused_recurrent.py")
    if args.source:
        source = args.source
        name = "logssms.kernel.linear_attention.vendor.fla.decode_baseline"
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        fn = module.fused_recurrent_gated_delta_rule_to_logstate
    results = []
    reference = None
    if args.compare_source:
        name = "logssms.kernel.linear_attention.vendor.fla.decode_reference"
        spec = importlib.util.spec_from_file_location(name, args.compare_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        reference = module.fused_recurrent_gated_delta_rule_to_logstate
    for batch in map(int, args.batches.split(",")):
        for count in map(int, args.replay_counts.split(",")):
            if reference:
                row = benchmark(reference, batch, 8, 24, count, False, args.compare_log_decay, args.warmup_ms)
                row["variant"] = "baseline"
                results.append(row)
                print(json.dumps(row), flush=True)
            row = benchmark(fn, batch, 8, 24, count, False, args.log_decay, args.warmup_ms)
            row["variant"] = "current"
            results.append(row)
            print(json.dumps(row), flush=True)
        if reference:
            row = benchmark(reference, batch, 8, 24, 0, True, args.compare_log_decay, args.warmup_ms)
            row["variant"] = "baseline"
            results.append(row)
        row = benchmark(fn, batch, 8, 24, 0, True, args.log_decay, args.warmup_ms)
        row["variant"] = "current"
        results.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(
        scope="Isolated decode arithmetic, not a serving or prefix-cache comparison",
        gpu=torch.cuda.get_device_name(), dtype="bfloat16", log_decay=args.log_decay,
        warmup_ms=args.warmup_ms,
        reference_log_decay=args.compare_log_decay,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), results=results,
        reference_sha256=(hashlib.sha256(args.compare_source.read_bytes()).hexdigest()
                          if args.compare_source else None),
    ), indent=2))


if __name__ == "__main__":
    main()
