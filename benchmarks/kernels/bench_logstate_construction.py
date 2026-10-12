"""Measure LogState construction GPU time without a model forward."""

import argparse

import torch
import triton.testing

from logssms.cache.logstate_pool.mha_pool import MHALogStatePool
from logssms.kernel.linear_attention.logstate import (
    _construction_args,
    _construction_dims,
    _load_construction_module,
)
from logssms.tvm import call_cuda_ffi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--repetitions-ms", type=int, default=100)
    args = parser.parse_args()
    assert args.tasks > 0 and args.segments > 0
    assert args.layers > 0 and args.heads > 0

    count = args.tasks * args.segments
    pool = MHALogStatePool(
        num_frames=8 * count + 32,
        page_size=64,
        granularity=16,
        num_attention_layers=args.layers,
        num_kv_heads=args.heads,
        head_dim=128,
        num_linear_layers=args.layers,
        num_key_heads=args.heads,
        num_value_heads=args.heads,
        head_k_dim=128,
        head_v_dim=128,
        conv_dim=128,
        state_len=3,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    allocated = pool.reserve(logs=count, works=count)
    assert allocated is not None
    logs, works = allocated[1:]
    pool.storage.view(torch.bfloat16).normal_(mean=0.0, std=0.01)
    for slot in logs:
        record = pool.allocator.slot(slot).decay
        size = pool.layout.log_decay_bytes
        begin = record.index * size
        pool.storage[record.frame, begin:begin + size].zero_()

    chains = [
        (
            None,
            [(logs[row * args.segments + col], 0, 64)
             for col in range(args.segments)],
            [(works[row * args.segments + col],)
             for col in range(args.segments)],
        )
        for row in range(args.tasks)
    ]

    module = _load_construction_module(64, 128, 128, pool.layout.granularity)
    prepared = []
    max_tasks = torch.cuda.get_device_properties(pool.device).multi_processor_count

    def vector(values):
        return torch.tensor(values, dtype=torch.int32, device=pool.device)

    for task_begin in range(0, args.tasks, max_tasks):
        group = chains[task_begin:task_begin + max_tasks]
        segments = []
        outputs = []
        task_offsets = [0]
        output_offsets = [0]
        for _base, task_segments, task_outputs in group:
            segments.extend(task_segments)
            task_offsets.append(len(segments))
            for slots in task_outputs:
                outputs.extend(slots)
                output_offsets.append(len(outputs))
        prepared.append((
            *_construction_args(pool),
            vector([-1 if base is None else base for base, _logs, _outputs in group]),
            vector(task_offsets),
            vector([slot for slot, _begin, _end in segments]),
            vector([begin for _slot, begin, _end in segments]),
            vector([end for _slot, _begin, end in segments]),
            vector(output_offsets), vector(outputs),
            *_construction_dims(pool),
        ))

    def construct() -> None:
        for call_args in prepared:
            call_cuda_ffi(module.construct, *call_args)

    construct()
    elapsed = triton.testing.do_bench_cudagraph(
        construct, rep=args.repetitions_ms,
    )
    print(
        f"tasks={args.tasks} segments={args.segments} "
        f"layers={args.layers} heads={args.heads} "
        f"cuda_ms={elapsed:.4f} affine_workspace_MiB=0.0",
        flush=True,
    )


if __name__ == "__main__":
    main()
