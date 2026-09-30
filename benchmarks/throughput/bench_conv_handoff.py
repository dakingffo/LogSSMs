"""Measure the prefill-to-decode convolution snapshot handoff."""

import argparse
import time

import torch

from logssms.cache.logstate_pool.frame_pool import FramedMHALogStatePool
from logssms.kernel.linear_attention.conv_copy import copy_framed_conv_snapshots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    pool = FramedMHALogStatePool(
        num_frames=256,
        page_size=64,
        granularity=32,
        num_attention_layers=4,
        num_kv_heads=4,
        head_dim=128,
        num_linear_layers=args.layers,
        num_key_heads=2,
        num_value_heads=2,
        head_k_dim=128,
        head_v_dim=128,
        conv_dim=192,
        state_len=3,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    allocated = pool.reserve(logs=args.requests, works=args.requests)
    assert allocated is not None
    source = torch.tensor(allocated[1], device="cuda", dtype=torch.int32)
    destination = torch.tensor(allocated[2], device="cuda", dtype=torch.int32)
    source_granule = torch.full_like(source, 1)
    zero = torch.zeros_like(destination)

    def old() -> None:
        for layer in range(pool.layout.num_linear_layers):
            pool.write_conv(
                destination, zero, layer,
                pool.read_conv(source, source_granule, layer),
            )

    def fused() -> None:
        copy_framed_conv_snapshots(
            pool, source, source_granule, destination, zero,
        )

    def view_copy() -> None:
        for source_slot, destination_slot in zip(
            allocated[1], allocated[2], strict=True,
        ):
            pool.copy_conv_snapshot(source_slot, 1, destination_slot)

    for name, operation in (
        ("per_layer", old), ("fused", fused), ("view_copy", view_copy),
    ):
        for _ in range(2):
            operation()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.repeats):
            operation()
        torch.cuda.synchronize()
        print(f"{name}: {(time.perf_counter() - start) * 1000 / args.repeats:.3f} ms/call")


if __name__ == "__main__":
    main()
