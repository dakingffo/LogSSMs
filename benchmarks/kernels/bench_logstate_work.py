"""Microbenchmark a full LogState decode window's recurrent work traffic.

Runs the production logstate recurrent entry point for every token in a window.
The reported time is CUDA-Graph GPU time per token, excluding conv and model layers.
"""

import argparse

import torch
import triton.testing

from logssms.cache.logstate_pool.mha_pool import MHALogStatePool
from logssms.cache.logstate_pool.layout import LOG_STATE_REPLAY_WINDOW
from logssms.kernel.linear_attention.vendor.fla import (
    fused_recurrent_gated_delta_rule_to_logstate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--granularity", type=int, default=16)
    parser.add_argument("--batch-sizes", default="1,8")
    parser.add_argument("--modes", default="eager,gemm")
    args = parser.parse_args()
    assert args.granularity in (16, 32, 64)
    torch.manual_seed(92)
    modes = args.modes.split(",")
    assert all(mode in ("eager", "gemm") for mode in modes)
    for batch_size in map(int, args.batch_sizes.split(",")):
        pool = MHALogStatePool(
            num_frames=max(256, batch_size * 16), page_size=64,
            granularity=args.granularity,
            num_attention_layers=1, num_kv_heads=1, head_dim=128,
            num_linear_layers=1, num_key_heads=2, num_value_heads=2,
            head_k_dim=128, head_v_dim=128, conv_dim=768, state_len=3,
            device=torch.device("cuda"), dtype=torch.bfloat16,
        )
        allocated = pool.reserve(logs=batch_size, works=batch_size)
        assert allocated is not None
        log_slots, work_slots = allocated[1:]
        q = torch.randn(batch_size, 1, 2, 128, device="cuda", dtype=pool.dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        gate = torch.randn(batch_size, 1, 2, device="cuda", dtype=pool.dtype)
        beta = torch.randn_like(gate)
        A_log = torch.zeros(2, device="cuda")
        dt_bias = torch.zeros_like(A_log)
        logs = torch.tensor(log_slots, device="cuda", dtype=torch.int32)
        work = torch.tensor(work_slots, device="cuda", dtype=torch.int32)
        offsets = [
            torch.full((batch_size,), offset, device="cuda", dtype=torch.int32)
            for offset in range(args.granularity)
        ]
        replay_counts = [
            torch.full((batch_size,), offset % LOG_STATE_REPLAY_WINDOW,
                       device="cuda", dtype=torch.int32)
            for offset in range(args.granularity)
        ]

        for mode in modes:
            for slot in work_slots:
                pool.zero_state(slot)

            def window() -> None:
                for token_offsets, counts in zip(offsets, replay_counts):
                    fused_recurrent_gated_delta_rule_to_logstate(
                        q, k, v, gate, beta, A_log, dt_bias,
                        pool, work, logs, token_offsets, 0, counts,
                        enable_eager_state=mode == "eager",
                    )

            window()
            per_token_us = (
                triton.testing.do_bench_cudagraph(window, rep=100)
                * 1000 / args.granularity
            )
            print(
                f"granularity={args.granularity} batch={batch_size} "
                f"mode={mode} recurrent_us_per_token={per_token_us:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
