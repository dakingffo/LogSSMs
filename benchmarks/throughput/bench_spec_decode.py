"""Speculative-decoding throughput benchmark: MTP vs non-MTP, controlled variables.

Both runs use the SAME seeded prompt batch, the SAME httpx load driver and the SAME
metrics -- the only thing that differs is the server's ``--speculative-algorithm`` flag
(``--speculative-num-draft-tokens`` defaults to 4). tp size comes
from ``--tp-size``. Server flags are parsed by the library's
``logssms.server.args.parse_args`` -- nothing is redefined here.

Examples:
  python benchmarks/throughput/bench_spec_decode.py --model-path <model> --tp-size 2 \
      --input-len 64 --output-len 256 --num-concurrent-sessions 32 \
      --speculative-num-draft-tokens 4
  python benchmarks/throughput/bench_spec_decode.py --model-path <model> \
      --mode mtp --out /tmp/mtp.json   # single-mode run for longer data collection
  python benchmarks/throughput/bench_spec_decode.py --model-path <model> \
      --dataset spec_bench --dataset-category coding,math_reasoning \
      --num-concurrent-sessions 8 --output-len 256
  python benchmarks/throughput/bench_spec_decode.py --model-path <model> \
      --dataset sharegpt --num-concurrent-sessions 8 --num-turns 4 --output-len 128
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bench_common import (
    input_label,
    launch_server,
    parse_full,
    parse_int_list,
    prepare_history,
    print_compare,
    print_stats,
    resolve_num_spec_tokens,
    resolve_port,
    run_benchmark_point,
    save_json,
    wait_server_ready,
    kill_server,
)


def main() -> int:
    server_args, bench, server_argv = parse_full(
        sys.argv[1:],
        extra={
            "--input-len": {"type": int, "default": 64, "help": "prompt length in tokens"},
            "--output-len": {"type": int, "default": 256, "help": "generation length in tokens"},
            "--mode": {"type": str, "choices": ["both", "nonmtp", "mtp"], "default": "both"},
            "--dt": {"action": "store_true",
                     "help": "enable drafter/target separation for the MTP run"},
        },
    )

    if bench.mode == "both":
        modes = [("nonmtp", False), ("mtp", True)]
    elif bench.mode == "nonmtp":
        modes = [("nonmtp", False)]
    else:
        modes = [("mtp", True)]

    port = resolve_port(server_argv, server_args)
    history_path, history = prepare_history("bench_spec_decode", server_args, bench)
    num_spec_tokens = resolve_num_spec_tokens(server_argv)

    results: dict[str, dict[str, dict]] = {}
    for label, enable_specualtive_decoding in modes:
        proc = launch_server(
            server_argv,
            port=port,
            enable_specualtive_decoding=enable_specualtive_decoding,
            num_spec_tokens=num_spec_tokens,
            enable_dt=(bench.dt or server_args.dt_separation)
                      and enable_specualtive_decoding,
        )
        try:
            base = wait_server_ready(port)
            for point_idx, c in enumerate(
                parse_int_list(bench.num_concurrent_sessions, 32)
            ):
                history_records = [] if history is not None else None
                stats, in_lens = asyncio.run(
                    run_benchmark_point(
                        base,
                        server_args.model_path,
                        bench.input_len,
                        bench.output_len,
                        c,
                        bench.seed + point_idx,
                        num_turns=bench.num_turns,
                        dataset=bench.dataset,
                        dataset_categories=bench.dataset_category,
                        mtp=enable_specualtive_decoding,
                        warmup=not bench.no_warmup,
                        pbar=not bench.no_pbar,
                        history_records=history_records,
                    )
                )
                assert stats["chunks_ok"], "SSE chunk count inconsistent (parse broken?)"
                note = (
                    f"avg_accept={stats['avg_accept']:.2f}"
                    if enable_specualtive_decoding else ""
                )
                input_desc = (
                    f"multi(avg={stats['in_tokens'] / stats['n']:.0f})"
                    if bench.num_turns > 1
                    else input_label(bench.dataset, in_lens, bench.input_len)
                )
                print_stats(
                    f"bench_spec_decode {label} tp={server_args.tp_info.size} "
                    f"in={input_desc} "
                    f"out={bench.output_len} conc={c} turns={bench.num_turns}",
                    stats,
                    note=note,
                )
                results.setdefault(label, {})[str(c)] = stats
                save_json(bench.out, results)
                if history is not None:
                    history["points"].setdefault(label, {})[str(c)] = {
                        "concurrency": c,
                        "sessions": history_records,
                    }
                    save_json(history_path, history)
        finally:
            kill_server(proc)

    if bench.mode == "both":
        for c in parse_int_list(bench.num_concurrent_sessions, 32):
            print(f"  concurrency = {c}")
            print_compare(results["nonmtp"][str(c)], results["mtp"][str(c)])
    return 0


if __name__ == "__main__":
    sys.exit(main())


# =========================================================================
# 吞吐量测试矩阵 (A800-80GB-NVLink * 2, Qwen3.6-27B, output=256, spec_bench=512)
# 每个 run 都加 --out <json> 落地,防崩溃丢块缓冲。
# spec_bench 用 --output-len 512:紧跟 1K 上下文测试之后跑;
#
# A. TP=2, DT 不分离 (--tensor-parallel-size 2)
#    随机乱码 (--input-len N, 不带 --dataset):
#      16K -> 1,2,4,8            8K -> 1,2,4,8,16
#       4K -> 1,2,4,8,16,32      2K -> 1,2,4,8,16,32,64
#       1K -> 1,2,4,8,16,32,64,128
#    spec_bench (--dataset spec_bench, --output-len 512): 1,2,4,8,16,32,64,128
#
# B. DT 分离 (--tensor-parallel-size 1 --enable-dt-separation --max-running-seq 16)
#    随机乱码:
#      16K -> 1        8K -> 1,2
#       4K -> 1,2,4    2K -> 1,2,4,8
#       1K -> 1,2,4,8,16
#    spec_bench (--output-len 512): 1,2,4,8,16
#
# 纯 latency 矩阵暂不做。
# =========================================================================
