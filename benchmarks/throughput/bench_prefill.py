"""Prefill-throughput benchmark: long prompts, tiny outputs.

Launches a LogSSMs HTTP server (tp-size from --tp-size), streams a batch of
``--input-len``-token prompts through ``/v1/chat/completions`` and reports prefill
throughput (input-tok/s), TTFT and req/s. Server flags are parsed by the library's
``logssms.server.args.parse_args`` -- nothing is redefined here.

Examples:
  python benchmarks/throughput/bench_prefill.py --model-path <model> --tp-size 2 \
      --input-len 2048 --num-concurrent-sessions 64
  python benchmarks/throughput/bench_prefill.py --model-path <model> --tp-size 2 \
      --dataset spec_bench --num-concurrent-sessions 64 --output-len 8
  python benchmarks/throughput/bench_prefill.py --model-path <model> --tp-size 2 \
      --dataset spec_bench --num-concurrent-sessions 16 --num-turns 2 --output-len 8
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
    print_stats,
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
            "--input-len": {"type": int, "default": 512, "help": "prompt length in tokens"},
            "--output-len": {"type": int, "default": 8, "help": "generation length in tokens"},
        },
    )
    port = resolve_port(server_argv, server_args)
    history_path, history = prepare_history("bench_prefill", server_args, bench)
    proc = launch_server(
        server_argv, port=port,
        enable_specualtive_decoding=False, num_spec_tokens=0,
    )
    try:
        base = wait_server_ready(port)
        results: dict[str, dict] = {}
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
                    warmup=not bench.no_warmup,
                    pbar=not bench.no_pbar,
                    history_records=history_records,
                )
            )
            assert stats["chunks_ok"], "chunk count != requested output tokens (SSE parse broken?)"
            input_desc = (
                f"multi(avg={stats['in_tokens'] / stats['n']:.0f})"
                if bench.num_turns > 1
                else input_label(bench.dataset, in_lens, bench.input_len)
            )
            print_stats(
                f"bench_prefill tp={server_args.tp_info.size} "
                f"in={input_desc} "
                f"out={bench.output_len} conc={c} turns={bench.num_turns}",
                stats,
            )
            results[str(c)] = stats
            save_json(bench.out, results)
            if history is not None:
                history["points"][str(c)] = {
                    "concurrency": c,
                    "sessions": history_records,
                }
                save_json(history_path, history)
        return 0
    finally:
        kill_server(proc)


if __name__ == "__main__":
    sys.exit(main())
