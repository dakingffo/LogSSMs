"""Decode-throughput benchmark: short prompts, long generations.

Launches a LogSSMs HTTP server (tp-size from --tp-size), streams a batch of
``--output-len``-token generations through ``/v1/chat/completions`` and reports decode
throughput (output-tok/s), total throughput, TPOT and TTFT. Server flags are parsed by
the library's ``logssms.server.args.parse_args`` -- nothing is redefined here.

Examples:
  python benchmarks/throughput/bench_decode.py --model-path <model> --tp-size 2 \
      --input-len 32 --output-len 256 --num-concurrent-sessions 32
  python benchmarks/throughput/bench_decode.py --model-path <model> --tp-size 2 \
      --dataset spec_bench --num-concurrent-sessions 32 --output-len 256
  python benchmarks/throughput/bench_decode.py --model-path <model> --tp-size 2 \
      --dataset sharegpt --num-concurrent-sessions 1,2,4 --num-sessions 16 \
      --num-turns 4 --output-len 128

For a multi-turn sweep, ``--num-concurrent-sessions`` denotes concurrent sessions and
``--num-sessions`` denotes the fixed workload size. The workload is sampled once and
reused verbatim; every concurrency point gets a fresh server/cache.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bench_common import (
    input_label,
    launch_server,
    make_conversations,
    parse_full,
    parse_int_list,
    print_conc_summary,
    print_stats,
    resolve_port,
    run_multiturn_bench,
    run_benchmark_point,
    save_json,
    split_warmup_conversation,
    wait_server_ready,
    kill_server,
)


def main() -> int:
    server_args, bench, server_argv = parse_full(
        sys.argv[1:],
        extra={
            "--input-len": {"type": int, "default": 32, "help": "prompt length in tokens"},
            "--output-len": {"type": int, "default": 256, "help": "generation length in tokens"},
            "--num-sessions": {
                "type": int,
                "default": None,
                "help": (
                    "fixed number of measured multi-turn sessions; defaults to the "
                    "largest --num-concurrent-sessions value"
                ),
            },
        },
    )
    concurrencies = parse_int_list(bench.num_concurrent_sessions, 32)
    if any(concurrency < 1 for concurrency in concurrencies):
        raise ValueError("all concurrency values must be positive")
    port = resolve_port(server_argv, server_args)
    rows = []
    results: dict[str, dict] = {}

    if bench.num_turns > 1:
        num_sessions = (
            max(concurrencies)
            if bench.num_sessions is None else bench.num_sessions
        )
        if num_sessions < 1:
            raise ValueError("--num-sessions must be positive")
        if num_sessions < max(concurrencies):
            raise ValueError(
                "--num-sessions must be at least the largest "
                "--num-concurrent-sessions value"
            )
        conversations = make_conversations(
            server_args.model_path,
            bench.input_len,
            num_sessions + int(not bench.no_warmup),
            bench.num_turns,
            bench.seed,
            dataset=bench.dataset,
            dataset_categories=bench.dataset_category,
        )
        warmup_conversation, conversations = split_warmup_conversation(
            conversations, not bench.no_warmup
        )

        for c in concurrencies:
            proc = launch_server(
                server_argv,
                port=port,
                enable_specualtive_decoding=False,
                num_spec_tokens=0,
            )
            try:
                base = wait_server_ready(port)
                stats = asyncio.run(
                    run_multiturn_bench(
                        base,
                        server_args.model_path,
                        conversations,
                        bench.output_len,
                        concurrency=c,
                        warmup_conversation=warmup_conversation,
                        pbar=not bench.no_pbar,
                        request_timeout=bench.request_timeout,
                    )
                )
            finally:
                kill_server(proc)
            assert stats["chunks_ok"], (
                "chunk count != requested output tokens (SSE parse broken?)"
            )
            input_desc = f"multi(avg={stats['in_tokens'] / stats['n']:.0f})"
            print_stats(
                f"bench_decode tp={server_args.tp_info.size} "
                f"in={input_desc} out={bench.output_len} "
                f"conc={c} sessions={num_sessions} turns={bench.num_turns}",
                stats,
            )
            rows.append((c, stats))
            results[str(c)] = stats
            save_json(bench.out, results)
    else:
        if bench.num_sessions is not None:
            raise ValueError("--num-sessions requires --num-turns greater than one")
        proc = launch_server(
            server_argv, port=port,
            enable_specualtive_decoding=False, num_spec_tokens=0,
        )
        try:
            base = wait_server_ready(port)
            for point_idx, c in enumerate(concurrencies):
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
                    )
                )
                assert stats["chunks_ok"], "chunk count != requested output tokens (SSE parse broken?)"
                input_desc = input_label(bench.dataset, in_lens, bench.input_len)
                print_stats(
                    f"bench_decode tp={server_args.tp_info.size} "
                    f"in={input_desc} "
                    f"out={bench.output_len} conc={c} turns={bench.num_turns}",
                    stats,
                )
                rows.append((c, stats))
                results[str(c)] = stats
                save_json(bench.out, results)
        finally:
            kill_server(proc)

    if len(rows) > 1:
        print_conc_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
