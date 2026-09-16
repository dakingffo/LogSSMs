"""Decode-throughput benchmark: short prompts, long generations.

Launches a LogSSMs HTTP server (tp-size from --tp-size), streams a batch of
``--output-len``-token generations through ``/v1/chat/completions`` and reports decode
throughput (output-tok/s), total throughput, TPOT and TTFT. Server flags are parsed by
the library's ``logssms.server.args.parse_args`` -- nothing is redefined here.

Examples:
  python benchmarks/throughput/bench_decode.py --model-path <model> --tp-size 2 \
      --input-len 32 --output-len 256 --num-prompts 32
  python benchmarks/throughput/bench_decode.py --model-path <model> --tp-size 2 \
      --dataset spec_bench --num-prompts 32 --output-len 256
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bench_common import (
    input_label,
    launch_server,
    make_prompts,
    parse_full,
    parse_int_list,
    print_conc_summary,
    print_stats,
    resolve_port,
    run_online_bench,
    save_json,
    split_warmup_prompt,
    wait_server_ready,
    kill_server,
)


def main() -> int:
    server_args, bench, server_argv = parse_full(
        sys.argv[1:],
        extra={
            "--input-len": {"type": int, "default": 32, "help": "prompt length in tokens"},
            "--output-len": {"type": int, "default": 256, "help": "generation length in tokens"},
        },
    )
    port = resolve_port(server_argv, server_args)
    proc = launch_server(
        server_argv, port=port,
        enable_specualtive_decoding=False, num_spec_tokens=0,
    )
    try:
        base = wait_server_ready(port)
        rows = []
        results: dict[str, dict] = {}
        for point_idx, c in enumerate(parse_int_list(bench.num_prompts, 32)):
            warmup = not bench.no_warmup
            prompts, in_lens = make_prompts(
                server_args.model_path,
                bench.input_len,
                c + int(warmup),
                bench.seed + point_idx,
                dataset=bench.dataset,
                dataset_categories=bench.dataset_category,
            )
            warmup_prompt, prompts, in_lens = split_warmup_prompt(
                prompts, in_lens, warmup
            )
            out_lens = [bench.output_len] * c
            stats = asyncio.run(
                run_online_bench(
                    base,
                    server_args.model_path,
                    prompts,
                    in_lens,
                    out_lens,
                    mtp=False,
                    warmup_prompt=warmup_prompt,
                    pbar=not bench.no_pbar,
                )
            )
            assert stats["chunks_ok"], "chunk count != requested output tokens (SSE parse broken?)"
            print_stats(
                f"bench_decode tp={server_args.tp_info.size} "
                f"in={input_label(bench.dataset, in_lens, bench.input_len)} "
                f"out={bench.output_len} conc={c}",
                stats,
            )
            rows.append((c, stats))
            results[str(c)] = stats
            save_json(bench.out, results)
        if len(rows) > 1:
            print_conc_summary(rows)
        return 0
    finally:
        kill_server(proc)


if __name__ == "__main__":
    sys.exit(main())
