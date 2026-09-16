"""Latency benchmark: TTFT and TPOT across input/output length sweeps.

Launches a LogSSMs HTTP server once and drives a streaming
``/v1/chat/completions`` workload for each (input_len, output_len) config,
reporting TTFT / TPOT percentiles per config plus a sweep summary table.

TTFT (time-to-first-token) is dominated by prefill, so it scales with
``--input-lens``; TPOT is the steady-state per-token latency. The two lists are
zipped (a length-1 list broadcasts), so the common sweeps are:

  # TTFT vs input length, fixed short output
  python benchmarks/latency/bench_latency.py --model-path <model> \
      --input-lens 32,512,2048 --output-lens 8,8,8

  # TPOT vs output length, fixed short input
  python benchmarks/latency/bench_latency.py --model-path <model> \
      --input-lens 32 --output-lens 16,64,256

  # Natural prompts from Spec-Bench (the first turn of multi-turn records)
  python benchmarks/latency/bench_latency.py --model-path <model> \
      --dataset spec_bench --dataset-category coding,math_reasoning \
      --num-prompts 8 --output-lens 64

Server flags are parsed by the library's ``logssms.server.args.parse_args`` --
nothing is redefined here.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bench_common import (
    launch_server,
    make_prompts,
    parse_full,
    parse_int_list,
    print_stats,
    resolve_port,
    run_online_bench,
    save_json,
    split_warmup_prompt,
    wait_server_ready,
    kill_server,
)


def _zip_lens(input_lens: list[int], output_lens: list[int]) -> list[tuple[int, int]]:
    if len(input_lens) == len(output_lens):
        return list(zip(input_lens, output_lens))
    if len(input_lens) == 1:
        return [(input_lens[0], o) for o in output_lens]
    if len(output_lens) == 1:
        return [(i, output_lens[0]) for i in input_lens]
    raise SystemExit(
        f"--input-lens ({len(input_lens)} values) != --output-lens "
        f"({len(output_lens)} values); use equal-length lists or broadcast one to length 1"
    )


def print_latency_summary(rows: list[tuple[int, int, int, dict]]) -> None:
    print("=" * 88)
    print("  latency sweep summary")
    print(f"  {'conc':>5} {'input':>6} {'output':>6} {'TTFT p50':>10} {'TTFT p90':>10} "
          f"{'TPOT p50':>10} {'TPOT p90':>10} {'out tok':>8}")
    for conc, in_len, out_len, s in rows:
        t, p = s["ttft_ms"], s["tpot_ms"]
        print(f"  {conc:>5} {in_len:>6} {out_len:>6} {t[1]:>10.3f} {t[2]:>10.3f} "
              f"{p[1]:>10.3f} {p[2]:>10.3f} {s['out_tokens']:>8}")
    print("=" * 88)


def main() -> int:
    server_args, bench, server_argv = parse_full(
        sys.argv[1:],
        extra={
            "--num-prompts": {"type": str, "default": None,
                              "help": "comma-separated concurrent request counts (default 8)"},
            "--input-lens": {"type": str, "default": None,
                             "help": "comma-separated prompt lengths (default 32)"},
            "--output-lens": {"type": str, "default": None,
                              "help": "comma-separated generation lengths (default 64)"},
        },
    )
    configs = _zip_lens(
        parse_int_list(bench.input_lens, 32),
        parse_int_list(bench.output_lens, 64),
    )
    concs = parse_int_list(bench.num_prompts, 8)
    port = resolve_port(server_argv, server_args)
    proc = launch_server(
        server_argv, port=port,
        enable_specualtive_decoding=False, num_spec_tokens=0,
    )
    try:
        base = wait_server_ready(port)
        rows: list[tuple[int, int, int, dict]] = []
        results: dict[str, dict] = {}
        point_idx = 0
        for c in concs:
            for in_len, out_len in configs:
                warmup = not bench.no_warmup
                prompts, in_lens = make_prompts(
                    server_args.model_path,
                    in_len,
                    c + int(warmup),
                    bench.seed + point_idx,
                    dataset=bench.dataset,
                    dataset_categories=bench.dataset_category,
                )
                warmup_prompt, prompts, in_lens = split_warmup_prompt(
                    prompts, in_lens, warmup
                )
                display_in_len = (
                    round(sum(in_lens) / len(in_lens)) if bench.dataset else in_len
                )
                out_lens = [out_len] * c
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
                    f"bench_latency tp={server_args.tp_info.size} "
                    f"in={display_in_len} out={out_len} conc={c}",
                    stats,
                )
                rows.append((c, display_in_len, out_len, stats))
                results[f"{c}:{display_in_len}:{out_len}"] = stats
                save_json(bench.out, results)
                point_idx += 1
        print_latency_summary(rows)
        return 0
    finally:
        kill_server(proc)


if __name__ == "__main__":
    sys.exit(main())
