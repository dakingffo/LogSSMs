# Online analysis ablation

Compare MARCONI and LogState's Default and Online eviction policies on the same
Claude Code workload using the production scheduler and allocator on CPU.

```bash
bash scripts/ablation_online_analysis.sh
```

The case uses Qwen3.5-4B's BF16 cache layout at TP=2, 16-token LogState
granularity (vs. MARCONI `block_size=128`, `bootstrap_multiplier=5`), and
58.33 GiB of cache per rank. It selects the first 16 complete
sessions whose requests all fit within 262,144 tokens including output. The
filter includes subagent requests. Up to 8 sessions can be open at once, with
an independent server limit of 64 running requests.

The virtual clock releases each request at its recorded offset from the session
start and permits concurrent subagent requests. An iteration advances virtual
time by 1 ms. Prompt and output token IDs are reconstructed from the
[public Claude Code trace](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126),
with recorded output lengths and EOS ignored. The replay verifies the reusable
prefix lengths and all policies use the same input trace. LogState enables hit
materialization. The workload measures cache behavior using the virtual clock.

The figure shows the cumulative historical prefill recomputation over processed
requests. The report includes overall and historical hit rates, recomputed
historical tokens, P95 and P99 recomputation, total prefill tokens, and
reclamation rounds.

The trace is read from `datasets/cc-traces-weka-062126/traces.jsonl`. Results
are written to `experiment/ablation_online_analysis/results/<timestamp>/`.
Each run saves its case, selected session IDs, trace checksum, per-policy logs,
and reports. Set `PYTHON` to choose an interpreter.

Each replay logs a heartbeat every 30 wall-clock seconds with the completion
count, pending queue head, and current timed operation. Each policy's
`*-timings.jsonl` records admission and reclamation planning call starts and
durations. A Python stack dump is emitted if the heartbeat thread is blocked
for two intervals. Use `--heartbeat-seconds` to change the interval.

Use `--resume` to reuse finished policy reports and rerun missing policies.

```bash
bash scripts/ablation_online_analysis.sh --resume --out experiment/ablation_online_analysis/results/<run>
bash scripts/ablation_online_analysis.sh --plot-only --out experiment/ablation_online_analysis/results/<run>
```
