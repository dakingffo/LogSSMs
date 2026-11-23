# Granularity ablation

Compare LogState granularity 16, 32, and 64 on one standard multiturn workload.

```bash
bash scripts/ablation_granularity.sh --model-path /path/to/Qwen3.6-27B
```

The script starts a fresh LogSSMs server for each point and stops it after the
client finishes. All points use 64 OASST2 trees, up to one concurrent request per
tree, and a 1024-token output cap with EOS enabled. The server uses TP=2,
a 0.9 memory ratio, CUDA graphs, and mixed scheduling with prefill budgets of
128 tokens for mixed batches and 8192 for pure prefill. Online analysis is enabled. Eager state and scheduler profiling are disabled.

The client warms up with four disjoint trees and excludes them from the
measurements. Each child request inherits the exact prompt and generated token
IDs of its parent in the current run. The runner checks that all points used the
same workload manifest. Compare actual generated token totals alongside latency
and throughput.

Results are saved in `results/<run>/` by default, with a report and request history
in each granularity's subdirectory, plus server logs and `summary.md`. The table
uses token-weighted history hit rate, TTFT P50/P99, TPOT P50, and output tok/s.
The default seed is 0, and the sequence limit is 18432 tokens. Set `PYTHON` to
select a Python interpreter. Use `--granularity 32 16` to repeat only selected
points or change their order.
