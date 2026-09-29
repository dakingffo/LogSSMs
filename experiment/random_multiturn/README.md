# Random multi-turn cache experiment

Run `python experiment/random_multiturn.py` from the repository root. The driver
starts a fresh LogSSMs server for each concurrency point. Within a session, turn
`t+1` sends exactly `prompt_t + generated_token_ids_t + random_increment_t` as
token IDs. It does not use a chat template or decode/re-encode model output.
Generation is greedy, `ignore_eos=True`, and every turn requests `--output-len`
tokens. User increments and initial prompts are seeded random non-special IDs.

Example (use the same workload flags for each cache policy):

```bash
python experiment/random_multiturn.py \
  --model-path /path/to/model --tensor-parallel-size 1 \
  --cache-type radix --page-size 64 \
  --input-len 512 --append-len 128 --output-len 256 \
  --num-turns 8 --num-sessions 32 --num-concurrent-sessions 8,16,32 \
  --seed 0 --name radix64-seed0

python experiment/random_multiturn.py \
  --model-path /path/to/model --tensor-parallel-size 1 \
  --cache-type logstate --granularity 32 \
  --input-len 512 --append-len 128 --output-len 256 \
  --num-turns 8 --num-sessions 32 --num-concurrent-sessions 8,16,32 \
  --seed 0 --name logstate32-seed0
```

Each run is written to `experiment/random_multiturn/<name>/`. `report.json`
contains throughput/latency and aggregate cache-hit metrics; `history-cN.json`
contains exact prompt and generated IDs per turn; `profile-cN.rank0.json`
contains the scheduler's cache-match samples. Each point first runs with profiling
to verify cache behavior, then starts a fresh server without profiling for timing.
`--profile-only` skips the second run and is intended for diagnostics, not final
performance numbers.

The report records whether the profiled and unprofiled runs had identical input
traces, plus an input-trace SHA-256 for cross-policy comparison. A long theoretical
reusable prefix with a short actual match can indicate eviction or admission loss;
it is not evidence that the workload failed to preserve its generated history.
For strict paired comparisons, require identical input hashes across cache policies;
the output hash additionally checks generation equivalence.
This synthetic workload is complementary to the ShareGPT experiment with the normal
chat template and EOS behavior; it does not model natural dialogue content.
