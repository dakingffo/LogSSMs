# Random multi-turn cache experiment

Run `python -m experiment.random_multiturn` from the repository root against an
already-running server. The client does not start or stop the service. Within a
session, turn `t+1` sends exactly `prompt_t + generated_token_ids_t +
random_increment_t` as token IDs. It does not use a chat template or
decode/re-encode model output. Generation is greedy with `ignore_eos=True`;
initial prompts and increments are seeded random non-special IDs.

Run one concurrency point per invocation, with a fresh server for each point.
The service launchers are documented in `experiment/launch_server/README.md`.

```bash
python -m experiment.random_multiturn \
  --api logssms --base-url http://127.0.0.1:28300 \
  --tokenizer /path/to/model --max-seq-len 8192 \
  --input-len 256 --append-len 64 --output-len 256 \
  --num-turns 16 --num-sessions 64 --num-concurrent-sessions 64 \
  --seed 0 --name logstate32-c64
```

For an SGLang server, change `--api` to `sglang` and choose a different name.
The client uses SGLang's native `/generate` endpoint with token-ID input and
reads actual generated `output_ids` from the SSE stream. Its
`meta_info.cached_tokens` is recorded per request. LogSSMs cache-hit metrics
require a separate scheduler profile; the client never enables profiling.

Each run writes `experiment/random_multiturn/<name>/report.json` and
`history-cN.json`. The history contains exact prompt and generated IDs for
every turn. The report contains input/output trace hashes; different hashes
across implementations mean later requests were not strictly paired.

For LogSSMs cache-hit measurements, start the server with
`LOGSSMS_PROFILE_SCHEDULER=1` and
`LOGSSMS_PROFILE_SCHEDULER_PATH=/path/to/profile.json`, run the client, stop
the server so it writes `profile.rank0.json`, then attach it:

```bash
python -m experiment.random_multiturn.profile \
  --run-dir experiment/random_multiturn/logstate32-c64 \
  --profile /path/to/profile.rank0.json
```

This synthetic workload complements ShareGPT with normal chat templates and
EOS behavior; it does not model natural dialogue content.

## Execution profiling (including CUDA graph replay)

The same `LOGSSMS_PROFILE_SCHEDULER=1` switch enables per-batch CPU and CUDA-event
timings. The rank-local JSON now includes:

- `engine_prepare`: pad, allocate, positions, input/write mapping, output-location
  lookup, cache preparation, attention/linear metadata, and sampler preparation.
- `engine_forward`: eager model execution or graph input copy, replay metadata
  updates and graph execution, followed by sampling and output copy.
- `prefill_*_batches`, `decode_*_batches`, `verify_*_batches`: scheduler preparation,
  submission and result processing, including output wait and process sub-stages.

Join records by `batch_id`. Each forward retains the CPU-side context lengths,
logical/padded batch size and, for LogState, replay counts and eager-state mode.
`replay_counts` describes bookkeeping; `effective_replay_counts` is zero in eager
mode. Aggregate `by_padded_size` and `replay_count_rows` include all measured batches.
`gpu_gap_before_ms` is the gap between consecutive engine forwards on its stream;
an intervening prefill is not counted as a decode idle gap.

Events are recorded outside capture, freshly for every replay, and harvested by
non-blocking queries. No per-batch synchronization or GPU-to-CPU metadata read is
added. `gpu_stages_ms` measures CUDA stream intervals (including idle time and
contention), not summed kernel durations. Do not add CPU and GPU measurements or
overlapping prepare/forward times to estimate TPOT. Profiling itself adds overhead;
use a separate non-profiled run for performance conclusions.

Detailed records retain the last 8192 batches per phase and record kind by default.
Use `LOGSSMS_PROFILE_MAX_BATCH_SAMPLES=32768` to retain more. Aggregate counters
still cover all batches; `dropped_samples` identifies truncation. A partial dump can
contain `null` GPU timings with `pending_gpu_events > 0`; stop the server cleanly
for final timings.

For individual kernels **inside** a replayed graph, collect a node-level Nsight
Systems trace. The Python layer decorators are not re-executed during replay:

```bash
LOGSSMS_PROFILE_SCHEDULER=1 \
LOGSSMS_PROFILE_SCHEDULER_PATH=/tmp/logssms-execution.json \
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
  --output=/tmp/logssms-execution \
  python -m experiment.launch_server.launch_logssms <server arguments>
```

NVTX ranges include phase and batch ID, so the kernel trace can be matched to JSON
records. Node tracing is more intrusive than graph-level tracing; use it for a
small diagnostic workload. See the [Nsight Systems CUDA Graph Trace documentation](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-graph-trace).
