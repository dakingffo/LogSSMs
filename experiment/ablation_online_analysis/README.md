# Online-analysis ablation

Compare unified online eviction with Log-first binary Checkpoint thinning using
recorded standard-multiturn token histories. Run the production Scheduler and
LogStateCacheManager on CPU, replacing GPU numerical operations with metadata
adapters. No inference server, GPU or model weights are required.

```bash
conda activate daking
bash scripts/ablation_online_analysis.sh
```

The script runs both 32/64/128-tree comparisons and replaces their outputs in
`results/aligned-prefill-distribution/` and `results/exponent-regularization/`.
It sets the Python import path and disables GPU visibility. Set `PYTHON` to select
an interpreter and `ABLATION_WORKERS` to change the default six replay processes.

To run either comparison separately:

```bash
export PYTHONPATH="$PWD/python:$PWD"
python -m experiment.ablation_online_analysis.run_online_analysis \
  --trees 32 64 128 --workers 6 \
  --out experiment/ablation_online_analysis/results/aligned-prefill-distribution
python -m experiment.ablation_online_analysis.run_regularization --workers 6
```

The regularization replay changes only the quadratic exponent penalty in its
worker process. Both report generators run automatically after replay completes.
The general online-versus-default driver also accepts a custom `--out`; without
one, it writes to `results/<timestamp>/`.

## Experiment modules

| Experiment | Runner | Report |
|---|---|---|
| Online versus default | `run_online_analysis.py` | `report_online_analysis.py` |
| Exponent regularization | `run_regularization.py` | `report_regularization.py` |

`replay_common.py` implements the shared replay lifecycle, and `cpu_common.py`
provides its CPU numerical adapters. `replay_regularization.py` applies the
experimental fitting change before calling the shared replay. Both experiments
use `cases.json`.

## Execution

The driver calls the production `Scheduler.overlap_loop()`, including request
arrival, FCFS admission, batch selection, result processing and deferred request
release. `Engine.prepare_batch()` performs cache allocation and preparation.
`PrefillAdder`, `PrefillManager`, `DecodeManager` and `LogStateCacheManager` execute
their normal implementations. Matching, branch protection, online observations,
memorized restoration, State adoption and KV deduplication are not reimplemented
by this experiment.

The scheduling policy is mixed, with an 8192-token ordinary prefill budget, a
128-token mixed prefill budget and the case's maximum running-request count as
the decode token budget. Work states are allocated on admission. Decode allocates
incrementally; it does not reserve the full generated suffix as physical storage.
The production overlap loop retains one outstanding forward and performs its
normal deferred release of completed requests.

`cpu_common.py` supplies the numerical boundary: metadata-backed `MHALogStatePool`
storage, synchronous completion events, CPU expansion of CUDA index mappings,
and recorded-token output instead of model execution. Production allocation,
frame descriptors, materialization bookkeeping and release remain in use.
The online-versus-default comparison does not patch cache-policy methods. The
exponent ablation replaces only the fitting function in its worker process.
Numerical State/Log/conv contents, CUDA graph
execution, GPU stream timing and asynchronous reclamation latency are not tested.
CPU replay therefore cannot establish service TTFT, TPOT, throughput, numerical
correctness or exact GPU-time-dependent scheduling.

## Workload

`cases.json` selects the recorded Qwen3.6-27B OASST2 histories and hardware
configuration. The 32-, 64- and 128-tree cases retain their actual outputs from
1024-, 512- and 256-token output-cap runs, including EOS stops. Every child must
contain its parent's complete prompt and actual generated token IDs; replay
aborts on a prefix mismatch. There is no chat-template rendering or
Detokenize/retokenize step.

Recorded dispatch order determines the per-tree request order. A tree has at most
one submitted request, and its next request waits for the previous completion and
its parent dependency. Different trees run concurrently. `--batch-size` limits new
arrivals per scheduler iteration; it does not force a prefill batch or bypass
production admission. `--workers` controls independent replay processes.

The recorded output length is used as each playback request's output cap. The
CPU decoder stops at that length rather than interpreting model EOS IDs; every
returned token is checked against the original history. This does not generate
extra tokens after the history's EOS. Both policies consume identical input and
output tokens, though pressure-dependent admission/completion order can differ.

## Memory model

Model two NVIDIA A800 80GB GPUs with TP=2 and BF16 storage. Derive head sharding,
shared K Logs, recurrent payloads, convolution snapshots and decay bitmaps from
[Qwen3.6-27B's configuration](https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json).
The per-rank cache budget is 47,605,350,400 bytes, taken from the recorded A800
allocation in
`experiment/standard_multiturn/qwen36-logstate32-trees128-o256-mixed-noprofile-20261026-001446/server.log`.
At 32-token granularity it provides 45,400 frames of 1 MiB. TP ranks mirror the
same logical ownership; their budgets are not pooled. Cache payload storage uses
the PyTorch meta device, so these bytes are not allocated in host RAM.

Per rank, a Checkpoint occupies 38 frames, a 32-token Log occupies 14 frames
plus one bitmap-shared decay record, and a decay frame holds seven records.
The minimum profitable materialization group contains three Log granules.
The pool and production manager account for all allocated work states, Logs,
Checkpoints, KV frames, decay records and dummy resources.

## Policies

- `default`: evict tail Logs first, remembering their shapes on CPU. If more
  space is needed, merge adjacent Checkpoints within each node, preferring the
  smallest coverage scale. Remove a leaf only after its remaining State can be
  pruned according to the production planner.
- `online`: rank tail Log/KV trimming, node-local Checkpoint merging and
  whole-leaf eviction by expected additional recomputation per released frame.
  Use the production Poisson density fit, amplitude/exponent priors, separate
  tail-hit probability and recency-weighted observations. The observation window
  defaults to 8192 tokens and half-life to 1024 admitted requests.

Owners with no internal hit observations use default candidate ordering: reclaim
available tail Logs before their Checkpoints, thin Checkpoints at the smallest
adjacent coverage scale, and allow whole-leaf Checkpoint pruning after those
merges. The fallback pool orders Logs by timestamp and Checkpoint pairs by scale,
then timestamp, using the default planner keys. Only its next action competes
with learned-owner actions by expected recomputation per frame. If no owner has
internal observations, the manager uses the default cold-window enumeration and
planner directly. Tail-only observations do not disable the fallback. Owners
with internal observations use the regularized density fit. Its exponent has a
quadratic penalty centered at 2, and amplitude and tail probability use pooled
priors.

Both modes retain KV during Checkpoint merging and release a terminal Checkpoint
with its leaf's KV. Materialization requires strictly positive frame savings.
Branch points and remembered protection endpoints are installed by the normal
production preparation and commit paths. Neither mode deletes Checkpoints merely
because a request or prefill batch completed.

## Outputs

- `report.md`, `comparison.png`, `comparison.pdf`: paired recomputation, hit-rate,
  entry-count and Checkpoint-span results.
- `frame-distribution.png`: per-rank cache resource allocation.
- `prefill-distribution.{json,csv,md}`: token distributions for all prefill
  forwards, mixed forwards, pure prefill forwards and per-request uncached input.
  Include mean, P50/P90/P99, maximum and fractions at token thresholds.
- `prefill-token-cdf.{png,pdf}`: prefill tokens per forward, grouped by tree count
  and policy. Mixed forwards count only prefill tokens. Statistics and plots are
  generated automatically by the report command, including single-case runs.
- `summary.json`, `summary.csv`: aggregate metrics.
- `{default,online}-trees<N>.json`: per-request admission hits, completion counts,
  frame usage, cache layouts, scheduler phase counts, eviction events and online
  observations. Snapshots cover prefill, eviction and completion transitions;
  iteration IDs preserve their order. Request records retain source dispatch
  indices and actual admission indices.
- `batches-trees<N>.csv`: transition snapshots and their resource breakdowns.
- `manifest.json`, `production.patch`, `case-trees<N>.json`: history hashes,
  software snapshot, `cpu_common.py` boundary hash and exact replay configuration.

Prefix hit rate is admitted recoverable tokens divided by input tokens.
Historical hit rate divides reusable parent tokens by the parent's complete
prompt-plus-output length. Historical recomputation excludes newly appended
input. Final integrity checks require every recorded output to be returned,
every request table/work slot to be released and all cache resources to reconcile.

## Retained results

- [Online versus default](results/aligned-prefill-distribution/report.md):
  paired 32-, 64- and 128-tree replays, including prefill-token distributions.
  Its online request records and cache-layout snapshots exactly match the
  regularized controls in the exponent ablation.
- [Exponent regularization](results/exponent-regularization/comparison.md):
  paired replays with the quadratic exponent penalty enabled or disabled.
  Amplitude and tail priors, zero-internal fallback and cache budgets are unchanged.
