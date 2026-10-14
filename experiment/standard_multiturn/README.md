# Standard multi-turn prefix-cache experiment

Figure 1 compares LogSSMs, vLLM and SGLang without profiling. Execute complete
OASST2 trees concurrently, using each server's actual generated token IDs.
Request behavior emerges from a seeded randomized topological order, completion
order and server load. There are no behavior quotas or artificial revisit delays.
Run `python -m experiment.standard_multiturn` from the repository root against an
already-running server. Server launch remains separate under
`experiment/launch_server/`; the client never starts or stops the service.

```bash
python -m experiment.standard_multiturn \
  --api logssms --base-url http://127.0.0.1:28300 \
  --tokenizer /root/autodl-tmp/models/Qwen3.6-27B \
  --dataset oasst2 --language en --num-trees 64 \
  --output-len 1024 --max-seq-len 18432 --seed 0 \
  --name qwen36-logstate32-trees64
```

Use `--api vllm` or `--api sglang` against the corresponding service; no imports
of those inference engines are needed in the client environment. For vLLM enable
prefix caching and prompt-token details, and set `--model` when the served name
differs from the tokenizer path. Change `--num-trees` to 128 or 256 for the other
points and use a fresh server and a unique run name for each point.

`--prepare-only` writes the manifest without making an HTTP request. It audits
the context bound and freezes the selected trees, encoded suffixes, tokenizer
template hash and random topological priorities. The default run directory is
`experiment/standard_multiturn/<name>/`; `--out-dir` changes the parent directory.

## Workload

- Reconstruct trees from `datasets/parser.py` using `message_tree_id`,
  `message_ids` and `parent_ids`. Deduplicate shared nodes by message ID;
  do not independently replay every root-to-leaf record.
- Fix language, quality filters, tree ordering and seed in a manifest. Shuffle
  eligible tree IDs once; use the first 64, 128 and 256 as nested samples.
- Preserve the selected trees' surviving topology. Do not restrict each tree to
  one fork, two user children or a fixed depth, and do not fabricate longer paths.
- Distinguish user-follow-up forks from alternative assistant replies in OASST2.
  Preserve their provenance rather than calling every edge a user fork.
  Recorded assistant text is never substituted for generated history.
- Generate each assistant node once in the execution graph. Its descendant
  requests reuse that node's actual generated IDs. Alternative assistant nodes
  have their own generation events, not duplicated recorded answers.
- Collect all generation requests across the selected trees. Build a seeded
  random topological order: choose randomly among nodes whose parents have already
  been placed in the order. Save this order in the manifest.
- Dispatch asynchronously using that order as priority. A request is executable
  only after its parent response has completed and its actual output IDs are
  available, not merely after its parent HTTP request was submitted.
- Skip a temporarily blocked request and dispatch subsequent executable requests;
  revisit blocked requests when their parents finish. Newly executable siblings
  rejoin the global ordering rather than being sent as a dedicated sibling batch.
- The selected trees' roots are executable from the start. Do not serialize whole
  trees or impose a sibling-completion dependency.
- Finish when the selected trees have completed or explicitly report failures.
  There are no cross-tree dependencies, traffic counters or per-turn barriers.

This is dataset-driven generation, not exact replay of recorded conversations:
the original follow-up texts may refer to answers different from those generated.
The random topological construction is not guaranteed to sample all possible
topological orders uniformly. It also does not guarantee a minimum distance
between siblings: when few requests are executable, siblings may still cluster.
Use the same manifest priorities across systems and record actual dispatch order,
which can differ because parent completion times differ.

## Prefix continuity

For shared history `H = parent_prompt_ids + parent_generated_ids`, form child
prompts as `H + child_suffix_ids`. Encode only the new user turn and required
chat boundaries. Never render the whole history again or decode/re-encode `H`.
Use one tokenizer and chat protocol across systems, retain all returned special
tokens, and avoid duplicating a terminal EOS in the next turn boundary.
The client renders chat scaffolds with `enable_thinking=False`,
`preserve_thinking=True` and a fixed template date. These kwargs and the template
hash are saved in the manifest. Only the new suffix is tokenized; returned
reasoning and other generated content are preserved verbatim as IDs regardless
of whether a template honors the thinking option.

Disable `ignore_eos`; the output length is a maximum, not a fixed response size.
Validate incremental template handling and token-level prefix continuity before
interpreting cache results. Generated outputs can differ across engines, so
record actual input/output lengths and trace hashes rather than claiming identical
token traces.

## Load and comparison controls

The matrix has 9 points: 3 tree counts × 3 systems. Increasing tree count increases
both offered load and retained-history pressure; it is not a fixed-client-concurrency
capacity sweep.

The client does not impose a 32-request or per-tree semaphore. HTTP connection
limits must accommodate ready requests without unintentionally serializing them;
record any transport-side waiting separately. The server queues and admits the
submitted traffic according to its normal scheduler and memory constraints.
Keep the server maximum running requests fixed at 32 across the three systems
and all tree-count points initially. This is distinct from the number of submitted
or pending requests.

Suggested common settings are Qwen3.6-27B BF16, TP=2, memory fraction 0.9,
prefill budget 8192, maximum output 1024, and CUDA graphs enabled. Use LogSSMs
LogState32 with memorized eviction and non-eager state, vLLM `align`, and SGLang
`extra_buffer` with tracking interval 256. Record merge flags, versions, graph
settings and startup capacities.

Audit path lengths with a conservative output-cap bound before fixing the common
context limit. Do not silently truncate shared history. Start a fresh server for
each point, warm up with independent trees and exclude warmup requests from metrics.
By default four disjoint trees warm up with output capped at 64 tokens. Their
history and stats are saved separately; use `--warmup-trees 0` for a cold-start
run. Warmup leaves entries in the cache and can share ordinary chat headers with
measured requests, but not their tree histories. Begin with seed 0; repeated matched
seeds can later quantify run-to-run variation.

## Figure and report

Use one bar-and-line plot. The x-axis has 64/128/256-tree groups, each containing
LogSSMs, vLLM and SGLang bars. Bars show mean TTFT over all measured requests on
the left axis. System lines show overall token-weighted cache-hit fraction on the
right axis, including cold roots in its denominator:

```
sum(cached_tokens) / sum(prompt_token_count)
```

Also report reuse of the verified shared history:

```
sum(min(cached_tokens, expected_reusable_len)) / sum(expected_reusable_len)
```

Read request admission counts through the existing response adapters:
LogSSMs/SGLang `meta_info.cached_tokens`, vLLM
`usage.prompt_tokens_details.cached_tokens`. Missing metadata is unmeasured,
not zero. Profiling and production responses use the same admission snapshot.

Report TTFT p50/p90/p99, TPOT, E2E, output throughput, actual token counts,
zero-hit/partial-hit fractions and achieved in-flight request counts. TTFT starts
at HTTP dispatch and includes server queuing; retain ready/dispatch timestamps
to expose client-side delay. Use actual output counts for EOS-terminated requests.
Empty responses have no first-token latency; responses with fewer than two tokens
have no TPOT sample. Counts for both cases and the number of latency samples are
included in the report, rather than interpreting them as zero-latency responses.

Save `report.json`, `history.json` and `manifest.json` under a named run directory
here. History retains tree/message/parent IDs, prompt/generated IDs, expected
shared prefix, observed cached tokens and timestamps. Figure 2 may analyze
realized revisit distance and capacity pressure; Figure 1 does not force requests
into three equally sized classes or claim that immediate submission creates a
deliberate delayed-return workload.

Completed requests are retained in a partial history if a request fails; the
report has `status=failed` and its error. These runs must not be plotted as completed
benchmarks. `report.json` includes overall and reusable-history cache fractions;
missing response cache details are explicitly unmeasured.

## Plot Figure 1

```bash
python -m experiment.standard_multiturn.plot \
  --reports experiment/standard_multiturn/qwen36-*/report.json \
  --out experiment/standard_multiturn/figures/figure1.png
```

Supply all three systems for each tree-count point. The plotter requires matching
manifest hashes and seeds, measured cache counts and completed runs. With repeated
seeds, bars show mean run TTFT with standard-deviation error bars; hit fractions
are weighted by input tokens across runs. The history hashes still expose any
differences in generated continuations between systems.
