# Granularity ablation

Compare LogState granularity 16, 32, and 64 at 16, 32, 48, and 64 concurrent
trees using CPU replay.

```bash
bash scripts/ablation_granularity.sh
```

All points replay the same 64 OASST2 trees from `data/history.json`. The
history contains the actual prompt and output token IDs recorded on
Qwen3.6-27B with a 1,024-token output cap and EOS enabled. Replay verifies
each continuation against its parent's exact prompt and generated tokens.
Admission uses the original 1,024-token budget, and decoding stops at the
recorded EOS token or at the budget.
One request per tree can be in flight. A tree retains its client slot until
the client observes its last response, then another tree takes the slot.
A finished response passes through one receive interval before it becomes
visible to the simulated client. This models the frontend round trip while
the scheduler retains its production deferred-release behavior. The
client-side history does not record exact server-side receive iterations.

The production scheduler and LogState allocator run with TP=2 metadata and
the same cache byte budget per rank at every point. The number of frames
depends on granularity. The server-side limit stays at 64 requests. Online
analysis and hit materialization are enabled.

The horizontal axis shows the unweighted mean of individual continuation
request hit fractions. The vertical axis shows the token-weighted historical
prefix hit rate. Both use the recoverable cache length at admission, capped
at the parent's prompt and output length. Roots have no parent history and
are excluded. Marker shapes identify the concurrent tree cap and colors
identify the granularity in tokens. Gray solid lines connect points at the
same concurrency.

Dashed curves connect points with the same granularity and extend to the plot
edges along their endpoint slopes.

The runner writes replay reports, `summary.json`, `granularity.pdf`, and
`granularity.png` to `results/<run>/`. Use `--out` to select another directory,
`--granularity` and `--concurrency` to run selected points, and `PYTHON` to
select the interpreter. The replay requires the LogSSMs Python dependencies
and Matplotlib, and runs without a GPU.

To redraw an existing complete run from its request-level reports:

```bash
bash scripts/ablation_granularity.sh --plot-only --out experiment/ablation_granularity/results/<run>
```
