# Prefix-cache locality experiments

`python -m experiment.prefix_cache_hit_rate` replays agentic trajectories offline. It reconstructs
every model invocation from the message history and finds its prefix match
endpoint by exact token-ID comparison. Native assistant tool calls and tool
responses are retained when applying the target model's chat template.

No inference server or model weights are required; only the tokenizer is loaded.

```bash
python -m experiment.prefix_cache_hit_rate \
  --tokenizer /path/to/model \
  --dataset swe_agent \
  --page-sizes 1,16,32,64,128,256 \
  --bin-width 256 \
  --plot-max-distance 2000 \
  --order round_robin
```

For the Thoughtworks agentic-coding corpus:

```bash
python -m experiment.prefix_cache_hit_rate \
  --tokenizer /path/to/model \
  --dataset agentic_coding_trajectories \
  --min-isl 8192 \
  --max-isl 16384 \
  --max-trajectories 100 \
  --order round_robin \
  --output-dir experiment/prefix_cache_hit_rate/agentic_coding_trajectories
```

`max_isl` in this dataset was measured with `cl100k_base`; requests are retokenized
with `--tokenizer` before prefix matching. The 8K--16K slice contains the SWE-smith
and Klear sources; OpenHands trajectories in this corpus start above 16K.

The experiment removes system messages before tokenization, so shared agent
instructions cannot dominate cross-session prefix hits. Page size 1 is the exact
token-prefix baseline; larger page sizes show the loss caused by page alignment.

Outputs under `experiment/results/prefix_cache_hit_rate/`:

- `density.csv`: bucket counts and `hit_count / bin_width / num_queries`.
- `ccdf.csv`: tail counts and `P(distance >= d)` at every bucket boundary.
- `summary.json`: experiment configuration and hit rates.
- `hit_density.png`: all prefix hits, including continuation hits at distance 0.
- `branch_hit_density.png`: only hits whose matched cached branch continues past
  the match endpoint.
- `branch_hit_ccdf.png`: log-log CCDF with a `d^-1` reference line, corresponding
  to a `d^-2` probability density.

The plot defaults to the 0--2000 token range. This only changes the x-axis;
`density.csv` always retains the complete long tail.

Use `--max-trajectories` and `--max-turns` for a quick smoke run. `sequential`
finishes one agent session before starting the next; `round_robin` interleaves one
turn from every active session and better approximates concurrent agents.
