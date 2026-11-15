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
  --plot-max-distance 1024 \
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
- `hit_density_ccdf.png`: vertically stacked density and CCDF panels with one
  enlarged legend inside the CCDF panel and no x-axis description. The upper panel shows all prefix hits;
  the lower panel shows the log-log CCDF of internal prefix-hit distances with
  a `d^-1` reference line, corresponding to a `d^-2` probability density.

The plot defaults to a maximum distance of 1024 tokens. The density panel uses a linear
x-axis starting at zero; the CCDF uses a logarithmic x-axis with its lower bound
chosen to align the highlighted boundary across both panels (324 tokens for
SWE-agent and 256 tokens for Thoughtworks at a 1024-token upper bound).
`density.csv` always retains the complete long tail.

Figures show page sizes 16, 32, 64, and 128, with a shaded recent region ending at
768 tokens for SWE-agent and 512 tokens for Thoughtworks. Only that boundary and the right endpoint are
labeled on the x-axis, with token units; y-axis tick labels are hidden. The CCDF y-axis starts at
`1e-3`. CSV files retain all measured page sizes and their original statistics.

Use `--max-trajectories` and `--max-turns` for a quick smoke run. `sequential`
finishes one agent session before starting the next; `round_robin` interleaves one
turn from every active session and better approximates concurrent agents.
