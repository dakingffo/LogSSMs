---
license: other
license_name: derivative-multi-source
license_link: LICENSE
task_categories:
  - text-generation
language:
  - en
tags:
  - agentic-coding
  - swe-bench
  - multi-turn
  - benchmarking
  - llm-evaluation
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files:
      - split: train
        path: sessions.parquet
---

# agentic-coding-trajectories

A unified, tokenized corpus of **15,000 multi-turn agentic-coding sessions** (618K turns, 41 turns/session avg) drawn from three publicly-released upstream datasets. Built for benchmarking LLM serving systems on realistic multi-turn coding-agent workloads.

## Why this exists

Most LLM serving benchmarks use single-shot prompts. Real coding agents work in long multi-turn loops where each turn appends to a growing prompt. This corpus captures that shape — context grows monotonically across turns, output flips between reasoning-heavy planning and short tool-calls — so you can measure server behavior (TTFT, TPOT, KV cache, prefix-cache hit rate) under workload patterns that match production.

## Sources

| Source | Sessions | Original |
|---|---:|---|
| `nebius-swe-rebench-openhands` | 5,000 | [Nebius/swe-rebench-openhands](https://huggingface.co/datasets/Nebius/SWE-rebench) |
| `swe-smith-claude-3-7-sonnet` | 5,000 | [SWE-bench/SWE-smith trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) |
| `kwai-klear-swe-smith-mini` | 5,000 | [Kwaipilot/Klear-SWE-Smith-Mini](https://huggingface.co/datasets/Kwaipilot/Klear-SWE-Smith-Mini) |

Every row preserves `source_dataset` and `source_id` so you can trace back to the upstream record. The build process is a thin transformation (normalize message schema, tokenize with `cl100k_base`, write parquet sorted by max ISL) — no content is generated or modified.

## Schema

| Column | Type | Description |
|---|---|---|
| `session_id` | string | `<source_dataset>::<source_id>` — globally unique |
| `source_dataset` | string | One of the three sources above |
| `source_id` | string | Original ID in the upstream dataset |
| `agent_framework` | string | E.g. `swe-agent`, `openhands`, `klear` |
| `recorded_model` | string | Model used in the upstream trajectory (may be `unknown`) |
| `messages_json` | string | Full conversation as JSON: `[{role, content, ...}, ...]` |
| `n_turns` | int64 | Number of assistant turns in the session |
| `max_isl` | int64 | Token count at the deepest turn (cl100k_base) |
| `total_tokens` | int64 | Sum of all message tokens |
| `ground_truth_meta_json` | string | Per-row metadata (e.g., `instance_id`, `resolved`, `patch_present`) |

## Distribution

**By max-ISL (deepest prompt in each session):**

| Range | Sessions | % |
|---|---:|---:|
| 0–8K | 1,072 | 7% |
| 8–16K | 4,197 | 28% |
| 16–24K | 2,389 | 16% |
| 24–32K | 1,584 | 11% |
| 32–64K | 4,748 | 32% |
| 64K+ | 1,010 | 7% |

p50 max-ISL is 23,840 tokens. p95 is 69,274 tokens. p99 is 89,876 tokens.

**By turns per session:** p50 = 36, p95 = 87, max = 120.

## Quickstart

```python
from datasets import load_dataset
import json

ds = load_dataset("thoughtworks/agentic-coding-trajectories", split="train")

# A session
sess = ds[0]
print(sess["session_id"], "→", sess["n_turns"], "turns,", sess["max_isl"], "max ISL")

# Iterate turns
messages = json.loads(sess["messages_json"])
for msg in messages:
    print(f"[{msg['role']}] {str(msg.get('content', ''))[:80]}")
```

**Filter by max-ISL window** (typical for benchmarking a fixed-context model):

```python
ds_8_16k = ds.filter(lambda r: 8192 <= r["max_isl"] < 16384)
print(len(ds_8_16k), "sessions in 8K-16K window")
```

**Sample N sessions for parallel multi-turn replay:**

```python
import random

def sample_sessions(ds, n: int, *, min_isl: int = 0, max_isl: int = 1_000_000, seed: int = 0):
    pool = [r for r in ds if min_isl <= r["max_isl"] < max_isl]
    random.Random(seed).shuffle(pool)
    return pool[:n]

users = sample_sessions(ds, n=16, min_isl=8192, max_isl=16384)
# Replay each user's session turn-by-turn against your serving endpoint
```

## Intended use

- **Benchmark LLM serving systems** under realistic multi-turn agentic loads (TTFT, TPOT, KV cache pressure, prefix-cache reuse, throughput vs concurrency)
- **Evaluate agent behavior** with controlled prompt-shape distributions
- **Study reasoning vs tool-call patterns** within long sessions

## Limitations

- **Tokenizer-specific counts.** `max_isl` and `total_tokens` are computed with `cl100k_base`. If you serve a model with a different tokenizer, recount before drawing conclusions.
- **Synthetic agent traces.** All trajectories are LLM-generated solving real GitHub repos. They are not human-authored.
- **No live tools.** This is replay data, not a tool-execution environment. The agent's tool calls and the simulated tool responses are baked into `messages_json`.
- **Source skew.** Two of the three upstreams come from SWE-bench / SWE-smith ecosystems. Generalization to other agentic domains (data analysis, product Q&A, etc.) is not guaranteed.

## Build reproducibility

The corpus is fully reproducible from the three upstream HF datasets in ~3 minutes on a laptop. The build script reads each upstream, normalizes message schema, tokenizes, and writes sorted parquet. We will publish the standalone build + benchmarking package separately.

## License

This is a derivative dataset. Each row's content is governed by the license of its `source_dataset`:

- **`nebius-swe-rebench-openhands`** — see [the Nebius SWE-rebench dataset card](https://huggingface.co/datasets/Nebius/SWE-rebench)
- **`swe-smith-claude-3-7-sonnet`** — see [the SWE-bench SWE-smith dataset card](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories). Trajectory text was generated using Anthropic's Claude 3.7 Sonnet; downstream use is also subject to [Anthropic's Usage Policy](https://www.anthropic.com/legal/aup).
- **`kwai-klear-swe-smith-mini`** — see [the Kwaipilot Klear dataset card](https://huggingface.co/datasets/Kwaipilot/Klear-SWE-Smith-Mini)

Filter by `source_dataset` if you need a uniform-license subset.

## Citation

```bibtex
@misc{agentic-coding-trajectories-2026,
  title  = {agentic-coding-trajectories: a unified multi-turn coding-agent corpus for serving benchmarks},
  author = {ThoughtWorks},
  year   = {2026},
  url    = {https://huggingface.co/datasets/thoughtworks/agentic-coding-trajectories}
}
```

Please also cite the three upstream datasets (links above).
