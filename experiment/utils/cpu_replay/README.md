# CPU cache replay

The replay runs the production scheduler and cache manager with recorded output
tokens and CPU cache metadata. Supported `--mode` policies are `default` and
`online` (LogState), and `marconi` (Marconi hybrid block cache with optional
`--block-size` and `--bootstrap-multiplier`). Choose a trace mode independently
of the eviction policy.

| Trace mode | Input | Request release |
| --- | --- | --- |
| `eager` (default) | OASST2 history JSON | One outstanding request per session tree. Submit the next eligible request when its parent response is visible. |
| `virtual_clock` | SemiAnalysis Claude Code JSONL or history JSON with a session-relative `t` in seconds on each request | Submit requests at their recorded offsets, including overlapping requests from the same session. |

`--tree-concurrency` limits open client sessions. In virtual-clock mode, the next
session starts when one of those sessions finishes. The server's
`max_running_requests` still controls admission independently.
`--batch-size` limits the number of arrivals delivered in one scheduler iteration.

Virtual time advances by `--clock-step-ms` per scheduler iteration (default
`1` ms) and jumps to the next arrival when idle. The step controls the
interleaving of model iterations and recorded arrivals. It does not estimate
GPU latency. Use the same step for policy comparisons. The report records
the original request offset, virtual arrival and virtual admission times.

The dataset parser in `datasets/parser.py` reads Agent JSONL and normalizes it
to replay records. Session-local 64-token block hashes are mapped to
deterministic token IDs. A later prompt on the same stream supplies any
recoverable earlier output IDs. Output lengths follow the trace with EOS ignored.
Prompt continuity is exact for the reconstructed token IDs. The source trace
only exposes block-level relationships. The virtual-clock replay uses recorded
arrival offsets directly, including arrivals from parallel subagents. The
recorded offsets do not wait for simulated output completion within a session.

```bash
PYTHONPATH=python:. python -m experiment.utils.cpu_replay.replay \
  --mode default --trace-mode virtual_clock \
  --history datasets/cc-traces-weka-062126/traces.jsonl \
  --case /path/to/agent-case.json --trace-sessions 1 --tree-concurrency 1 \
  --skip-layout-snapshots --out /tmp/agent-replay.json
```

Set the case's context limit and cache capacity to fit the selected sessions.
`--trace-sessions` selects the first N rows. Without it, all sessions are read.
