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
