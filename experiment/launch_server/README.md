# Standalone server launchers

These entrypoints only launch services. `experiment.random_multiturn` is a
separate HTTP client and never installs, imports, starts, or stops SGLang or vLLM.

LogSSMs flags are handled by its own server parser:

```bash
python experiment/launch_server/launch_logssms.py \
  --model-path /path/to/model --tensor-parallel-size 2 \
  --cache-type logstate --granularity 32 --port 28300
```

SGLang flags are passed to `sglang.launch_server` in a separate Python
environment. Only `--sglang-python` and optional `--sglang-root` belong to
the wrapper; the remaining flags belong to SGLang:

```bash
python experiment/launch_server/launch_sglang.py \
  --sglang-python /root/autodl-tmp/sglang-venv/bin/python \
  --sglang-root /root/autodl-tmp/sglang \
  --model-path /path/to/model --tp-size 2 --port 28300
```

Run the client from another terminal once the server is ready. For a clean
comparison, restart the service before each benchmark point.

vLLM uses its own installed environment. Only `--vllm-python` belongs to the
wrapper; all other arguments are passed unchanged to `vllm serve`. Unlike the
other launchers, its model path is a positional argument:

```bash
python experiment/launch_server/launch_vllm.py \
  --vllm-python /root/autodl-tmp/vllm-venv/bin/python \
  /root/autodl-tmp/models/Qwen3.6-27B \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.9 \
  --enable-prefix-caching --enable-prompt-tokens-details --port 28300
```

The launcher does not translate LogSSMs/SGLang flags into vLLM flags, and removes
inherited `PYTHONPATH` to avoid importing another checkout into the child process.
Use `--api vllm` in `random_multiturn` once the server is ready. Enable prefix
caching and `--enable-prompt-tokens-details` to expose actual cached-token counts
in the response; client results distinguish missing cache metrics from zero hits.

For exact token-prefix continuity, vLLM 0.24.0 accepts token IDs through
`POST /v1/completions`, not through ordinary chat `messages`:

```json
{
  "model": "/root/autodl-tmp/models/Qwen3.6-27B",
  "prompt": [100, 200, 300],
  "max_tokens": 512,
  "ignore_eos": true,
  "return_token_ids": true,
  "stream": true
}
```

Streaming `choices[].token_ids` contains each chunk's generated token IDs;
`choices[].prompt_token_ids` is returned in the first chunk. Build the next turn
from the original prompt IDs, actual generated IDs, and new input IDs, without
detokenizing/retokenizing or applying a chat template. Verify the returned prompt
IDs and observed cache hits before interpreting performance.
