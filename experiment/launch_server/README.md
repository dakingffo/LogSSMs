# Standalone server launchers

These entrypoints only launch services. `experiment.random_multiturn` is a
separate HTTP client and never installs, imports, starts, or stops SGLang.

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
