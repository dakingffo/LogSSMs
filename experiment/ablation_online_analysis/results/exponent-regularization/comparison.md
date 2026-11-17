# Exponent regularization ablation

Paired production CPU Scheduler/LogState replay using the same recorded prompts and actual generated token IDs. Both variants use online eviction. The off variant removes only -(k-2)^2/2 from the fitted objective, implemented by removing -(k-2) from its derivative. Amplitude and tail priors, exponent bounds, zero-internal fallback, weighted exposures and cache capacity are unchanged. The production implementation retains the penalty. GPU numerical correctness and service latency are not measured.

| Trees | Exponent penalty | Prefix hit | Historical hit | Historical recomputation (tokens) | KV gap | State gap | Final Checkpoints | Final Logs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | on | 92.1002% | 96.5595% | 6148 | 5843 | 305 | 265 | 2207 |
| 32 | off | 92.0833% | 96.5416% | 6180 | 5875 | 305 | 272 | 2190 |
| 64 | on | 88.5229% | 95.8881% | 8244 | 7812 | 432 | 291 | 2152 |
| 64 | off | 88.5087% | 95.8622% | 8296 | 8056 | 240 | 300 | 2129 |
| 128 | on | 80.9831% | 91.8964% | 18656 | 17036 | 1620 | 361 | 1950 |
| 128 | off | 80.8785% | 91.7826% | 18918 | 17458 | 1460 | 361 | 1955 |

## Historical recomputation changes with penalty removed

- 32 trees: +0.52%; `{"better": 0, "extra_tokens": 32, "saved_tokens": 0, "unchanged": 304, "worse": 1}`
- 64 trees: +0.63%; `{"better": 2, "extra_tokens": 256, "saved_tokens": 204, "unchanged": 589, "worse": 4}`
- 128 trees: +1.40%; `{"better": 8, "extra_tokens": 683, "saved_tokens": 421, "unchanged": 1119, "worse": 14}`

## Sparse model snapshots

Statistics below describe saved model-update snapshots with weighted internal hit count in (0,4]. These are not independent exponent fits. Production profiling retains at most the latest 8192 model updates; dropped counts are preserved in comparison.json.

| Trees | Exponent penalty | Snapshots | Mean k | Mean abs(k-2) | Minimum k | Maximum k |
|---:|---|---:|---:|---:|---:|---:|
| 32 | on | 3222 | 0.4649 | 1.5351 | 0.0000 | 1.9051 |
| 32 | off | 3224 | 0.1520 | 1.8480 | 0.0000 | 1.7987 |
| 64 | on | 3569 | 0.4733 | 1.5293 | 0.0000 | 2.1337 |
| 64 | off | 3544 | 0.1063 | 1.8988 | 0.0000 | 2.2176 |
| 128 | on | 4352 | 0.5885 | 1.4143 | 0.0000 | 2.2424 |
| 128 | off | 4291 | 0.0903 | 1.9144 | 0.0000 | 2.3967 |
