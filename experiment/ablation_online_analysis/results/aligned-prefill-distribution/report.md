# Online-analysis ablation

Qwen3.6-27B token playback through the production Scheduler overlap loop, Engine cache preparation, PrefillAdder, DecodeManager and LogStateCacheManager. Two A800 TP ranks have mirrored logical frame ownership. Both policies use the same recorded tokens and arrival rules. Work states are allocated on admission, and decode allocates incrementally. GPU numerical kernels and completion events are replaced with synchronous CPU metadata adapters. No model arithmetic or GPU timing is simulated.

| Trees | Policy | Prefix hit | Historical prefix hit | Historical recompute tokens | Mean Checkpoints before matching | Mean Logs before matching |
|---:|---|---:|---:|---:|---:|---:|
| 32 | default | 91.24% | 95.67% | 7,746 | 259.42 | 910.08 |
| 32 | online | 92.10% | 96.56% | 6,148 | 184.16 | 1099.13 |
| 64 | default | 87.72% | 95.01% | 9,995 | 268.47 | 931.03 |
| 64 | online | 88.52% | 95.89% | 8,244 | 180.27 | 1151.95 |
| 128 | default | 79.13% | 90.00% | 23,020 | 285.15 | 903.09 |
| 128 | online | 80.98% | 91.90% | 18,656 | 201.19 | 1112.74 |

Historical recomputation is max(0, parent prompt length + actual generated length - recoverable prefix). Prefill tokens include both historical recomputation and new input. Prefix hit rate divides recoverable tokens by all input tokens; historical prefix hit rate uses only inherited parent tokens. The 32/64/128-tree histories retain their original 1024/512/256 output caps and actual EOS lengths. Compare policies within each case.

Each JSON includes per-request recovery, per-batch Checkpoint span histograms and Log counts, per-rank frame usage/free space, memory-layout constants, model-update samples and eviction events. Both TP ranks have identical logical ownership; their bytes are not pooled into a single allocator.

## Historical recomputation breakdown

KV gap is inherited history beyond the token-prefix match. It includes both missing cacheable KV and the parent's unretained alignment tail. State gap is inherited history whose KV prefix matches but whose State endpoint is unavailable. Both are model-prefill tokens; replaying retained Logs to reconstruct work State is not counted here.

| Trees | Policy | KV gap tokens | Cacheable KV gap | Alignment tail | State gap tokens |
|---:|---|---:|---:|---:|---:|
| 32 | default | 7,601 | 5,504 | 2,097 | 145 |
| 32 | online | 5,843 | 3,840 | 2,003 | 305 |
| 64 | default | 9,867 | 6,016 | 3,851 | 128 |
| 64 | online | 7,812 | 3,968 | 3,844 | 432 |
| 128 | default | 22,444 | 14,208 | 8,236 | 576 |
| 128 | online | 17,036 | 9,376 | 7,660 | 1,620 |

## Frame distribution

Request-weighted means before admission batches, per TP rank. One frame is 1 MiB. Log includes payload, convolution snapshots and bitmap-shared decay frames, including the sentinel Log. Checkpoint includes payload and convolution. Work contains admitted requests' actual work slots and the sentinel work slot. KV includes the sentinel KV frame. Every snapshot reconciles with the case's frame budget.

| Trees | Policy | Log frames | Decay subset | Checkpoint frames | KV frames | Work frames | Free frames |
|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | default | 17571.8 | 190.0 | 9857.9 | 2003.9 | 973.8 | 14992.5 |
| 32 | online | 20238.0 | 209.6 | 6998.2 | 2191.2 | 973.8 | 14998.7 |
| 64 | default | 18359.5 | 210.3 | 10201.9 | 2075.7 | 1863.7 | 12899.1 |
| 64 | online | 21475.2 | 234.8 | 6850.2 | 2290.0 | 1862.9 | 12921.8 |
| 128 | default | 18888.9 | 216.4 | 10835.7 | 2134.1 | 3650.7 | 9890.6 |
| 128 | online | 21852.8 | 239.5 | 7645.4 | 2316.0 | 3652.9 | 9932.9 |

## Prefill token distributions

[Statistics by forward phase and request](prefill-distribution.md) include means, quantiles and threshold fractions. [The forward-token CDF](prefill-token-cdf.png) compares the two policies.
