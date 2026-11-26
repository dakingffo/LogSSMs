# Online analysis ablation

| Policy | Overall hit rate | Historical hit rate | Historical recompute (tokens) | P95 / P99 (tokens) | Model prefill (tokens) | Reclamation rounds |
|---|---:|---:|---:|---:|---:|---:|
| MARCONI | 94.9435% | 98.9455% | 978,947 | 3,584 / 24,000 | 4,996,992 | 624 |
| LogState Default | 95.6163% | 99.1777% | 763,379 | 256 / 20,224 | 4,332,096 | 35,814 |
| LogState Online | 95.7693% | 99.3388% | 613,779 | 256 / 24,064 | 4,180,864 | 35,842 |

| Policy | Peak sampled State (GiB/rank) | Final Log (GiB/rank) | Final Checkpoint (GiB/rank) |
|---|---:|---:|---:|
| MARCONI | 12.01 | 0.00 | 9.81 |
| LogState Default | 53.50 | 0.97 | 23.40 |
| LogState Online | 53.50 | 0.72 | 16.55 |
