# Actual prefill token distributions

Forward counts include only valid prefill tokens before Engine.prepare_batch advances requests. Mixed forwards exclude decode tokens; graph padding, State reconstruction and Log materialization are excluded. The request scope counts each request's total uncached input, including all prefill chunks. Both totals must equal the replay's prefill-token total. Quantiles use NumPy linear interpolation. Empty scopes have no quantiles or threshold fractions.

| Trees | Policy | Scope | Count | Mean | P50 | P90 | P99 | Max | <=32 | <=64 | <=128 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | default | all | 270 | 61.61 | 31.50 | 90.00 | 742.79 | 1077 | 52.96% | 82.22% | 94.81% |
| 32 | default | mixed | 256 | 37.06 | 30.00 | 71.50 | 116.70 | 128 | 55.86% | 86.72% | 100.00% |
| 32 | default | prefill | 14 | 510.50 | 422.50 | 909.40 | 1056.46 | 1077 | 0.00% | 0.00% | 0.00% |
| 32 | default | request | 305 | 54.54 | 31.00 | 72.60 | 676.28 | 1077 | 55.08% | 87.54% | 96.07% |
| 32 | online | all | 270 | 55.57 | 31.00 | 89.10 | 638.67 | 949 | 54.44% | 83.33% | 95.56% |
| 32 | online | mixed | 258 | 36.65 | 30.00 | 71.30 | 116.58 | 128 | 56.98% | 87.21% | 100.00% |
| 32 | online | prefill | 12 | 462.25 | 326.50 | 837.30 | 938.66 | 949 | 0.00% | 0.00% | 0.00% |
| 32 | online | request | 305 | 49.19 | 30.00 | 70.60 | 619.00 | 949 | 56.39% | 88.52% | 96.72% |
| 64 | default | all | 428 | 64.31 | 43.00 | 105.00 | 459.49 | 923 | 42.29% | 65.19% | 94.63% |
| 64 | default | mixed | 405 | 47.30 | 39.00 | 92.00 | 121.00 | 125 | 44.69% | 68.89% | 100.00% |
| 64 | default | prefill | 23 | 363.70 | 366.00 | 531.00 | 843.58 | 923 | 0.00% | 0.00% | 0.00% |
| 64 | default | request | 595 | 46.26 | 29.00 | 79.60 | 429.84 | 527 | 57.82% | 86.05% | 95.80% |
| 64 | online | all | 431 | 59.70 | 42.00 | 104.00 | 417.90 | 652 | 42.00% | 64.97% | 95.82% |
| 64 | online | mixed | 413 | 47.97 | 40.00 | 93.00 | 121.00 | 125 | 43.83% | 67.80% | 100.00% |
| 64 | online | prefill | 18 | 328.78 | 316.50 | 518.30 | 630.58 | 652 | 0.00% | 0.00% | 0.00% |
| 64 | online | request | 595 | 43.25 | 29.00 | 78.60 | 330.18 | 515 | 58.15% | 86.39% | 96.47% |
| 128 | default | all | 658 | 87.30 | 57.00 | 126.30 | 654.66 | 2170 | 31.76% | 54.56% | 90.12% |
| 128 | default | mixed | 593 | 53.34 | 50.00 | 96.00 | 123.08 | 127 | 35.24% | 60.54% | 100.00% |
| 128 | default | prefill | 65 | 397.11 | 281.00 | 665.20 | 1812.88 | 2170 | 0.00% | 0.00% | 0.00% |
| 128 | default | request | 1141 | 50.34 | 30.00 | 90.00 | 345.40 | 579 | 54.60% | 82.30% | 92.99% |
| 128 | online | all | 666 | 78.61 | 53.00 | 116.50 | 582.30 | 2170 | 33.63% | 58.71% | 92.19% |
| 128 | online | mixed | 614 | 52.00 | 48.00 | 96.00 | 122.00 | 126 | 36.48% | 63.68% | 100.00% |
| 128 | online | prefill | 52 | 392.77 | 255.00 | 691.60 | 1885.42 | 2170 | 0.00% | 0.00% | 0.00% |
| 128 | online | request | 1141 | 45.88 | 29.00 | 85.00 | 305.20 | 579 | 57.41% | 84.93% | 94.39% |
