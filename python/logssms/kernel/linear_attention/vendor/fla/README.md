# Vendored FLA kernels

This directory contains the NVIDIA inference-only forward path derived from
`flash-linear-attention` 0.5.2. Training, backward, context-parallel, alternative
hardware backends, and backend dispatch are intentionally omitted.

LogSSMs's `chunk_gated_delta_rule` additionally returns the recurrent state
after every 64-token chunk. The serving layer selects page-boundary checkpoints
from those states and writes them into its persistent state pool.

The original implementation is Copyright (c) 2023-2026 Songlin Yang, Yu Zhang,
Zhiyuan Li and is distributed under the MIT license in `LICENSE`.
