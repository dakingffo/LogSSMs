# Vendored FLA kernels

This directory contains the NVIDIA inference-only forward path derived from
`flash-linear-attention` 0.5.2. Training, backward, context-parallel, alternative
hardware backends, and backend dispatch are intentionally omitted.

`chunk_gated_delta_rule_to_cache` reads and writes recurrent states directly in
the persistent pool, with optional direct Log writes. The recurrent decode
paths likewise update persistent state or Log slots; verify returns per-token
states for acceptance processing.

The original implementation is Copyright (c) 2023-2026 Songlin Yang, Yu Zhang,
Zhiyuan Li and is distributed under the MIT license in `LICENSE`.
