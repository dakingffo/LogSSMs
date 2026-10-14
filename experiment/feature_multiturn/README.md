# Feature multi-turn experiment

Placeholder for Figure 2: fix 128 OASST2 trees and enable profiling. Unlike the
overall mixed-workload plot in Figure 1, its x-axis has three behavior groups:
immediate forks, delayed revisits and pressure revisits. Each group compares
LogSSMs, vLLM and SGLang to explain the individual features.

Reuse `standard_multiturn` selection manifests, token-exact history construction,
request classifications and scheduling policies. Do not introduce a separate
definition of cache hits. Profiling consumes the same request admission snapshot
returned in production responses.
Record actual revisit distance and capacity pressure for analysis. Figure 1
uses randomized topological priorities and has no imposed behavior labels or one-third
quotas. Whether Figure 2 uses observed traffic groups or separate controlled
feature workloads remains to be decided; do not infer deliberate delayed
revisits from server queuing alone.

The plotted quantities are not finalized. Candidates include recomputed prefill
tokens, persistent cache bytes and free capacity, and state recovery/merge/eviction
work. Separate comparable cross-system measurements from implementation-specific
diagnostics. Profiling overhead must not be presented as normal serving latency;
Figure 1 supplies the non-profiled performance comparison.

No profiling client or plotting code is implemented in this directory yet.
