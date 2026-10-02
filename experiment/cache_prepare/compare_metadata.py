"""Validate and summarize the saved local metadata baseline comparison."""
import json
from pathlib import Path
import statistics


root = Path("benchmark-results/metadata-study")
baseline = json.loads((root / "baseline-repeat.json").read_text())
optimized = json.loads((root / "optimized-repeat.json").read_text())
comparison = []
for before, after in zip(baseline["metadata"], optimized["metadata"], strict=True):
    identity = ("bs", "phase", "page_size", "linear")
    assert all(before[key] == after[key] for key in identity)
    comparison.append({
        **{key: after[key] for key in identity},
        "cpu_submit_baseline_p50_ms": before["cpu_submit_p50_ms"],
        "cpu_submit_optimized_p50_ms": after["cpu_submit_p50_ms"],
        "cpu_submit_reduction_fraction": 1 - after["cpu_submit_p50_ms"] / before["cpu_submit_p50_ms"],
        "explicit_h2d_transfers_baseline": before.get("explicit_h2d_transfers", before.get("h2d_transfers")),
        "explicit_h2d_transfers_optimized": after.get("explicit_h2d_transfers", after.get("h2d_transfers")),
    })

e2e = []
traces = []
for name in ("e2e-baseline", "e2e-optimized"):
    history = json.loads((root / name / "client/history-c4.json").read_text())
    for session in history["sessions"]:
        for previous, current in zip(session["turns"], session["turns"][1:]):
            prefix = previous["prompt_token_ids"] + previous["generated_token_ids"]
            assert current["prompt_token_ids"][:len(prefix)] == prefix
    traces.append([[(turn["prompt_token_ids"], turn["generated_token_ids"])
                    for turn in session["turns"]] for session in history["sessions"]])
    report = json.loads((root / name / "client/report.json").read_text())
    profile = json.loads((root / name / "scheduler.rank0.json").read_text())
    assert profile["engine_forward"]["phases"]["decode"]["batches"] > 0
    samples = profile["engine_prepare"]["decode_batches"]
    e2e.append(dict(
        name=name, cache=report["points"]["4"]["cache"],
        released_state_slots_from_merge=profile["cache"]["aggregate"]["released_state_slots_from_merge"],
        profiled_decode_prepare_p50_ms=statistics.median(sample["cpu_total_ms"] for sample in samples),
    ))
assert traces[0] == traces[1], "Different model outputs or subsequent request histories"
assert e2e[0]["cache"] == e2e[1]["cache"]
cache_before = json.loads((root / "cache-baseline.json").read_text())
cache_after = json.loads((root / "cache-optimized.json").read_text())
assert cache_before["runs"][0]["matches"] == cache_after["runs"][0]["matches"]
assert cache_before["runs"][0]["layout"] == cache_after["runs"][0]["layout"]
result = dict(
    scope="Local RTX4060 microbench + graph-enabled Qwen3.5-0.8B correctness E2E; not server performance",
    metadata=comparison, reclamation=dict(baseline=baseline["reclamation"], optimized=optimized["reclamation"],
        note="Artificial pending-kernel delay: measures removal of CPU blocking, not serving TTFT"),
    e2e=e2e, exact_prompt_and_generated_token_ids_equal=True,
    cache_layout_and_hits_equal=True,
    e2e_timing_note="Cold JIT differs; do not compare total duration/throughput as an optimization result",
    prefill_budget="Unchanged, no budget sweep; default 8192 left for future server experiment",
)
(root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
