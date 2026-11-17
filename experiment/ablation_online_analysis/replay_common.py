from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from logssms.distributed.info import DistributedInfo, tp_override
from experiment.ablation_online_analysis.cpu_common import ReplayScheduler, make_manager, numerical_boundaries


EXECUTION = "production-scheduler-v1"


def frame_usage(allocator):
    usage = Counter(log=0, checkpoint=0, work=0)
    for slot in allocator.allocated_slots:
        record = allocator.slot(slot)
        usage[record.kind] += len(record.payload) + sum(map(len, record.conv))
    usage["log_payload_conv"] = usage["log"]
    usage["decay"] = allocator.profile_usage()["decay_frames"]
    usage["log"] += usage["decay"]
    usage["kv"] = len(allocator.allocated_kv)
    usage["free"] = allocator.free_frames
    assert sum(usage[k] for k in ("log", "checkpoint", "work", "kv", "free")) == allocator.num_frames
    return dict(usage)


def layout_snapshot(tree):
    entries = tree.iter_entries()
    spans = Counter(str(e.end - e.begin) for e in entries if e.kind.value == "checkpoint")
    runs = Counter()
    for node in tree._iter_nodes():
        run, last = 0, -1
        for entry in node.entries:
            assert last <= entry.begin
            if entry.kind.value == "log":
                if run and entry.begin != last:
                    runs[str(run)] += 1
                    run = 0
                run += entry.end - entry.begin
            elif run:
                runs[str(run)] += 1
                run = 0
            last = entry.end
        if run:
            runs[str(run)] += 1
    return dict(checkpoint=sum(e.kind.value == "checkpoint" for e in entries),
                log=sum(e.kind.value == "log" for e in entries),
                checkpoint_spans=dict(spans), log_run_spans=dict(runs))


def replay(case, history, mode, batch_size=8, window=8192, half_life=1024):
    torch.set_num_threads(1)
    requests = sorted(json.loads(history.read_text())["requests"], key=lambda r: r["dispatch_index"])
    if not requests or mode not in ("default", "online") or batch_size < 1:
        raise ValueError("Replay requires requests, a valid policy and positive arrival batch size")
    by_id = {r["uid"]: r for r in requests}
    if len(by_id) != len(requests):
        raise ValueError("Duplicate request UID")
    totals = Counter()
    for record in requests:
        if not record["prompt_token_ids"] or not record["generated_token_ids"]:
            raise ValueError(f"Empty recorded prompt or output: {record['uid']}")
        if record["parent"] is not None:
            parent = by_id[record["parent"]]
            prefix = parent["prompt_token_ids"] + parent["generated_token_ids"]
            if record["prompt_token_ids"][:len(prefix)] != prefix:
                raise ValueError(f"Broken parent token prefix: {record['uid']}")
            totals["verified_parent_prefixes"] += 1
    max_seq_len = max(len(r["prompt_token_ids"]) + len(r["generated_token_ids"]) for r in requests) + 1
    samples, batches = {}, []
    cycle = 0
    with numerical_boundaries(), tp_override(DistributedInfo(0, case["tp_size"])):
        manager = make_manager(case, mode, max_seq_len, window, half_life)
        tree, allocator = manager.tree, manager._allocator

        def observe_admissions(batch):
            for req in batch.prefill_reqs:
                if req.uid in samples:
                    continue
                record = requests[req.uid]
                parent = by_id.get(record["parent"])
                parent_end = 0 if parent is None else len(parent["prompt_token_ids"]) + len(parent["generated_token_ids"])
                hit = req.prefix_cached_len
                trace = req.cache_handle.match_trace
                matched = trace.matched_len if trace is not None else hit
                sample = dict(
                    uid=record["uid"], parent=record["parent"], depth=record["depth"],
                    dispatch_index=record["dispatch_index"], admission_index=len(samples),
                    admitted_iteration=cycle, input_tokens=len(record["prompt_token_ids"]),
                    tree_match=matched, recoverable=hit,
                    historical_recompute=max(0, parent_end - hit),
                    historical_kv_gap=max(0, parent_end - matched),
                    historical_state_gap=max(0, min(parent_end, matched) - min(parent_end, hit)),
                    historical_cacheable_kv_gap=max(0, max(0, (parent_end - 1) // 32 * 32) - matched),
                    prefill_tokens=len(record["prompt_token_ids"]) - hit,
                )
                assert sample["historical_kv_gap"] + sample["historical_state_gap"] == sample["historical_recompute"]
                for key in ("historical_kv_gap", "historical_state_gap", "historical_cacheable_kv_gap"):
                    totals[key] += sample[key]
                totals["historical_alignment_gap"] += sample["historical_kv_gap"] - sample["historical_cacheable_kv_gap"]
                totals["rebuilt_state_tokens_on_existing_kv"] += max(0, matched - hit)
                totals["input_tokens"] += sample["input_tokens"]
                totals["recoverable_tokens"] += hit
                totals["historical_recompute_tokens"] += sample["historical_recompute"]
                totals["prefill_tokens"] += sample["prefill_tokens"]
                totals["history_tokens"] += parent_end
                totals["history_hit_tokens"] += min(parent_end, hit)
                samples[req.uid] = sample

        scheduler = ReplayScheduler(manager, requests, case, batch_size, max_seq_len, observe_admissions)
        original_event = manager.evictor._record_lifecycle_event

        def record_event(event):
            event.update(iteration=cycle, completed=len(scheduler.finished))
            original_event(event)

        manager.evictor._record_lifecycle_event = record_event
        data = None
        forward_counts = Counter()
        min_free_frames = allocator.free_frames
        while (len(scheduler.finished) < len(requests) or data is not None
               or scheduler._deferred_req_frees):
            cycle += 1
            before_frames = frame_usage(allocator)
            before = Counter(e.kind.value for e in tree.iter_entries())
            admitted_before, completed_before = len(samples), len(scheduler.finished)
            evictions_before = len(manager._profile_lifecycle_events)
            had_inflight = data is not None
            data = scheduler.overlap_loop(data)
            min_free_frames = min(min_free_frames, allocator.free_frames)
            batch = None if data is None else data[0].batch
            if batch is not None:
                forward_counts[batch.phase] += 1
            if data is None and not had_inflight and len(scheduler.finished) < len(requests):
                raise RuntimeError(f"Production scheduler stalled at iteration {cycle}; "
                                   f"pending={len(scheduler.prefill_manager.pending_list)}, "
                                   f"running={len(scheduler.ar_manager.running_reqs)}, "
                                   f"free_frames={allocator.free_frames}")
            if (len(samples) != admitted_before or len(scheduler.finished) != completed_before
                    or len(manager._profile_lifecycle_events) != evictions_before
                    or (batch is not None and batch.prefill_count)):
                tree.check_integrity()
                allocator.check_integrity()
                fresh = list(samples.values())[admitted_before:]
                batches.append(dict(
                    batch=len(batches) + 1, iteration=cycle,
                    phase="drain" if batch is None else batch.phase,
                    completed=len(scheduler.finished), requests=len(fresh),
                    prefill_requests=0 if batch is None else batch.prefill_count,
                    input_tokens=sum(r["input_tokens"] for r in fresh),
                    recoverable_tokens=sum(r["recoverable"] for r in fresh),
                    checkpoint_before=before["checkpoint"], log_before=before["log"],
                    free_frames_before=before_frames["free"], frames_before=before_frames,
                    frames_after=frame_usage(allocator), free_frames=allocator.free_frames,
                    used_bytes_per_rank=(allocator.num_frames - allocator.free_frames) * allocator.frame_bytes,
                    **layout_snapshot(tree),
                ))
        manager.check_integrity()
        assert len(samples) == len(requests)
        assert scheduler.table_manager.available_size == case["max_running_requests"]
        assert all(scheduler.returned_tokens[i] == r["generated_token_ids"] for i, r in enumerate(requests))
        assert sum(step["tokens"] for step in scheduler.prefill_steps) == totals["prefill_tokens"]
        return dict(
            execution=EXECUTION, scheduling_policy=scheduler.config.scheduling_policy,
            prefill_budget=scheduler.prefill_budget, mixed_prefill_tokens=scheduler.config.mixed_prefill_tokens,
            arrival_batch_size=batch_size, overlap=True, numerical_values=False,
            mode=mode, hardware=case, trace_sha256=hashlib.sha256(history.read_bytes()).hexdigest(),
            requests=len(requests), totals=dict(totals),
            hit_rate=totals["recoverable_tokens"] / totals["input_tokens"],
            historical_hit_rate=totals["history_hit_tokens"] / max(1, totals["history_tokens"]),
            frame_bytes=allocator.frame_bytes, num_frames=allocator.num_frames,
            checkpoint_frames=allocator.payload_frames + allocator.conv_frames,
            log_frames=allocator.log_payload_frames + allocator.num_granules * allocator.conv_frames,
            decay_records_per_frame=allocator.decay_records_per_frame,
            materialize_log_count=allocator.materialize_log_count,
            min_free_frames=min(min_free_frames, manager.logstate_pool.min_free_frames),
            prefill_steps=scheduler.prefill_steps,
            forward_batches=dict(forward_counts),
            batches=batches, requests_detail=[samples[i] for i in range(len(requests))],
            analysis=manager.analyzer.profile_result(), evictions=dict(manager._profile_stats),
            eviction_events=manager._profile_lifecycle_events,
            final_layout=[dict(
                prefix_sha256=hashlib.sha256(torch.cat([p.key for p in tree.path(node)]).numpy().tobytes()).hexdigest(),
                begin=node.begin, end=node.depth, protected=node.is_protected,
                entries=[dict(begin=e.begin, end=e.end, kind=e.kind.value) for e in node.entries],
            ) for node in tree._iter_nodes() if not node.is_root()],
        )

def main():
    parser = argparse.ArgumentParser(description="Replay standard multiturn cache policies on CPU.")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", choices=("default", "online"), required=True)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Maximum new arrivals per production scheduler iteration")
    parser.add_argument("--min-sampling-len", type=int, default=8192)
    parser.add_argument("--hit-half-life", type=int, default=1024)
    args = parser.parse_args()
    result = replay(
        json.loads(args.case.read_text()),
        args.history,
        args.mode,
        args.batch_size,
        args.min_sampling_len,
        args.hit_half_life,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                mode=args.mode,
                requests=result["requests"],
                hit_rate=result["hit_rate"],
                historical_recompute_tokens=result["totals"]["historical_recompute_tokens"],
            )
        )
    )


if __name__ == "__main__":
    main()
