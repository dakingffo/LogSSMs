from __future__ import annotations

import argparse
from contextlib import nullcontext
from collections import Counter
import hashlib
import json
import math
from statistics import mean
from pathlib import Path
from unittest.mock import patch

import torch

from logssms.distributed.info import DistributedInfo, tp_override
from experiment.utils.cpu_replay.diagnostics import ReplayDiagnostics
from experiment.utils.cpu_replay.scheduler import ReplayScheduler, VirtualClock, make_manager, numerical_boundaries
from datasets.parser import load_replay_history, replay_history_prefix


EXECUTION = "production-scheduler-v1"


def frame_usage(manager):
    if hasattr(manager, "logstate_pool"):
        allocator = manager._allocator
        if hasattr(allocator, "frame_usage_summary"):
            usage = allocator.frame_usage_summary()
            assert sum(usage[k] for k in ("log", "checkpoint", "work", "kv", "free")) == allocator.num_frames
            return usage
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
    pool = manager.pool
    ckpt_blocks = len(pool.state_blocks) * manager.plan.layout.state_blocks
    free = len(pool.allocator.free)
    kv_blocks = manager.plan.num_blocks - free - ckpt_blocks
    return dict(log=0, checkpoint=ckpt_blocks, work=0, kv=kv_blocks, free=free)


def _free_frames(manager) -> int:
    if hasattr(manager, "_allocator"):
        return manager._allocator.free_frames
    return len(manager.pool.allocator.free)


def _pool_frame_bytes(manager) -> int:
    if hasattr(manager, "_allocator"):
        return manager._allocator.frame_bytes
    return manager.plan.layout.block_bytes(manager.pool.dtype.itemsize)


def _pool_num_frames(manager) -> int:
    if hasattr(manager, "_allocator"):
        return manager._allocator.num_frames
    return manager.plan.num_blocks


def _pool_min_free_frames(manager) -> int:
    if hasattr(manager, "logstate_pool"):
        return manager.logstate_pool.min_free_frames
    return len(manager.pool.allocator.free)


def _marconi_layout_snapshot(tree):
    ckpts = sum(node.state is not None for node in tree.nodes)
    spans = Counter(str(node.end - node.begin) for node in tree.nodes if node.state is not None)
    return dict(checkpoint=ckpts, log=0, checkpoint_spans=dict(spans), log_run_spans={})


def _marconi_final_layout(tree):
    return [
        dict(
            prefix_sha256=hashlib.sha256(torch.tensor(node.tokens, dtype=torch.int32).numpy().tobytes()).hexdigest(),
            begin=node.begin,
            end=node.end,
            protected=node.locks > 0,
            entries=(
                [dict(begin=node.begin, end=node.end, kind="checkpoint")]
                if node.state is not None else []
            ),
        )
        for node in sorted(tree.nodes, key=lambda n: (n.begin, n.end))
    ]


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


def replay(case, history, mode, batch_size=8, window=8192, half_life=1024, *, enable_hit_materialization=False,
           block_size=128, bootstrap_multiplier=5,
           tree_concurrency=None, record_layout=True, record_state_usage=False,
           trace_mode="eager", clock_step_ms=1., trace_sessions=None, trace_max_context=None,
           progress_every=0, heartbeat_seconds=0., timing_path=None):
    torch.set_num_threads(1)
    if trace_mode not in ("eager", "virtual_clock"):
        raise ValueError(f"Unknown trace mode: {trace_mode}")
    clock = VirtualClock(clock_step_ms) if trace_mode == "virtual_clock" else None
    trace = load_replay_history(history, session_limit=trace_sessions, max_context=trace_max_context)
    requests = trace.requests
    if not requests or mode not in ("default", "online", "marconi") or batch_size < 1:
        raise ValueError("Replay requires requests, a valid policy and positive arrival batch size")
    tree_count = len({record["tree_id"] for record in requests})
    max_tree_concurrency = tree_count if clock else min(tree_count, case["max_running_requests"])
    if tree_concurrency is not None and not 1 <= tree_concurrency <= max_tree_concurrency:
        raise ValueError("Tree concurrency must fit the history and the server request limit")
    by_id = {r["uid"]: r for r in requests}
    if len(by_id) != len(requests):
        raise ValueError("Duplicate request UID")
    output_cap = case["output_cap"]
    eos_token_id = case["eos_token_id"]
    if output_cap < 1:
        raise ValueError("Replay requires a positive output cap")
    totals = Counter()
    history_lengths = {}
    required_seq_len = 0
    for record in requests:
        if not record["prompt_token_ids"] or not record["generated_token_ids"]:
            raise ValueError(f"Empty recorded prompt or output: {record['uid']}")
        if clock and ("t" not in record or not math.isfinite(record["t"]) or record["t"] < 0):
            raise ValueError(f"Expected a non-negative arrival time in seconds: {record['uid']}")
        history_len = 0
        if record["parent"] is not None:
            parent = by_id[record["parent"]]
            if parent["tree_id"] != record["tree_id"]:
                raise ValueError(f"Cross-tree parent: {record['uid']}")
            parent_end = len(parent["prompt_token_ids"]) + len(parent["generated_token_ids"])
            history_len = replay_history_prefix(record, parent)
            if history_len == parent_end:
                totals["verified_parent_prefixes"] += 1
            elif clock is None:
                raise ValueError(f"Broken parent token prefix: {record['uid']}")
        history_lengths[record["uid"]] = history_len
        output = record["generated_token_ids"]
        cap = record.get("output_cap", output_cap)
        ignore_eos = record.get("ignore_eos", False)
        if (len(output) > cap or (ignore_eos and len(output) != cap)
                or (not ignore_eos and (eos_token_id in output[:-1]
                    or (len(output) < cap and output[-1] != eos_token_id)))):
            raise ValueError(f"Recorded output does not end at EOS or the request cap: {record['uid']}")
        required_seq_len = max(required_seq_len, len(record["prompt_token_ids"]) + cap)
    max_seq_len = case.get("max_seq_len", required_seq_len)
    if max_seq_len < required_seq_len:
        raise ValueError("The context limit must fit all prompts at their requested output cap")
    samples, batches = {}, []
    cycle = 0
    time_override = (patch("logssms.cache.logstate.prefix_cache.time", clock)
                     if clock is not None and mode != "marconi" else nullcontext())
    diagnostics = ReplayDiagnostics(heartbeat_seconds, timing_path) if heartbeat_seconds or timing_path else None
    with (time_override, numerical_boundaries(mode), tp_override(DistributedInfo(0, case["tp_size"])),
          diagnostics if diagnostics is not None else nullcontext()):
        manager = make_manager(
            case, mode, max_seq_len, window, half_life,
            enable_hit_materialization=enable_hit_materialization,
            block_size=block_size, bootstrap_multiplier=bootstrap_multiplier,
        )
        pool = getattr(manager, "logstate_pool", getattr(manager, "pool", None))
        allocator = getattr(manager, "_allocator", pool.allocator)
        tree = getattr(manager, "tree", manager.prefix_cache)
        granularity = getattr(manager, "granularity", manager.page_size)

        def observe_admissions(batch):
            for req in batch.prefill_reqs:
                if req.uid in samples:
                    continue
                record = requests[req.uid]
                parent_end = history_lengths[record["uid"]]
                hit = req.prefix_cached_len
                trace_info = getattr(req.cache_handle, "match_trace", None)
                matched = trace_info.matched_len if trace_info is not None else hit
                sample = dict(
                    uid=record["uid"], parent=record["parent"], depth=record["depth"],
                    dispatch_index=record["dispatch_index"], admission_index=len(samples),
                    admitted_iteration=cycle, input_tokens=len(record["prompt_token_ids"]),
                    tree_match=matched, recoverable=hit,
                    output_cap=req.output_len,
                    history_tokens=parent_end, history_hit_tokens=min(parent_end, hit),
                    historical_recompute=max(0, parent_end - hit),
                    historical_kv_gap=max(0, parent_end - matched),
                    historical_state_gap=max(0, min(parent_end, matched) - min(parent_end, hit)),
                    historical_cacheable_kv_gap=max(0, max(0, (parent_end - 1) // granularity * granularity) - matched),
                    prefill_tokens=len(record["prompt_token_ids"]) - hit,
                )
                if clock is not None:
                    sample.update(trace_time_s=record["t"], arrival_time_s=scheduler.arrival_times[req.uid],
                                  admission_time_s=clock.now / 1_000_000_000)
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

        scheduler = ReplayScheduler(
            manager, requests, case, batch_size, max_seq_len, observe_admissions,
            tree_concurrency=tree_concurrency, clock=clock,
        )
        if diagnostics is not None:
            diagnostics.attach(scheduler, manager)
        evictor = getattr(manager, "evictor", None)
        if evictor is not None:
            original_event = evictor._record_lifecycle_event

            def record_event(event):
                event.update(iteration=cycle, completed=len(scheduler.finished))
                if clock is not None:
                    event["time_s"] = clock.now / 1_000_000_000
                original_event(event)

            evictor._record_lifecycle_event = record_event

        state_usage = []

        def sample_state_usage():
            frames = frame_usage(manager)
            sample = dict(completed=len(scheduler.finished),
                          log_frames=frames["log"], checkpoint_frames=frames["checkpoint"])
            if state_usage and state_usage[-1]["completed"] == sample["completed"]:
                state_usage[-1] = sample
            else:
                state_usage.append(sample)

        if record_state_usage:
            sample_state_usage()

        data = None
        forward_counts = Counter()
        min_free_frames = _free_frames(manager)
        frame_bytes = _pool_frame_bytes(manager)
        num_frames = _pool_num_frames(manager)
        next_progress = progress_every
        while (len(scheduler.finished) < len(requests) or data is not None
               or scheduler._deferred_req_frees):
            cycle += 1
            if diagnostics is not None:
                diagnostics.iteration = cycle
            if record_layout:
                before_frames = frame_usage(manager)
                before = (manager.layout_snapshot() if hasattr(manager, "layout_snapshot")
                          else _marconi_layout_snapshot(tree) if mode == "marconi"
                          else Counter(e.kind.value for e in tree.iter_entries()))
                admitted_before, completed_before = len(samples), len(scheduler.finished)
                evictions_before = len(manager._profile_lifecycle_events)
            had_inflight = data is not None
            data = scheduler.overlap_loop(data)
            min_free_frames = min(min_free_frames, _free_frames(manager))
            completed = len(scheduler.finished)
            if progress_every and completed >= next_progress:
                hit = totals["recoverable_tokens"] / max(1, totals["input_tokens"])
                history_hit = totals["history_hit_tokens"] / max(1, totals["history_tokens"])
                print(f"{completed}/{len(requests)} completed, {len(samples)} admitted, "
                      f"hit={hit:.4%}, historical={history_hit:.4%}, "
                      f"prefill={totals['prefill_tokens']:,}, "
                      f"historical_recompute={totals['historical_recompute_tokens']:,}", flush=True)
                next_progress = (completed // progress_every + 1) * progress_every
            if record_state_usage and len(scheduler.finished) != state_usage[-1]["completed"]:
                sample_state_usage()
            batch = None if data is None else data[0].batch
            if batch is not None:
                forward_counts[batch.phase] += 1
            if record_layout and (
                len(samples) != admitted_before or len(scheduler.finished) != completed_before
                or len(manager._profile_lifecycle_events) != evictions_before
                or (batch is not None and batch.prefill_count)
            ):
                if mode == "marconi":
                    manager.check_integrity()
                elif hasattr(tree, "check_integrity") and hasattr(tree, "iter_entries"):
                    tree.check_integrity()
                    allocator.check_integrity()
                else:
                    allocator.check_integrity()
                fresh = list(samples.values())[admitted_before:]
                snap = (manager.layout_snapshot() if hasattr(manager, "layout_snapshot")
                        else _marconi_layout_snapshot(tree) if mode == "marconi"
                        else layout_snapshot(tree))
                after_frames = frame_usage(manager)
                free_now = _free_frames(manager)
                batches.append(dict(
                    batch=len(batches) + 1, iteration=cycle,
                    phase="drain" if batch is None else batch.phase,
                    completed=len(scheduler.finished), requests=len(fresh),
                    prefill_requests=0 if batch is None else batch.prefill_count,
                    input_tokens=sum(r["input_tokens"] for r in fresh),
                    recoverable_tokens=sum(r["recoverable"] for r in fresh),
                    checkpoint_before=before["checkpoint"], log_before=before["log"],
                    free_frames_before=before_frames["free"], frames_before=before_frames,
                    frames_after=after_frames, free_frames=free_now,
                    used_bytes_per_rank=(num_frames - free_now) * frame_bytes,
                    **snap,
                ))
                if clock is not None:
                    batches[-1]["time_s"] = clock.now / 1_000_000_000
            if (data is None and not had_inflight and len(scheduler.finished) < len(requests)
                    and not scheduler.client_has_pending_completions):
                if scheduler.advance_to_next_arrival():
                    continue
                raise RuntimeError(f"Production scheduler stalled at iteration {cycle}; "
                                   f"pending={len(scheduler.prefill_manager.pending_list)}, "
                                   f"running={len(scheduler.ar_manager.running_reqs)}, "
                                   f"free_frames={_free_frames(manager)}")
            if clock is not None:
                clock.now += clock.step
        if record_state_usage:
            sample_state_usage()
        manager.check_integrity()
        assert len(samples) == len(requests)
        assert scheduler.table_manager.available_size == case["max_running_requests"]
        assert all(scheduler.returned_tokens[i] == r["generated_token_ids"] for i, r in enumerate(requests))
        assert sum(step["tokens"] for step in scheduler.prefill_steps) == totals["prefill_tokens"]
        if mode == "marconi":
            m_stats = dict(manager.tree.stats)
            eviction_summary = {
                **m_stats,
                "eviction_rounds": m_stats.get("leaf_evictions", 0) + m_stats.get("intermediate_evictions", 0),
            }
            extra_fields = dict(
                enable_hit_materialization=False,
                checkpoint_frames=manager.plan.layout.state_blocks,
                log_frames=0,
                decay_records_per_frame=0,
                materialize_log_count=0,
                marconi_alpha=manager.tree.alpha,
                marconi_tune_results=manager.tune_results,
            )
        else:
            eviction_summary = (manager.eviction_stats() if hasattr(manager, "eviction_stats")
                                else dict(manager._profile_stats))
            extra_fields = dict(
                enable_hit_materialization=enable_hit_materialization,
                checkpoint_frames=allocator.payload_frames + allocator.conv_frames,
                log_frames=allocator.log_payload_frames + allocator.num_granules * allocator.conv_frames,
                decay_records_per_frame=allocator.decay_records_per_frame,
                materialize_log_count=allocator.materialize_log_count,
            )
        return dict(
            execution=EXECUTION, trace_mode=trace_mode,
            clock_step_ms=clock_step_ms if clock else None, trace_sessions=trace_sessions,
            trace_max_context=trace_max_context,
            scheduling_policy=scheduler.config.scheduling_policy,
            prefill_budget=scheduler.prefill_budget, mixed_prefill_tokens=scheduler.config.mixed_prefill_tokens,
            arrival_batch_size=batch_size, overlap=True, numerical_values=False,
            mode=mode, hardware=case, trace_sha256=trace.sha256(),
            requests=len(requests), totals=dict(totals),
            tree_count=tree_count, tree_concurrency=tree_concurrency,
            peak_tree_concurrency=scheduler.peak_tree_concurrency,
            granularity=granularity,
            request_hit_rate=mean(
                sample["history_hit_tokens"] / sample["history_tokens"]
                for sample in samples.values() if sample["history_tokens"]
            ) if totals["history_tokens"] else None,
            hit_rate=totals["recoverable_tokens"] / totals["input_tokens"],
            historical_hit_rate=totals["history_hit_tokens"] / max(1, totals["history_tokens"]),
            frame_bytes=frame_bytes, num_frames=num_frames,
            min_free_frames=min(min_free_frames, _pool_min_free_frames(manager)),
            prefill_steps=scheduler.prefill_steps, state_usage=state_usage,
            forward_batches=dict(forward_counts),
            batches=batches, requests_detail=[samples[i] for i in range(len(requests))],
            analysis=(manager.analyzer.profile_result() if hasattr(manager, "analyzer") else None),
            evictions=eviction_summary,
            eviction_events=manager._profile_lifecycle_events,
            final_layout=(manager.final_layout_snapshot() if hasattr(manager, "final_layout_snapshot")
                          else _marconi_final_layout(tree) if mode == "marconi"
                          else [dict(
                prefix_sha256=hashlib.sha256(torch.cat([p.key for p in tree.path(node)]).numpy().tobytes()).hexdigest(),
                begin=node.begin, end=node.depth, protected=node.is_protected,
                entries=[dict(begin=e.begin, end=e.end, kind=e.kind.value) for e in node.entries],
            ) for node in tree._iter_nodes() if not node.is_root()]),
            **extra_fields,
        )

def main(*, extra_report=None):
    parser = argparse.ArgumentParser(description="Replay multiturn cache policies on CPU.")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", choices=("default", "online", "marconi"), required=True)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Maximum new arrivals per production scheduler iteration")
    parser.add_argument("--min-sampling-len", type=int, default=8192)
    parser.add_argument("--hit-half-life", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--bootstrap-multiplier", type=int, default=5)
    parser.add_argument("--tree-concurrency", type=int)
    parser.add_argument("--trace-mode", choices=("eager", "virtual_clock"), default="eager")
    parser.add_argument("--clock-step-ms", type=float, default=1.)
    parser.add_argument("--trace-sessions", type=int, help="Read the first N eligible Agent trace sessions")
    parser.add_argument("--trace-max-context", type=int, help="Filter whole Agent sessions by peak input + output")
    parser.add_argument("--progress-every", type=int, default=0, help="Report progress every N completions")
    parser.add_argument("--heartbeat-seconds", type=float, default=0.,
                        help="Log scheduler progress at this wall-clock interval")
    parser.add_argument("--timings", type=Path, help="Write per-call admission and eviction planning timings")
    parser.add_argument("--hit-materialization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-layout-snapshots", action="store_true")
    parser.add_argument("--record-state-usage", action="store_true")
    args = parser.parse_args()
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if not math.isfinite(args.heartbeat_seconds) or args.heartbeat_seconds < 0.:
        parser.error("--heartbeat-seconds must be finite and non-negative")
    result = replay(
        json.loads(args.case.read_text()),
        args.history,
        args.mode,
        args.batch_size,
        args.min_sampling_len,
        args.hit_half_life,
        tree_concurrency=args.tree_concurrency,
        trace_mode=args.trace_mode,
        clock_step_ms=args.clock_step_ms,
        trace_sessions=args.trace_sessions,
        trace_max_context=args.trace_max_context,
        progress_every=args.progress_every,
        heartbeat_seconds=args.heartbeat_seconds,
        timing_path=args.timings,
        record_layout=not args.skip_layout_snapshots,
        record_state_usage=args.record_state_usage,
        enable_hit_materialization=args.hit_materialization,
        block_size=args.block_size,
        bootstrap_multiplier=args.bootstrap_multiplier,
    )
    if extra_report is not None:
        result.update(extra_report)
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
