"""Replay actual token histories to compare CPU cache maintenance in isolation.

--revision reads the selected cache sources directly from Git in this process;
it never checks out or changes the worktree. The pool microbenchmark measures
metadata and retirement only, not reconstruction arithmetic or serving TTFT.
"""

import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

revision_args = argparse.ArgumentParser(add_help=False)
revision_args.add_argument("--revision")
revision_args.add_argument("--source-root", type=Path)
source_args = revision_args.parse_known_args()[0]
revision = source_args.revision
if source_args.source_root is not None:
    assert revision is None, "Choose either --revision or --source-root"
    sys.path.insert(0, str(source_args.source_root.resolve()))
source_paths = [
    "python/logssms/cache/logstate/manager.py",
    "python/logssms/cache/logstate/evictor.py",
    "python/logssms/cache/logstate/online_analysis_scheduler.py",
    "python/logssms/cache/logstate/recycler.py",
    "python/logssms/cache/logstate/prefix_cache.py",
    "python/logssms/cache/logstate/frame_pool.py",
    "python/logssms/kernel/linear_attention/logstate.py",
    "python/logssms/utils/torch_utils.py",
]
source_blobs = {
    path: subprocess.check_output(["git", "show", f"{revision}:{path}"])
    if revision else (source_args.source_root / path.removeprefix("python/")
                     if source_args.source_root else Path(path)).read_bytes()
    for path in source_paths
}
if revision:
    class RevisionLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path=None, target=None):
            filename = "python/" + fullname.replace(".", "/") + ".py"
            if filename in source_blobs:
                return importlib.util.spec_from_loader(fullname, self, origin=filename)

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            filename = module.__spec__.origin
            module.__file__ = str(Path(filename).resolve())
            exec(compile(source_blobs[filename], filename, "exec"), module.__dict__)

    sys.meta_path.insert(0, RevisionLoader())

from logssms.cache.logstate.manager import (
    LogStateCacheManager,
)
from logssms.cache.logstate.online_analysis_scheduler import (
    OnlineAnalysisScheduler,
)
from logssms.cache.logstate.prefix_cache import LogRadixTree, LogStateEntry, LogStateKind
from logssms.cache.logstate.frame_pool import MHALogStatePool


class TreeManager(LogStateCacheManager):
    @property
    def device(self):
        return torch.device("cpu")

    @property
    def logstate_pool(self):
        return self.cpu_pool

    def _commit_materialize(self, plans):
        for plan in plans:
            entries = self.tree.entries_for_slot(plan.log_slot)
            assert len(entries) == 1
            old_entry = entries[0]
            owner = self.tree.entry_owner(old_entry)
            assert self.tree.detach_entry(plan.log_slot) == old_entry
            self.tree.attach_entry(owner, LogStateEntry(
                slot=plan.log_slot, kind=LogStateKind.CHECKPOINT,
                begin=plan.begin, end=plan.end, span_pages=1,
            ))
        self.cpu_pool.materialize_slots([plan.log_slot for plan in plans])


def run(history, batch_size):
    tree = LogRadixTree(64, torch.device("cpu"), granularity=32)
    analyzer = OnlineAnalysisScheduler(64, 32)
    manager = object.__new__(TreeManager)
    manager.prefix_cache, manager.page_size = tree, 64
    manager._profile_enabled = False
    manager._lazy_free_states = None
    manager.cpu_pool = SimpleNamespace(materialize=lambda _: None,
                                      materialize_slots=lambda _: None,
                                      retire_slots=lambda _: None,
                                      release_slots=lambda _: None)
    manager._return_slots = lambda *_: None
    samples = []
    next_slot = 0
    matches = []
    sessions = history["sessions"]
    for turn in range(len(sessions[0]["turns"])):
        for begin in range(0, len(sessions), batch_size):
            records = [s["turns"][turn] for s in sessions[begin:begin + batch_size]]
            inputs = [torch.tensor(r["prompt_token_ids"], dtype=torch.int32) for r in records]
            matches.extend(tree.match_prefix(v).cuda_handle.cached_len for v in inputs)
            tic = time.perf_counter_ns()
            analyzer._batch_branch_points(inputs)
            branch_ms = (time.perf_counter_ns() - tic) / 1e6
            tic = time.perf_counter_ns()
            plan = analyzer.analyze(tree, inputs)
            analysis_ms = (time.perf_counter_ns() - tic) / 1e6
            tic = time.perf_counter_ns()
            manager._commit_materialize(plan.materialize)
            released, _, skips = manager._commit_coalesce(plan.coalesce)
            commit_ms = (time.perf_counter_ns() - tic) / 1e6
            assert not skips
            samples.append(dict(turn=turn + 1, batch_size=len(records), branch_ms=branch_ms,
                                analysis_ms=analysis_ms, commit_ms=commit_ms,
                                materialized=len(plan.materialize), released=released))
            for record in records:
                values = record["prompt_token_ids"] + record["generated_token_ids"]
                end = (len(values) - 1) // 64 * 64
                values = torch.tensor(values[:end], dtype=torch.int32)
                start = tree.match_prefix(values).cuda_handle.cached_len
                entries = []
                for pos in range(start, end, 64):
                    entries.append(LogStateEntry(next_slot, LogStateKind.LOG, pos, pos + 64))
                    next_slot += 1
                tree.insert(values, torch.arange(end, dtype=torch.int32), tuple(entries), cache_len=end)
    tree.check_integrity()
    layout = sorted((e.kind.name, e.begin, e.end, e.span_pages) for e in tree.iter_entries())
    return dict(samples=samples, matches=matches, layout=layout)


def pool_benchmark(device, slots, repeats):
    pool = MHALogStatePool(
        num_frames=2048, page_size=64, granularity=32,
        num_attention_layers=2, num_kv_heads=1, head_dim=32,
        num_linear_layers=2, num_key_heads=2, num_value_heads=2,
        head_k_dim=32, head_v_dim=32, conv_dim=192, state_len=3,
        device=torch.device(device), dtype=torch.float16,
    )
    samples = []
    for iteration in range(repeats + 5):
        if device == "cuda":
            torch.cuda.synchronize()
        tic = time.perf_counter_ns()
        allocated = pool.reserve(logs=slots)
        assert allocated is not None
        logs = allocated[1]
        reserved = time.perf_counter_ns()
        if not hasattr(pool, "materialize_slots"):
            for slot in logs:
                pool.materialize(slot)
        else:
            pool.materialize_slots(logs)
        retired = time.perf_counter_ns()
        if device == "cuda":
            torch.cuda.synchronize()
        ready = time.perf_counter_ns()
        pool.poll_reclamation()
        reclaimed = time.perf_counter_ns()
        pool.release_slots(logs)
        released = time.perf_counter_ns()
        if device == "cuda":
            torch.cuda.synchronize()
        finished = time.perf_counter_ns()
        pool.allocator.check_integrity()
        assert not pool.allocator.allocated_slots
        if iteration >= 5:
            samples.append(dict(
                reserve_ms=(reserved - tic) / 1e6,
                retire_ms=(retired - reserved) / 1e6,
                wait_ms=(ready - retired) / 1e6,
                reclaim_ms=(reclaimed - ready) / 1e6,
                release_ms=(released - reclaimed) / 1e6,
                total_ms=(finished - tic) / 1e6,
            ))
    return dict(device=device, slots=slots, repeats=repeats, samples=samples,
                median_ms={key: statistics.median(s[key] for s in samples) for key in samples[0]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--revision")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pool-device", choices=("cpu", "cuda"))
    parser.add_argument("--pool-slots", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(1)
    raw = args.history.read_bytes()
    history = json.loads(raw)
    for session in history["sessions"]:
        for previous, current in zip(session["turns"], session["turns"][1:]):
            prior = previous["prompt_token_ids"] + previous["generated_token_ids"]
            assert current["prompt_token_ids"][:len(prior)] == prior
    run(history, args.batch_size)  # untimed warmup
    runs = [run(history, args.batch_size) for _ in range(args.repeats)]
    assert all(r["matches"] == runs[0]["matches"] and r["layout"] == runs[0]["layout"] for r in runs)
    result = dict(kind="CPU maintenance replay; not a serving benchmark", history_path=str(args.history.resolve()), history_sha256=hashlib.sha256(raw).hexdigest(),
                  source_sha256={p: hashlib.sha256(source_blobs[p]).hexdigest() for p in source_paths},
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
                  revision=args.revision, repeats=args.repeats, batch_size=args.batch_size, runs=runs)
    result["median_totals_ms"] = {
        key: statistics.median(sum(s[key] for s in r["samples"]) for r in runs)
        for key in ["branch_ms", "analysis_ms", "commit_ms"]
    }
    if args.pool_device:
        result["pool"] = pool_benchmark(args.pool_device, args.pool_slots, 25)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["median_totals_ms"]))
    if args.pool_device:
        print(json.dumps(result["pool"]["median_ms"]))


if __name__ == "__main__":
    main()
