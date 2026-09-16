"""Cache-hit e2e: a hybrid_radix prefix hit must produce byte-identical output.

The whole point of the hybrid RadixTreeNode is that a second request sharing page-
aligned prefix tokens reuses the cached KV pages AND the cached per-page linear
states (borrowed via state_table) instead of recomputing them. This driver proves
that reusing them is CORRECT: a partial-hit run of a prompt produces exactly the
same stream as a cold run of that prompt.

Design (one scheduler per mode, compared by --compare; Engine forbids two
Schedulers per process):
  P   = tokens [0,150)  (3 pages: 0-63, 64-127, 128-149)
  P2  = tokens [0,170)  (shares P's pages [0,128))
  P6  = tokens [0,175)  (extends past P2's partial page 2)

  cold mode: run P, P2, P6 on an EMPTY cache -> cold_P, cold_P2, cold_P6
  hit  mode: run P2 first (populates pages [0,128) of the tree), then P and P6.
             P  matches [0,128) -> borrows 2 KV pages + 2 state slots, computes
             only [128,150) fresh. Same for P6 with [128,175) fresh.
  assert: hit_P == cold_P and hit_P6 == cold_P6.

If the borrowed states were stale or misaligned (the design claim being tested),
the hit runs would diverge from their cold baselines.

Usage:
  python tests/mtp/test_cache_hit.py --cold --out /tmp/cache_cold.json
  python tests/mtp/test_cache_hit.py --hit  --out /tmp/cache_hit.json
  python tests/mtp/test_cache_hit.py --compare /tmp/cache_cold.json /tmp/cache_hit.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("CUDA_HOME", "/home/daking/.conda/envs/daking")
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "python"))
sys.path.insert(0, str(_REPO))

import torch

from logssms.core import SamplingParams
from logssms.message import UserMsg
from test_mtp_e2e import OfflineScheduler, make_config
from transformers import AutoTokenizer

MODEL = os.environ.get("QWEN3_NEXT_MODEL", "/home/daking/models/huggingface/Qwen3.5-0.8B")
LEN_P, LEN_P2, LEN_P6 = 150, 170, 175
MAX_TOKENS = 16


def build_prompts():
    tok = AutoTokenizer.from_pretrained(MODEL)
    # a single long text; P / P2 / P6 are strict prefixes sharing pages [0,128)
    text = ("The theory of everything must explain both quantum mechanics and "
            "general relativity, which are the two pillars of modern physics. ")
    ids = tok(text * 12, return_tensors="pt").input_ids[0]
    assert len(ids) >= LEN_P6, f"text too short: {len(ids)}"
    return tok, {n: ids[:L].to(torch.int32) for n, L in
                 (("P", LEN_P), ("P2", LEN_P2), ("P6", LEN_P6))}


def run(mode: str, enable_specualtive_decoding: bool = True) -> dict:
    tok, prompts = build_prompts()

    def msg(name, uid):
        return UserMsg(
            uid=uid, input_ids=prompts[name],
            sampling_params=SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS),
        )

    if mode == "cold":
        msgs = [msg("P", 0), msg("P2", 1), msg("P6", 2)]
    else:  # hit: P2 populates first; P and P6 hit its pages [0,128)
        msgs = [msg("P2", 1)]

    sched = OfflineScheduler(make_config(enable_specualtive_decoding), msgs)

    # record the matched cached_len each prefill batch actually got (0 = cold, >0 = hit)
    max_cached: dict[int, int] = {}
    orig_snb = sched.prefill_manager.schedule_next_batch

    def snb(adder):
        b = orig_snb(adder)
        if b is not None:
            for req in b.reqs:
                max_cached[req.uid] = max(max_cached.get(req.uid, 0), req.cached_len)
        return b

    sched.prefill_manager.schedule_next_batch = snb

    data = None
    idle = 0
    appended = mode != "hit"
    for _ in range(5000):
        if mode == "hit" and not appended and any(
            r.uid == 1 and r.finished for r in sched.results
        ):
            # P2 fully drained -> its pages are in the tree; now feed the hitters
            sched._pending.append(msg("P", 0))
            sched._pending.append(msg("P6", 2))
            appended = True
        if data is None and not (
            sched.prefill_manager.runnable
            or sched.decode_manager.runnable
            or (sched.verify_manager is not None and sched.verify_manager.runnable)
        ):
            idle += 1
            if idle >= 2:
                break
        else:
            idle = 0
        data = sched.overlap_loop(data)

    streams = {}
    for r in sched.results:
        u = r.uid
        toks = r.next_token if isinstance(r.next_token, list) else [r.next_token]
        streams.setdefault(u, []).extend(toks)

    integrity_ok, integrity_msg = True, "ok"
    try:
        sched.cache_manager.check_integrity()
    except Exception as e:  # noqa: BLE001
        integrity_ok, integrity_msg = False, str(e)
    sched.shutdown()

    name_of = {0: "P", 1: "P2", 2: "P6"}
    return {
        "mode": mode,
        "enable_specualtive_decoding": enable_specualtive_decoding,
        "streams": {name_of[u]: t for u, t in streams.items()},
        "max_cached": {name_of[u]: c for u, c in max_cached.items()},
        "integrity_ok": integrity_ok,
        "integrity_msg": integrity_msg,
        "n_msgs": len(sched.results),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--hit", action="store_true")
    ap.add_argument("--mtp", type=int, choices=[0, 1], default=1)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--compare", nargs=2, metavar=("COLD", "HIT"))
    args = ap.parse_args()

    if args.compare:
        cold, hit = (json.load(open(p)) for p in args.compare)
        assert (
            cold["enable_specualtive_decoding"]
            == hit["enable_specualtive_decoding"]
        ), "modes must match"
        ok = True
        for name in ("P", "P6"):
            c = cold["streams"].get(name, [])
            h = hit["streams"].get(name, [])
            same = c == h
            ok &= same
            print(f"  {name:3s} cold {len(c):3d} tok, hit {len(h):3d} tok, "
                  f"{'IDENTICAL' if same else '*** DIFFER ***'}")
            if not same:
                print(f"    cold: {c}")
                print(f"    hit : {h}")
        # prove the hits actually fired: in hit mode P/P6 must have matched >= 1 page
        hit_p = hit.get("max_cached", {}).get("P", 0)
        hit_p6 = hit.get("max_cached", {}).get("P6", 0)
        ok &= hit_p > 0 and hit_p6 > 0
        print(f"  hit matched cached_len: P={hit_p} P6={hit_p6} "
              f"cold P={cold.get('max_cached', {}).get('P', '?')}")
        ok &= cold.get("integrity_ok", False)
        ok &= hit.get("integrity_ok", False)
        print(f"integrity: cold={cold.get('integrity_ok')} hit={hit.get('integrity_ok')}")
        print(f"CACHE HIT TEST (mtp={cold['enable_specualtive_decoding']}): "
              f"{'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)

    assert (args.cold or args.hit) and args.out
    result = run(
        "cold" if args.cold else "hit",
        enable_specualtive_decoding=bool(args.mtp),
    )
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
