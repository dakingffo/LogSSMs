"""Step 8f: multi-req batch + mid-flight abort.

Drives the MTP scheduler with 5 concurrent requests (same suite as the e2e harness),
then injects AbortBackendMsg for one request mid-generation. Verifies:

  * the aborted request's stream stops (does not reach max_tokens);
  * the other requests continue and complete;
  * page integrity still holds after drain (no per-round page leak from the aborted
    request's outstanding verify-window pages).

The abort is injected a fixed number of overlap_loop iterations in, chosen so the
target request has already entered a verify round (so the abort lands mid-round and the
round's allocated window pages must be freed).

Run: /home/daking/.conda/envs/daking/bin/python tests/mtp/test_mtp_abort.py
"""
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
from logssms.message import AbortBackendMsg, UserMsg
from logssms.scheduler.config import SchedulerConfig
from logssms.distributed import DistributedInfo
from test_mtp_e2e import OfflineScheduler, make_config
from transformers import AutoTokenizer

MODEL = os.environ.get("QWEN3_NEXT_MODEL", "/home/daking/models/huggingface/Qwen3.5-0.8B")
PROMPTS = [
    ("q1", "The meaning of life is"),
    ("q2", "一次函数 y = kx + b,其中 k 和 b 分别是"),
    ("q3", "def quicksort(arr):\n    if len(arr) <= 1:"),
    ("q4", "The capital of France is"),
    ("q5", "碳达峰指的是"),
]
MAX_TOKENS = 24
ABORT_UID = 2  # q3
ABORT_ITER = 5  # inject after this many overlap_loop iterations


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    msgs = []
    for uid, (name, text) in enumerate(PROMPTS):
        ids = tokenizer(text, return_tensors="pt").input_ids[0].to(torch.int32)
        msgs.append(
            UserMsg(
                uid=uid,
                input_ids=ids,
                sampling_params=SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS),
            )
        )
    sched = OfflineScheduler(make_config(True), msgs)

    data = None
    injected = False
    idle = 0
    for it in range(5000):
        if it == ABORT_ITER and not injected:
            sched._pending.append(AbortBackendMsg(uid=ABORT_UID))
            injected = True
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

    # streams per uid
    names = [n for n, _ in PROMPTS]
    streams = {
        u: [
            tok
            for r in sched.results
            if r.uid == u
            for tok in (r.next_token if isinstance(r.next_token, list) else [r.next_token])
        ]
        for u in range(5)
    }
    done = {r.uid for r in sched.results if r.finished}

    # integrity
    integrity_ok, integrity_msg = True, "ok"
    try:
        sched.cache_manager.check_integrity()
    except Exception as e:  # noqa: BLE001
        integrity_ok, integrity_msg = False, str(e)

    sched.shutdown()

    print(f"aborted req: {names[ABORT_UID]} (uid {ABORT_UID})")
    for u in range(5):
        print(f"  {names[u]:5s}: n={len(streams[u]):3d}  finished={'yes' if u in done else 'NO'}  "
              f"{'' if u != ABORT_UID else '(aborted)'}")
    print(f"integrity: {'PASS' if integrity_ok else 'FAIL'}  {integrity_msg}")
    print(f"free_pages={len(sched.cache_manager.free_pages)}/{sched.cache_manager.num_pages}")

    aborted = len(streams[ABORT_UID])
    others_ok = all(u == ABORT_UID or (len(streams[u]) == MAX_TOKENS and u in done) for u in range(5))
    aborted_ok = aborted < MAX_TOKENS
    print(f"\nABORT TEST: {'PASS' if (aborted_ok and others_ok and integrity_ok) else 'FAIL'}")


if __name__ == "__main__":
    main()
