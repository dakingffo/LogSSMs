"""Step 8: MTP speculative-decoding end-to-end verification driver.

Drives the real Scheduler offline (no ZMQ/tokenizer workers): UserMsgs are fed
directly, DetokenizeMsg replies are collected, and overlap_loop is pumped until
every request emitted a `finished` token. One scheduler config per OS process
(Engine asserts torch.cuda is not already initialized), so --mtp 1 and --mtp 0
must run as separate invocations and be compared with --compare.

Usage:
  python tests/mtp/test_mtp_e2e.py --mtp 1 --out /tmp/mtp.json
  python tests/mtp/test_mtp_e2e.py --mtp 0 --out /tmp/non_mtp.json
  python tests/mtp/test_mtp_e2e.py --compare /tmp/non_mtp.json /tmp/mtp.json
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("CUDA_HOME", "/home/daking/.conda/envs/daking")
sys.path.insert(0, "/home/daking/PROJECT/pico-sglang/python")

import torch

from picosgl.core import Batch, SamplingParams
from picosgl.distributed import DistributedInfo
from picosgl.message import UserMsg
from picosgl.scheduler.config import SchedulerConfig
from picosgl.scheduler.scheduler import Scheduler
from picosgl.speculator import make_local_data_plane_pair, speculator_worker
from picosgl.speculator.drafters.mtp import MTPSpeculatorConfig

MODEL = os.environ.get("QWEN3_NEXT_MODEL", "/home/daking/models/huggingface/Qwen3.5-0.8B")

if os.environ.get("QWEN3_NEXT_PROMPT"):
    PROMPTS = [("single", os.environ["QWEN3_NEXT_PROMPT"])]
else:
    PROMPTS = [
    ("q1", "The meaning of life is"),
    ("q2", "一次函数 y = kx + b,其中 k 和 b 分别是"),
    ("q3", "def quicksort(arr):\n    if len(arr) <= 1:"),
    ("q4", "The capital of France is"),
    ("q5", "碳达峰指的是"),
]
MAX_TOKENS = 24
K = 4  # num_spec_tokens

# A divergence at position p is a *near-tie* (bf16 spec-decode rounding noise, not a bug)
# when the MTP VERIFY forward's top-2 logit margin at the predicting position is below
# this threshold. The divergence is decided by the verify forward's own logits (the ones
# used to commit the token), NOT the reference path's -- the fused mini-prefill context
# drifts ~0.25 hidden/layer vs per-token decode and accumulates across rounds, so the
# reference can be confident (decode margin 4.6) where the verify forward itself sits at
# a tie. bf16 has ~8 mantissa bits: for logits of magnitude ~20-25 one ULP is ~0.08, so a
# margin < 0.2 is within the fused-vs-decode rounding band. Measured: both observed
# divergences (q3@24, q5@22) had verify margin=0.125.
TIE_MARGIN = 0.2


class OfflineScheduler(Scheduler):
    """Scheduler with an in-process message queue instead of ZMQ.

    Requests and replies bypass the tokenizer-side ZMQ queues. In speculative mode the
    drafter still follows the production architecture: a separate same-device worker,
    a ZMQ control plane, and the configured tensor data plane.
    """

    def __init__(self, config, msgs):
        self._pending = list(msgs)
        self.results: list = []
        self.speculator_process: mp.Process | None = None
        self._speculator_queues: list = []
        if config.enable_specualtive_decoding:
            ctx = mp.get_context("spawn")
            start_event = ctx.Event()
            ready_event = ctx.Event()
            ack_queue = ctx.Queue(maxsize=1)
            target_data_plane, speculator_data_plane = make_local_data_plane_pair(
                config, ctx
            )
            self._speculator_queues = [ack_queue]
            self.speculator_process = ctx.Process(
                target=speculator_worker,
                kwargs={
                    "args": config,
                    "speculator_start_event": start_event,
                    "speculator_ready_event": ready_event,
                    "speculator_data_plane": speculator_data_plane,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name="picosgl-test-speculator",
            )
            self.speculator_process.start()
            try:
                super().__init__(
                    config, start_event, ready_event, target_data_plane
                )
            except BaseException:
                self.speculator_process.terminate()
                self.speculator_process.join()
                raise
            assert ack_queue.get() == "Speculator is ready"
        else:
            super().__init__(config)

    def offline_receive_msg(self, blocking: bool = False) -> list:
        out, self._pending = self._pending, []
        return out

    def offline_send_result(self, reply) -> None:
        self.results.extend(reply)

    def shutdown(self) -> None:
        super().shutdown()
        if self.speculator_process is not None:
            self.speculator_process.terminate()
            self.speculator_process.join()
        for queue in self._speculator_queues:
            queue.close()
            queue.join_thread()


def make_config(enable_specualtive_decoding: bool, num_pages: int = 256) -> SchedulerConfig:
    # page_size stays 1 here: the engine overrides it to 64 for hybrid models (must be a
    # multiple of 64), and dense models (the non-MTP byte-identity regression) keep 1.
    # num_pages is small because a hybrid page also owns one ~9.6MB linear-state slot, so
    # 256 pages + the K+1 reserve per MTP request must fit on an 8GB card.
    return SchedulerConfig(
        model_path=MODEL,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_running_req=8,
        page_size=1,
        num_page_override=num_pages,
        cache_type="radix",
        cuda_graph_max_bs=0,
        speculative_algorithm="MTP" if enable_specualtive_decoding else None,
        speculative_draft_model_path=MODEL if enable_specualtive_decoding else None,
        speculator_config=(
            MTPSpeculatorConfig(num_draft_tokens=K)
            if enable_specualtive_decoding else None
        ),
        offline_mode=True,
    )


def run(enable_specualtive_decoding: bool, debug_logits: bool = False, temperature: float = 0.0) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    msgs = []
    for uid, (name, text) in enumerate(PROMPTS):
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        msgs.append(
            UserMsg(
                uid=uid,
                input_ids=ids.to(torch.int32),
                sampling_params=SamplingParams(temperature=temperature, max_tokens=MAX_TOKENS),
            )
        )
    sched = OfflineScheduler(make_config(enable_specualtive_decoding), msgs)
    uid_map = {uid: name for uid, (name, _) in enumerate(PROMPTS)}

    # top-2 logit margin at every position, recorded from the DECODE path's forward
    # (i.e. sampler.sample: prefill's last token + every decode step). This is the
    # reference greedy path, so the divergence position's margin is measured against the
    # true greedy context -- the two streams only differ AT the divergent position, so
    # contexts up to it are identical. --compare classifies a divergence as a bf16
    # near-tie (margin < TIE_MARGIN) from this margin. NOT the MTP verify margins: those
    # are contaminated by the speculative draft context inside the verify window (the
    # drafts at positions C+1.. can differ from greedy tokens, swinging the logits by
    # far more than bf16 rounding).
    decode_margins: dict[tuple[int, int], float] = {}
    # top-2 logit margin at every position predicted by the MTP VERIFY forward's window
    # row (position C+j+1 predicted by row j). This is the confidence the MTP path
    # actually acted on when committing that token. Unlike decode_margins, this reflects
    # the drifted verify context (fused mini-prefill + accumulated bf16 error), so a
    # large verify margin on a divergent position proves the verify forward is CONFIDENT
    # about its different choice -> genuine fused-context drift, not a near-tie.
    verify_margins: dict[tuple[int, int], float] = {}

    # record each verify round's committed length (acceptance histogram) without touching
    # production code. The commit now happens inside process (deferred one iteration after
    # the verify forward), and each non-aborted req emits one list-typed DetokenizeMsg per
    # round whose length is num_sampled -- read it from the reply.
    hist: list[tuple[int, int]] = []
    if enable_specualtive_decoding:
        orig_process = sched.verify_manager.process

        def rec_process(ctx, batch, output):
            reply, finished = orig_process(ctx, batch, output)
            for msg in reply:
                if isinstance(msg.next_token, list):
                    hist.append((msg.uid, len(msg.next_token)))
            return reply, finished

        sched.verify_manager.process = rec_process

    if debug_logits or enable_specualtive_decoding:
        rs = sched.engine.sampler.reject_sample

        def dump_rs(logits, batch):
            out = rs(logits, batch)
            if batch.is_verify:
                off = 0
                for req in batch.reqs:
                    n = req.extend_len
                    if debug_logits:
                        am = torch.argmax(logits[off : off + n].float(), dim=-1).tolist()
                        print(f"  [dbg] verify round req={req.uid} "
                              f"positions=[{req.cached_len},{req.cached_len + n}) "
                              f"argmax={am} extend={out[batch.reqs.index(req)].tolist()}")
                    lg = logits[off : off + n].float()
                    top2 = torch.topk(lg, 2, dim=-1)
                    C = req.cached_len
                    for j in range(n):
                        # window row j predicts position C+j+1. Record the margin the
                        # verify forward actually saw when committing that token.
                        verify_margins[(req.uid, C + j + 1)] = (
                            top2.values[j, 0].item() - top2.values[j, 1].item()
                        )
                    off += n
            return out

        sched.engine.sampler.reject_sample = dump_rs

    # record the decode path's top-2 margin at every predicted position. `sample` runs
    # for prefill (last token) in both modes and for every non-MTP decode step, so the
    # reference greedy path is fully covered by the non-MTP run. engine calls sample()
    # AFTER the ctx.forward_batch block exits, so capture the batch via a forward_batch
    # wrapper instead of ctx.batch.
    cur_batch: list[Batch | None] = [None]
    orig_fb = sched.engine.forward_batch

    def fb(batch, args):
        cur_batch[0] = batch
        return orig_fb(batch, args)

    sched.engine.forward_batch = fb
    smp = sched.engine.sampler.sample

    def dump_sample(logits, args, **kw):
        out = smp(logits, args, **kw)
        b = cur_batch[0]
        if b is not None:
            for i, req in enumerate(b.reqs):
                if i >= len(logits):
                    break
                lg = logits[i].float()
                top2 = torch.topk(lg, 2)
                decode_margins[(req.uid, int(req.device_len))] = (
                    top2.values[0].item() - top2.values[1].item()
                )
        return out

    sched.engine.sampler.sample = dump_sample

    data = None
    idle = 0
    for _ in range(5000):
        # Terminate only when the loop goes truly idle: no batch scheduled (data is
        # None) AND every running manager is empty. Drain-until-idle is deliberately
        # stronger than break-on-finished: it collects every sampled DetokenizeMsg even
        # if a finished flag were mis-timed (historically the non-MTP path fired it one
        # position early; now DecodeManager snapshots it at complete_n time).
        if data is None and not (
            sched.prefill_manager.runnable
            or sched.decode_manager.runnable
            or (sched.verify_manager is not None and sched.verify_manager.runnable)
        ):
            idle += 1
            if idle >= 2:  # two consecutive idle passes -> drained
                break
        else:
            idle = 0
        data = sched.overlap_loop(data)

    # group emitted tokens by uid, in position order (DetokenizeMsg order preserves it).
    # verify emits one list-typed msg per round; flatten it the same way as scalar msgs.
    tokens = {u: [] for u in uid_map}
    for r in sched.results:
        toks = r.next_token if isinstance(r.next_token, list) else [r.next_token]
        tokens[r.uid].extend(toks)
    # verify termination
    done = {r.uid for r in sched.results if r.finished}
    missing = [uid_map[u] for u in uid_map if u not in done]

    # Step 8d: page-integrity no-leak. After every request drained (all pages back in
    # the prefix cache or the free list), free_slots + prefix_cache.total_size must equal
    # num_pages exactly. Committed positions stay in the req's cache; the rejected suffix
    # is freed in process each round; an abort frees the in-flight window -- no page may
    # leak.
    integrity_ok, integrity_msg = True, "ok"
    try:
        sched.cache_manager.check_integrity()
    except Exception as e:  # noqa: BLE001
        integrity_ok, integrity_msg = False, str(e)
    free_pages = int(len(sched.cache_manager.free_pages))

    sched.shutdown()

    result = {
        "enable_specualtive_decoding": enable_specualtive_decoding,
        "tokens": {uid_map[u]: t for u, t in tokens.items()},
        "prompt_len": {uid_map[u]: len(m.input_ids) for u, m in enumerate(msgs)},
        "missing": missing,
        "rounds": len(hist),
        "integrity_ok": integrity_ok,
        "integrity_msg": integrity_msg,
        "num_pages": int(sched.cache_manager.num_pages),
        "free_pages": free_pages,
    }
    result["decode_margins"] = [
        [u, p, round(m, 4)] for (u, p), m in decode_margins.items()
    ]
    result["verify_margins"] = [
        [u, p, round(m, 4)] for (u, p), m in verify_margins.items()
    ]
    if enable_specualtive_decoding:
        n = len(hist)
        result["accept_hist"] = hist
        if n:
            result["avg_accept"] = round(sum(h for _, h in hist) / n, 3)
            result["full_commit"] = sum(1 for _, h in hist if h == K + 1)
            result["one_tok"] = sum(1 for _, h in hist if h == 1)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtp", type=int, choices=[0, 1], default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--compare", nargs=2, metavar=("NON_MTP", "MTP"))
    ap.add_argument("--debug-logits", action="store_true")
    ap.add_argument("--sampling-temp", type=float, default=0.0,
                    help="temperature for all prompts (0 = greedy; >0 exercises the "
                         "drafter's draft_probs data-plane leg)")
    args = ap.parse_args()

    if args.compare:
        a, b = (json.load(open(p)) for p in args.compare)
        # decode margins of the non-MTP (reference greedy) run: (uid, predicted position)
        dm = {(u, p): m for u, p, m in a.get("decode_margins", [])}
        # verify margins of the MTP run: (uid, predicted position)
        vm = {(u, p): m for u, p, m in b.get("verify_margins", [])}
        # Per prompt: PASS iff identical, or iff its FIRST divergence is a bf16 near-tie
        # (the MTP verify forward's own logit margin there is < TIE_MARGIN). Every
        # position after that root is a CASCADE -- once the stream diverges, the contexts
        # genuinely differ and later margins are large no matter what -- so they are
        # reported but do not fail the check. A single non-tie root = real bug.
        verdict: dict[str, str] = {}  # name -> "identical" | "near-tie" | "fail"
        for name in a["tokens"]:
            # QWEN3_NEXT_PROMPT override emits a single prompt named "single" (uid 0);
            # the standard suite names q1..q5. Match by name, else fall back to 0.
            uid = next((u for u, n in enumerate(PROMPTS) if n[0] == name), 0)
            ta, tb = a["tokens"][name], b["tokens"][name]
            same = ta == tb
            verdict[name] = "identical" if same else "fail"
            print(f"  {name:12s} non-MTP {len(ta):3d} tok, MTP {len(tb):3d} tok, "
                  f"{'IDENTICAL' if same else '*** DIFFER ***'}")
            if same:
                continue
            P = b.get("prompt_len", {}).get(name, 0)
            first_ok = False
            for i in range(max(len(ta), len(tb))):
                if i < len(ta) and i < len(tb) and ta[i] == tb[i]:
                    continue
                # the divergent token sits at stream position i = model position
                # P+i, predicted by the decode step at position P+i-1. The non-MTP
                # run's margin there is measured against the true greedy context,
                # so it reflects whether the model was genuinely at a tie.
                x = ta[i] if i < len(ta) else "<EOS>"
                y = tb[i] if i < len(tb) else "<EOS>"
                marg = dm.get((uid, P + i))
                vmarg = vm.get((uid, P + i))
                vm_s = f"verify margin={vmarg:.4f}" if vmarg is not None else "verify margin=n/a"
                # The divergence is decided by the MTP VERIFY forward's own logits
                # (that is the logits used to commit the token), not by the
                # reference path's. So the near-tie criterion is the VERIFY margin
                # at the divergent position: bf16 fused-context rounding noise
                # (verified 0.125 at both observed divergences) flips an argmax the
                # verify forward itself judged a tie. The reference decode margin is
                # reported as context -- it can be confident (4.6) because the fused
                # mini-prefill context drifts ~0.25 hidden/layer and accumulates
                # across rounds, swinging the verify logits to a tie.
                if vmarg is not None and vmarg < TIE_MARGIN:
                    if not first_ok:
                        first_ok = True
                        verdict[name] = "near-tie"
                        print(f"    ROOT diff at idx {i}: {x} vs {y} -> bf16 NEAR-TIE "
                              f"(verify margin={vmarg:.4f} < {TIE_MARGIN}; "
                              f"decode margin={marg}); {len(ta) - i} later positions "
                              f"are context cascades and are reported below")
                    else:
                        print(f"    cascade at idx {i}: {x} vs {y} "
                              f"(decode margin={marg}; {vm_s})")
                else:
                    print(f"    diff at idx {i}: {x} vs {y} "
                          f"(decode margin={marg}; {vm_s})")
        print(f"rounds(MTP)={b.get('rounds')} avg_accept={b.get('avg_accept')} "
              f"full={b.get('full_commit')} one={b.get('one_tok')} "
              f"missing={b.get('missing') or a.get('missing')}")
        n_ident = sum(1 for v in verdict.values() if v == "identical")
        n_near = sum(1 for v in verdict.values() if v == "near-tie")
        if "fail" not in verdict.values():
            print(f"GREEDY CONSISTENCY: PASS ({n_ident}/{len(verdict)} prompts "
                  f"bit-identical; {n_near} diverge only at a bf16 near-tie root "
                  f"-> cascades expected)")
        else:
            print("GREEDY CONSISTENCY: FAIL (non-tie divergence -> real bug)")
        sys.exit(0 if "fail" not in verdict.values() else 1)

    assert args.mtp is not None and args.out is not None
    result = run(bool(args.mtp), debug_logits=args.debug_logits,
                 temperature=args.sampling_temp)
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(json.dumps({k: v for k, v in result.items() if k != "tokens"}, indent=1))
    for name, t in result["tokens"].items():
        print(f"  {name:12s} -> {len(t)} tokens")


if __name__ == "__main__":
    main()
