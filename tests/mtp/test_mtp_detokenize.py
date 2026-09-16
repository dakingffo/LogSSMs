"""DetokenizeManager int/list dispatch test.

DetokenizeMsg.next_token is `int` (non-MTP decode: one token per round) or
`list[int]` (MTP verify: a round's committed tokens in one msg). The worker must
append scalars / extend lists so that the concatenated incremental_outputs always
reconstruct the exact decoded text, and a trailing EOS on a finished msg must never
be user-visible (the stream stops before it).

Background: before the int/list dispatch existed, MTP emitted one scalar msg per
committed token; a batch carrying several msgs of one uid re-emitted the previous
msgs' deltas (the chat stutter, `你好！！很高兴见到你你。`). Batching the whole
round into one list per req per round removes that hazard; this test pins the new
contract: int and list shapes over the same tokens must join to identical text.

Usage: python tests/mtp/test_mtp_detokenize.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "python"))
sys.path.insert(0, str(_REPO))

from transformers import AutoTokenizer

from logssms.message import DetokenizeMsg
from logssms.tokenizer.tokenizer import DetokenizeManager

MODEL = "/home/daking/models/huggingface/Qwen3.5-0.8B"
EOS = AutoTokenizer.from_pretrained(MODEL).eos_token_id

TEXTS = [
    "你好！很高兴见到你。有什么我可以帮助你的吗？",
    "The quick brown fox jumps over the lazy dog. Hello world!",
    "一次函数 y = kx + b，其中 k 和 b 分别是斜率与截距。",
]


def token_stream(tok, text: str, chunk: int, eos_at_end: bool, uid: int = 0) -> list[DetokenizeMsg]:
    """One uid's DetokenizeMsg sequence for `text`.

    chunk == 1 -> the non-MTP shape: one scalar int msg per token, the final token
                  being EOS (finished=True), which the worker filters out.
    chunk  > 1 -> the MTP shape: one list msg per round of `chunk` committed tokens;
                  the final round carries a trailing EOS with finished=True
                  (eos_at_end, the EOS-stop path) or just finished=True with no EOS
                  (a max_tokens stop).
    """
    ids = tok(text).input_ids
    msgs: list[DetokenizeMsg] = []
    pos = 0
    while pos < len(ids):
        round_ids = [int(x) for x in ids[pos : pos + chunk]]
        pos += len(round_ids)
        last = pos >= len(ids)
        if chunk == 1:
            msgs.append(DetokenizeMsg(uid=uid, next_token=int(round_ids[0]), finished=last))
            # a trailing EOS is a separate follow-up msg (the real non-MTP flow emits the
            # last real token, and the finish is signalled by the sampled EOS, if any)
            if last and eos_at_end:
                msgs.append(DetokenizeMsg(uid=uid, next_token=int(EOS), finished=True))
        else:
            tok_val: int | list[int] = round_ids + ([EOS] if last and eos_at_end else [])
            msgs.append(DetokenizeMsg(uid=uid, next_token=tok_val, finished=last))
    return msgs


def join_stream(tok, msgs: list[DetokenizeMsg]) -> str:
    """Feed the msgs one per detokenize call; return the concatenated increments."""
    mgr = DetokenizeManager(tok)
    parts: list[str] = []
    for msg in msgs:
        parts.extend(mgr.detokenize([msg]))
    return "".join(parts)


def join_streams(tok, uid_msgs: dict[int, list[DetokenizeMsg]]) -> dict[int, str]:
    """Feed several uids' msg streams interleaved in one batch per step (the real
    server shape: a detokenize batch can carry one msg of each in-flight uid)."""
    mgr = DetokenizeManager(tok)
    joined = {u: "" for u in uid_msgs}
    iters = {u: iter(msgs) for u, msgs in uid_msgs.items()}
    while iters:
        batch = []
        for u in list(iters):
            try:
                batch.append(next(iters[u]))
            except StopIteration:
                del iters[u]
        if batch:
            out = mgr.detokenize(batch)
            for msg, incr in zip(batch, out, strict=True):
                joined[msg.uid] += incr
    return joined


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    ok = True
    for text in TEXTS:
        reference = join_stream(tok, token_stream(tok, text, 1, eos_at_end=True))
        print(f"int (non-MTP)  : {text[:24]!r} -> {reference!r}")
        for chunk in (2, 3, 5):
            for eos_at_end in (True, False):
                joined = join_stream(tok, token_stream(tok, text, chunk, eos_at_end))
                same = joined == reference
                ok &= same
                print(f"  list chunk={chunk} eos={eos_at_end}: "
                      f"{'identical' if same else 'MISMATCH'}")
                if not same:
                    print(f"    ref : {reference!r}")
                    print(f"    got : {joined!r}")
        # multi-uid interleaved batch: two uids share detokenize calls
        two = {0: token_stream(tok, text, 3, True, 0), 1: token_stream(tok, TEXTS[1], 2, True, 1)}
        joined2 = join_streams(tok, two)
        ok &= joined2[0] == reference
        ok &= joined2[1] == join_stream(tok, token_stream(tok, TEXTS[1], 1, True))
        print(f"  multi-uid interleaved: "
              f"{'identical' if joined2[0] == reference else 'MISMATCH'}")
        if not ok:
            break
    print("\nDETOKENIZE INT/LIST DISPATCH:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
