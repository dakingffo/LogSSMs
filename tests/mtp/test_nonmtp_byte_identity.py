"""Step 8g: Qwen3-1.7B non-MTP regression — transformers greedy reference comparison.

Loads the non-MTP LogSSMs run JSON (tests/mtp/test_mtp_e2e.py --mtp 0) and a
transformers greedy decode of the same 5 prompts, then compares token-for-token.

Qwen3-1.7B is dense (num_linear_layers=0), so the depth-D linear state pool is inert and
VerifyManager is not created without --speculative-algorithm. This test proves the
non-MTP path still
produces a byte-identical greedy stream to the reference HF implementation — i.e. the MTP
refactor did not perturb it.

Run: /home/daking/.conda/envs/daking/bin/python tests/mtp/test_nonmtp_byte_identity.py /tmp/non_mtp_17b.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "python"))
sys.path.insert(0, str(_REPO))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "/home/daking/models/huggingface/Qwen3-1.7B"
PROMPTS = [
    ("q1", "The meaning of life is"),
    ("q2", "一次函数 y = kx + b,其中 k 和 b 分别是"),
    ("q3", "def quicksort(arr):\n    if len(arr) <= 1:"),
    ("q4", "The capital of France is"),
    ("q5", "碳达峰指的是"),
]
MAX_TOKENS = 24


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/non_mtp_17b.json"
    result = json.load(open(src))
    assert result["enable_specualtive_decoding"] is False, "this is a non-MTP regression"
    assert not result["missing"], f"unfinished reqs: {result['missing']}"
    assert result["integrity_ok"], f"integrity failed: {result['integrity_msg']}"

    tok = AutoTokenizer.from_pretrained(MODEL)
    eos = tok.eos_token_id

    # LogSSMs non-MTP greedy streams (from the harness run)
    sgl = {name: [int(t) for t in result["tokens"][name]] for name, _ in PROMPTS}

    # transformers greedy reference (raw-tokenized prompt, no chat template — the harness
    # feeds UserMsg.input_ids = tokenizer(text).input_ids verbatim)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    model.cuda().eval()
    ref = {}
    with torch.no_grad():
        for name, text in PROMPTS:
            ids = tok(text, return_tensors="pt").input_ids.cuda()
            out = model.generate(
                ids,
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
                temperature=1.0,
                pad_token_id=eos,
                use_cache=True,
            )
            ref[name] = out[0, len(ids[0]):].tolist()

    ok = True
    for name, _ in PROMPTS:
        a, b = sgl[name], ref[name]
        same = a == b
        ok &= same
        print(f"  {name:12s} pico {len(a):3d} tok, HF {len(b):3d} tok, "
              f"{'IDENTICAL' if same else '*** DIFFER ***'}")
        if not same:
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    print(f"    diff at idx {i}: pico={x} vs HF={y}")
                    break
    print("\nNON-MTP REGRESSION (Qwen3-1.7B):", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
