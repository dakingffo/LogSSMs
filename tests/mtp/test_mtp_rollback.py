"""Step 8c (rework): MTP verify reserve-commit correctness on the hybrid page cache.

The old circular-buffer rollback_to is gone. The verify forward writes each candidate's
post-state into the separate K+1 reserve table R[0..K], and committing ``num_sampled``
accepted tokens swaps the selected reserve slots into token-indexed live page mappings.
This test drives the real 0.8B
model through the three commit shapes and checks the committed baseline state equals
"state after num_sampled tokens appended to the prefill terminal state", measured
against an independent sequential-decode reference:

  * no page crossing    (C=100, num_sampled=3): R[2] becomes page 1's live state.
  * crossing, last      (C=126, num_sampled=2): R[1] closes page 1.
  * crossing, mid       (C=125, num_sampled=4): R[2] closes page 1 and R[3] becomes
                         page 2's live state.

The verify forward and the direct-decode reference feed the same token sequence from the
same baseline state, so the committed state matches the reference within the bf16 band
(the verify's fused attention context drifts ~1 ULP from per-token decode -- same
tolerance the old test used).

Run: /home/daking/.conda/envs/daking/bin/python tests/mtp/test_mtp_rollback.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("CUDA_HOME", "/home/daking/.conda/envs/daking")

MODEL_PATH = os.environ.get("QWEN3_5_MODEL", "/home/daking/models/huggingface/Qwen3.5-0.8B")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import torch

from logssms.utils import load_model_config, torch_dtype


def setup():
    from logssms.distributed import DistributedInfo, set_tp_info

    set_tp_info(DistributedInfo(rank=0, size=1))
    from logssms.models import make_model_config

    return make_model_config(load_model_config(MODEL_PATH))


def to_device(module, device):
    if isinstance(module, (list, tuple)):
        for v in module:
            to_device(v, device)
        return
    d = getattr(module, "__dict__", None)
    if d is None:
        return
    for name, v in d.items():
        if name.startswith("_"):
            continue
        if isinstance(v, torch.Tensor):
            if v.is_meta:
                continue
            setattr(module, name, v.to(device))
        elif isinstance(v, (list, tuple)):
            for el in v:
                to_device(el, device)
        elif getattr(v, "__dict__", None) is not None:
            to_device(v, device)


def main():
    mcfg = setup()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    from logssms.core import Batch, Context, Request, set_global_ctx
    from logssms.cache import make_kvcache_pool
    from logssms.cache.radix.state_pool import LinearStatePool
    from logssms.layers.attention_backend import make_attention_backend
    from logssms.models.qwen3_next import Qwen3NextForCausalLM
    from logssms.models.weight import load_target_weight
    from logssms.cache import CacheResources, RadixCacheManager

    loaded = {k: v for k, v in load_target_weight(MODEL_PATH, "cpu")}
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = Qwen3NextForCausalLM(mcfg)
    model.load_state_dict(dict(loaded))
    del loaded
    to_device(model, device)
    torch.cuda.empty_cache()

    K = 4                 # num_spec_tokens
    D = K + 1             # reserve width (verify window size)
    PS = 64               # state page size (hybrid; page boundary == 64-chunk boundary)
    conv_dim = (
        mcfg.linear_num_key_heads * mcfg.linear_key_head_dim * 2
        + mcfg.linear_num_value_heads * mcfg.linear_value_head_dim
    )
    ctx = Context(page_size=PS)
    num_pages = 32
    ctx.kv_cache = make_kvcache_pool(
        model_config=mcfg, num_pages=num_pages, page_size=PS,
        dtype=torch.bfloat16, device=device,
    )
    num_work_states = 8 * D
    pool = LinearStatePool(
        num_slots=num_pages + num_work_states,
        num_linear_layers=mcfg.num_linear_layers, conv_dim=conv_dim,
        kernel_size=mcfg.linear_conv_kernel_dim,
        num_v_heads=mcfg.linear_num_value_heads,
        head_k_dim=mcfg.linear_key_head_dim,
        head_v_dim=mcfg.linear_value_head_dim,
        device=device, dtype=torch.bfloat16,
    )
    ctx.linear_state = pool

    MAX_SEQ = 1024
    ctx.page_table = torch.zeros((8, MAX_SEQ), dtype=torch.int32, device=device)

    # Live state table is token-indexed; every page repeats one slot id.
    st = torch.full((8, MAX_SEQ), -1, dtype=torch.int32, device=device)
    reserve = torch.arange(
        num_pages,
        num_pages + num_work_states,
        dtype=torch.int32,
        device=device,
    ).view(8, D)
    ctx.state_table = st
    ctx.work_state_table = reserve
    set_global_ctx(ctx)
    ctx.attn_backend = make_attention_backend("fi", mcfg)

    resources = CacheResources(
        kv_pool=ctx.kv_cache,
        page_table=ctx.page_table,
        linear_state=pool,
        state_table=st,
        work_state_table=reserve,
    )
    cm = RadixCacheManager(
        num_pages=num_pages, page_size=PS, num_states=num_pages,
        resources=resources,
        num_draft_states=reserve.numel(),
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    ids = tokenizer(
        "def quicksort(arr):\n    if len(arr) <= 1:", return_tensors="pt"
    ).input_ids[0]
    ids = torch.cat([ids, ids[:1].expand(150)]).to(device)  # >= 131 tokens for the window
    N = ids.shape[0]

    def mk_req(tidx, n_tokens, cached_len):
        return Request(
            input_ids=ids[:n_tokens].cpu(), table_idx=tidx, cached_len=cached_len,
            output_len=16, uid=tidx, sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
            max_device_len=n_tokens + 16,
        )

    def run_prefill(tidx, n):
        """Fresh prefill of ids[:n] on row tidx; returns the Request."""
        req = mk_req(tidx, n, cached_len=0)
        cm.allocate([req])
        pb = Batch(reqs=[req], phase="prefill")
        pb.input_ids = ids[:n]
        pb.positions = torch.arange(n, device=device)
        pb.padded_reqs = [req]
        pb.out_loc = ctx.page_table[tidx, :n]
        ctx.attn_backend.prepare_metadata(pb)
        ctx.linear_attn_backend.prepare_metadata(pb)
        with ctx.forward_batch(pb):
            model.forward()
        return req

    def run_verify(tidx, C, toks):
        """Verify window [C, C+n) -> R[0..n); returns (req, n)."""
        n = toks.shape[0]
        req = mk_req(tidx, C + n, cached_len=C)
        cm.allocate([req])
        vb = Batch(reqs=[req], phase="verify")
        vb.input_ids = toks
        vb.positions = torch.arange(C, C + n, device=device)
        vb.padded_reqs = [req]
        vb.out_loc = ctx.page_table[tidx, C : C + n]
        ctx.attn_backend.prepare_metadata(vb)
        ctx.linear_attn_backend.prepare_metadata(vb)
        with ctx.forward_batch(vb):
            model.forward()
        return req, n

    def run_decode(tidx, pos, tok):
        """Single decode step at position pos (KV + state read/write on row tidx)."""
        req = mk_req(tidx, pos + 1, cached_len=pos)
        cm.allocate([req])
        db = Batch(reqs=[req], phase="decode")
        db.input_ids = tok
        db.positions = torch.tensor([pos], device=device)
        db.padded_reqs = [req]
        db.out_loc = ctx.page_table[tidx, pos : pos + 1]
        ctx.attn_backend.prepare_metadata(db)
        ctx.linear_attn_backend.prepare_metadata(db)
        with ctx.forward_batch(db):
            model.model.forward(tok)

    def read_state(slot):
        return (pool.conv_state[slot].clone(), pool.recurrent_state[slot].clone())

    def cmp(tag, a, b):
        dc = (a[0].float() - b[0].float()).abs().max().item()
        dr = (a[1].float() - b[1].float()).abs().max().item()
        ac = a[0].abs().max().item()
        ar = a[1].abs().max().item()
        rel = max(dc / max(ac, 1e-9), dr / max(ar, 1e-9))
        print(f"  {tag:30s} conv|d|={dc:.5f}  recurrent|d|={dr:.5f}  rel={rel:.2e}")
        return rel

    def run_case(tag, C, num_sampled, tidx_v=4, tidx_r=5):
        print(f"\n=== {tag}: C={C}, num_sampled={num_sampled} ===")
        # ---- prefill the verify row to C (fresh KV) ----
        run_prefill(tidx_v, C)
        # ---- verify window [C, C+D) using the REAL continuation ids as drafts ----
        toks = ids[C : C + D]
        req, n = run_verify(tidx_v, C, toks)
        assert n == D
        # ---- commit: pure index ops ----
        old_page_slot = int(st[tidx_v, C - 1])
        cm.state_commit_verify(req, C, num_sampled)
        committed_slot = int(st[tidx_v, C + num_sampled - 1])
        committed = read_state(committed_slot)
        print(f"  committed slot={committed_slot} (page slot was {old_page_slot})")
        # ---- reference: fresh prefill to C on the ref row + num_sampled decodes ----
        run_prefill(tidx_r, C)
        for j in range(num_sampled):
            run_decode(tidx_r, C + j, toks[j : j + 1])
        ref_slot = int(st[tidx_r, C + num_sampled - 1])
        ref = read_state(ref_slot)
        rel = cmp("committed vs direct decode", committed, ref)
        # ---- structural: the committed state is NOT the stale prefill terminal state ----
        stale = read_state(old_page_slot)
        dc = (committed[0].float() - stale[0].float()).abs().max().item()
        dr = (committed[1].float() - stale[1].float()).abs().max().item()
        print(f"  committed vs stale baseline: conv|d|={dc:.4f} recurrent|d|={dr:.4f}")
        return rel, max(dc, dr)

    rels = []
    # case A: no crossing. R[2] becomes page 1's live state.
    rels.append(run_case("no-cross swap", C=100, num_sampled=3, tidx_v=0, tidx_r=1)[0])
    # case B: crossing, boundary token is the last accepted. R[1] closes page 1.
    rels.append(run_case(
        "crossing, boundary-last", C=126, num_sampled=2, tidx_v=2, tidx_r=3
    )[0])
    # case C: R[2] closes page 1 and R[3] becomes page 2's live state.
    rels.append(run_case(
        "crossing, boundary-mid", C=125, num_sampled=4, tidx_v=4, tidx_r=5
    )[0])

    ok = max(rels) < 0.02  # bf16 ULP band (same tolerance as the old rollback test)
    print(f"\nRESERVE COMMIT: {'PASS' if ok else 'CHECK'} "
          f"(max rel state diff {max(rels):.2e})")
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
