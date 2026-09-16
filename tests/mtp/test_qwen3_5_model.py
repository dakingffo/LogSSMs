"""Qwen3.5 verification using a Qwen3.5-2B checkpoint.

Run with: /home/daking/.conda/envs/daking/bin/python tests/mtp/test_qwen3_5_model.py

Tests:
  1. Weight loading: model.state_dict() keys vs checkpoint keys (no missing/unexpected).
  2. GatedDeltaNet standalone math: prefill (chunked rule) == step-by-step decode
     (recurrent rule), and chunked continuation (use_state=True) == full prefill.
  3. Full model (paged flashinfer path): prefill-vs-decode logits consistency.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("CUDA_HOME", "/home/daking/.conda/envs/daking")

import torch

# Qwen3.6 checkpoints use the same Hugging Face architecture implementation.
#   QWEN3_5_MODEL=/path/to/model python tests/mtp/test_qwen3_5_model.py
MODEL_PATH = os.environ.get("QWEN3_5_MODEL", "/home/daking/models/huggingface/Qwen3.5-2B")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from logssms.utils import load_model_config, torch_dtype


def setup():
    from logssms.distributed import DistributedInfo, set_tp_info

    set_tp_info(DistributedInfo(rank=0, size=1))
    from logssms.models import make_model_config

    mcfg = make_model_config(load_model_config(MODEL_PATH))
    return mcfg


def load_model(mcfg, loaded=None):
    from logssms.models.qwen3_next import Qwen3NextForCausalLM
    from logssms.models.weight import load_target_weight

    if loaded is None:
        loaded = {k: v for k, v in load_target_weight(MODEL_PATH, "cpu")}
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = Qwen3NextForCausalLM(mcfg)
    # load_state_dict pops keys from the passed dict; pass a copy to keep `loaded` intact
    model.load_state_dict(dict(loaded))
    return model, loaded


def test1_weight_loading(mcfg, model, loaded):
    print("=" * 60)
    print("Test 1: weight loading")
    sd = model.state_dict()
    ckpt_keys, model_keys = set(loaded), set(sd)
    missing = sorted(ckpt_keys - model_keys)
    unexpected = sorted(model_keys - ckpt_keys)
    print(f"  checkpoint keys: {len(ckpt_keys)}, model keys: {len(model_keys)}")
    print(f"  missing (in ckpt, not in model): {missing}")
    print(f"  unexpected (in model, not in ckpt): {unexpected}")
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    # spot-check shapes on representative target-model layers
    lt = "model.layers.0.linear_attn"
    at = "model.layers.3.self_attn"
    for k in (f"{lt}.conv1d.weight", f"{lt}.in_proj_qkv.weight", f"{at}.qkv_proj.weight",
              f"{at}.q_norm.weight", "model.norm.weight"):
        assert k in sd, f"missing {k}"
        print(f"  {k}: {tuple(sd[k].shape)} {sd[k].dtype}")
    assert not any(k.startswith(("mtp.", "model.mtp.")) for k in sd)
    print("  PASS")


def make_req(table_idx, cached_len, device_len, output_len=4):
    from logssms.core import Request

    return Request(
        input_ids=torch.tensor([0] * device_len, dtype=torch.int32),
        table_idx=table_idx,
        cached_len=cached_len,
        output_len=output_len,
        uid=table_idx,
        sampling_params=None,  # type: ignore
        cache_handle=None,  # type: ignore
        max_device_len=device_len + output_len,
    )


def make_batch(reqs, phase, input_ids, positions):
    from logssms.core import Batch

    batch = Batch(reqs=reqs, phase=phase)
    batch.input_ids = input_ids
    batch.positions = positions
    batch.padded_reqs = reqs
    return batch


def to_device(module, device):
    """Move every tensor in a BaseOP tree to ``device`` in place, one at a time.

    load_state_dict assigns the SAME tensor objects into the tree, so replacing each attr
    frees the CPU source as it goes. This avoids the double-copy peak (CPU+GPU model +
    transient GPU state_dict) that overflows an 8GB card for a ~3.9GB bf16 model.
    """
    if isinstance(module, (list, tuple)):
        for v in module:
            to_device(v, device)
        return
    d = getattr(module, "__dict__", None)
    if d is None:
        return  # int/str/None/etc: nothing to move
    for name, v in d.items():
        if name.startswith("_"):
            continue
        if isinstance(v, torch.Tensor):
            if v.is_meta:
                continue  # tied lm_head.weight never materialized; never used
            setattr(module, name, v.to(device))
        elif isinstance(v, (list, tuple)):
            for el in v:
                to_device(el, device)
        elif getattr(v, "__dict__", None) is not None:
            to_device(v, device)


def _set_tree(module, name, v):
    """Set a dotted-key tensor inside a BaseOP tree (plain setattr can't traverse dots)."""
    parts = name.split(".")
    obj = module
    for p in parts[:-1]:
        obj = obj.__dict__[p]
    setattr(obj, parts[-1], v)


def test2_gated_delta_net_math(mcfg):
    print("=" * 60)
    print("Test 2: GatedDeltaNet prefill vs decode math (random weights)")
    from logssms.core import Context, get_global_ctx, set_global_ctx
    from logssms.cache import LinearStatePool
    from logssms.layers import GatedDeltaNet

    hidden = mcfg.hidden_size
    conv_dim = (
        mcfg.linear_num_key_heads * mcfg.linear_key_head_dim * 2
        + mcfg.linear_num_value_heads * mcfg.linear_value_head_dim
    )
    ps = 64  # hybrid: page boundaries must align with the 64-token chunk rule
    ctx = Context(page_size=ps)
    ctx.linear_state = LinearStatePool(
        num_slots=32, num_linear_layers=1, conv_dim=conv_dim,
        kernel_size=mcfg.linear_conv_kernel_dim,
        num_v_heads=mcfg.linear_num_value_heads,
        head_k_dim=mcfg.linear_key_head_dim,
        head_v_dim=mcfg.linear_value_head_dim,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    # one state slot per page the layer will write (reqs 0,1,2, pages 0..3 cover 256 tokens)
    # the pool is a pure buffer now (free-list lives in CacheManager); hand out distinct slots
    ctx.state_table = torch.full((4, 256), -1, dtype=torch.int32)
    slot = iter(range(ctx.linear_state.conv_state.shape[0]))
    for tidx in (0, 1, 2):
        for p in range(4):
            ctx.state_table[tidx, p * ps : (p + 1) * ps] = next(slot)
    set_global_ctx(ctx)

    torch.manual_seed(0)
    layer = GatedDeltaNet(
        hidden_size=mcfg.hidden_size,
        num_key_heads=mcfg.linear_num_key_heads,
        num_value_heads=mcfg.linear_num_value_heads,
        head_k_dim=mcfg.linear_key_head_dim,
        head_v_dim=mcfg.linear_value_head_dim,
        conv_kernel_size=mcfg.linear_conv_kernel_dim,
        rms_norm_eps=mcfg.rms_norm_eps,
        layer_idx=0,
    )
    # random fp32 weights (nested params via _set_tree)
    for name, p in layer.state_dict().items():
        _set_tree(layer, name, torch.randn_like(p) * 0.02)

    L = 200  # > one 64-token chunk
    x = torch.randn(L, hidden)

    def prefill_run(seq, req):
        batch = make_batch([req], "prefill", None, None)
        ctx.linear_attn_backend.prepare_metadata(batch)
        with ctx.forward_batch(batch):
            return layer.forward(seq)

    # reference: full prefill of all L tokens
    req = make_req(0, 0, L)
    out_ref = prefill_run(x, req)

    # (a) step-by-step: prefill token 0, then decode one token at a time
    req = make_req(1, 0, 1)
    out0 = prefill_run(x[:1], req)
    err = (out0 - out_ref[:1]).abs().max().item()
    assert err < 1e-4, f"pos0 prefill mismatch {err}"
    for i in range(1, L):
        req = make_req(1, i, i + 1)
        batch = make_batch([req], "decode", None, None)
        ctx.linear_attn_backend.prepare_metadata(batch)
        with ctx.forward_batch(batch):
            out_i = layer.forward(x[i : i + 1])
        err = (out_i - out_ref[i : i + 1]).abs().max().item()
        assert err < 1e-3, f"decode pos {i} mismatch {err}"
    print("  step-by-step decode == full prefill  ✓")

    # (b) chunked continuation: prefill 0..64, then prefill 64..L with cached_len=64
    K = 64
    req = make_req(2, 0, K)
    prefill_run(x[:K], req)
    req = make_req(2, K, L)
    out_chunk = prefill_run(x[K:], req)
    err = (out_chunk - out_ref[K:]).abs().max().item()
    assert err < 1e-3, f"chunked continuation mismatch {err}"
    print("  chunked continuation (use_state=True) == full prefill  ✓")
    print("  PASS")
    from logssms.core import clear_global_ctx

    clear_global_ctx()


def test3_full_model(mcfg, loaded=None):
    print("=" * 60)
    print("Test 3: full model prefill vs decode logits (paged flashinfer)")
    from logssms.core import Context, get_global_ctx, set_global_ctx, Request
    from logssms.cache import make_kvcache_pool, LinearStatePool
    from logssms.layers.attention_backend import make_attention_backend

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    model, loaded = load_model(mcfg, loaded=loaded)
    # move weights cpu->device in place (double-copy load_state_dict peaks at 2x model = OOM)
    to_device(model, device)

    conv_dim = (
        mcfg.linear_num_key_heads * mcfg.linear_key_head_dim * 2
        + mcfg.linear_num_value_heads * mcfg.linear_value_head_dim
    )
    ctx = Context(page_size=64)
    num_pages = 4096
    ctx.kv_cache = make_kvcache_pool(
        model_config=mcfg, num_pages=num_pages, page_size=1,
        dtype=torch.bfloat16, device=device,
    )
    ctx.linear_state = LinearStatePool(
        num_slots=8, num_linear_layers=mcfg.num_linear_layers, conv_dim=conv_dim,
        kernel_size=mcfg.linear_conv_kernel_dim,
        num_v_heads=mcfg.linear_num_value_heads,
        head_k_dim=mcfg.linear_key_head_dim,
        head_v_dim=mcfg.linear_value_head_dim,
        device=device, dtype=torch.bfloat16,
    )
    # one state slot per page the layer writes. All prefill/decode here stays in page 0
    # (the prompt is ~15 tokens < 64); the prefill chunk callback writes page 0 for each
    # request's row, decode reads/writes page 0 too.
    ctx.state_table = torch.full((8, 1024), -1, dtype=torch.int32, device=device)
    slot = iter(range(ctx.linear_state.conv_state.shape[0]))
    for tidx in (3, 4):
        ctx.state_table[tidx, :ctx.page_size] = next(slot)
    ctx.page_table = torch.zeros((8, 8192), dtype=torch.int32, device=device)
    # sequential pages per req
    base = torch.arange(num_pages, device=device, dtype=torch.int32).view(1, -1)
    ctx.page_table[:, : num_pages] = base  # req r token i -> page r*num_pages + i
    set_global_ctx(ctx)  # backend reads ctx.kv_cache via get_global_ctx()
    ctx._debug_attn = True
    ctx.attn_backend = make_attention_backend("fi", mcfg)

    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(MODEL_PATH)
    tokens = tokenizer("The quick brown fox jumps over the lazy dog and the", return_tensors="pt")
    ids = tokens.input_ids[0].tolist()
    N = len(ids) - 1  # positions 0..N
    id_t = torch.tensor(ids, dtype=torch.int32, device=device)

    def run(batch):
        print(f"    [run] phase={batch.phase} input_ids={tuple(batch.input_ids.shape)} "
              f"out_loc={tuple(batch.out_loc.shape)} positions={tuple(batch.positions.shape)}")
        with ctx.forward_batch(batch):
            return model.forward()

    def mk_req(table_idx, n_tokens, cached_len, uid):
        # device_len is derived from input_ids length
        return Request(
            input_ids=id_t[:n_tokens].cpu(), table_idx=table_idx, cached_len=cached_len,
            output_len=64, uid=uid, sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
            max_device_len=n_tokens + 64,
        )

    # reference per-position logits via fresh prefill 0..i
    ref = []
    for i in range(N + 1):
        req = mk_req(3, i + 1, 0, uid=0)
        batch = make_batch([req], "prefill", id_t[: i + 1],
                           torch.arange(i + 1, device=device))
        batch.out_loc = ctx.page_table[3, : i + 1]
        ctx.attn_backend.prepare_metadata(batch)
        logits = run(batch)
        ref.append(logits[0])  # prefill returns last-token logits

    # (a) single full prefill: last-token logit must equal ref[N]
    req = mk_req(4, N + 1, 0, uid=1)
    batch = make_batch([req], "prefill", id_t, torch.arange(N + 1, device=device))
    batch.out_loc = ctx.page_table[4, : N + 1]
    ctx.attn_backend.prepare_metadata(batch)
    logits_full = run(batch)
    err = (logits_full[0] - ref[N]).abs().max().item()
    assert err < 1e-2, f"full prefill vs fresh prefill mismatch {err}"
    print("  full prefill last-token logits == incremental prefill  ✓")

    # (b) step-by-step decode
    # no pool.reset: the fresh prefill below (cached_len=0, baseline=None) overwrites
    # page 0's slot from zero, so stale state from the reference prefills can't leak in.
    # prefill token 0 on req slot 3
    req = mk_req(3, 1, 0, uid=2)
    batch = make_batch([req], "prefill", id_t[:1], torch.zeros(1, device=device, dtype=torch.int64))
    batch.out_loc = ctx.page_table[3, :1]
    ctx.attn_backend.prepare_metadata(batch)
    logit0 = run(batch)
    err = (logit0[0] - ref[0]).abs().max().item()
    assert err < 1e-2, f"decode step 0 mismatch {err}"
    # decode == fresh-prefill: the prefill (chunked rule) and decode (recurrent rule) are
    # mathematically equivalent. The remaining diff is bf16 quantization noise accumulated
    # through 24 layers + lm_head amplification (~0.2-0.4 on ~30-magnitude logits; all exact
    # powers of 2). A real bug (bad state write/index, bad conv handoff) gives O(1-15), so a
    # ~0.5 ceiling cleanly separates noise from bugs. (Single-layer decode matches to ~0.001.)
    DECODE_TOL = 0.5
    decode_errs = []
    for i in range(1, N + 1):
        req = mk_req(3, i + 1, i, uid=2)  # full history, cached_len=i -> extend_len=1
        batch = make_batch([req], "decode", id_t[i : i + 1],
                           torch.tensor([i], device=device, dtype=torch.int64))
        batch.out_loc = ctx.page_table[3, i : i + 1]
        ctx.attn_backend.prepare_metadata(batch)
        logits = run(batch)
        err = (logits[0] - ref[i]).abs().max().item()
        decode_errs.append(err)
    print(f"  step-by-step decode ({N} steps) vs incremental prefill:")
    print(f"    per-step max|logits diff|: {[f'{e:.4g}' for e in decode_errs]}")
    worst = max(decode_errs)
    assert worst < DECODE_TOL, f"decode worst mismatch {worst} >= {DECODE_TOL}"
    print(f"    worst = {worst:.4g} < {DECODE_TOL}  ✓")
    print("  PASS")


if __name__ == "__main__":
    mcfg = setup()
    model, loaded = load_model(mcfg)  # single disk read, reused by test3
    test1_weight_loading(mcfg, model, loaded)
    test2_gated_delta_net_math(mcfg)
    if torch.cuda.is_available():
        test3_full_model(mcfg, loaded=loaded)
    else:
        print("No CUDA available; skipping test 3")
    print("\nALL TESTS PASSED")
