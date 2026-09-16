"""Numerical TP=2 vs TP=1 sharding verification on a single GPU.

The seetacloud Qwen3.6-27B produces garbage logits at tp=2 (but tp=1 is byte-identical
to HF). Every collective in the main model sits at the END of its layer's forward, so we
can verify each TP op independently without a real 2-GPU run:

  * build the op three times: full (tp=1) and two rank-sharded copies (tp=2),
  * feed each the SAME full-width input,
  * with a RecordingDistributedImpl that returns the row-parallel partial unchanged
    (all_reduce) or a shape-correct dummy (all_gather) while recording the real input,
  * combine the two partials exactly like the real collectives would
      - all_reduce  -> p0 + p1          (row-parallel / vocab-parallel sum)
      - all_gather  -> cat([p0, p1])    (vocab-parallel logits, rank0 block first)
  * the combined tp=2 result must equal the tp=1 full output.

Any layer whose sharding math (loader `_shard_tensor` + local head counts + projection
sizes) is wrong shows up as a mismatch here.

Run: /home/daking/.conda/envs/daking/bin/python tests/mtp/test_tp2_numeric.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/home/daking/.conda/envs/daking")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import torch
import torch.nn.functional as F

from logssms.distributed import DistributedInfo, set_tp_info
from logssms.distributed.impl import DistributedCommunicator, DistributedImpl, NoopDistributedImpl
from logssms.models.config import RotaryConfig
from logssms.models.qwen3_next import Qwen3_5Config


# -------------------------------------------------------------------------------------
# Synthetic hybrid config (shapes mirror Qwen3.5 structure, small enough for one GPU)
# -------------------------------------------------------------------------------------
def make_config() -> Qwen3_5Config:
    return Qwen3_5Config(
        num_layers=2,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=64,
        hidden_size=512,
        vocab_size=5000,  # odd -> exercises the div_ceil vocab shard edge
        intermediate_size=1024,
        rms_norm_eps=1e-6,
        rotary_config=RotaryConfig(
            rotary_dim=32, max_position=4096, base=10000, scaling=None
        ),
        hidden_act="silu",
        tie_word_embeddings=False,
        architectures=["Qwen3_5ForCausalLM"],
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=4,
        linear_num_value_heads=12,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        shared_expert_intermediate_size=0,
        norm_topk_prob=False,
        mlp_only_layers=[0, 1],
        decoder_sparse_step=1,
        mtp_num_hidden_layers=0,
    )


class RecordingImpl(DistributedImpl):
    """Records every collective input. all_reduce returns x unchanged (row-parallel
    partial); all_gather returns a shape-correct dummy (the real partial is recorded)."""

    def __init__(self, records: list, tp_size: int):
        self.records = records
        self.tp_size = tp_size

    def all_reduce(self, x):
        self.records.append(("reduce", x.detach().clone()))
        return x

    def all_gather(self, x):
        self.records.append(("gather", x.detach().clone()))
        shp = list(x.shape)
        shp[0] *= self.tp_size
        return x.new_zeros(shp)


def _set_tree(module, name, v):
    parts = name.split(".")
    obj = module
    for p in parts[:-1]:
        if p.isdigit():
            obj = obj.op_list[int(p)]
        else:
            obj = obj.__dict__[p]
    setattr(obj, parts[-1], v)


def reset_tp(rank, size):
    import logssms.distributed.info as _info

    _info._TP_INFO = None
    set_tp_info(DistributedInfo(rank=rank, size=size))


def _load_from(op, sd, prefix, device="cuda:0"):
    """Load tensors whose key starts with `prefix` into `op`, stripping the prefix."""
    for k, v in sd.items():
        if prefix and not k.startswith(prefix):
            continue
        key = k[len(prefix):] if prefix else k
        _set_tree(op, key, v.to(device))


def build_full_sd(config: Qwen3_5Config, seed: int) -> dict:
    """Build the tp=1 full model and return its state_dict with REAL values (on cpu)."""
    reset_tp(0, 1)
    torch.manual_seed(seed)
    from logssms.models.qwen3_next import Qwen3NextForCausalLM

    model = Qwen3NextForCausalLM(config)
    for name, p in model.state_dict().items():
        _set_tree(model, name, torch.randn_like(p) * 0.02)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def shard_sd(full_sd, rank, size, num_kv_heads, config=None):
    from logssms.models.weight import _shard_tensor

    qkv_regions = None
    if config is not None and config.is_hybrid and config.linear_num_key_heads:
        qk = config.linear_num_key_heads * config.linear_key_head_dim
        v = config.linear_num_value_heads * config.linear_value_head_dim
        qkv_regions = (qk, qk, v)
    result = {}
    for k, v in full_sd.items():
        if ".self_attn.qkv_proj" in k and config is not None and config.is_hybrid:
            q_rows = config.num_qo_heads * config.head_dim * 2
            kv_rows = config.num_kv_heads * config.head_dim
            q, key, value = v.split([q_rows, kv_rows, kv_rows], dim=0)
            result[k] = torch.cat(
                [
                    _shard_tensor(
                        k.replace(".qkv_proj", ".q_proj"),
                        q,
                        rank,
                        size,
                        num_kv_heads,
                    ),
                    _shard_tensor(
                        k.replace(".qkv_proj", ".k_proj"),
                        key,
                        rank,
                        size,
                        num_kv_heads,
                    ),
                    _shard_tensor(
                        k.replace(".qkv_proj", ".v_proj"),
                        value,
                        rank,
                        size,
                        num_kv_heads,
                    ),
                ],
                dim=0,
            )
        else:
            result[k] = _shard_tensor(k, v, rank, size, num_kv_heads, qkv_regions)
    return result


def check(name, full, p0, p1, combine="sum", tol=1e-3):
    if combine == "sum":
        combined = p0 + p1
    elif combine == "cat":
        combined = torch.cat([p0, p1], dim=-1)
    err = (combined - full).abs().max().item()
    rel = err / (full.abs().max().item() + 1e-12)
    status = "OK " if (err < tol and rel < tol) else "FAIL"
    print(f"  [{status}] {name:28s} abs_err={err:.3e} rel_err={rel:.3e}")
    if status == "FAIL":
        print(f"    full range [{full.min().item():.4f}, {full.max().item():.4f}]")
        print(f"    comb range [{combined.min().item():.4f}, {combined.max().item():.4f}]")
        return False
    return True


def run():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    config = make_config()
    n = 2
    num_kv_heads = config.num_kv_heads

    full_sd = build_full_sd(config, seed=42)

    ok = True
    T = 8

    def run_tp2(op_cls, build_kwargs, sd_prefix, full_instance, fn, combine):
        """Run op at tp=1 (full_instance) and tp=2 (two ranks), combine partials, check."""
        nonlocal ok
        parts = []
        for r in range(n):
            reset_tp(r, n)
            op = op_cls(**build_kwargs)
            _load_from(op, shard_sd(full_sd, r, n, num_kv_heads, config), sd_prefix)
            recs = []
            DistributedCommunicator.impl = RecordingImpl(recs, n)
            parts.append(fn(op).detach())
            assert len(recs) == 1, f"{op_cls.__name__}: expected 1 collective, got {len(recs)}"
        ok &= check(op_cls.__name__, fn(full_instance), parts[0], parts[1], combine)
        return parts

    # ---------------- VocabParallelEmbedding ----------------
    print("=" * 60)
    print("VocabParallelEmbedding (all_reduce)")
    from logssms.layers.embedding import VocabParallelEmbedding

    reset_tp(0, 1)
    emb_full = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
    _load_from(emb_full, full_sd, "model.embed_tokens.")
    ids = torch.randint(0, config.vocab_size, (T,), device=device)
    run_tp2(
        VocabParallelEmbedding,
        dict(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size),
        "model.embed_tokens.",
        emb_full,
        lambda m: m.forward(ids),
        "sum",
    )

    # ---------------- GatedMLP ----------------
    print("=" * 60)
    print("GatedMLP (down_proj all_reduce)")
    from logssms.layers import GatedMLP

    reset_tp(0, 1)
    mlp_kwargs = dict(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
    )
    mlp_full = GatedMLP(**mlp_kwargs)
    _load_from(mlp_full, full_sd, "model.layers.1.mlp.")
    x = torch.randn(T, config.hidden_size, device=device)
    run_tp2(
        GatedMLP,
        mlp_kwargs,
        "model.layers.1.mlp.",
        mlp_full,
        lambda m: m.forward(x),
        "sum",
    )

    # ---------------- GatedRotaryAttention (local reference attention) ----------------
    print("=" * 60)
    print("GatedRotaryAttention projection + o_proj all_reduce")
    from logssms.layers import GatedRotaryAttention

    positions = torch.arange(T, dtype=torch.int64, device=device)
    rotary_config = config.rotary_config
    attention_kwargs = dict(
        hidden_size=config.hidden_size,
        head_dim=config.head_dim,
        num_qo_heads=config.num_qo_heads,
        num_kv_heads=config.num_kv_heads,
        layer_id=1,
        rotary_dim=rotary_config.rotary_dim,
        max_position=rotary_config.max_position,
        rope_base=rotary_config.base,
        rope_scaling=(
            tuple(rotary_config.scaling.items()) if rotary_config.scaling else None
        ),
        qk_norm_eps=config.rms_norm_eps,
        zero_centered_norm=True,
    )
    reset_tp(0, 1)
    attn_full = GatedRotaryAttention(**attention_kwargs)
    _load_from(attn_full, full_sd, "model.layers.1.self_attn.")

    def attention_forward(module):
        query, gate, key, value = module.proj(x, positions)
        num_repeats = module.num_qo_heads // module.num_kv_heads
        key = key.repeat_interleave(num_repeats, dim=1).transpose(0, 1)
        value = value.repeat_interleave(num_repeats, dim=1).transpose(0, 1)
        query = query.transpose(0, 1)
        score = torch.matmul(query, key.transpose(-1, -2)) * module.head_dim**-0.5
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
        )
        score = F.softmax(
            score.masked_fill(mask, float("-inf")), dim=-1, dtype=torch.float32
        )
        output = torch.matmul(score.to(value.dtype), value).transpose(0, 1)
        output *= torch.sigmoid(gate)
        return module.o_proj.forward(output.flatten(-2))

    run_tp2(
        GatedRotaryAttention,
        attention_kwargs,
        "model.layers.1.self_attn.",
        attn_full,
        attention_forward,
        "sum",
    )

    # ---------------- GatedDeltaNet (out_proj all_reduce) ----------------
    print("=" * 60)
    print("GatedDeltaNet prefill (out_proj all_reduce)")
    from logssms.cache.linear_pool.state_pool import LinearStatePool
    from logssms.core import Batch, Context, Request, clear_global_ctx, set_global_ctx
    from logssms.layers import GatedDeltaNet

    def make_gated_delta_net():
        return GatedDeltaNet(
            hidden_size=config.hidden_size,
            num_key_heads=config.linear_num_key_heads,
            num_value_heads=config.linear_num_value_heads,
            head_k_dim=config.linear_key_head_dim,
            head_v_dim=config.linear_value_head_dim,
            conv_kernel_size=config.linear_conv_kernel_dim,
            rms_norm_eps=config.rms_norm_eps,
            layer_idx=0,
        )

    conv_dim_full = (
        config.linear_num_key_heads * config.linear_key_head_dim * 2
        + config.linear_num_value_heads * config.linear_value_head_dim
    )
    conv_dim_local = conv_dim_full // n
    ps = 64
    L = 80  # > one 64-token chunk

    def make_req(table_idx, cached_len, device_len):
        return Request(
            input_ids=torch.tensor([0] * device_len, dtype=torch.int32),
            table_idx=table_idx, cached_len=cached_len, output_len=4,
            uid=table_idx, sampling_params=None, cache_handle=None,
            max_device_len=device_len + 4,
        )

    def make_batch(req):
        b = Batch(reqs=[req], phase="prefill")
        b.padded_reqs = b.reqs
        return b

    def setup_ctx(conv_dim, num_v_heads):
        ctx = Context(page_size=ps)
        ctx.linear_state = LinearStatePool(
            num_slots=16, num_linear_layers=1, conv_dim=conv_dim,
            kernel_size=config.linear_conv_kernel_dim, num_v_heads=num_v_heads,
            head_k_dim=config.linear_key_head_dim, head_v_dim=config.linear_value_head_dim,
            device=device, dtype=torch.float32,
        )
        st = torch.full((4, 128), -1, dtype=torch.int32, device=device)
        # hand out one distinct slot per page the layer will snapshot (L=80 -> pages 0,1)
        for p in (0, 1):
            st[0, p * ps : (p + 1) * ps] = p
        ctx.state_table = st
        set_global_ctx(ctx)
        return ctx

    xg = torch.randn(L, config.hidden_size, device=device)
    req = make_req(0, 0, L)

    reset_tp(0, 1)
    gdn_full = make_gated_delta_net()
    _load_from(gdn_full, full_sd, "model.layers.0.linear_attn.")
    ctx = setup_ctx(conv_dim_full, config.linear_num_value_heads)
    batch = make_batch(req)
    ctx.linear_attn_backend.prepare_metadata(batch)
    with ctx.forward_batch(batch):
        gdn_out_full = gdn_full.forward(xg).detach()
    clear_global_ctx()

    gdn_parts = []
    for r in range(n):
        reset_tp(r, n)
        gdn = make_gated_delta_net()
        _load_from(gdn, shard_sd(full_sd, r, n, num_kv_heads, config), "model.layers.0.linear_attn.")
        ctx = setup_ctx(conv_dim_local, config.linear_num_value_heads // n)
        recs = []
        DistributedCommunicator.impl = RecordingImpl(recs, n)
        batch = make_batch(req)
        ctx.linear_attn_backend.prepare_metadata(batch)
        with ctx.forward_batch(batch):
            gdn_parts.append(gdn.forward(xg).detach())
        clear_global_ctx()
        assert len(recs) == 1
    ok &= check("gated_delta_net", gdn_out_full, gdn_parts[0], gdn_parts[1], "sum")

    # ---------------- ParallelLMHead (all_gather) ----------------
    print("=" * 60)
    print("ParallelLMHead (all_gather vocab)")
    from logssms.layers.embedding import ParallelLMHead

    reset_tp(0, 1)
    lm_full = ParallelLMHead(config.vocab_size, config.hidden_size, tie_word_embeddings=False)
    _load_from(lm_full, full_sd, "lm_head.")
    b = Batch(reqs=[make_req(0, 0, 1)], phase="decode")
    b.padded_reqs = b.reqs
    xl = torch.randn(T, config.hidden_size, device=device)

    def lm_forward(m):
        ctxb = Context(page_size=ps)
        set_global_ctx(ctxb)
        try:
            with ctxb.forward_batch(b):
                return m.forward(xl)
        finally:
            clear_global_ctx()

    lm_parts = []
    for r in range(n):
        reset_tp(r, n)
        lm = ParallelLMHead(config.vocab_size, config.hidden_size, tie_word_embeddings=False)
        _load_from(lm, shard_sd(full_sd, r, n, num_kv_heads, config), "lm_head.")
        recs = []
        DistributedCommunicator.impl = RecordingImpl(recs, n)
        lm_forward(lm)
        assert len(recs) == 1 and recs[0][0] == "gather"
        lm_parts.append(recs[0][1])  # the REAL pre-gather partial (T, vocab/tp), not the dummy
    ok &= check("lm_head", lm_forward(lm_full), lm_parts[0], lm_parts[1], "cat")

    DistributedCommunicator.impl = NoopDistributedImpl()
    print("=" * 60)
    print("ALL PASS" if ok else "FAILURES FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    run()
