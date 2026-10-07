from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import FalconH1Config as HFConfig, FalconH1ForCausalLM as HFModel

from logssms.cache.kv_pool.mha_pool import MHAKVCachePool
from logssms.cache.linear_pool.state_pool import LinearStatePool
from logssms.cache.logstate_pool.frame_pool import FramedMHALogStatePool
from logssms.core import Batch, Context, Request, SamplingParams, clear_global_ctx, set_global_ctx
from logssms.distributed import DistributedInfo, tp_override
from logssms.engine.prefill_graph import PrefillGraphRunner
from logssms.layers.linear_attention_backend import make_linear_attention_backend
from logssms.models import make_model, make_model_config, load_target_weight
from logssms.utils import torch_dtype


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class AttentionOracle:
    def __init__(self, pool):
        self.pool = pool

    def forward(self, query, layer, batch):
        req = batch.reqs[0]
        indices = torch.arange(req.device_len, device=query.device)
        k = self.pool.k_cache(layer).flatten(0, 1)[indices]
        v = self.pool.v_cache(layer).flatten(0, 1)[indices]
        repeats = query.shape[1] // k.shape[1]
        k, v = (tensor.repeat_interleave(repeats, 1).transpose(0, 1).unsqueeze(0) for tensor in (k, v))
        q = query.transpose(0, 1).unsqueeze(0)
        mask = torch.arange(req.device_len, device=query.device)[None, :] <= batch.positions[:, None]
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask).squeeze(0).transpose(0, 1)


@pytest.mark.parametrize("cache_type", ["radix", "logstate"])
@pytest.mark.parametrize("gated_norm", [False, True])
def test_falcon_h1_matches_transformers_and_prefill_graph(tmp_path, cache_type, gated_norm):
    torch.manual_seed(11)
    raw = HFConfig(
        architectures=["FalconH1ForCausalLM"], hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        vocab_size=128, mamba_d_ssm=128, mamba_n_heads=4, mamba_d_head=32,
        mamba_d_state=32, mamba_n_groups=1, mamba_d_conv=4, mamba_conv_bias=True,
        mamba_rms_norm=gated_norm, mamba_norm_before_gate=False,
        max_position_embeddings=256, mamba_chunk_size=16,
        embedding_multiplier=2.0, lm_head_multiplier=0.5,
        attention_in_multiplier=0.7, attention_out_multiplier=0.3, key_multiplier=0.9,
        ssm_in_multiplier=0.8, ssm_out_multiplier=0.6,
        ssm_multipliers=[0.5, 0.7, 0.8, 0.9, 0.6], mlp_multipliers=[0.8, 0.7],
    )
    reference = HFModel(raw).eval().cuda().bfloat16()
    reference.save_pretrained(tmp_path)
    config = make_model_config(raw)
    clear_global_ctx()
    with tp_override(DistributedInfo(0, 1)), torch.inference_mode():
        with torch.device("cuda"), torch_dtype(torch.bfloat16):
            model = make_model(config)
        model.load_state_dict(dict(load_target_weight(str(tmp_path), torch.device("cuda"))))
        ctx = Context(64)
        if cache_type == "radix":
            pool = LinearStatePool(5, 2, 192, 4, 4, 32, 32, torch.device("cuda"), torch.bfloat16)
            ctx.kv_cache = MHAKVCachePool(2, 4, 64, 2, 64, torch.bfloat16, torch.device("cuda"))
            ctx.state_table = (torch.arange(256, device="cuda", dtype=torch.int32) // 64)[None]
        else:
            pool = FramedMHALogStatePool(
                num_frames=128, page_size=64, granularity=32,
                num_attention_layers=2, num_kv_heads=2, head_dim=64,
                num_linear_layers=2, num_key_heads=1, num_value_heads=4,
                head_k_dim=32, head_v_dim=32, conv_dim=192, state_len=3,
                device=torch.device("cuda"), dtype=torch.bfloat16, recurrent_type="mamba2",
            )
            reserved = pool.reserve(kv=4, logs=4, works=1)
            kv, logs, works = reserved
            ctx.kv_cache = pool
            ctx.state_table = torch.tensor(logs, device="cuda", dtype=torch.int32).repeat_interleave(64)[None]
            ctx.work_state_table = torch.tensor([works], device="cuda", dtype=torch.int32)
            pool.zero_state(works[0])
            pool.zero_work_conv(works[0])
        ctx.page_table = torch.arange(256, device="cuda", dtype=torch.int32)[None]
        ctx.linear_state = pool
        ctx.linear_attn_backend = make_linear_attention_backend(cache_type, recurrent_type="mamba2")
        ctx.attn_backend = AttentionOracle(ctx.kv_cache)
        set_global_ctx(ctx)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        runner = PrefillGraphRunner(model, 64, stream)
        ctx.prefill_graph_runner = runner
        try:
            for length in (17, 47, 33):
                ids = torch.randint(1, 128, (length,), device="cuda", dtype=torch.int32)
                expected = reference(ids.long()[None], use_cache=False).logits[:, -1].float()
                req = Request(ids.cpu(), 0, 0, 1, 0, SamplingParams(), None, 256)
                batch = Batch([req], "prefill")
                batch.padded_reqs = batch.reqs
                batch.input_ids = ids
                batch.positions = torch.arange(length, device="cuda", dtype=torch.int32)
                batch.out_loc = batch.positions
                if cache_type == "logstate":
                    pool.zero_state(works[0])
                    pool.zero_work_conv(works[0])
                    # Use contiguous physical KV frames for the oracle's lookup.
                    batch.out_loc = torch.tensor(kv, device="cuda", dtype=torch.int32).repeat_interleave(64)[:length] * 64 + batch.positions % 64
                    ctx.page_table[0, :length] = batch.out_loc
                    ctx.attn_backend.forward = _framed_attention(pool)
                batch.attn_metadata = SimpleNamespace(get_last_indices=lambda bs: torch.tensor([length-1], device="cuda"))
                ctx.linear_attn_backend.prepare_metadata(batch)
                with ctx.forward_batch(batch):
                    runner.enabled = False
                    eager = model.forward().clone()
                    if cache_type == "logstate":
                        pool.zero_state(works[0])
                        pool.zero_work_conv(works[0])
                    runner.enabled = True
                    graphed = model.forward().clone()
                torch.testing.assert_close(graphed, eager, rtol=2e-2, atol=2e-2)
                torch.testing.assert_close(eager.float(), expected, rtol=5e-2, atol=3e-2)
            # Continue the actual prefix through a replay-window flush and a
            # physical page boundary; compare against fresh HF recomputation.
            replay_count = 0
            for _ in range(35):
                position = len(ids)
                ids = torch.cat((ids, torch.randint(1, 128, (1,), device="cuda", dtype=torch.int32)))
                expected = reference(ids.long()[None], use_cache=False).logits[:, -1].float()
                req = Request(ids.cpu(), 0, position, 1, 0, SamplingParams(), None, 256)
                batch = Batch([req], "decode")
                batch.padded_reqs = batch.reqs
                batch.input_ids = ids[-1:]
                batch.positions = torch.tensor([position], device="cuda", dtype=torch.int32)
                batch.out_loc = batch.positions.clone()
                if cache_type == "logstate":
                    batch.out_loc.fill_(kv[position // 64] * 64 + position % 64)
                    ctx.page_table[0, position] = batch.out_loc[0]
                    batch.logstate_replay_counts = [replay_count]
                    replay_count = 0 if (position + 1) % 16 == 0 else replay_count + 1
                batch.attn_metadata = SimpleNamespace(get_last_indices=lambda bs: torch.zeros(1, device="cuda", dtype=torch.int64))
                ctx.linear_attn_backend.prepare_metadata(batch)
                with ctx.forward_batch(batch):
                    actual = model.forward()
                torch.testing.assert_close(actual.float(), expected, rtol=6e-2, atol=4e-2)
            assert runner.replays == 12
        finally:
            runner.destroy_cuda_graphs()
            clear_global_ctx()


def _framed_attention(pool):
    def forward(query, layer, batch):
        from logssms.core import get_global_ctx

        req = batch.reqs[0]
        out_loc = get_global_ctx().page_table[req.table_idx, :req.device_len].long()
        k = pool.k_cache(layer).flatten(0, 1)[out_loc.long()]
        v = pool.v_cache(layer).flatten(0, 1)[out_loc.long()]
        k, v = (tensor.repeat_interleave(2, 1).transpose(0, 1).unsqueeze(0) for tensor in (k, v))
        mask = torch.arange(req.device_len, device=query.device)[None, :] <= batch.positions[:, None]
        result = F.scaled_dot_product_attention(query.transpose(0, 1)[None], k, v, attn_mask=mask)
        return result.squeeze(0).transpose(0, 1)
    return forward
