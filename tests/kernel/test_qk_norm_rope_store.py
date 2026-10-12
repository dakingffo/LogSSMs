from types import SimpleNamespace

import pytest
import torch

from logssms.kernel import qk_norm_rope_store


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _reference(x, weight, eps, zero_centered, positions, cache):
    values = x.float()
    if weight is not None:
        weights = weight.float() + (1.0 if zero_centered else 0.0)
        values = (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
                  * weights).to(x.dtype).float()
    half = cache.shape[1] // 2
    cos = cache[positions.long(), :half, None].transpose(1, 2)
    sin = cache[positions.long(), half:, None].transpose(1, 2)
    result = values.clone()
    result[..., :half] = values[..., :half] * cos - values[..., half:2 * half] * sin
    result[..., half:2 * half] = values[..., half:2 * half] * cos + values[..., :half] * sin
    return result.to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("norm", ["none", "rms", "gemma"])
@pytest.mark.parametrize("dims", [(64, 64), (128, 64), (256, 64), (512, 128)])
@pytest.mark.parametrize("layout", ["radix", "frame"])
def test_qk_norm_rope_store_matches_reference(dtype, norm, dims, layout):
    from logssms.layers.rotary import RotaryEmbedding

    torch.manual_seed(31)
    tokens, q_heads, kv_heads = 19, 4, 2
    head_dim, rotary_dim = dims
    raw = torch.randn(tokens, (2 * q_heads + 2 * kv_heads) * head_dim,
                      device="cuda", dtype=dtype)
    q_gate, key, value = raw.split(
        [2 * q_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim], -1)
    query, gate = q_gate.view(tokens, q_heads, 2 * head_dim).chunk(2, -1)
    key, value = key.view(tokens, kv_heads, head_dim), value.view(tokens, kv_heads, head_dim)
    original = raw.clone()
    with torch.device("cuda"):
        rotary = RotaryEmbedding(head_dim, rotary_dim, 1024, 10000.0)
    q_weight = torch.randn(head_dim, device="cuda", dtype=dtype) * 0.1 if norm != "none" else None
    k_weight = torch.randn(head_dim, device="cuda", dtype=dtype) * 0.1 if norm != "none" else None
    eps, zero_centered = 1e-6, norm == "gemma"
    positions = torch.randint(0, 1024, (tokens,), device="cuda", dtype=torch.int64)
    positions[0] = 0
    locations = torch.randperm(64, device="cuda")[:tokens].to(torch.int32)
    locations[-1] = -1
    if layout == "radix":
        backing = torch.full((2, 4, 16, kv_heads, head_dim), 7.0, device="cuda", dtype=dtype)
        k_cache, v_cache = backing[0], backing[1]
    else:
        backing = torch.full((4, 6, 16, kv_heads, head_dim), 7.0, device="cuda", dtype=dtype)
        k_cache, v_cache = backing[:, 2], backing[:, 3]
    expected_backing = backing.clone()
    if layout == "radix":
        expected_k, expected_v = expected_backing[0], expected_backing[1]
    else:
        expected_k, expected_v = expected_backing[:, 2], expected_backing[:, 3]
    valid = locations >= 0
    pages, offsets = locations[valid].long() // 16, locations[valid].long() % 16
    expected_k[pages, offsets] = _reference(
        key, k_weight, eps, zero_centered, positions, rotary._cos_sin_cache)[valid]
    expected_v[pages, offsets] = value[valid]
    expected_q = _reference(query, q_weight, eps, zero_centered, positions, rotary._cos_sin_cache)
    actual = qk_norm_rope_store(
        query, key, value, positions, rotary._cos_sin_cache, locations, k_cache, v_cache,
        q_weight=q_weight, k_weight=k_weight, q_eps=eps, k_eps=eps,
        q_zero_centered=zero_centered, k_zero_centered=zero_centered,
    )
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(actual, expected_q, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(backing, expected_backing, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(raw, original, rtol=0, atol=0)
    assert actual.is_contiguous() and gate.stride(-2) == head_dim * 2


@pytest.mark.parametrize("layout", ["radix", "frame"])
@pytest.mark.parametrize("page_size", [1, 16, 32, 64, 128])
def test_qk_norm_rope_store_graph_replay_updates_mapping(layout, page_size):
    from logssms.cache.kv_pool.mha_pool import MHAKVCachePool
    from logssms.cache.logstate_pool.mha_pool import MHALogStatePool
    from logssms.distributed import DistributedInfo, tp_override
    from logssms.layers.norm import RMSNorm
    from logssms.layers.rotary import RotaryEmbedding

    if layout == "frame" and page_size not in (16, 32, 64):
        pytest.skip("LogState granularity is 16/32/64")
    dtype, device = torch.bfloat16, torch.device("cuda")
    torch.manual_seed(7)
    if layout == "frame":
        pool = MHALogStatePool(
            num_frames=32, page_size=64, granularity=page_size,
            num_attention_layers=2, num_kv_heads=2, head_dim=256,
            num_linear_layers=2, num_key_heads=2, num_value_heads=2,
            head_k_dim=128, head_v_dim=128, conv_dim=20, state_len=3,
            device=device, dtype=dtype,
        )
        backing = pool.storage.view(dtype)
    else:
        with tp_override(DistributedInfo(0, 1)):
            pool = MHAKVCachePool(2, 2, 256, 32, page_size, dtype, device)
        backing = pool._kv_buffer
    backing.fill_(7.0)
    query = torch.randn(4, 4, 256, device=device, dtype=dtype)
    key = torch.randn(4, 2, 256, device=device, dtype=dtype)
    value = torch.randn_like(key)
    positions = torch.zeros(4, device=device, dtype=torch.int32)
    locations = torch.arange(4, device=device, dtype=torch.int64)
    norm = RMSNorm(256, 1e-6, True)
    norm.weight = torch.randn(256, device=device, dtype=dtype) * 0.1
    with torch.device(device):
        rotary = RotaryEmbedding(256, 64, 1024, 10000.0)

    def forward():
        return qk_norm_rope_store(
            query, key, value, positions, rotary._cos_sin_cache,
            locations, pool.k_cache(1), pool.v_cache(1),
            q_weight=norm.weight, k_weight=norm.weight,
            q_zero_centered=True, k_zero_centered=True,
        )

    forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = forward()
    for step in range(3):
        query.normal_()
        key.normal_()
        value.normal_()
        positions.copy_(torch.arange(4, device=device) + 17 * step)
        locations.copy_(torch.arange(4, device=device) + 4 * step)
        expected_q, expected_k = rotary.forward(
            positions, norm.forward(query), norm.forward(key))
        expected_backing = backing.clone()
        graph.replay()
        torch.cuda.synchronize()
        actual_backing = backing.clone()
        backing.copy_(expected_backing)
        pool.store_kv(expected_k.flatten(-2), value.flatten(-2), locations, 1)
        torch.testing.assert_close(actual, expected_q, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(actual_backing, backing, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_rotary_attention_fused_path_matches_projection(monkeypatch, gated, normalize):
    import logssms.layers.attention as attention
    from logssms.cache.kv_pool.mha_pool import MHAKVCachePool
    from logssms.distributed import DistributedInfo, tp_override

    if gated and not normalize:
        pytest.skip("Gated attention always normalizes QK")
    torch.manual_seed(13)
    with tp_override(DistributedInfo(0, 1)), torch.device("cuda"):
        common = dict(hidden_size=128, head_dim=64, num_qo_heads=4, num_kv_heads=2,
                      layer_id=1, rotary_dim=32, max_position=512, rope_base=10000.0,
                      rope_scaling=None)
        if gated:
            module = attention.GatedRotaryAttention(**common, qk_norm_eps=1e-6,
                                                    zero_centered_norm=True)
        else:
            module = attention.RotaryAttention(**common, qk_norm_eps=1e-6 if normalize else None)
        pool = MHAKVCachePool(2, 2, 64, 4, 16, torch.bfloat16, torch.device("cuda"))
    for name, weight in module.state_dict().items():
        replacement = torch.randn_like(weight, dtype=torch.bfloat16) * 0.1
        owner = module
        path = name.split(".")
        for segment in path[:-1]:
            owner = getattr(owner, segment)
        setattr(owner, path[-1], replacement)
    hidden = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(7, device="cuda", dtype=torch.int32) + 5
    locations = torch.arange(7, device="cuda", dtype=torch.int32) + 19
    batch = SimpleNamespace(positions=positions, out_loc=locations)

    class Backend:
        def forward(self, q, layer_id, received_batch):
            assert layer_id == 1 and received_batch is batch
            return q

    ctx = SimpleNamespace(batch=batch, kv_cache=pool, attn_backend=Backend())
    monkeypatch.setattr(attention, "get_global_ctx", lambda: ctx)
    expected = module.proj(hidden, positions)
    raw = module._project(hidden)
    if gated:
        query, gate, key, value = raw
        expected_q, expected_gate, expected_k, expected_v = expected
        torch.testing.assert_close(gate, expected_gate)
    else:
        query, key, value = raw
        expected_q, expected_k, expected_v = expected
    actual = module.attend(query, key, value, positions)
    torch.testing.assert_close(actual, expected_q, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(pool.k_cache(1)[1, 3:10], expected_k, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(pool.v_cache(1)[1, 3:10], expected_v, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["radix", "frame"])
def test_fused_qkv_with_flashinfer_prefill_decode_verify(monkeypatch, layout):
    import logssms.layers.attention as attention
    import logssms.layers.attention_backend.fi as fi
    from logssms.cache.kv_pool.mha_pool import MHAKVCachePool
    from logssms.cache.logstate_pool.mha_pool import MHALogStatePool
    from logssms.core import Batch, Request, SamplingParams
    from logssms.distributed import DistributedInfo, tp_override

    torch.manual_seed(23)
    device, dtype = torch.device("cuda"), torch.bfloat16
    if layout == "frame":
        pool = MHALogStatePool(
            num_frames=32, page_size=64, granularity=32,
            num_attention_layers=2, num_kv_heads=2, head_dim=256,
            num_linear_layers=2, num_key_heads=2, num_value_heads=2,
            head_k_dim=128, head_v_dim=128, conv_dim=20, state_len=3,
            device=device, dtype=dtype,
        )
    else:
        with tp_override(DistributedInfo(0, 1)):
            pool = MHAKVCachePool(2, 2, 256, 8, 32, dtype, device)
    ctx = SimpleNamespace(
        kv_cache=pool,
        page_table=torch.arange(64, device=device, dtype=torch.int32).view(1, 64),
    )
    monkeypatch.setattr(attention, "get_global_ctx", lambda: ctx)
    monkeypatch.setattr(fi, "get_global_ctx", lambda: ctx)
    with tp_override(DistributedInfo(0, 1)), torch.device(device):
        module = attention.GatedRotaryAttention(
            hidden_size=128, head_dim=256, num_qo_heads=4, num_kv_heads=2,
            layer_id=1, rotary_dim=64, max_position=512, rope_base=10000.0,
            rope_scaling=None, qk_norm_eps=1e-6, zero_centered_norm=True,
        )
        ctx.attn_backend = fi.FlashInferBackend(SimpleNamespace(
            num_qo_heads=4, num_kv_heads=2, head_dim=256))
    module.qkv_proj.weight = torch.randn_like(module.qkv_proj.weight, dtype=dtype) * 0.1
    module.q_norm.weight = torch.randn(256, device=device, dtype=dtype) * 0.1
    module.k_norm.weight = torch.randn_like(module.q_norm.weight) * 0.1
    cached_len = 0
    for phase, tokens in (("prefill", 5), ("decode", 1), ("verify", 3)):
        request = Request(
            input_ids=torch.zeros(cached_len + tokens, dtype=torch.int32),
            table_idx=0, cached_len=cached_len, output_len=4, uid=0,
            sampling_params=SamplingParams(), cache_handle=None,
            max_device_len=cached_len + tokens + 4,
        )
        batch = Batch([request], phase)
        batch.padded_reqs = batch.reqs
        batch.positions = torch.arange(cached_len, cached_len + tokens, device=device, dtype=torch.int32)
        batch.out_loc = batch.positions.clone()
        ctx.batch = batch
        ctx.attn_backend.prepare_metadata(batch)
        hidden = torch.randn(tokens, 128, device=device, dtype=dtype)
        q, _, k, v = module.proj(hidden, batch.positions)
        pool.store_kv(k.flatten(-2), v.flatten(-2), batch.out_loc, 1)
        expected = ctx.attn_backend.forward(q, 1, batch).clone()
        old_prefix = pool.k_cache(1)[0, :cached_len].clone()
        q, _, k, v = module._project(hidden)
        actual = module.attend(q, k, v, batch.positions)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(pool.k_cache(1)[0, :cached_len], old_prefix, rtol=0, atol=0)
        cached_len += tokens
