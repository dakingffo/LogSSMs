from types import SimpleNamespace

import pytest
import torch

from logssms.core import Batch, Context, set_global_ctx
from logssms.kernel import qk_norm_rope_store
from logssms.layers.attention import MHAAttentionImpl
from logssms.layers.attention_backend.fi import FIMetadata, FlashInferBackend
from tests.utils import clear_global_ctx


P, D = 128, 4
PAGE, MAX_SEQ = 16, 512
Q_HEADS, KV_HEADS, HEAD_DIM = 4, 2, 64
LENGTHS = [128, 1, 31, 32, 63, 64, 65, 127, 128]


@pytest.fixture
def mixed_fi():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    pytest.importorskip("flashinfer")
    clear_global_ctx()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(731)
    pages_per_req = MAX_SEQ // PAGE
    shape = ((D + 1) * pages_per_req, PAGE, KV_HEADS, HEAD_DIM)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    cache = SimpleNamespace(k_cache=lambda _: k, v_cache=lambda _: v)
    # Use real wrappers/kernels with small scratch, avoiding production's
    # 128 MiB base workspace and unrelated distributed/model initialization.
    backend = FlashInferBackend.__new__(FlashInferBackend)
    backend.device = device
    backend.kvcache = cache
    backend.float_workspace_buffer = torch.zeros(8 << 20, device=device, dtype=torch.uint8)
    backend.last_event = torch.cuda.Event()
    backend.last_event.record()
    ctx = Context(page_size=PAGE)
    ctx.kv_cache = cache
    ctx.attn_backend = backend
    set_global_ctx(ctx)
    # Noncontiguous physical pages detect accidental contiguous-cache assumptions.
    table = torch.randperm(shape[0], generator=torch.Generator().manual_seed(93))
    table = table.reshape(D + 1, pages_per_req)
    try:
        yield SimpleNamespace(ctx=ctx, backend=backend, k=k, v=v, table=table,
                              generator=generator, device=device)
        torch.cuda.synchronize()
        assert torch.cuda.max_memory_allocated() - baseline < 512 << 20
    finally:
        clear_global_ctx()


def _part(env, lengths, histories, request_ids, phase):
    q_indptr = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32)
    seq_lens = torch.tensor([h + n for h, n in zip(histories, lengths)], dtype=torch.int32)
    page_counts = (seq_lens + PAGE - 1) // PAGE
    kv_indptr = torch.tensor([0] + list(page_counts.cumsum(0)), dtype=torch.int32)
    indices = torch.cat([env.table[r, :int(n)] for r, n in zip(request_ids, page_counts)])
    batch = Batch(reqs=[None] * len(lengths), phase=phase)
    batch.padded_reqs = batch.reqs
    batch.input_ids = torch.zeros(sum(lengths), device=env.device, dtype=torch.int32)
    batch.attn_metadata = FIMetadata(
        cu_seqlens_q_cpu=q_indptr,
        cu_seqlens_k_cpu=kv_indptr,
        cu_seqlens_q_gpu=q_indptr.to(env.device),
        indices=indices.to(device=env.device, dtype=torch.int32),
        last_page_len_cpu=(seq_lens - 1) % PAGE + 1,
        num_qo_heads=Q_HEADS, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM,
        page_size=PAGE, pos_encoding_mode="NONE", seq_lens_cpu=seq_lens,
        dtype=torch.bfloat16, wrapper=None,
    )
    return batch


def _parts(env, length, history, decode_histories):
    return (_part(env, [length], [history], [0], "prefill"),
            _part(env, [1] * len(decode_histories), decode_histories,
                  list(range(1, len(decode_histories) + 1)), "decode"))


def _locations(env, length, history, decode_histories):
    locations = torch.full((P + D,), -1, dtype=torch.int32)
    positions = torch.zeros(P + D, dtype=torch.int32)
    positions[:length] = torch.arange(history, history + length)
    for row in range(length):
        position = history + row
        locations[row] = env.table[0, position // PAGE] * PAGE + position % PAGE
    for lane, position in enumerate(decode_histories):
        positions[P + lane] = position
        locations[P + lane] = env.table[lane + 1, position // PAGE] * PAGE + position % PAGE
    return positions.to(env.device), locations.to(env.device)


@pytest.mark.parametrize("use_tensor_cores", [False, True])
def test_mixed_graph_actual_lengths_and_masked_cache(mixed_fi, monkeypatch, use_tensor_cores):
    env = mixed_fi
    backend = env.backend
    backend.use_tensor_cores = use_tensor_cores
    q = torch.randn(P + D, Q_HEADS, HEAD_DIM, device=env.device,
                    dtype=torch.bfloat16, generator=env.generator)
    key = torch.randn(P + D, KV_HEADS, HEAD_DIM, device=env.device,
                      dtype=q.dtype, generator=env.generator)
    value = torch.randn_like(key)
    # Identity RoPE makes the expected cache writes independently exact.
    cos_sin = torch.zeros(MAX_SEQ, HEAD_DIM, device=env.device, dtype=torch.float32)
    cos_sin[:, :HEAD_DIM // 2] = 1
    positions, locations = _locations(env, P, 17, [1, 31, 127, 255])
    captured = backend.make_mixed_capture(*_parts(env, P, 17, [1, 31, 127, 255]), MAX_SEQ)
    parent = Batch(reqs=[None] * (D + 1), phase="mixed")
    parent.mixed_parts = captured
    parent.prefill_token_capacity = P
    layer = MHAAttentionImpl(0)

    def forward():
        prepared = qk_norm_rope_store(q, key, value, positions, cos_sin,
                                      locations, env.k, env.v)
        return layer.forward(prepared)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), env.ctx.forward_batch(parent):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()

    def no_cat(*args, **kwargs):
        raise AssertionError("mixed attention caller must copy into its shared output")

    # Capture the real caller layer, not a test-local split/concatenation.
    with monkeypatch.context() as patch:
        patch.setattr(torch, "cat", no_cat)
        with env.ctx.forward_batch(parent), torch.cuda.graph(graph, stream=stream):
            output = forward()

    histories = [0, 15, 16, 63, 128, 257, 31, 255, 7]
    decode_counts = [4, 1, 3, 2, 4, 1, 2, 3, 4]
    for step, (length, history, count) in enumerate(zip(LENGTHS, histories, decode_counts)):
        decode_histories = [(step * 47 + lane * 71) % 400 for lane in range(count)]
        runtime = _parts(env, length, history, decode_histories)
        new_positions, new_locations = _locations(env, length, history, decode_histories)
        positions.copy_(new_positions)
        locations.copy_(new_locations)
        q.normal_(generator=env.generator)
        key.normal_(generator=env.generator)
        value.normal_(generator=env.generator)
        env.k.normal_(generator=env.generator)
        env.v.normal_(generator=env.generator)
        before_k, before_v = env.k.clone(), env.v.clone()
        expected_k, expected_v = before_k.clone(), before_v.clone()
        active = locations >= 0
        write_indices = locations[active].long()
        expected_k.view(-1, KV_HEADS, HEAD_DIM).index_copy_(0, write_indices, key[active])
        expected_v.view(-1, KV_HEADS, HEAD_DIM).index_copy_(0, write_indices, value[active])

        backend.prepare_mixed_replay(captured, runtime)
        graph.replay()
        # Check the entire allocation: history, padded lanes, unused pages, and
        # page tails must all survive; only actual tokens may update the cache.
        torch.testing.assert_close(env.k, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(env.v, expected_v, rtol=0, atol=0)
        untouched = torch.ones(env.k.shape[0] * PAGE, device=env.device, dtype=torch.bool)
        untouched[write_indices] = False
        assert torch.equal(env.k.flatten(0, 1)[untouched], before_k.flatten(0, 1)[untouched])
        assert torch.equal(env.v.flatten(0, 1)[untouched], before_v.flatten(0, 1)[untouched])

        # Fresh eager plans use actual L/d and share only float scratch with
        # captured wrappers. Alternating eager/graph catches planning aliasing.
        backend.prepare_mixed_metadata(runtime)
        prefill_ref = backend.forward(q[:length], 0, runtime[0])
        decode_ref = backend.forward(q[P:P + count], 0, runtime[1])
        torch.testing.assert_close(output[:length], prefill_ref, rtol=1e-2, atol=1e-2,
                                   msg=f"prefill L={length}, history={history}")
        torch.testing.assert_close(output[P:P + count], decode_ref, rtol=1e-2, atol=1e-2,
                                   msg=f"decode L={length}, d={count}")
        assert torch.count_nonzero(output[length:P]) == 0
        assert torch.isfinite(output).all()
        padded_decode = _part(
            env, [1] * D,
            decode_histories + [decode_histories[0]] * (D - count),
            list(range(1, count + 1)) + [1] * (D - count), "decode",
        )
        backend.prepare_mixed_metadata((runtime[0], padded_decode))
        padded_decode_ref = backend.forward(q[P:], 0, padded_decode)
        torch.testing.assert_close(output[P:], padded_decode_ref, rtol=1e-2, atol=1e-2)
        padded_ref = backend.forward(q[:P], 0, runtime[0])
        torch.testing.assert_close(padded_ref[:length], prefill_ref, rtol=0, atol=0)
        assert torch.count_nonzero(padded_ref[length:]) == 0
        # Eager planning must not overwrite either captured integer plan.
        graph.replay()
        torch.testing.assert_close(output[:length], prefill_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(output[P:], padded_decode_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(env.k, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(env.v, expected_v, rtol=0, atol=0)
