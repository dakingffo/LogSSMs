from types import SimpleNamespace

import pytest
import torch

from logssms.core import Batch, Context, Request, set_global_ctx
from logssms.layers.attention_backend.fi import FlashInferBackend
from logssms.utils.batch import BatchIndices
from tests.utils import clear_global_ctx


def test_plan_waits_only_for_the_same_wrapper(monkeypatch):
    class Event:
        def __init__(self):
            self.waits = self.records = 0

        def synchronize(self):
            self.waits += 1

        def record(self):
            self.records += 1

    class Wrapper:
        def plan(self, **kwargs):
            pass

    monkeypatch.setattr(torch.cuda, "Event", Event)
    backend = FlashInferBackend.__new__(FlashInferBackend)
    backend.profile_plan = False
    wrappers = [Wrapper(), Wrapper()]
    names = ("cu_seqlens_q_cpu", "cu_seqlens_k_cpu", "indices", "last_page_len_cpu",
             "num_qo_heads", "num_kv_heads", "head_dim", "page_size", "pos_encoding_mode",
             "seq_lens_cpu", "dtype")

    def plan(wrapper):
        metadata = SimpleNamespace(initialized=False, wrapper=wrapper,
                                   **dict.fromkeys(names))
        backend._initialize_metadata_once(metadata)
        backend._initialize_metadata_once(metadata)

    for wrapper in wrappers:
        plan(wrapper)
    assert all(event.waits == 0 and event.records == 1
               for event in backend._plan_events.values())
    plan(wrappers[0])
    assert backend._plan_events[wrappers[0]].waits == 1
    assert backend._plan_events[wrappers[1]].waits == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("head_dim", [64, 256])
@pytest.mark.parametrize("decodes", [1, 3])
@pytest.mark.parametrize("prefills", [1, 2, 4, 8])
@pytest.mark.parametrize("capacity", [16, 32, 64, 128])
def test_mixed_graph_matches_separate_attention(head_dim, decodes, prefills, capacity):
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper

    clear_global_ctx()
    torch.manual_seed(741)
    device = torch.device("cuda")
    page, q_heads, kv_heads = 16, 6, 1
    shape = ((decodes + prefills) * 512 // page, page, kv_heads, head_dim)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    backend = FlashInferBackend.__new__(FlashInferBackend)
    backend.profile_plan = True
    backend.device = device
    backend.config = SimpleNamespace(head_dim=head_dim)
    backend.qo_head_local, backend.kv_head_local = q_heads, kv_heads
    backend.use_tensor_cores = True
    backend.kvcache = SimpleNamespace(k_cache=lambda _: k, v_cache=lambda _: v,
                                     dtype=k.dtype, kv_page_size=page)
    backend.float_workspace_buffer = torch.zeros(32 << 20, device=device, dtype=torch.uint8)
    backend.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(backend.float_workspace_buffer, kv_layout="NHD", backend="fa2")
    backend.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(backend.float_workspace_buffer, kv_layout="NHD", backend="fa2", use_tensor_cores=True)
    ctx = Context(page)
    table = torch.randperm(shape[0], device=device).reshape(decodes + prefills, -1)
    ctx.page_table = (table[:, :, None] * page + torch.arange(page, device=device)).reshape(decodes + prefills, -1).int()
    set_global_ctx(ctx)

    def batch(length):
        lengths = [length // prefills + (i < length % prefills) for i in range(prefills)]
        reqs = [Request(torch.zeros(64 + n, dtype=torch.int32), i, 64, 1, i, None, None, 512)
                for i, n in enumerate(lengths)]
        reqs += [Request(torch.zeros(18 + i, dtype=torch.int32), i + prefills, 17 + i, 1, i + prefills, None, None, 512) for i in range(decodes)]
        result = Batch(reqs, "mixed", num_prefill_reqs=prefills)
        result.prefill_token_capacity = capacity
        result.padded_reqs = reqs
        result.prepared_indices = BatchIndices.make_mixed(result, device, page)[0]
        backend.prepare_mixed_metadata(result)
        return result

    graph = None
    try:
        static = batch(capacity)
        backend.make_mixed_capture(static, 512)
        q = torch.randn(capacity + decodes, q_heads, head_dim, device=device, dtype=k.dtype)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            backend.forward(q, 0, static)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = backend.forward(q, 0, static)
        before = k.clone(), v.clone()
        for length in (capacity, 1, 31, 32, 63, 17, capacity):
            length = min(capacity, max(length, prefills))
            runtime = batch(length)
            assert runtime.attn_metadata.prefill.wrapper is not runtime.attn_metadata.decode.wrapper
            expected_indices = torch.cat((
                runtime.prepared_indices.cu_seqlens_q_gpu[1:prefills + 1] - 1,
                torch.arange(decodes, device=device, dtype=torch.int32) + capacity,
            ))
            torch.testing.assert_close(runtime.attn_metadata.get_last_indices(prefills + decodes),
                                       expected_indices, check_dtype=False)
            q.normal_()
            backend.prepare_mixed_replay(static, runtime)
            graph.replay()
            actual = torch.cat((output[:length], output[capacity:])).clone()
            p, d = Batch(runtime.reqs[:prefills], "prefill"), Batch(runtime.reqs[prefills:], "decode")
            for part in (p, d):
                part.padded_reqs = part.reqs
                backend.prepare_metadata(part)
            expected = torch.cat((backend.forward(q[:length], 0, p), backend.forward(q[capacity:], 0, d)))
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
            assert runtime.attn_metadata.plan_cpu_ms["workspace_wait"] >= 0
        torch.testing.assert_close(k, before[0], rtol=0, atol=0)
        torch.testing.assert_close(v, before[1], rtol=0, atol=0)
    finally:
        if graph is not None:
            graph.reset()
        clear_global_ctx()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("prefills,capacity", [(1,32), (4,128)])
def test_24_slot_mixed_graph_replays_prefill_and_decode(prefills, capacity):
    test_mixed_graph_matches_separate_attention(256, 24-prefills, prefills, capacity)
