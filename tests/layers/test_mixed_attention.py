from types import SimpleNamespace

import pytest
import torch

from logssms.core import Batch, Context, Request, set_global_ctx
from logssms.distributed import DistributedInfo, tp_override
from logssms.layers.attention_backend.fi import FlashInferBackend, FIPODMetadata
from logssms.utils.batch import BatchIndices
from tests.utils import clear_global_ctx


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(autouse=True)
def single_rank():
    with tp_override(DistributedInfo(rank=0, size=1)):
        yield


@pytest.mark.parametrize("lengths,histories,decode_histories", [
    ([36, 52, 40], [0, 16, 64], [15, 31]),
    ([1, 31], [47, 96], [0, 255, 63]),
    ([64], [256], [511]),
])
@pytest.mark.parametrize("head_dim", [64, 256])
def test_batch_pod_matches_separate_paged_attention(lengths, histories, decode_histories, head_dim):
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper

    clear_global_ctx()
    device = torch.device("cuda")
    page, max_seq, q_heads, kv_heads = 16, 512, 6, 1
    count = len(lengths) + len(decode_histories)
    shape = (count * max_seq // page, page, kv_heads, head_dim)
    torch.manual_seed(741)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    backend = FlashInferBackend.__new__(FlashInferBackend)
    backend.profile_plan = True
    backend.device = device
    backend.config = SimpleNamespace(head_dim=head_dim)
    backend.qo_head_local, backend.kv_head_local = q_heads, kv_heads
    backend.kvcache = SimpleNamespace(
        k_cache=lambda _: k, v_cache=lambda _: v, dtype=k.dtype, kv_page_size=page,
    )
    backend.float_workspace_buffer = torch.zeros(32 << 20, device=device, dtype=torch.uint8)
    backend.last_event = torch.cuda.Event()
    backend.last_event.record()
    backend.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
        backend.float_workspace_buffer, kv_layout="NHD", backend="fa2",
    )
    backend.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
        backend.float_workspace_buffer, kv_layout="NHD", backend="fa2", use_tensor_cores=True,
    )
    ctx = Context(page)
    table = torch.randperm(shape[0], device=device).reshape(count, -1)
    ctx.page_table = (table[:, :, None] * page + torch.arange(page, device=device)).reshape(count, -1).int()
    set_global_ctx(ctx)

    def part(start, sizes, cached, phase):
        reqs = [
            Request(torch.zeros(begin + size, dtype=torch.int32), start + index,
                    begin, 1, start + index, None, None, begin + size + 1)
            for index, (size, begin) in enumerate(zip(sizes, cached))
        ]
        result = Batch(reqs, phase)
        result.padded_reqs = reqs
        result.prepared_indices = BatchIndices.make(result, device, page)
        return result

    try:
        before_warmup = k.clone(), v.clone()
        dummy = Request(torch.zeros(1, dtype=torch.int32), 0, 0, 1, -1, None, None, 2)
        backend.warmup_mixed(dummy)
        torch.testing.assert_close(k, before_warmup[0], rtol=0, atol=0)
        torch.testing.assert_close(v, before_warmup[1], rtol=0, atol=0)
        prefill = part(0, lengths, histories, "prefill")
        decode = part(len(lengths), [1] * len(decode_histories), decode_histories, "decode")
        batch = Batch(prefill.reqs + decode.reqs, "mixed", num_prefill_reqs=prefill.size)
        batch.mixed_parts = prefill, decode
        backend.prepare_mixed_metadata(batch)
        assert isinstance(batch.attn_metadata, FIPODMetadata)
        q = torch.randn(sum(lengths) + decode.size, q_heads, head_dim, device=device, dtype=k.dtype)
        before = k.clone(), v.clone()
        calls = []
        wrapper = batch.attn_metadata.wrapper
        plan = wrapper.plan
        def plan_once(*args, **kwargs):
            calls.append(True)
            return plan(*args, **kwargs)
        wrapper.plan = plan_once
        for _ in range(2):
            actual = backend.forward(q, 0, batch).clone()
        assert len(calls) == 1
        assert batch.attn_metadata.plan_cpu_ms["workspace_wait"] >= 0
        assert batch.attn_metadata.plan_cpu_ms["plan"] >= 0
        expected = torch.cat((
            backend.forward(q[:sum(lengths)], 0, prefill),
            backend.forward(q[sum(lengths):], 0, decode),
        ))
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(k, before[0], rtol=0, atol=0)
        torch.testing.assert_close(v, before[1], rtol=0, atol=0)
        # Replanning the shared wrapper must not reuse the preceding batch's lengths.
        prefill = part(0, [1] * len(lengths), histories, "prefill")
        batch = Batch(prefill.reqs + decode.reqs, "mixed", num_prefill_reqs=prefill.size)
        batch.mixed_parts = prefill, decode
        backend.prepare_mixed_metadata(batch)
        q = q[:prefill.size + decode.size]
        actual = backend.forward(q, 0, batch)
        expected = torch.cat((
            backend.forward(q[:prefill.size], 0, prefill),
            backend.forward(q[prefill.size:], 0, decode),
        ))
        assert len(calls) == 2
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    finally:
        clear_global_ctx()
