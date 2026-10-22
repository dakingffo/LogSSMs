from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import torch
import pytest

from logssms.engine import prefill_warmup


def test_complete_prefill_forward_after_capture(monkeypatch):
    active = []
    forwarded = []
    prepared = []
    released = []
    runner = SimpleNamespace(buckets=[1, 2, 4, 8], replays=5, padded_tokens=7)

    @contextmanager
    def storage(engine, lengths):
        active.append(tuple(lengths))
        try:
            yield
        finally:
            released.append(active.pop())

    @contextmanager
    def forward_batch(batch):
        assert batch.is_prefill
        assert len(batch.input_ids) == sum(active[-1])
        assert len(batch.sample_indices) == len(active[-1])
        yield

    def forward():
        forwarded.append(active[-1])
        runner.replays += 1
        runner.padded_tokens += 1

    monkeypatch.setattr(prefill_warmup, "prefill_storage", storage)
    monkeypatch.setattr(prefill_warmup.logger, "info_rank0", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(prefill_warmup, "make_batch_mapping", lambda packet, *_args, **_kw: (
        torch.arange(packet.num_tokens),
        (torch.zeros(packet.num_tokens), torch.zeros(packet.num_tokens)),
        torch.arange(packet.num_tokens),
    ))
    engine = SimpleNamespace(
        piecewise_graph_runner=runner, max_seq_len=8, stream=None, device=torch.device("cpu"),
        config=SimpleNamespace(model_config=SimpleNamespace(is_hybrid=True),
                               cache_config=SimpleNamespace(prefill_alignment=32), max_running_req=4),
        ctx=SimpleNamespace(page_table=None, forward_batch=forward_batch),
        kv_cache=SimpleNamespace(kv_page_size=32),
        attn_backend=SimpleNamespace(uses_paged_indices=True, prepare_metadata=lambda b: prepared.append(b)),
        linear_attn_backend=SimpleNamespace(prepare_metadata=lambda b: None),
        model=SimpleNamespace(forward=forward),
    )
    prefill_warmup.warmup_prefill(engine)
    assert {(1,), (2,), (4,), (8,)} <= set(forwarded)
    assert {len(case) for case in forwarded} == {1, 2, 3, 4}
    assert len(prepared) == len(forwarded)
    assert forwarded == released
    assert (runner.replays, runner.padded_tokens) == (5, 7)
    assert not active


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("fail", [False, True])
def test_prefill_storage_releases_frames_and_restores_tables(fail):
    from logssms.cache.logstate_pool.mha_pool import MHALogStatePool

    device = torch.device("cuda")
    pool = MHALogStatePool(
        num_frames=128, page_size=64, granularity=32,
        num_attention_layers=1, num_kv_heads=1, head_dim=64,
        num_linear_layers=1, num_key_heads=2, num_value_heads=2,
        head_k_dim=32, head_v_dim=32, conv_dim=192, state_len=3,
        device=device, dtype=torch.bfloat16, recurrent_type="gdn",
    )
    tables = (torch.full((4, 128), -1, dtype=torch.int32, device=device),
              torch.full((4, 128), -1, dtype=torch.int32, device=device),
              torch.full((4, 1), -1, dtype=torch.int32, device=device))
    ctx = SimpleNamespace(page_table=tables[0], state_table=tables[1],
                          work_state_table=tables[2], linear_state=pool)
    engine = SimpleNamespace(ctx=ctx, kv_cache=pool, device=device,
                             config=SimpleNamespace(page_size=64))
    available = pool.allocator.free_frames
    try:
        with prefill_warmup.prefill_storage(engine, (65, 33)):
            mappings = torch.cat((ctx.page_table[0, :65], ctx.page_table[1, :33]))
            assert len(mappings.unique()) == 98
            assert pool.allocator.free_frames < available
            if fail:
                raise ValueError("test cleanup")
    except ValueError:
        assert fail
    assert pool.allocator.free_frames == available
    assert ctx.page_table is tables[0]
    assert ctx.state_table is tables[1]
    assert ctx.work_state_table is tables[2]
    assert all(bool((table == -1).all()) for table in tables)
