from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from logssms.cache import CacheResources
from logssms.cache.logstate.frame_pool import MHALogStatePool
from logssms.cache.logstate.manager import LogStateCacheManager
from logssms.core import Batch, Request, SamplingParams
from logssms.scheduler.prefill import PendingRequest


PAGE_SIZE = 64
GRANULARITY = 64
MAX_RUNNING_REQ = 4
MAX_SEQ_LEN = 1024


@pytest.fixture(autouse=True)
def _cpu_copy_states():
    with patch("logssms.kernel.linear_attention.logstate.copy_states"):
        yield


def _stub_cpu_kernels(pool: MHALogStatePool) -> None:
    if pool.device.type == "cpu":
        pool.kernels = SimpleNamespace(
            restore=lambda *args, **kwargs: None,
            copy_conv=lambda *args, **kwargs: None,
            copy_states=lambda *args, **kwargs: None,
        )


def make_manager(
    *,
    num_pages: int = 8,
    num_states: int = 8,
    num_work_states: int = MAX_RUNNING_REQ + 1,
    max_seq_len: int = MAX_SEQ_LEN,
    granularity: int = GRANULARITY,
    device: str | torch.device = "cpu",
    enable_merge_adjustment: bool = False,
    enable_hit_materialization: bool = True,
    value_head_ratio: int = 1,
) -> LogStateCacheManager:
    dev = torch.device(device)
    cuda = dev.type == "cuda"
    pool = MHALogStatePool(
        num_frames=(num_pages + 1) + (num_states + num_work_states + 1) * 7,
        page_size=granularity,
        granularity=granularity,
        num_attention_layers=1,
        num_kv_heads=2,
        head_dim=64 if cuda else 4,
        num_linear_layers=1,
        num_key_heads=2,
        num_value_heads=2 * value_head_ratio,
        head_k_dim=32 if cuda else 4,
        head_v_dim=32 if cuda else 4,
        conv_dim=192 if cuda else 16,
        state_len=3,
        device=dev,
        dtype=torch.bfloat16 if cuda else torch.float32,
    )
    _stub_cpu_kernels(pool)
    rows = num_work_states + 1
    resources = CacheResources(
        pool,
        torch.zeros(rows, max_seq_len, dtype=torch.int32, device=dev),
        pool,
        torch.full((rows, max_seq_len), -1, dtype=torch.int32, device=dev),
        torch.full((rows, 1), -1, dtype=torch.int32, device=dev),
    )
    return LogStateCacheManager(
        num_pages=pool.allocator.num_frames - 1,
        num_states=num_states,
        num_work_states=num_work_states,
        page_size=granularity,
        resources=resources,
        granularity=granularity,
        enable_merge_adjustment=enable_merge_adjustment,
        enable_hit_materialization=enable_hit_materialization,
    )


def new_req(
    manager: LogStateCacheManager,
    tokens: torch.Tensor,
    *,
    uid: int = 0,
    table_idx: int = 0,
    max_tokens: int = 16,
    session_id: str | None = None,
) -> Request:
    pending = PendingRequest(
        uid=uid,
        input_ids=tokens.to(torch.int32),
        sampling_params=SamplingParams(max_tokens=max_tokens),
        cache_session_id=session_id,
    )
    adm = manager.begin_prefill_admission((), ())
    res = manager.admit_prefill(
        pending, adm, chunk_budget=lambda rem: rem, allocate_table=lambda: table_idx,
    )
    assert not isinstance(res, str), f"admission rejected: {res}"
    return Request(
        input_ids=tokens.to(torch.int32),
        table_idx=res.table_idx,
        cached_len=res.cached_len,
        output_len=max_tokens,
        uid=uid,
        sampling_params=SamplingParams(max_tokens=max_tokens),
        cache_handle=res.handle,
        max_device_len=len(tokens) + max_tokens,
    )


def run_prefill_and_cache(
    manager: LogStateCacheManager,
    tokens: torch.Tensor,
    *,
    uid: int = 0,
    table_idx: int = 0,
    decode_steps: int = 0,
    session_id: str | None = None,
) -> Request:
    req = new_req(
        manager, tokens, uid=uid, table_idx=table_idx,
        max_tokens=max(1, decode_steps + 1), session_id=session_id,
    )
    batch = Batch([req], "prefill")
    manager.allocate(batch.reqs, phase="prefill")
    manager.prepare_batch(batch)
    manager.after_forward(batch)
    req.cached_len = req.device_len
    if decode_steps > 0:
        manager.cache_req(req, finished=False)
        for step in range(decode_steps):
            req.append_host(torch.tensor([10_000 + uid * 100 + step], dtype=torch.int32))
            req.device_len += 1
            db = Batch([req], "decode")
            db.padded_reqs = db.reqs
            manager.allocate(db.reqs, phase="decode")
            manager.prepare_batch(db)
            manager.after_forward(db)
            req.cached_len = req.device_len
    manager.cache_req(req, finished=True)
    return req


def test_prefill_decode_prefix_hit_and_integrity():
    manager = make_manager(granularity=32, num_pages=64)
    base = torch.arange(128, dtype=torch.int32)
    run_prefill_and_cache(manager, base, uid=0, table_idx=0, decode_steps=40)
    manager.check_integrity()

    second_tokens = torch.cat((base, torch.arange(500, 564, dtype=torch.int32)))
    req2 = new_req(manager, second_tokens, uid=1, table_idx=1)
    assert req2.cached_len == len(base)
    b = Batch([req2], "prefill")
    manager.allocate([req2], phase="prefill")
    manager.prepare_batch(b)
    assert b.cache_metadata is not None
    assert manager.work_state_table[1, 0].item() >= 0
    req2.cached_len = req2.device_len
    manager.cache_req(req2, finished=True)
    manager.check_integrity()


def test_chunked_prefill_advances_across_chunks_and_caches_full_prefix():
    manager = make_manager(granularity=32, num_pages=64)
    tokens = torch.arange(160, dtype=torch.int32)
    pending = PendingRequest(0, tokens, SamplingParams(max_tokens=4))
    adm = manager.begin_prefill_admission((), ())
    res = manager.admit_prefill(
        pending, adm, chunk_budget=lambda rem: min(rem, 64), allocate_table=lambda: 0,
    )
    assert not isinstance(res, str)
    assert res.chunk_size == 64
    req = Request(
        input_ids=tokens,
        table_idx=0,
        cached_len=0,
        output_len=4,
        uid=0,
        sampling_params=SamplingParams(max_tokens=4),
        cache_handle=res.handle,
        max_device_len=len(tokens) + 4,
    )
    for end in (64, 128, 160):
        req.device_len = end
        b = Batch([req], "prefill")
        manager.allocate([req], phase="prefill")
        manager.prepare_batch(b)
        req.cached_len = end
        manager.cache_req(req, finished=(end == 160))
    manager.check_integrity()

    probe = new_req(manager, torch.cat((tokens, torch.tensor([999], dtype=torch.int32))), uid=1, table_idx=1)
    assert probe.cached_len == 160
    manager.free_to_cached(probe)
    manager.cache_req(probe, finished=True)
    manager.check_integrity()


@pytest.mark.parametrize("enable_merge_adjustment", [False, True])
def test_eviction_and_checkpoint_merge_under_memory_pressure(enable_merge_adjustment):
    manager = make_manager(
        granularity=32,
        num_pages=8,
        num_states=4,
        enable_merge_adjustment=enable_merge_adjustment,
        enable_hit_materialization=True,
    )
    for idx in range(18):
        tokens = torch.arange(idx * 1000, idx * 1000 + 128, dtype=torch.int32)
        run_prefill_and_cache(
            manager, tokens, uid=idx, table_idx=idx % 4, decode_steps=8, session_id=f"s{idx % 3}",
        )
    manager.check_integrity()
    stats = manager.eviction_stats()
    assert stats["eviction_rounds"] > 0
    assert stats["evicted_kv_tokens"] > 0 or stats["pressure_dropped_checkpoints"] > 0
    layout = manager.layout_snapshot()
    assert layout["checkpoint"] + layout["log"] > 0


def test_abort_releases_uncommitted_frames_cleanly():
    manager = make_manager(granularity=32, num_pages=64)
    initial_free = manager._allocator.free_frames
    tokens = torch.arange(96, dtype=torch.int32)
    req = new_req(manager, tokens, uid=0, table_idx=0)
    b = Batch([req], "prefill")
    manager.allocate([req], phase="prefill")
    manager.prepare_batch(b)
    assert manager._allocator.free_frames < initial_free
    manager.free_to_cached(req)
    manager.cache_req(req, finished=True)
    manager.check_integrity()
    assert manager._allocator.free_frames == initial_free
