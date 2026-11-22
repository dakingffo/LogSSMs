import pytest
from types import MethodType, SimpleNamespace

import torch

from logssms.cache.base import BaseCacheManager, PrefillAdmission, PrefillAllocation
from logssms.core import ChunkedRequest, Request, SamplingParams
from logssms.scheduler.prefill import PendingRequest, PrefillAdder, PrefillManager


class FakeAdder:
    def try_add_one(self, pending: PendingRequest):
        req = Request(
            input_ids=pending.input_ids,
            table_idx=pending.uid,
            cached_len=0,
            output_len=pending.output_len,
            uid=pending.uid,
            cache_handle=None,
            sampling_params=pending.sampling_params,
            max_device_len=pending.input_len + pending.output_len,
        )
        return req


def test_prefill_overlap_carries_previous_batch_reservation_into_adder():
    manager = PrefillManager(torch.empty((2, 16), dtype=torch.int32))
    params = SamplingParams(max_tokens=8)
    manager.pending_list = [
        PendingRequest(0, torch.arange(4, dtype=torch.int32), params),
        PendingRequest(1, torch.arange(4, dtype=torch.int32), params),
    ]

    first = manager.schedule_next_batch(FakeAdder())
    assert first is not None
    assert manager.inflight_reqs
    cache = make_cache_stub(_reservation_size=lambda n, **_: n)
    admission = cache.begin_prefill_admission((), tuple(manager.inflight_reqs.values()))
    assert admission.reserved == 16

    manager.pending_list.append(
        PendingRequest(2, torch.arange(4, dtype=torch.int32), params)
    )
    second = manager.schedule_next_batch(FakeAdder())
    assert second is not None
    assert [req.uid for req in second.reqs] == [2]
    assert set(manager.inflight_reqs) == {2}


def make_cache_stub(**methods):
    stub = SimpleNamespace(_schedule_profile_step=None, **methods)
    for name in ("begin_prefill_admission", "_prefill_match_transaction", "admit_prefill", "_acquire_prefill",
                 "_account_prefill", "record_schedule_step"):
        setattr(stub, name, MethodType(getattr(BaseCacheManager, name), stub))
    return stub


def test_cold_prefill_initializes_cache_handle() -> None:
    calls = []
    handle = SimpleNamespace(cached_len=0)
    cache_manager = make_cache_stub(
        page_size=64,
        _reservation_size=lambda tokens, **_: (tokens + 63) // 64 * 64,
        match_req=lambda _req: SimpleNamespace(cuda_handle=handle),
        plan_prefill_chunk=lambda remain, budget: min(remain, budget),
        _try_pin_prefill=lambda *_args, **_kwargs: True,
        record_match=lambda *_args, **_kwargs: None,
        init_with_handle=lambda index, value: calls.append((index, value)),
    )
    pending = PendingRequest(
        7, torch.arange(16, dtype=torch.int32), SamplingParams(max_tokens=8)
    )

    result = cache_manager.admit_prefill(pending, PrefillAdmission(), lambda _: 64, lambda: 3)
    assert result == PrefillAllocation(handle, 3, 0, 16)
    assert calls == [(3, handle)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Pinned transfer requires CUDA")
def test_cache_match_snapshot_is_taken_only_after_successful_admission():
    from logssms.cache.base import BaseCacheManager

    matches = iter([8, 4])
    admissions = iter([False, True])
    manager = make_cache_stub(
        _match_profile_enabled=False,
        _reservation_size=lambda tokens, **_: tokens,
        match_req=lambda _: SimpleNamespace(cuda_handle=SimpleNamespace(cached_len=next(matches))),
        plan_prefill_chunk=lambda remain, budget: min(remain, budget),
        _try_pin_prefill=lambda *_args, **_kwargs: next(admissions),
        admission_capacity=64,
        init_with_handle=lambda *_args: None,
        record_prefill_wait=lambda *_args: None,
    )
    manager.record_match = lambda req, match, **kwargs: BaseCacheManager.record_match(
        manager, req, match, **kwargs
    )
    table = SimpleNamespace(
        available_size=1, allocate=lambda: 0,
        token_pool=torch.empty((1, 64), dtype=torch.int32),
    )
    pending = PendingRequest(0, torch.arange(20, dtype=torch.int32), SamplingParams(max_tokens=8))
    adder = PrefillAdder(4, manager, table, PrefillAdmission())
    assert adder.try_add_one(pending) is None
    assert pending.prefix_cached_len == 0
    assert adder.try_add_one(pending) is not None
    assert pending.prefix_cached_len == 4


class FakeChunkAdder:
    def try_add_one(self, pending: PendingRequest):
        cached_len = 0 if pending.chunked_req is None else pending.chunked_req.device_len
        device_len = min(cached_len + 4, pending.input_len)
        cls = ChunkedRequest if device_len < pending.input_len else Request
        req = cls(
            input_ids=pending.input_ids[:device_len],
            table_idx=pending.uid,
            cached_len=cached_len,
            output_len=pending.output_len,
            uid=pending.uid,
            cache_handle=None,
            sampling_params=pending.sampling_params,
            max_device_len=pending.input_len + pending.output_len,
        )
        return req


def test_same_request_can_schedule_next_chunk_before_previous_commit():
    manager = PrefillManager(torch.empty((1, 32), dtype=torch.int32))
    params = SamplingParams(max_tokens=8)
    manager.pending_list = [
        PendingRequest(0, torch.arange(12, dtype=torch.int32), params)
    ]

    first = manager.schedule_next_batch(FakeChunkAdder())
    assert first is not None

    output_mapping = (
        torch.tensor([], dtype=torch.int64),
        torch.tensor([], dtype=torch.int64),
    )
    output = SimpleNamespace(next_tokens_gpu=torch.tensor([], dtype=torch.int32))
    manager.advance_for_next_schedule(None, (first, None, None, output_mapping), output)
    second = manager.schedule_next_batch(FakeChunkAdder())

    assert second is not None
    assert first.reqs[0].cached_len == 4
    assert first.reqs[0].device_len == 4
    assert second.reqs[0].cached_len == 4
    assert second.reqs[0].device_len == 8
    # The newer frontier replaces, rather than adds to, the older reservation.
    assert manager.inflight_reqs[0] is second.reqs[0]
    assert manager.inflight_reqs[0].remain_len == 12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Pinned transfer requires CUDA")
def test_prefill_adder_continues_from_inflight_device_frontier():
    params = SamplingParams(max_tokens=8)
    pending = PendingRequest(0, torch.arange(12, dtype=torch.int32), params)
    previous = ChunkedRequest(
        input_ids=pending.input_ids[:4],
        table_idx=0,
        cached_len=0,
        output_len=8,
        uid=0,
        cache_handle=None,
        sampling_params=params,
        max_device_len=pending.input_len + pending.output_len,
    )
    previous.complete_to_device_len()
    pending.chunked_req = previous
    table_manager = SimpleNamespace(
        token_pool=torch.empty((1, 32), dtype=torch.int32)
    )
    cache_manager = make_cache_stub(
        page_size=1,
        _reservation_size=lambda tokens, **_: tokens,
        record_prefill_wait=lambda _wait_ms: None,
        plan_prefill_chunk=lambda remain_len, token_budget: min(
            remain_len, token_budget
        ),
    )
    admission = cache_manager.begin_prefill_admission((), (previous,))
    adder = PrefillAdder(4, cache_manager, table_manager, admission)

    current = adder.try_add_one(pending)

    assert current is not None
    assert current.cached_len == previous.device_len == 4
    assert current.device_len == 8
    assert admission.reserved == previous.remain_len


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Pinned transfer requires CUDA")
def test_hybrid_chunked_prefill_is_state_checkpoint_aligned():
    params = SamplingParams(max_tokens=8)
    pending = PendingRequest(0, torch.arange(200, dtype=torch.int32), params)
    table_manager = SimpleNamespace(
        token_pool=torch.empty((1, 256), dtype=torch.int32)
    )
    cache_manager = SimpleNamespace(
        page_size=64,
        record_prefill_wait=lambda _wait_ms: None,
        plan_prefill_chunk=lambda remain_len, token_budget: 64,
    )
    adder = PrefillAdder(100, cache_manager, table_manager)

    chunk_size = cache_manager.plan_prefill_chunk(pending.input_len, adder.token_budget)
    current = adder._add_one_req(
        pending, None, table_idx=0, cached_len=0, chunk_size=chunk_size
    )

    assert isinstance(current, ChunkedRequest)
    assert current.device_len == 64
    assert adder.token_budget == 36
