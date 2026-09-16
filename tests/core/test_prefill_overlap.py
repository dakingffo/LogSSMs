from types import SimpleNamespace

import torch

from logssms.core import ChunkedRequest, Request, SamplingParams
from logssms.scheduler.prefill import PendingRequest, PrefillAdder, PrefillManager


class FakeAdder:
    def __init__(self, reserved_size=0):
        self.reserved_size = reserved_size

    def try_add_one(self, pending: PendingRequest):
        self.reserved_size += len(pending.input_ids) + pending.output_len
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
    manager = PrefillManager(torch.empty((2, 16), dtype=torch.int32), 1)
    params = SamplingParams(max_tokens=8)
    manager.pending_list = [
        PendingRequest(0, torch.arange(4, dtype=torch.int32), params),
        PendingRequest(1, torch.arange(4, dtype=torch.int32), params),
    ]

    first = manager.schedule_next_batch(FakeAdder())
    assert first is not None
    assert manager.inflight_reqs
    assert manager.need_tokens == 16

    manager.pending_list.append(
        PendingRequest(2, torch.arange(4, dtype=torch.int32), params)
    )
    adder = FakeAdder(manager.need_tokens)
    second = manager.schedule_next_batch(adder)
    assert second is not None
    assert [req.uid for req in second.reqs] == [2]
    assert set(manager.inflight_reqs) == {2}
    assert manager.need_tokens == 8
    assert adder.reserved_size == 28


class FakeChunkAdder:
    def __init__(self):
        self.reserved_size = 0

    def try_add_one(self, pending: PendingRequest):
        cached_len = 0 if pending.chunked_req is None else pending.chunked_req.device_len
        device_len = min(cached_len + 4, pending.input_len)
        cls = ChunkedRequest if device_len < pending.input_len else Request
        self.reserved_size += pending.input_len - cached_len + pending.output_len
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
    manager = PrefillManager(torch.empty((1, 32), dtype=torch.int32), 1)
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
    assert manager.need_tokens == 12

    assert manager.inflight_reqs[0] is second.reqs[0]
    assert manager.need_tokens == 12


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
    cache_manager = SimpleNamespace(
        page_size=1,
        plan_prefill_chunk=lambda remain_len, token_budget: min(
            remain_len, token_budget
        ),
    )
    adder = PrefillAdder(4, 0, cache_manager, table_manager)

    current = adder.try_add_one(pending)

    assert current is not None
    assert current.cached_len == previous.device_len == 4
    assert current.device_len == 8


def test_hybrid_chunked_prefill_is_state_checkpoint_aligned():
    params = SamplingParams(max_tokens=8)
    pending = PendingRequest(0, torch.arange(200, dtype=torch.int32), params)
    table_manager = SimpleNamespace(
        token_pool=torch.empty((1, 256), dtype=torch.int32)
    )
    cache_manager = SimpleNamespace(
        page_size=64,
        plan_prefill_chunk=lambda remain_len, token_budget: 64,
    )
    adder = PrefillAdder(100, 0, cache_manager, table_manager)

    chunk_size = cache_manager.plan_prefill_chunk(pending.input_len, adder.token_budget)
    current = adder._add_one_req(
        pending, None, table_idx=0, cached_len=0, chunk_size=chunk_size
    )

    assert isinstance(current, ChunkedRequest)
    assert current.device_len == 64
    assert adder.token_budget == 36
