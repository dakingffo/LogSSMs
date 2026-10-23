"""
Test that CacheManager._allocate correctly handles eviction with page_size > 1.
"""

from __future__ import annotations

import pytest
import torch
from unittest.mock import patch
import os

import logssms.core as core
from logssms.cache import CacheResources, RadixCacheManager, RadixPrefixValue
from logssms.core import Request, SamplingParams
from logssms.message import UserMsg
from logssms.scheduler.prefill import PendingRequest


@pytest.fixture(autouse=True)
def reset_global_ctx():
    """Reset global context before and after each test."""
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = old_ctx


def _make_cache_manager(num_pages: int, page_size: int) -> RadixCacheManager:
    """Helper to create a CacheManager with radix cache on CPU."""
    page_table = torch.empty((1,))
    ctx = core.Context(page_size=page_size)
    core.set_global_ctx(ctx)
    resources = CacheResources(
        kv_pool=None,  # type: ignore[arg-type]
        page_table=page_table,
        linear_state=None,
        state_table=None,
        work_state_table=None,
    )
    return RadixCacheManager(
        num_pages, page_size, resources,
    )


def _insert_evictable(
    cm: RadixCacheManager,
    input_ids: torch.Tensor,
    indices: torch.Tensor,
):
    """Insert a prefix into the radix cache so it becomes evictable."""
    cm.prefix_cache.insert_prefix(input_ids, RadixPrefixValue(indices))


def _assert_all_page_aligned(tensor: torch.Tensor, page_size: int, label: str = ""):
    """Assert every element in tensor is a multiple of page_size."""
    if len(tensor) == 0:
        return
    misaligned = tensor[tensor % page_size != 0]
    assert (
        len(misaligned) == 0
    ), f"{label} contains non-page-aligned values: {misaligned.tolist()}, page_size={page_size}"


def _assert_no_overlap(pages: torch.Tensor, page_size: int):
    """Assert that page-aligned starts, when expanded, produce no overlapping ranges."""
    if len(pages) <= 1:
        return
    expanded = set()
    for p in pages.tolist():
        token_range = set(range(p, p + page_size))
        overlap = expanded & token_range
        assert len(overlap) == 0, f"Overlapping tokens: {overlap}"
        expanded.update(token_range)


def test_radix_lifecycle_profile_tracks_arrival_admission_and_eviction():
    with patch.dict(os.environ, {"LOGSSMS_PROFILE_SCHEDULER": "1"}):
        manager = _make_cache_manager(num_pages=4, page_size=4)
    manager.resources.page_table = torch.full((2, 32), -1, dtype=torch.int32)
    tokens = torch.arange(9, dtype=torch.int32)
    allocated, _ = manager._allocate(2)
    assert allocated is not None
    manager.page_table[0, :8] = manager._page_to_token(allocated)

    pending = PendingRequest(
        uid=0, input_ids=tokens, sampling_params=SamplingParams(max_tokens=4),
        profile_tag="session/turn-1",
    )
    handle = manager.match_req(pending).cuda_handle
    manager.lock(handle)
    req = Request(
        input_ids=tokens, table_idx=0, cached_len=8, output_len=4,
        uid=0, sampling_params=pending.sampling_params, cache_handle=handle,
        max_device_len=12, profile_tag=pending.profile_tag,
    )
    manager.cache_req(req, finished=True)

    next_tokens = torch.cat([tokens[:4], torch.tensor([99], dtype=torch.int32)])
    arrival = UserMsg(
        uid=1, input_ids=next_tokens,
        sampling_params=SamplingParams(max_tokens=4),
        profile_tag="session/turn-2",
    )
    original = manager.prefix_cache.root_node.children.copy()
    child = next(iter(original.values()))
    timestamp = child.timestamp
    manager.record_arrival_profile(arrival)
    assert manager.prefix_cache.root_node.children == original
    assert child.timestamp == timestamp
    assert child.length == 8

    pending = PendingRequest(
        uid=arrival.uid, input_ids=arrival.input_ids,
        sampling_params=arrival.sampling_params, profile_tag=arrival.profile_tag,
    )
    match = manager.match_req(pending)
    manager.record_match(pending, match, input_len=len(next_tokens))
    profile = manager.profile_result()
    assert [event["event"] for event in profile["lifecycle_events"]] == [
        "complete", "arrival", "admission",
    ]
    assert profile["lifecycle_events"][1]["tree_match_len"] == 4
    assert profile["lifecycle_events"][1]["previous_inserted_len"] == 8
    assert profile["match_samples"][0]["arrival_tree_match_len"] == 4

    # Admission split the compressed edge. An unrelated allocation evicts
    # the unlocked suffix; its original node must appear in the event.
    manager._allocate(2)
    manager._allocate(1)
    events = manager.profile_result()["lifecycle_events"]
    assert events[-1]["event"] == "eviction"
    assert events[-1]["nodes"][0]["uuid"] == child.uuid
    assert events[-1]["kv_tokens"] == 4


@pytest.mark.parametrize("profile", [False, True])
def test_prefix_match_snapshot_survives_chunked_prefill(profile):
    from types import SimpleNamespace
    from logssms.core import ChunkedRequest
    from logssms.scheduler.prefill import PrefillAdder

    with patch.dict(os.environ, {"LOGSSMS_PROFILE_SCHEDULER": str(int(profile))}):
        manager = _make_cache_manager(num_pages=16, page_size=4)
    manager.resources.page_table = torch.full((1, 64), -1, dtype=torch.int32)
    tokens = torch.arange(20, dtype=torch.int32)
    _insert_evictable(manager, tokens[:8], torch.arange(8, dtype=torch.int32))
    table = SimpleNamespace(
        available_size=1, allocate=lambda: 0,
        token_pool=torch.empty((1, 64), dtype=torch.int32),
    )
    pending = PendingRequest(0, tokens, SamplingParams(max_tokens=8))
    requests = []
    for _ in range(3):
        request = PrefillAdder(4, 0, manager, table).try_add_one(pending)
        assert request is not None
        requests.append(request)
        request.complete_to_device_len()
        pending.chunked_req = request if isinstance(request, ChunkedRequest) else None
    assert [req.cached_len for req in requests] == [12, 16, 20]
    assert all(req.prefix_cached_len == 8 for req in requests)
    assert pending.prefix_cached_len == 8
    requests[-1].complete_n(1)
    assert requests[-1].prefix_cached_len == 8
    result = manager.profile_result()
    if profile:
        assert len(result["match_samples"]) == 1
        assert result["match_samples"][0]["admitted_cached_len"] == 8
    else:
        assert result is None
        assert manager._match_profile_samples == []


class TestAllocateEvictPageAlignment:
    """Tests for _allocate handling eviction with page_size > 1."""

    def test_allocate_after_evict_returns_page_aligned(self):
        """Allocated pages after eviction must be page-aligned."""
        page_size = 4
        num_pages = 4
        cm = _make_cache_manager(num_pages, page_size)

        # Exhaust all free pages
        cm._allocate(num_pages)
        assert len(cm.free_pages) == 0

        # Insert 2 pages worth of data into the cache (evictable)
        input_ids = torch.arange(page_size * 2, dtype=torch.int32)
        # Simulate page table entries: page 0 = [0,1,2,3], page 1 = [4,5,6,7]
        indices = torch.arange(page_size * 2, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 1 page — triggers eviction
        allocated = cm._allocate(1)[0]
        _assert_all_page_aligned(allocated, page_size, "allocated")
        _assert_all_page_aligned(cm.free_pages, page_size, "_free_slots after evict")

    def test_consecutive_allocations_after_evict_no_overlap(self):
        """Multiple allocations after eviction must not produce overlapping pages."""
        page_size = 4
        num_pages = 4
        cm = _make_cache_manager(num_pages, page_size)

        # Exhaust all free pages
        cm._allocate(num_pages)

        # Insert 2 pages into cache
        input_ids = torch.arange(page_size * 2, dtype=torch.int32)
        indices = torch.arange(page_size * 2, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 2 pages one by one
        page_a = cm._allocate(1)[0]
        page_b = cm._allocate(1)[0]
        all_pages = torch.cat([page_a, page_b])

        _assert_all_page_aligned(all_pages, page_size, "all_pages")
        _assert_no_overlap(all_pages, page_size)

    def test_free_slots_stay_page_aligned_after_evict(self):
        """_free_slots must remain page-aligned after eviction refills them."""
        page_size = 8
        num_pages = 8
        cm = _make_cache_manager(num_pages, page_size)

        # Exhaust all free pages
        cm._allocate(num_pages)

        # Insert 4 pages worth of data (4 * 8 = 32 tokens)
        n_tokens = page_size * 4
        input_ids = torch.arange(n_tokens, dtype=torch.int32)
        # Indices: page starts at 0, 8, 16, 24
        indices = torch.arange(n_tokens, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 1 page — evicts and refills _free_slots
        cm._allocate(1)

        # All remaining free slots must be page-aligned
        _assert_all_page_aligned(cm.free_pages, page_size, "_free_slots")

    def test_allocate_exact_pages_needed_from_evict(self):
        """When exactly N pages are needed, eviction must provide at least N pages."""
        page_size = 4
        num_pages = 4
        cm = _make_cache_manager(num_pages, page_size)

        # Exhaust all
        cm._allocate(num_pages)

        # Insert 3 pages into cache
        n_tokens = page_size * 3
        input_ids = torch.arange(n_tokens, dtype=torch.int32)
        indices = torch.arange(n_tokens, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 2 pages at once — needs eviction of at least 2 pages
        allocated = cm._allocate(2)[0]
        assert len(allocated) == 2
        _assert_all_page_aligned(allocated, page_size, "allocated")
        _assert_no_overlap(allocated, page_size)

    def test_page_to_token_expansion_correct_after_evict(self):
        """_page_to_token on eviction-allocated pages must produce correct consecutive ranges."""
        page_size = 4
        num_pages = 4
        cm = _make_cache_manager(num_pages, page_size)

        cm._allocate(num_pages)

        # Insert 2 pages: tokens [0..7] with indices [0..7]
        input_ids = torch.arange(page_size * 2, dtype=torch.int32)
        indices = torch.arange(page_size * 2, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 2 pages via eviction
        pages = cm._allocate(2)[0]
        tokens = cm._page_to_token(pages)

        # Each page should expand to page_size consecutive tokens
        assert len(tokens) == 2 * page_size
        for i, page_start in enumerate(pages.tolist()):
            chunk = tokens[i * page_size : (i + 1) * page_size].tolist()
            expected = list(range(page_start, page_start + page_size))
            assert chunk == expected, f"Page {page_start}: got {chunk}, expected {expected}"

    def test_check_integrity_passes_after_evict_cycle(self):
        """CacheManager integrity check should pass after allocate-free-evict cycles."""
        page_size = 4
        num_pages = 8
        cm = _make_cache_manager(num_pages, page_size)

        # Allocate 2 pages (simulating a request using them)
        pages_for_cache = cm._allocate(2)[0]
        # free_slots: 6 pages remaining

        # Insert those 2 pages into radix cache (simulating a finished request)
        # The token indices come from _page_to_token expansion of the allocated pages
        token_indices = cm._page_to_token(pages_for_cache)
        n_tokens = len(token_indices)
        input_ids = torch.arange(n_tokens, dtype=torch.int32)
        _insert_evictable(cm, input_ids, token_indices)
        # Now: free=6, cache=2, total=8

        # This should not raise
        cm.check_integrity()

        # Now exhaust free slots and trigger eviction
        cm._allocate(6)
        assert len(cm.free_pages) == 0

        # Allocate 1 more page — must evict from cache
        allocated = cm._allocate(1)[0]
        _assert_all_page_aligned(allocated, page_size, "allocated after evict")
        _assert_all_page_aligned(cm.free_pages, page_size, "_free_slots after evict")


if __name__ == "__main__":
    pytest.main([__file__])
