from unittest.mock import patch

import pytest
import torch

from logssms.cache.logstate import _LivePage, _RequestCacheState
from logssms.core import Batch, SamplingParams
from logssms.scheduler.prefill import PendingRequest
from tests.core.test_logstate_cache_manager import (
    cpu_cache_copy, make_manager, new_req,
)


def test_live_page_cursor_matches_scan_across_partial_pages_and_backtracking():
    pages = [_LivePage(16, 64, 0, 1), _LivePage(64, 128, 64, 2),
             _LivePage(192, 256, 192, 3)]
    state = _RequestCacheState(10, 16, 300, live_pages=pages)
    for position in (16, 63, 64, 127, 128, 191, 192, 255, 256, 15, 80, 16):
        expected = next((p for p in pages if p.begin <= position < p.end), None)
        assert state.live_page_at(position) is expected
    state.live_pages = []
    assert state.live_page_at(16) is None


def test_decode_tail_lookup_does_not_iterate_history():
    class IndexedPages(list):
        reads = 0

        def __iter__(self):
            raise AssertionError("Decode must not scan page history")

        def __getitem__(self, index):
            self.reads += 1
            return super().__getitem__(index)

    pages = IndexedPages(_LivePage(i * 64, (i + 1) * 64, i * 64, i)
                         for i in range(1024))
    state = _RequestCacheState(2000, 0, 65536, live_pages=pages)
    assert state.live_page_at(65535).begin == 65472
    pages.reads = 0
    for _ in range(20):
        assert state.live_page_at(65535).begin == 65472
    assert pages.reads == 20


def test_pending_match_precedes_table_state_and_table_reuse_starts_fresh():
    manager = make_manager()
    tokens = torch.arange(128, dtype=torch.int32)
    req = new_req(manager, tokens)
    manager.allocate([req], phase="prefill")
    req.cached_len = len(tokens)
    manager.cache_req(req, finished=True)
    assert manager._request_states == {}

    history = torch.cat((tokens, torch.tensor([999], dtype=torch.int32)))
    pending = PendingRequest(99, history, SamplingParams(max_tokens=16))
    first = manager.match_req(pending)
    second = manager.match_req(pending)
    assert first.cuda_handle.cached_len == second.cuda_handle.cached_len == 128
    assert manager._request_states == {}

    req = new_req(manager, history, table_idx=2)
    old_state = manager._request_states[2]
    assert old_state.work_committed_pos == 128
    assert old_state.live_pages == [] and old_state.live_page_cursor == 0
    manager.allocate([req], phase="prefill")
    manager.prepare_batch(Batch([req], "prefill"))
    req.cached_len = len(history)
    manager.cache_req(req, finished=True)
    assert 2 not in manager._request_states
    req = new_req(manager, torch.arange(2000, 2010, dtype=torch.int32), table_idx=2)
    state = manager._request_states[2]
    assert state is not old_state
    assert state.work_committed_pos == 0 and state.live_pages == []
    assert state.live_page_cursor == 0 and state.recycle_match is None
    manager.cache_req(req, finished=True)
    manager.check_integrity()


@pytest.mark.parametrize("finish", [False, True])
def test_page_replacement_resets_cursor_and_releases_state_on_finish(finish):
    manager = make_manager(num_pages=16, num_states=16)
    req = new_req(manager, torch.arange(193, dtype=torch.int32))
    manager.allocate([req])
    state = manager._request_states[req.table_idx]
    assert state.live_page_at(192).begin == 192
    assert state.live_page_cursor == 3
    req.cached_len = 70
    manager.free_to_cached(req)
    assert state.live_page_cursor == 0 and len(state.live_pages) == 2
    assert state.live_page_at(69).begin == 64
    manager.cache_req(req, finished=finish)
    if not finish:
        assert state.live_page_cursor == 0
        assert state.live_page_at(69).begin == 64
        manager.cache_req(req, finished=True)
    assert req.table_idx not in manager._request_states
    manager.check_integrity()


def test_mixed_allocate_prepare_order_only_analyzes_prefill():
    manager = make_manager(num_pages=16, num_states=16)
    decoding = new_req(manager, torch.arange(35, dtype=torch.int32))
    manager.allocate([decoding], phase="prefill")
    manager.prepare_batch(Batch([decoding], "prefill"))
    decoding.cached_len = 35
    decoding.device_len = 36
    prefilling = new_req(manager, torch.arange(1000, 1080, dtype=torch.int32), table_idx=1)
    with patch.object(manager.analyzer, "_find_protections", wraps=manager.analyzer._find_protections) as protections, patch.object(
        manager.analyzer, "begin_round", wraps=manager.analyzer.begin_round,
    ) as analysis:
        manager.allocate([prefilling], phase="prefill")
        pending = manager._prefill_protections
        key = manager._prefill_protection_key
        manager.allocate([decoding], phase="decode")
        decode_batch = Batch([decoding], "decode")
        decode_batch.padded_reqs = [decoding]
        manager.prepare_batch(decode_batch)
        assert decode_batch.logstate_replay_counts == [0]
        assert manager._prefill_protections is pending
        assert manager._prefill_protection_key == key
        assert manager._request_states[1].work_committed_pos == 0
        analysis.assert_not_called()
        protections.assert_called_once()
        manager.prepare_batch(Batch([prefilling], "prefill"))
        analysis.assert_called_once()
        protections.assert_called_once()
    assert manager._request_states[0].work_committed_pos == 35
    assert manager._request_states[1].work_committed_pos == 80
    assert manager._prefill_protections is None
    for req in (decoding, prefilling):
        manager.cache_req(req, finished=True)
    manager.check_integrity()
