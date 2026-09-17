from __future__ import annotations

from unittest.mock import patch

import torch

from logssms.cache import CacheResources
from logssms.cache.linear_pool.logstate_pool import LogStatePool
from logssms.cache.logstate import LogStateCacheManager
from logssms.cache.prefix_cache.logstate import LogStateCacheHandle
from logssms.cache.prefix_cache.logstate import LogStateKind
from logssms.core import Batch, Request, SamplingParams
from logssms.scheduler.prefill import PendingRequest


PAGE_SIZE = 64
GRANULARITY = 16
MAX_RUNNING_REQ = 3
MAX_SEQ_LEN = 512


def make_manager(
    num_pages: int = 8,
    num_states: int = 8,
) -> LogStateCacheManager:
    num_work_states = MAX_RUNNING_REQ
    pool = LogStatePool(
        num_slots=num_states + num_work_states,
        page_size=PAGE_SIZE,
        num_linear_layers=1,
        num_key_heads=1,
        num_value_heads=2,
        head_k_dim=4,
        head_v_dim=4,
        conv_dim=16,
        state_len=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    resources = CacheResources(
        kv_pool=None,  # type: ignore[arg-type]
        page_table=torch.zeros(
            MAX_RUNNING_REQ + 1, MAX_SEQ_LEN, dtype=torch.int32
        ),
        linear_state=pool,
        state_table=torch.full(
            (MAX_RUNNING_REQ + 1, MAX_SEQ_LEN), -1, dtype=torch.int32
        ),
        work_state_table=torch.arange(
            num_states, num_states + num_work_states, dtype=torch.int32
        ).view(MAX_RUNNING_REQ, 1),
    )
    return LogStateCacheManager(
        num_pages=num_pages,
        num_states=num_states,
        num_work_states=num_work_states,
        page_size=PAGE_SIZE,
        resources=resources,
        granularity=GRANULARITY,
    )


def new_req(
    manager : LogStateCacheManager,
    input_ids: torch.Tensor,
    table_idx: int = 0,
) -> Request:
    pending = PendingRequest(
        uid=table_idx,
        input_ids=input_ids,
        sampling_params=SamplingParams(max_tokens=16),
    )
    handle = manager.match_req(pending).cuda_handle
    assert isinstance(handle, LogStateCacheHandle)
    manager.lock(handle)
    req = Request(
        input_ids=input_ids,
        table_idx=table_idx,
        cached_len=handle.cached_len,
        output_len=16,
        uid=table_idx,
        sampling_params=pending.sampling_params,
        cache_handle=handle,
        max_device_len=len(input_ids) + 16,
    )
    if handle.cached_len > 0:
        manager.init_with_handle(table_idx, handle)
    return req


def test_complete_pages_move_from_request_to_tree_and_tail_is_released():
    manager = make_manager()
    req = new_req(manager, torch.arange(130, dtype=torch.int32))

    manager.allocate([req])
    assert len(manager.free_pages) == 5
    assert len(manager.free_states) == 5

    req.cached_len = 130
    manager.cache_req(req, finished=False)

    assert req.cache_handle.cached_len == 128
    assert len(manager.tree.iter_entries()) == 2
    assert len(manager._live_pages[req.table_idx]) == 1
    assert len(manager.free_pages) == 5
    assert len(manager.free_states) == 5

    manager.cache_req(req, finished=True)

    assert req.table_idx not in manager._live_pages
    assert len(manager.free_pages) == 6
    assert len(manager.free_states) == 6
    assert torch.all(manager.state_table[req.table_idx] == -1)
    manager.check_integrity()


def test_sixteen_token_branch_hit_carries_recovery_into_work_state_plan():
    manager = make_manager()
    original = new_req(manager, torch.arange(130, dtype=torch.int32))
    manager.allocate([original])
    original.cached_len = 130
    manager.cache_req(original, finished=True)

    branched = torch.cat([
        torch.arange(80, dtype=torch.int32),
        torch.arange(1000, 1032, dtype=torch.int32),
    ])
    pending = PendingRequest(
        uid=1,
        input_ids=branched,
        sampling_params=SamplingParams(max_tokens=16),
    )
    handle = manager.match_req(pending).cuda_handle
    assert isinstance(handle, LogStateCacheHandle)
    assert handle.cached_len == 80
    assert handle.recovery.position == 80
    assert len(handle.recovery.log_slots) == 2
    assert handle.recovery.conv_granule == 0

    manager.lock(handle)
    manager.init_with_handle(1, handle)
    assert manager.pending_recoveries[1] == handle.recovery
    assert torch.equal(manager.page_table[1, :80], handle.kv_indices)
    manager.unlock(handle)


def test_work_state_slots_are_reserved_outside_general_allocator():
    manager = make_manager(num_pages=4, num_states=4)
    work_slots = set(manager.work_state_table.flatten().tolist())

    assert set(manager.free_states.tolist()) == {0, 1, 2, 3}
    assert work_slots == {4, 5, 6}
    assert set(manager.free_states.tolist()).isdisjoint(work_slots)


def test_branch_split_keeps_one_physical_log_slot_on_both_routes():
    manager = make_manager(num_pages=12, num_states=12)
    original_ids = torch.arange(130, dtype=torch.int32)
    original = new_req(manager, original_ids)
    manager.allocate([original])
    original.cached_len = len(original_ids)
    manager.cache_req(original, finished=True)

    branch_ids = torch.cat([
        original_ids[:80],
        torch.arange(1000, 1070, dtype=torch.int32),
    ])
    branch = new_req(manager, branch_ids, table_idx=1)
    assert branch.cached_len == 80
    manager.allocate([branch])
    branch.cached_len = len(branch_ids)
    manager.cache_req(branch, finished=True)

    rematch = new_req(manager, branch_ids, table_idx=2)
    assert rematch.cached_len == 144
    assert rematch.cache_handle.recovery.log_slots == (0, 1, 3)
    assert manager.tree.slot_is_fragmented(1)
    manager.unlock(rematch.cache_handle)
    manager.check_integrity()


def test_fragmented_log_slot_is_returned_only_after_all_routes_are_evicted():
    manager = make_manager(num_pages=12, num_states=12)
    original_ids = torch.arange(130, dtype=torch.int32)
    original = new_req(manager, original_ids)
    manager.allocate([original])
    original.cached_len = len(original_ids)
    manager.cache_req(original, finished=True)

    branch_ids = torch.cat([
        original_ids[:80],
        torch.arange(1000, 1070, dtype=torch.int32),
    ])
    branch = new_req(manager, branch_ids, table_idx=1)
    manager.allocate([branch])
    branch.cached_len = len(branch_ids)
    manager.cache_req(branch, finished=True)

    pages, states = manager._allocate(12, 12)
    assert len(torch.unique(pages)) == 12
    assert len(torch.unique(states)) == 12
    assert not manager.tree.has_slot(1)

    manager._return_slots(pages, states)
    manager.check_integrity()


def test_prepare_batch_applies_branch_protection_and_commits_plan():
    manager = make_manager(num_pages=12, num_states=12)
    original_ids = torch.arange(130, dtype=torch.int32)
    original = new_req(manager, original_ids)
    manager.allocate([original])
    original.cached_len = len(original_ids)
    manager.cache_req(original, finished=True)

    branch_ids = torch.cat([
        original_ids[:80],
        torch.arange(1000, 1032, dtype=torch.int32),
    ])
    branch = new_req(manager, branch_ids, table_idx=1)
    with patch("logssms.cache.logstate.reconstruct_log_states"):
        manager.prepare_batch(Batch(reqs=[branch], phase="prefill"))

    assert manager.last_plan is not None
    assert manager.tree.entry_is_protected(1)
    assert any(
        entry.kind is LogStateKind.CHECKPOINT
        and entry.begin == 0
        and entry.end == 64
        for entry in manager.tree.iter_entries()
    )
    manager.unlock(branch.cache_handle)
    manager.check_integrity()


def test_prepare_batch_materializes_then_coalesces_checkpoints():
    manager = make_manager(num_pages=12, num_states=12)
    input_ids = torch.arange(130, dtype=torch.int32)
    original = new_req(manager, input_ids)
    manager.allocate([original])
    original.cached_len = len(input_ids)
    manager.cache_req(original, finished=True)

    request = new_req(manager, input_ids, table_idx=1)
    batch = Batch(reqs=[request], phase="prefill")
    with patch("logssms.cache.logstate.reconstruct_log_states"):
        manager.prepare_batch(batch)

    checkpoints = manager.tree.iter_entries()
    assert len(checkpoints) == 2
    assert all(
        entry.kind is LogStateKind.CHECKPOINT
        and entry.merge_level == 0
        for entry in checkpoints
    )

    manager.prepare_batch(batch)
    checkpoints = manager.tree.iter_entries()
    assert len(checkpoints) == 1
    assert checkpoints[0].kind is LogStateKind.CHECKPOINT
    assert checkpoints[0].begin == 0
    assert checkpoints[0].end == 128
    assert checkpoints[0].merge_level == 1

    manager.unlock(request.cache_handle)
    manager.check_integrity()
