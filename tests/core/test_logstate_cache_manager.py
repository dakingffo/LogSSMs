from __future__ import annotations

from unittest.mock import patch

import torch

from logssms.cache import CacheResources
from logssms.cache.linear_pool.logstate_pool import LogStatePool
from logssms.cache.logstate import LogStateCacheManager
from logssms.cache.prefix_cache.logstate import LogStateCacheHandle
from logssms.cache.prefix_cache.logstate import LogStateEntry
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
        num_key_heads=2,
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


def test_allocate_maps_only_valid_suffix_of_last_physical_page():
    manager = make_manager()
    req = new_req(manager, torch.arange(MAX_SEQ_LEN, dtype=torch.int32))
    req.cached_len = MAX_SEQ_LEN - 48
    req.max_device_len = MAX_SEQ_LEN

    manager.allocate([req])

    pages = manager._live_pages[req.table_idx]
    assert len(pages) == 1
    page = pages[0]
    assert (page.begin, page.end) == (MAX_SEQ_LEN - 48, MAX_SEQ_LEN + 16)
    torch.testing.assert_close(
        manager.page_table[req.table_idx, page.begin:MAX_SEQ_LEN],
        page.kv_page + torch.arange(48, dtype=torch.int32),
    )
    assert torch.all(
        manager.state_table[req.table_idx, page.begin:MAX_SEQ_LEN]
        == page.state_slot
    )

    manager._release_live_pages(req.table_idx, pages)
    manager._live_pages[req.table_idx] = []
    manager.unlock(req.cache_handle)
    manager.check_integrity()


def test_cache_boundary_carries_conv_state_into_request_work_slot():
    manager = make_manager()
    req = new_req(manager, torch.arange(PAGE_SIZE, dtype=torch.int32))
    manager.allocate([req])
    page = manager._live_pages[req.table_idx][0]
    source = manager.logstate_pool.conv_state[page.state_slot, :, -1]
    source.copy_(torch.arange(source.numel()).view_as(source))

    req.cached_len = PAGE_SIZE
    manager.cache_req(req, finished=False)

    work_slot = int(manager.work_state_table[req.table_idx, 0])
    torch.testing.assert_close(
        manager.logstate_pool.conv_state[work_slot, :, 0], source
    )


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


def test_branch_at_first_granule_reuses_kv_through_sixteen_tokens():
    manager = make_manager(num_pages=12, num_states=12)
    original_ids = torch.arange(130, dtype=torch.int32)
    original = new_req(manager, original_ids)
    manager.allocate([original])
    original.cached_len = len(original_ids)
    manager.cache_req(original, finished=True)

    branch_ids = torch.cat([
        original_ids[:16],
        torch.arange(1000, 1032, dtype=torch.int32),
    ])
    branch = new_req(manager, branch_ids, table_idx=1)

    assert branch.cached_len == 16
    assert len(branch.cache_handle.kv_indices) == 16
    torch.testing.assert_close(
        branch.cache_handle.kv_indices,
        manager.tree.matched_kv(branch.cache_handle.node),
    )
    manager.init_with_handle(branch.table_idx, branch.cache_handle)
    torch.testing.assert_close(
        manager.page_table[branch.table_idx, :16],
        branch.cache_handle.kv_indices,
    )

    manager.unlock(branch.cache_handle)
    manager.check_integrity()


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
    with (
        patch("logssms.cache.logstate.reconstruct_log_states"),
        patch("logssms.cache.logstate.materialize_log_states_inplace"),
    ):
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
    with (
        patch("logssms.cache.logstate.reconstruct_log_states"),
        patch("logssms.cache.logstate.materialize_log_states_inplace"),
    ):
        manager.prepare_batch(batch)

    checkpoints = manager.tree.iter_entries()
    assert len(checkpoints) == 1
    assert checkpoints[0].kind is LogStateKind.CHECKPOINT
    assert checkpoints[0].begin == 0
    assert checkpoints[0].end == 128
    assert checkpoints[0].merge_level == 1

    manager.unlock(request.cache_handle)
    manager.check_integrity()


def test_state_admission_can_end_before_existing_kv_prefix():
    manager = make_manager(num_pages=12, num_states=12)
    original_ids = torch.arange(130, dtype=torch.int32)
    original = new_req(manager, original_ids)
    manager.allocate([original])
    original.cached_len = len(original_ids)
    manager.cache_req(original, finished=True)

    entries = manager.tree.iter_entries()
    assert len(entries) == 2
    left, right = entries
    leaf = manager.tree.leaves()[0]
    manager.tree.detach_entry(left.slot)
    manager.tree.detach_entry(right.slot)
    manager.tree.attach_entry(
        leaf,
        LogStateEntry(
            slot=right.slot,
            kind=LogStateKind.CHECKPOINT,
            begin=0,
            end=128,
            merge_level=1,
        ),
    )
    manager._return_slots(
        torch.empty(0, dtype=torch.int32),
        torch.tensor([left.slot], dtype=torch.int32),
    )

    branch_ids = torch.cat([
        original_ids[:96],
        torch.arange(1000, 1014, dtype=torch.int32),
    ])
    branch = new_req(manager, branch_ids, table_idx=1)
    assert branch.cached_len == 0
    manager.allocate([branch])
    branch.cached_len = len(branch_ids)
    manager.cache_req(branch, finished=False)

    assert branch.cache_handle.cached_len == 64
    assert any(
        entry.kind is LogStateKind.LOG
        and entry.begin == 0
        and entry.end == 64
        for entry in manager.tree.iter_entries()
    )
    manager.cache_req(branch, finished=True)
    manager.check_integrity()


def test_active_branch_recovery_anchor_survives_same_round_maintenance():
    manager = make_manager(num_pages=12, num_states=12)
    history = torch.arange(128, dtype=torch.int32)
    original = new_req(manager, history)
    manager.allocate([original])
    original.cached_len = len(history)
    manager.cache_req(original, finished=True)

    entries = manager.tree.iter_entries()
    assert len(entries) == 2
    for entry in entries:
        owner = manager.tree.entry_owner(entry)
        manager.tree.detach_entry(entry.slot)
        manager.tree.attach_entry(
            owner,
            LogStateEntry(
                slot=entry.slot,
                kind=LogStateKind.CHECKPOINT,
                begin=entry.begin,
                end=entry.end,
                merge_level=0,
            ),
        )

    branch_ids = torch.cat([
        history[:96],
        torch.arange(1000, 1032, dtype=torch.int32),
    ])
    branch = new_req(manager, branch_ids, table_idx=1)
    assert branch.cached_len == 64
    with patch("logssms.cache.logstate.reconstruct_log_states"):
        manager.prepare_batch(Batch(reqs=[branch], phase="prefill"))

    checkpoints = [
        entry
        for entry in manager.tree.iter_entries()
        if entry.kind is LogStateKind.CHECKPOINT
    ]
    assert {(entry.begin, entry.end) for entry in checkpoints} == {
        (0, 64), (64, 128)
    }

    manager.allocate([branch])
    branch.cached_len = len(branch_ids)
    manager.cache_req(branch, finished=False)
    assert branch.cache_handle.cached_len == 128
    manager.cache_req(branch, finished=True)
    manager.check_integrity()
