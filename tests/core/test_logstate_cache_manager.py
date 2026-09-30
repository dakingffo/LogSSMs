from __future__ import annotations

import os
from unittest.mock import patch
from types import SimpleNamespace

import pytest
import torch

from logssms.cache import CacheResources
from logssms.cache.logstate_pool.frame_pool import FramedMHALogStatePool
from logssms.cache.logstate import (
    LogStateCacheManager, LogStateCapacityError, PageMaterialize,
)
from logssms.cache.prefix_cache.logstate import LogStateCacheHandle
from logssms.cache.prefix_cache.logstate import LogStateEntry
from logssms.cache.prefix_cache.logstate import LogStateKind
from logssms.cache.logstate_recycle import LogStateRecycleBin, RecycledBranch, RecycledMatch
from logssms.core import Batch, Request, SamplingParams
from logssms.message import UserMsg
from logssms.scheduler.prefill import PendingRequest


PAGE_SIZE = 64
GRANULARITY = 16
MAX_RUNNING_REQ = 3
MAX_SEQ_LEN = 512


def make_manager(
    num_pages             : int  = 8,
    num_states            : int  = 8,
    enable_merge_adjustment: bool = False,
    enable_memorized_evict: bool = True,
    max_seq_len           : int = MAX_SEQ_LEN,
    granularity           : int = GRANULARITY,
) -> LogStateCacheManager:
    num_work_states = MAX_RUNNING_REQ + 1
    pool = FramedMHALogStatePool(
        num_frames=(num_pages + 1) + (num_states + num_work_states + 1) * 7,
        page_size=PAGE_SIZE,
        granularity=granularity,
        num_attention_layers=1,
        num_kv_heads=2,
        head_dim=4,
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
        kv_pool=pool,
        page_table=torch.zeros(
            MAX_RUNNING_REQ + 1, max_seq_len, dtype=torch.int32
        ),
        linear_state=pool,
        state_table=torch.full(
            (MAX_RUNNING_REQ + 1, max_seq_len), -1, dtype=torch.int32
        ),
        work_state_table=torch.full(
            (num_work_states, 1), -1, dtype=torch.int32,
        ),
    )
    return LogStateCacheManager(
        num_pages=pool.allocator.num_frames - 1,
        num_states=num_states,
        num_work_states=num_work_states,
        page_size=PAGE_SIZE,
        resources=resources,
        granularity=granularity,
        enable_memorized_evict=enable_memorized_evict,
        enable_merge_adjustment=enable_merge_adjustment,
    )


def test_disable_memorized_evict_has_no_recycle_bin() -> None:
    manager = make_manager(enable_memorized_evict=False)
    assert manager.recycle_bin is None
    checkpoint = manager.tree.insert(
        torch.arange(PAGE_SIZE, dtype=torch.int32),
        torch.arange(PAGE_SIZE, dtype=torch.int32) + 100,
        (LogStateEntry(1, LogStateKind.CHECKPOINT, 0, PAGE_SIZE, span_pages=1),),
    ).node
    log = manager.tree.insert(
        torch.arange(PAGE_SIZE, dtype=torch.int32) + 200,
        torch.arange(PAGE_SIZE, dtype=torch.int32) + 300,
        (LogStateEntry(2, LogStateKind.LOG, 0, PAGE_SIZE),),
    ).node
    checkpoint.timestamp = 1
    log.timestamp = 2
    assert list(manager.tree.iter_evict_candidates()) == [log, checkpoint]


def test_restore_unfused_chain_before_inplace_materialization() -> None:
    manager = make_manager()
    segment = SimpleNamespace(
        slot=4, begin=0, end=GRANULARITY, physical_begin=0
    )
    restore = SimpleNamespace(
        base_slot=None, segments=(segment,), outputs=((3,),), materialize=False
    )
    full_segment = SimpleNamespace(
        slot=4, begin=0, end=PAGE_SIZE, physical_begin=0
    )
    materialize = SimpleNamespace(
        base_slot=None, segments=(full_segment,), outputs=((4,),), materialize=True
    )
    entry = LogStateEntry(4, LogStateKind.LOG, 0, PAGE_SIZE)
    plan = PageMaterialize(4, 0, PAGE_SIZE, None, (entry,))
    order = []
    with (
        patch.object(manager, "_make_construction_chains", return_value=(materialize, restore)),
        patch(
            "logssms.cache.logstate.restore_framed_log_state_chains",
            side_effect=lambda *_args: order.append("restore"),
        ),
        patch(
            "logssms.cache.logstate.construct_framed_log_state_chains",
            side_effect=lambda *_args: order.append("materialize") or 0,
        ),
    ):
        manager._construct((), (plan,))
    assert order == ["restore", "materialize"]


def test_shared_log_materialization_rebases_branches_across_task_groups() -> None:
    manager = make_manager()
    shared = LogStateEntry(1, LogStateKind.LOG, 0, PAGE_SIZE)
    branches = [
        LogStateEntry(slot, LogStateKind.LOG, PAGE_SIZE, 2 * PAGE_SIZE)
        for slot in range(2, 8)
    ]
    plans = [PageMaterialize(1, 0, PAGE_SIZE, None, (shared,))]
    plans.extend(
        PageMaterialize(entry.slot, PAGE_SIZE, 2 * PAGE_SIZE,
                        None, (shared, entry))
        for entry in branches
    )
    independent = LogStateEntry(8, LogStateKind.LOG, 0, PAGE_SIZE)
    plans.append(PageMaterialize(8, 0, PAGE_SIZE, None, (independent,)))
    chains = manager._make_construction_chains((), plans)
    waves = manager._schedule_construction_waves(chains, plans)

    assert len(waves) == 2
    assert len(waves[0]) == 2
    assert {
        slot for chain in waves[0]
        for outputs in chain.outputs
        for slot in outputs
    } == {1, 2, 8}
    assert len(waves[1]) == 5  # crosses the kernel's max_tasks=4 boundary
    assert all(chain.base_slot == 1 for chain in waves[1])
    assert all(
        [entry.slot for entry in chain.segments] == [branch.slot]
        for chain, branch in zip(waves[1], branches[1:])
    )

    launches = []
    with patch(
        "logssms.cache.logstate.construct_framed_log_state_chains",
        side_effect=lambda _pool, group: launches.append(group) or 0,
    ):
        manager._construct((), plans)
    assert len(launches) == 2
    assert len(launches[0]) == 2
    assert len(launches[1]) == 5
    assert all(base == 1 for base, _segments, _outputs in launches[1])


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
    manager.init_with_handle(table_idx, handle)
    return req


def test_profile_tracks_complete_arrival_and_admission_without_mutating_tree():
    with patch.dict(os.environ, {"LOGSSMS_PROFILE_SCHEDULER": "1"}):
        manager = make_manager()
    tokens = torch.arange(128, dtype=torch.int32)
    req = new_req(manager, tokens)
    req.profile_tag = "session/turn-1"
    manager.allocate([req])
    req.cached_len = 128
    manager.cache_req(req, finished=True)

    next_tokens = torch.cat([tokens, torch.tensor([999], dtype=torch.int32)])
    arrival = UserMsg(
        uid=1,
        input_ids=next_tokens,
        sampling_params=SamplingParams(max_tokens=16),
        profile_tag="session/turn-2",
    )
    revision = manager.tree.revision
    manager.record_arrival_profile(arrival)
    assert manager.tree.revision == revision
    pending = PendingRequest(
        uid=arrival.uid,
        input_ids=next_tokens,
        sampling_params=arrival.sampling_params,
        profile_tag=arrival.profile_tag,
    )
    match = manager.match_req(pending)
    manager.record_match_profile(pending, match, input_len=len(next_tokens))

    profile = manager.profile_result()
    assert profile is not None
    events = profile["lifecycle_events"]
    assert [event["event"] for event in events] == [
        "complete", "arrival", "admission",
    ]
    assert events[0]["inserted_len"] == 128
    assert events[1]["previous_inserted_len"] == 128
    assert events[1]["tree_match_len"] == 128
    sample = profile["match_samples"][-1]
    assert sample["arrival_tree_match_len"] == 128
    assert sample["tree_match_len"] == 128
    assert sample["arrival_seq"] < sample["admission_seq"]


def test_allocation_profile_separates_work_prefill_and_pool_reserve():
    with patch.dict(os.environ, {"LOGSSMS_PROFILE_SCHEDULER": "1"}):
        manager = make_manager()
    req = new_req(manager, torch.arange(130, dtype=torch.int32))
    manager.allocate([req], phase="prefill")

    allocation = manager.profile_result()["allocation_profile"]
    phases = allocation["phases"]
    assert phases["work"]["calls"] == 1
    assert phases["prefill"]["calls"] == 1
    sample = next(
        value for value in allocation["samples"]
        if value["phase"] == "prefill"
    )
    assert sample["needed_pages"] > 0
    assert sample["timing_ms"]["initial_descriptor_write_ms"] >= 0
    assert sample["timing_ms"]["mapping_ms"] >= 0
    assert sample["counts"]["initial_descriptor_count"] == sample["needed_states"]
    assert phases["prefill"]["timing_ms"]["total_ms"] == sample["timing_ms"]["total_ms"]


def test_prefill_passes_physical_first_offset_without_device_sync():
    manager = make_manager()
    req = new_req(manager, torch.arange(130, dtype=torch.int32))
    manager.allocate([req], phase="prefill")
    batch = Batch(reqs=[req], phase="prefill")
    manager.prepare_batch(batch)
    assert batch.logstate_first_offsets == [0]

    # A continued chunk starts part-way through a physical 64-token page.
    req.cached_len = 70
    next_batch = Batch(reqs=[req], phase="prefill")
    manager.prepare_batch(next_batch)
    assert next_batch.logstate_first_offsets == [6]


@pytest.mark.parametrize("granularity", [16, 32, 64])
def test_decode_replay_count_tracks_prefill_tail_and_page_boundary(granularity):
    manager = make_manager(num_pages=12, num_states=12, granularity=granularity)
    req = new_req(manager, torch.arange(35, dtype=torch.int32))
    req.max_device_len = 100
    manager.allocate([req], phase="prefill")
    prefill = Batch(reqs=[req], phase="prefill")
    manager.prepare_batch(prefill)

    for token in range(35, 66):
        req.cached_len = token
        req.device_len = token + 1
        manager.allocate([req], phase="decode")
        batch = Batch(reqs=[req], phase="decode")
        batch.padded_reqs = [req, req]
        manager.prepare_batch(batch)
        assert batch.logstate_replay_counts == [
            token - max(35, (token // 16) * 16), 0,
        ]


def test_decode_allocation_skips_recycled_prefill_branch_analysis():
    manager = make_manager()
    req = new_req(manager, torch.arange(65, dtype=torch.int32))
    manager._recycle_matches[req.table_idx] = RecycledMatch(
        RecycledBranch(tuple(req.input_ids.tolist()), (), frozenset()), 64
    )
    with patch.object(manager.analyzer, "_batch_branch_points", return_value={}) as branches:
        manager.allocate([req], phase="decode")
        branches.assert_not_called()
        manager.allocate([req], phase="prefill")
        branches.assert_called_once()


def test_partial_log_hit_restores_only_matched_granule():
    manager = make_manager()
    original = new_req(manager, torch.arange(128, dtype=torch.int32))
    manager.allocate([original])
    original.cached_len = 128
    manager.cache_req(original, finished=True)

    input_ids = torch.cat([
        torch.arange(16, dtype=torch.int32),
        torch.arange(1000, 1032, dtype=torch.int32),
    ])
    req = new_req(manager, input_ids, table_idx=1)
    assert req.cached_len == 16
    tasks = manager._make_restore_tasks(Batch(reqs=[req], phase="prefill"))
    assert len(tasks) == 1
    assert tasks[0].logs[-1].end == 16

    chains = manager._make_construction_chains(tasks, ())
    assert len(chains) == 1
    assert chains[0].segments[-1].end == 16
    assert chains[0].outputs[-1] == (tasks[0].output_slot,)

    source = LogStateEntry(tasks[0].logs[0].slot, LogStateKind.LOG, 0, 64)
    materialize = PageMaterialize(
        log_slot=source.slot,
        begin=source.begin,
        end=source.end,
        base_checkpoint_slot=None,
        replay_logs=(source,),
    )
    fused = manager._make_construction_chains(tasks, (materialize,))
    assert len(fused) == 1
    assert [(entry.begin, entry.end) for entry in fused[0].segments] == [
        (0, 16), (16, 64),
    ]
    assert fused[0].outputs == ((tasks[0].output_slot,), (source.slot,))


def test_complete_pages_move_from_request_to_tree_and_tail_is_released():
    manager = make_manager()
    req = new_req(manager, torch.arange(130, dtype=torch.int32))

    initial_free = manager._allocator.free_frames
    manager.allocate([req])
    allocated_free = manager._allocator.free_frames
    assert allocated_free < initial_free

    req.cached_len = 130
    manager.cache_req(req, finished=False)

    assert req.cache_handle.cached_len == 128
    assert len(manager.tree.iter_entries()) == 2
    assert len(manager._live_pages[req.table_idx]) == 1
    assert manager._allocator.free_frames == allocated_free

    manager.cache_req(req, finished=True)

    assert req.table_idx not in manager._live_pages
    assert manager._allocator.free_frames > allocated_free
    assert torch.all(manager.state_table[req.table_idx] == -1)
    manager.check_integrity()


def test_cache_req_profile_tracks_prefill_conv_handoff():
    with patch.dict(os.environ, {"LOGSSMS_PROFILE_SCHEDULER": "1"}):
        manager = make_manager()
    req = new_req(manager, torch.arange(128, dtype=torch.int32))
    req.profile_tag = "session/turn-1"
    manager.allocate([req])
    cached_pages = [page.kv_page for page in manager._live_pages[req.table_idx][:2]]
    req.cached_len = 128

    manager.cache_req(req, finished=False)

    sample = manager.profile_result()["cache_req_samples"][-1]
    assert sample["tag"] == req.profile_tag
    assert sample["adopted_pages"] == 2
    assert not sample["finished"]
    assert sample["conv_copy_ms"] >= 0
    assert sample["total_ms"] >= sample["conv_copy_ms"]
    assert all(manager._cached_page_tokens[page] == PAGE_SIZE for page in cached_pages)


def test_adaptive_sampling_records_hit_only_after_successful_admission():
    manager = make_manager(
        num_pages=12,
        num_states=12,
        enable_merge_adjustment=True,
    )
    input_ids = torch.arange(130, dtype=torch.int32)
    original = new_req(manager, input_ids)
    manager.allocate([original])
    original.cached_len = len(input_ids)
    manager.cache_req(original, finished=True)
    source = manager.tree.merge_owners()[0].merge_stats
    assert source is not None

    pending = PendingRequest(
        uid=1,
        input_ids=input_ids,
        sampling_params=SamplingParams(max_tokens=16),
    )
    failed = manager.match_req(pending).cuda_handle
    assert not manager.prefill_acquire(manager.available_size + 1, failed)
    assert tuple(source.hit_positions) == ()

    admitted = manager.match_req(pending).cuda_handle
    assert manager.prefill_acquire(PAGE_SIZE, admitted)
    assert tuple(source.hit_positions) == (128,)
    manager.unlock(admitted)


def test_adaptive_sampling_owner_moves_to_the_latest_chain_node():
    manager = make_manager(enable_merge_adjustment=True)
    req = new_req(manager, torch.arange(130, dtype=torch.int32))
    stats = req.cache_handle.merge_stats
    assert stats is not None
    manager.allocate([req])

    req.cached_len = PAGE_SIZE
    manager.cache_req(req, finished=False)
    first_node = req.cache_handle.node
    assert first_node.merge_stats is stats

    req.cached_len = 2 * PAGE_SIZE
    manager.cache_req(req, finished=False)
    assert first_node.merge_stats is None
    assert req.cache_handle.node.merge_stats is stats

    manager.unlock(req.cache_handle)


def test_concurrent_same_prefix_discards_duplicate_log_pages():
    manager = make_manager(num_pages=12, num_states=12)
    input_ids = torch.arange(130, dtype=torch.int32)
    first = new_req(manager, input_ids, table_idx=0)
    second = new_req(manager, input_ids, table_idx=1)

    manager.allocate([first, second])
    first.cached_len = len(input_ids)
    second.cached_len = len(input_ids)
    manager.cache_req(first, finished=True)
    manager.cache_req(second, finished=True)

    entries = manager.tree.iter_entries()
    assert [(entry.begin, entry.end) for entry in entries] == [
        (0, PAGE_SIZE),
        (PAGE_SIZE, 2 * PAGE_SIZE),
    ]
    manager._allocator.check_integrity()
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
    assert (page.begin, page.end) == (MAX_SEQ_LEN - 48, MAX_SEQ_LEN)
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


def test_allocate_maps_multiple_requests_and_partial_pages():
    manager = make_manager(num_pages=12, num_states=12)
    first = new_req(manager, torch.arange(130, dtype=torch.int32), table_idx=0)
    second = new_req(manager, torch.arange(99, dtype=torch.int32), table_idx=1)
    first.max_device_len = 130
    second.max_device_len = 99
    manager.allocate([first, second])
    for req in (first, second):
        pages = manager._live_pages[req.table_idx]
        assert len(pages) == (len(req.input_ids) + PAGE_SIZE - 1) // PAGE_SIZE
        for page in pages:
            end = min(page.end, req.max_device_len)
            expected = page.kv_page + torch.arange(end - page.begin, dtype=torch.int32)
            torch.testing.assert_close(
                manager.page_table[req.table_idx, page.begin:end], expected
            )
            assert torch.all(
                manager.state_table[req.table_idx, page.begin:end] == page.state_slot
            )


def test_recycled_checkpoint_uses_one_direct_state_slot() -> None:
    manager = make_manager(num_pages=12, num_states=12)
    history = torch.arange(128, dtype=torch.int32)
    ghost_tree = type(manager.tree)(PAGE_SIZE, torch.device("cpu"), GRANULARITY)
    ghost = ghost_tree.insert(
        history, history + 1000,
        (LogStateEntry(50, LogStateKind.CHECKPOINT, 0, 128, span_pages=2),),
    ).node
    manager.recycle_bin = LogStateRecycleBin(16 * 1024)
    manager.recycle_bin.remember(ghost_tree, ghost)
    tokens = torch.cat((history, torch.tensor([201, 202], dtype=torch.int32)))
    req = new_req(manager, tokens)

    manager.allocate([req], phase="prefill")
    pages = manager._live_pages[req.table_idx]
    assert [page.state_slot for page in pages[:2]] == [None, pages[1].state_slot]
    assert pages[1].checkpoint_begin == 0
    assert manager._allocator.slot(pages[1].state_slot).kind == "work"
    assert torch.all(manager.state_table[req.table_idx, :128] == -1)
    assert pages[2].state_slot is not None

    batch = Batch([req], "prefill")
    manager.prepare_batch(batch)
    assert batch.logstate_checkpoint_writes == [(127, pages[1].state_slot)]
    assert batch.logstate_checkpoint_ranges == [(0, 0, 128)]

    checkpoint_slot = pages[1].state_slot
    assert checkpoint_slot is not None
    source = manager.logstate_pool.conv_view(checkpoint_slot, 0)
    source.fill_(7)
    req.cached_len = 128
    manager.cache_req(req, finished=False)
    assert manager._allocator.slot(checkpoint_slot).kind == "checkpoint"
    assert manager.tree.matched_entries(req.cache_handle.node) == (
        LogStateEntry(checkpoint_slot, LogStateKind.CHECKPOINT, 0, 128, span_pages=2),
    )
    target = manager.logstate_pool.conv_view(manager._work_slots[req.table_idx], 0)
    torch.testing.assert_close(target, source)


def test_recycled_shape_participates_in_same_round_merge() -> None:
    manager = make_manager(num_pages=20, num_states=20, max_seq_len=640)
    history = torch.arange(512, dtype=torch.int32)
    ghost_tree = type(manager.tree)(PAGE_SIZE, torch.device("cpu"), GRANULARITY)
    ghost_entries = (
        LogStateEntry(50, LogStateKind.CHECKPOINT, 0, 256, span_pages=4),
        *(LogStateEntry(51 + i, LogStateKind.LOG, 256 + 64 * i, 320 + 64 * i)
          for i in range(4)),
    )
    ghost = ghost_tree.insert(history, history + 1000, ghost_entries).node
    manager.recycle_bin = LogStateRecycleBin(32 * 1024)
    manager.recycle_bin.remember(ghost_tree, ghost)
    req = new_req(
        manager, torch.cat((history, torch.tensor([999], dtype=torch.int32)))
    )
    manager.allocate([req], phase="prefill")
    req.cached_len = 512
    with patch.object(manager, "_construct", return_value={}):
        manager.cache_req(req, finished=True)

    entries = manager.tree.iter_entries()
    assert [(entry.kind, entry.begin, entry.end, entry.span_pages) for entry in entries] == [
        (LogStateKind.CHECKPOINT, 0, 256, 4),
        (LogStateKind.CHECKPOINT, 256, 384, 2),
        (LogStateKind.CHECKPOINT, 384, 512, 2),
    ]
    manager.check_integrity()


def test_direct_recycled_checkpoint_aligns_after_granularity_hit() -> None:
    manager = make_manager(num_pages=20, num_states=20, max_seq_len=640)
    history = torch.arange(512, dtype=torch.int32)
    pool = manager.logstate_pool
    allocated = pool.reserve(kv=1, logs=1)
    assert allocated is not None
    kv = allocated[0][0] * PAGE_SIZE + torch.arange(16, dtype=torch.int32)
    manager.tree.insert(
        history[:16], kv,
        (LogStateEntry(allocated[1][0], LogStateKind.LOG, 0, 16),),
        cache_len=16,
    )
    manager._cached_page_tokens[allocated[0][0] * PAGE_SIZE] = len(kv)
    ghost_tree = type(manager.tree)(PAGE_SIZE, torch.device("cpu"), GRANULARITY)
    ghost = ghost_tree.insert(history, history + 1000, (
        LogStateEntry(50, LogStateKind.CHECKPOINT, 256, 512, span_pages=4),
    )).node
    manager.recycle_bin = LogStateRecycleBin(32 * 1024)
    manager.recycle_bin.remember(ghost_tree, ghost)
    req = new_req(
        manager, torch.cat((history, torch.tensor([999], dtype=torch.int32)))
    )
    assert req.cached_len == 16
    manager.allocate([req], phase="prefill")
    pages = manager._live_pages[req.table_idx]
    assert (pages[0].begin, pages[0].end) == (16, 64)
    assert (pages[4].begin, pages[4].end) == (256, 320)
    assert all(page.state_slot is None for page in pages[4:7])
    assert (pages[7].begin, pages[7].end, pages[7].checkpoint_begin) == (
        448, 512, 256,
    )


def test_cache_boundary_carries_conv_state_into_request_work_slot():
    manager = make_manager()
    req = new_req(manager, torch.arange(PAGE_SIZE, dtype=torch.int32))
    manager.allocate([req])
    page = manager._live_pages[req.table_idx][0]
    source = manager.logstate_pool.conv_view(page.state_slot, 3)
    source.copy_(torch.arange(source.numel()).view_as(source))

    req.cached_len = PAGE_SIZE
    manager.cache_req(req, finished=False)

    work_slot = int(manager.work_state_table[req.table_idx, 0])
    torch.testing.assert_close(
        manager.logstate_pool.conv_view(work_slot, 0), source
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
    torch.testing.assert_close(
        manager.page_table[branch.table_idx, :16],
        branch.cache_handle.kv_indices,
    )

    manager.unlock(branch.cache_handle)
    manager.check_integrity()


def test_work_state_slots_return_to_shared_allocator():
    manager = make_manager(num_pages=4, num_states=4)
    initial_free = manager._allocator.free_frames
    req = new_req(manager, torch.arange(8, dtype=torch.int32))
    work_slot = int(manager.work_state_table[req.table_idx, 0])
    assert work_slot in manager._work_slots.values()
    work_units = manager._allocator.payload_frames + manager._allocator.conv_frames
    assert manager._allocator.free_frames == initial_free - work_units
    manager.cache_req(req, finished=True)
    assert int(manager.work_state_table[req.table_idx, 0]) == -1
    assert manager._allocator.free_frames == initial_free
    manager.check_integrity()


def test_admission_reserves_capacity_without_allocating_future_pages():
    manager = make_manager(num_pages=4, num_states=4)
    initial_free = manager._allocator.free_frames
    input_ids = torch.tensor([7], dtype=torch.int32)
    pending = PendingRequest(
        uid=0,
        input_ids=input_ids,
        sampling_params=SamplingParams(max_tokens=127),
    )
    handle = manager.match_req(pending).cuda_handle
    assert manager.prefill_acquire(128, handle, request_len=128)
    assert manager._allocator.free_frames == initial_free
    manager.init_with_handle(0, handle)
    work_units = manager._allocator.payload_frames + manager._allocator.conv_frames
    assert manager._allocator.free_frames == initial_free - work_units

    req = Request(
        input_ids=input_ids,
        table_idx=0,
        cached_len=0,
        output_len=127,
        uid=0,
        sampling_params=pending.sampling_params,
        cache_handle=handle,
        max_device_len=128,
    )
    manager.allocate([req])
    first_page_free = manager._allocator.free_frames
    assert first_page_free < initial_free - work_units
    req.extend_device_n(64)
    manager.allocate([req])
    assert manager._allocator.free_frames < first_page_free
    manager.cache_req(req, finished=True)
    assert manager._allocator.free_frames == initial_free
    manager.check_integrity()


def test_admission_does_not_count_released_checkpoint_frames_twice():
    manager = make_manager(num_pages=4, num_states=4)
    req = new_req(manager, torch.arange(65, dtype=torch.int32))
    manager.allocate([req])
    req.cached_len = 65
    manager.cache_req(req, finished=True)

    slot = manager.tree.unique_slots()[0]
    entry = manager.tree.entries_for_slot(slot)[0]
    capacity = manager.available_size
    free_frames = manager._allocator.free_frames
    manager._commit_materialize((PageMaterialize(
        log_slot=slot,
        begin=entry.begin,
        end=entry.end,
        base_checkpoint_slot=None,
        replay_logs=(entry,),
    ),))

    assert manager._allocator.free_frames > free_frames
    assert manager.available_size == capacity
    manager.check_integrity()


def test_admission_rejects_unfulfillable_logical_reservation():
    manager = make_manager(num_pages=4, num_states=4)
    pending = PendingRequest(
        uid=0,
        input_ids=torch.tensor([7], dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=16),
    )
    handle = manager.match_req(pending).cuda_handle
    initial_free = manager._allocator.free_frames

    assert not manager.prefill_acquire(
        initial_free * manager.page_size, handle,
        request_len=(manager.num_pages + 1) * manager.page_size,
    )
    assert manager._allocator.free_frames == initial_free
    manager.check_integrity()


def test_logical_admission_does_not_evict_cached_prefix():
    manager = make_manager(num_pages=4, num_states=4)
    cached = new_req(manager, torch.arange(128, dtype=torch.int32))
    manager.allocate([cached])
    cached.cached_len = 128
    manager.cache_req(cached, finished=True)

    free_units = manager._allocator.free_frames
    cached_size = manager.tree.size_info.evictable_size
    pending = PendingRequest(
        uid=1,
        input_ids=torch.tensor([999], dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=16),
    )
    handle = manager.match_req(pending).cuda_handle
    assert manager.prefill_acquire(manager.available_size, handle)
    assert manager._allocator.free_frames == free_units
    assert manager.tree.size_info.evictable_size == cached_size
    manager.unlock(handle)
    manager.check_integrity()


def test_allocation_evicts_only_as_many_cached_leaves_as_needed():
    with patch.dict(os.environ, {"LOGSSMS_PROFILE_SCHEDULER": "1"}):
        manager = make_manager(num_pages=4, num_states=4)

    def ids_for(index: int) -> torch.Tensor:
        return torch.cat([
            torch.tensor([1000 + index], dtype=torch.int32),
            torch.arange(64, dtype=torch.int32),
        ])

    for index in range(8):
        req = new_req(manager, ids_for(index))
        manager.allocate([req])
        req.cached_len = len(req.input_ids)
        manager.cache_req(req, finished=True)
    assert manager.tree.size_info.total_size == 8 * PAGE_SIZE

    input_ids = ids_for(8)
    pending = PendingRequest(
        uid=8,
        input_ids=input_ids,
        sampling_params=SamplingParams(max_tokens=16),
    )
    handle = manager.match_req(pending).cuda_handle
    assert manager.prefill_acquire(2 * PAGE_SIZE, handle)
    manager.init_with_handle(0, handle)
    req = Request(
        input_ids=input_ids,
        table_idx=0,
        cached_len=0,
        output_len=16,
        uid=8,
        sampling_params=pending.sampling_params,
        cache_handle=handle,
        max_device_len=len(input_ids) + 16,
    )
    with patch.object(
        manager.tree, "evict_nodes", wraps=manager.tree.evict_nodes
    ) as evict:
        manager.allocate([req])
    assert evict.call_count == 1
    req.cached_len = len(input_ids)
    manager.cache_req(req, finished=True)

    # One 64-token leaf was enough to satisfy the reservation.  The former
    # missing_frames * page_size estimate evicted four leaves here.
    assert manager.tree.size_info.total_size == 8 * PAGE_SIZE
    profile = manager.profile_result()
    assert profile is not None
    evictions = [
        event for event in profile["lifecycle_events"]
        if event["event"] == "eviction"
    ]
    assert evictions
    assert all(event["nodes"] for event in evictions)
    assert any(
        manager.recycle_bin.match(ids_for(index), start=0, granularity=GRANULARITY)
        is not None
        for index in range(8)
    )
    manager.check_integrity()


def test_admitted_requests_can_allocate_their_full_reserved_suffixes():
    manager = make_manager(num_pages=4, num_states=4)
    for index in range(5):
        input_ids = torch.cat([
            torch.tensor([1000 + index], dtype=torch.int32),
            torch.arange(64, dtype=torch.int32),
        ])
        cached = new_req(manager, input_ids)
        manager.allocate([cached])
        cached.cached_len = len(input_ids)
        manager.cache_req(cached, finished=True)

    reqs = []
    for index in range(2):
        pending = PendingRequest(
            uid=index,
            input_ids=torch.tensor([2000 + index], dtype=torch.int32),
            sampling_params=SamplingParams(max_tokens=127),
        )
        handle = manager.match_req(pending).cuda_handle
        assert manager.prefill_acquire((index + 1) * 2 * PAGE_SIZE, handle)
        manager.init_with_handle(index, handle)
        reqs.append(Request(
            input_ids=pending.input_ids,
            table_idx=index,
            cached_len=0,
            output_len=127,
            uid=index,
            sampling_params=pending.sampling_params,
            cache_handle=handle,
            max_device_len=128,
        ))

    manager.allocate(reqs)
    for req in reqs:
        req.extend_device_n(PAGE_SIZE)
    manager.allocate(reqs)
    for req in reqs:
        # The synthetic test allocates future pages without generating their
        # token IDs, so finish without inserting those pages into the tree.
        req.cached_len = 0
        manager.cache_req(req, finished=True)
    manager.check_integrity()


def test_unadmittable_request_stays_pending_without_allocating():
    from logssms.scheduler.prefill import PrefillAdder, PrefillManager
    from logssms.scheduler.table import TableManager

    manager = make_manager(num_pages=4, num_states=4)
    active = new_req(manager, torch.arange(449, dtype=torch.int32))
    manager.allocate([active])

    table = TableManager(MAX_RUNNING_REQ, manager.page_table)
    prefill = PrefillManager(table.token_pool, PAGE_SIZE)
    pending = PendingRequest(
        uid=0,
        input_ids=torch.tensor([7], dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=127),
    )
    prefill.pending_list.append(pending)

    assert prefill.schedule_next_batch(PrefillAdder(64, 0, manager, table)) is None
    assert prefill.pending_list == [pending]
    assert table.available_size == MAX_RUNNING_REQ
    assert set(manager._work_slots) == {active.table_idx}

    active.cached_len = len(active.input_ids)
    manager.cache_req(active, finished=True)
    batch = prefill.schedule_next_batch(PrefillAdder(64, 0, manager, table))
    assert batch is not None and batch.size == 1
    assert prefill.pending_list == []
    manager.allocate(batch.reqs)
    req = batch.reqs[0]
    req.cached_len = 0  # No generated token IDs in this synthetic lifecycle.
    manager.cache_req(req, finished=True)
    manager.check_integrity()


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
    assert rematch.cached_len == 128
    slots = rematch.cache_handle.recovery.log_slots
    assert len(slots) == 3 and len(set(slots)) == 3
    assert manager.tree.slot_is_fragmented(slots[1])
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

    pages, states, _ = manager._allocate(12, 12)
    assert len(torch.unique(pages)) == 12
    assert len(torch.unique(states)) == 12
    assert not manager.tree.has_slot(1)

    manager._return_slots(pages, states)
    manager.check_integrity()


def test_unfulfillable_eviction_plan_preserves_cached_prefix():
    manager = make_manager()
    original_ids = torch.arange(130, dtype=torch.int32)
    req = new_req(manager, original_ids)
    manager.allocate([req])
    req.cached_len = len(original_ids)
    manager.cache_req(req, finished=True)
    revision = manager.tree.revision
    free_frames = manager._allocator.free_frames

    with pytest.raises(LogStateCapacityError):
        manager._allocate(1000, 1000)
    assert manager.tree.revision == revision
    assert manager._allocator.free_frames == free_frames
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
    with patch(
        "logssms.cache.logstate.construct_framed_log_state_chains",
        return_value=0,
    ) as construct:
        manager.prepare_batch(Batch(reqs=[branch], phase="prefill"))

    assert construct.call_count == 1
    assert len(construct.call_args.args[1][0][1]) == 2
    assert manager.last_plan is not None
    assert any(manager.tree.entry_is_protected(entry.slot) for entry in manager.tree.iter_entries())
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
    with patch("logssms.cache.logstate.construct_framed_log_state_chains", return_value=0):
        manager.prepare_batch(batch)

    checkpoints = manager.tree.iter_entries()
    assert len(checkpoints) == 1
    assert checkpoints[0].kind is LogStateKind.CHECKPOINT
    assert checkpoints[0].begin == 0
    assert checkpoints[0].end == 128
    assert checkpoints[0].span_pages == 2

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
            span_pages=2,
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
                span_pages=1,
            ),
        )

    branch_ids = torch.cat([
        history[:96],
        torch.arange(1000, 1032, dtype=torch.int32),
    ])
    branch = new_req(manager, branch_ids, table_idx=1)
    assert branch.cached_len == 64
    with patch("logssms.cache.logstate.construct_framed_log_state_chains", return_value=0):
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
