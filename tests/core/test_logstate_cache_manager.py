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
from logssms.cache.logstate import LogStateRecycleBin, RecycledBranch, RecycledMatch
from logssms.core import Batch, Request, SamplingParams
from logssms.message import UserMsg
from logssms.scheduler.prefill import PendingRequest


PAGE_SIZE = 64
GRANULARITY = 16
MAX_RUNNING_REQ = 3
MAX_SEQ_LEN = 512


@pytest.fixture(autouse=True)
def cpu_convolution_copy():
    # These tests check tree/allocator transitions, not CUDA kernel arithmetic.
    # Preserve actual snapshot-copy semantics without sending CPU pointers to Triton.
    def copy(pool, sources, source_granules, outputs, output_granules):
        for source, source_granule, output, output_granule in zip(
            sources.tolist(), source_granules.tolist(), outputs.tolist(), output_granules.tolist(), strict=True,
        ):
            pool.copy_conv_snapshot(source, source_granule, output, output_granule)

    with patch("logssms.kernel.linear_attention.AFFINE_LOG_STATE_KERNELS.copy_conv", side_effect=copy):
        yield


def make_manager(
    num_pages             : int  = 8,
    num_states            : int  = 8,
    enable_merge_adjustment: bool = False,
    enable_memorized_evict: bool = True,
    max_seq_len           : int = MAX_SEQ_LEN,
    granularity           : int = GRANULARITY,
    device                : torch.device = torch.device("cpu"),
) -> LogStateCacheManager:
    num_work_states = MAX_RUNNING_REQ + 1
    cuda = device.type == "cuda"
    pool = FramedMHALogStatePool(
        num_frames=(num_pages + 1) + (num_states + num_work_states + 1) * 7,
        page_size=PAGE_SIZE,
        granularity=granularity,
        num_attention_layers=1,
        num_kv_heads=2,
        head_dim=64 if cuda else 4,
        num_linear_layers=1,
        num_key_heads=2,
        num_value_heads=2,
        head_k_dim=32 if cuda else 4,
        head_v_dim=32 if cuda else 4,
        conv_dim=192 if cuda else 16,
        state_len=3,
        device=device,
        dtype=torch.bfloat16 if cuda else torch.float32,
    )
    resources = CacheResources(
        kv_pool=pool,
        page_table=torch.zeros(
            MAX_RUNNING_REQ + 1, max_seq_len, dtype=torch.int32, device=device,
        ),
        linear_state=pool,
        state_table=torch.full(
            (MAX_RUNNING_REQ + 1, max_seq_len), -1, dtype=torch.int32, device=device,
        ),
        work_state_table=torch.full(
            (num_work_states, 1), -1, dtype=torch.int32, device=device,
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
    slots = manager.logstate_pool.reserve(logs=2)[1]
    manager.logstate_pool.allocator.materialize(slots[0])
    checkpoint = manager.tree.insert(
        torch.arange(PAGE_SIZE, dtype=torch.int32),
        torch.arange(PAGE_SIZE, dtype=torch.int32) + 100,
        (LogStateEntry(slots[0], LogStateKind.CHECKPOINT, 0, PAGE_SIZE, span_pages=1),),
    ).node
    log = manager.tree.insert(
        torch.arange(PAGE_SIZE, dtype=torch.int32) + 200,
        torch.arange(PAGE_SIZE, dtype=torch.int32) + 300,
        (LogStateEntry(slots[1], LogStateKind.LOG, 0, PAGE_SIZE),),
    ).node
    checkpoint.timestamp = 1
    log.timestamp = 2
    assert list(manager.tree.iter_evict_candidates()) == [log, checkpoint]


@pytest.mark.parametrize("enabled", [False, True])
def test_schedule_profile_capacity_and_work_breakdown(monkeypatch, enabled):
    from logssms.scheduler.profile import SchedulerProfiler

    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", str(int(enabled)))
    manager = make_manager()
    profiler = SchedulerProfiler(torch.device("cpu"), manager)
    pending = PendingRequest(
        7, torch.arange(PAGE_SIZE, dtype=torch.int32),
        SamplingParams(max_tokens=PAGE_SIZE), profile_tag="session/turn-1",
    )
    available_before = manager.available_size
    free_frames_before = manager._allocator.free_frames
    with profiler.record_schedule() as sample:
        with profiler.record_schedule_step("match_req"):
            handle = manager.match_req(pending).cuda_handle
        with profiler.record_schedule_step("prefill_acquire"):
            assert manager.prefill_acquire(PAGE_SIZE, handle)
        with profiler.record_schedule_step("init_with_handle"):
            manager.init_with_handle(0, handle)
        if sample is not None:
            sample["phase"] = "prefill"
    assert manager._schedule_profile_step is None
    assert handle.cached_len == 0
    assert manager.available_size <= available_before
    assert manager._allocator.free_frames < free_frames_before
    assert manager.work_state_table[0, 0].item() == manager._work_slots[0]
    manager.logstate_pool.allocator.check_integrity()
    if not enabled:
        assert not profiler.schedule_stage_calls
        return
    calls = dict(profiler.schedule_stage_calls)
    for stage in ("reclamation_poll", "kv_capacity", "state_capacity", "frame_capacity"):
        assert calls[stage] == 2
    for stage in ("capacity_before_lock", "capacity_after_lock", "lock_handle",
                  "prefix_match", "recycle_match", "work_allocate", "work_mapping",
                  "state_mapping_clear"):
        assert calls[stage] == 1
    # Capacity queries outside schedule must not leak into its attribution.
    assert profiler.schedule_stage_calls == calls


def test_admission_counts_pending_merge_frames_without_premature_reuse():
    manager = make_manager()
    pool = manager.logstate_pool
    logs = pool.reserve(logs=4)[1]
    pool.reserve(kv=len(pool.allocator._free))
    assert manager.available_size == 0
    # Exercise retirement itself, including its capacity notification, while
    # keeping the frames unavailable until the mocked GPU event completes.
    ready = [False]
    event = SimpleNamespace(record=lambda _stream: None, query=lambda: ready[0])
    with (
        patch.object(type(pool), "device", property(lambda _pool: torch.device("cuda"))),
        patch("torch.cuda.Event", return_value=event),
        patch("torch.cuda.current_stream", return_value=None),
    ):
        pool.retire_slots(logs)
    assert manager.available_size >= PAGE_SIZE
    assert pool.allocator.free_frames == 0
    assert pool.allocator.reserve(kv=1, logs=1) is None
    assert set(logs) <= pool.allocator.allocated_slots
    pool.allocator.check_integrity()
    ready[0] = True
    pool.poll_reclamation()
    assert not pool.retiring_slots
    assert not set(logs).intersection(pool.allocator.allocated_slots)
    pool.allocator.check_integrity()


def test_capacity_queries_do_not_scan_tree_or_linear_slots():
    manager = make_manager()
    history = torch.arange(256, dtype=torch.int32)
    req = new_req(manager, history)
    manager.allocate([req], phase="prefill")
    req.cached_len = len(history)
    manager.cache_req(req, finished=True)
    expected = manager._allocator.admissible_pages(
        evictable_kv_frames=manager.tree.evictable_kv_frames(),
        evictable_slots=manager.tree.evictable_slots(),
    ) * PAGE_SIZE
    with (
        patch.object(manager.tree, "_iter_nodes", side_effect=AssertionError),
        patch.object(manager.tree, "evictable_slots", side_effect=AssertionError),
    ):
        assert manager.available_size == expected
        handle = manager.tree.match_prefix(history[:80]).cuda_handle
        manager.lock(handle)
        assert manager.available_size <= expected
        manager.unlock(handle)
        assert manager.available_size == expected
    manager.check_integrity()


def test_admission_capacity_still_rechecks_after_lock_and_rolls_back():
    manager = make_manager()
    history = torch.arange(256, dtype=torch.int32)
    req = new_req(manager, history)
    manager.allocate([req], phase="prefill")
    req.cached_len = len(history)
    manager.cache_req(req, finished=True)
    handle = manager.tree.match_prefix(history).cuda_handle
    before = manager.available_size
    manager.lock(handle)
    after = manager.available_size
    manager.unlock(handle)
    assert after < before
    assert not manager.prefill_acquire(after + 1, handle)
    assert manager.available_size == before
    assert not handle.lock_node.is_locked
    manager.check_integrity()


def test_failed_admission_projects_locks_without_mutation():
    manager = make_manager()
    history = torch.arange(256, dtype=torch.int32)
    req = new_req(manager, history)
    manager.allocate([req], phase="prefill")
    req.cached_len = len(history)
    manager.cache_req(req, finished=True)
    handle = manager.tree.match_prefix(history).cuda_handle
    before = manager.admission_capacity
    manager.lock(handle)
    after = manager.admission_capacity
    manager.unlock(handle)
    epoch = manager.tree.lock_revision
    with patch.object(manager.tree, "lock_handle", side_effect=AssertionError):
        for _ in range(10):
            assert not manager.prefill_acquire(after + 1, handle)
    assert manager.tree.lock_revision == epoch
    assert manager.admission_capacity == before
    manager.check_integrity()


def test_admission_final_capacity_check_still_rolls_back():
    manager = make_manager()
    req = new_req(manager, torch.arange(128, dtype=torch.int32))
    manager.allocate([req], phase="prefill")
    req.cached_len = 128
    manager.cache_req(req, finished=True)
    handle = manager.match_req(PendingRequest(
        7, torch.arange(192, dtype=torch.int32), SamplingParams(max_tokens=16),
    )).cuda_handle
    from unittest.mock import PropertyMock

    with patch.object(type(manager), "admission_capacity", new_callable=PropertyMock,
                      side_effect=[1000, 0]):
        assert not manager.prefill_acquire(1, handle)
    assert not handle.lock_node.is_locked
    manager.check_integrity()


def test_lock_projection_matches_shared_frames_and_fragmented_slots():
    import random

    manager = make_manager()
    history = torch.arange(256, dtype=torch.int32)
    req = new_req(manager, history)
    manager.allocate([req], phase="prefill")
    req.cached_len = len(history)
    manager.cache_req(req, finished=True)
    for position in (16, 80, 144):
        manager.tree.match_prefix(history[:position])
    nodes = [node for node in manager.tree._iter_nodes() if not node.is_root()]
    rng = random.Random(0)
    locks = []
    for _ in range(100):
        node = rng.choice(nodes)
        before = manager.admission_capacity
        kv, slots = manager.tree.lock_footprint(node)
        cost = kv + manager._allocator.lock_frame_cost(slots)
        assert manager.tree.lock_footprint(node) == (kv, slots)
        manager.tree.lock(node)
        assert before - manager.admission_capacity == cost
        locks.append(node)
        if rng.randrange(2):
            manager.tree.lock(locks.pop(rng.randrange(len(locks))), unlock=True)
    for node in locks:
        manager.tree.lock(node, unlock=True)
    manager.check_integrity()


def test_match_and_prefill_reuse_trace_until_tree_revision_changes():
    manager = make_manager(enable_merge_adjustment=True)
    history = torch.arange(256, dtype=torch.int32)
    req = new_req(manager, history)
    manager.allocate([req], phase="prefill")
    req.cached_len = len(history)
    manager.cache_req(req, finished=True)
    tokens = torch.cat((history, history[:64] + 1000))
    with patch.object(manager.tree, "trace", wraps=manager.tree.trace) as trace:
        continuation = new_req(manager, tokens)
    assert trace.call_count == 1  # Owner lookup shares the prefix match.
    expected = manager.analyzer.analyze(manager.tree, [tokens])
    with patch.object(manager.tree, "trace", side_effect=AssertionError):
        actual = manager.analyzer.analyze(
            manager.tree, [tokens], traces=manager._prefill_traces([continuation]),
        )
    assert actual == expected
    owner = manager.tree.entry_owner(manager.tree.unique_slots()[-1])
    manager.tree.protect(owner)
    with patch.object(manager.tree, "trace", wraps=manager.tree.trace) as trace:
        traces = manager._prefill_traces([continuation])
    assert trace.call_count == 1
    assert manager.analyzer.analyze(manager.tree, [tokens], traces=traces) == (
        manager.analyzer.analyze(manager.tree, [tokens])
    )


def test_cache_req_uses_insert_endpoint_without_rematching_tokens():
    manager = make_manager()
    req = new_req(manager, torch.arange(256, dtype=torch.int32))
    manager.allocate([req], phase="prefill")
    req.cached_len = len(req.input_ids)
    with patch.object(manager.tree, "match_prefix", side_effect=AssertionError):
        manager.cache_req(req, finished=True)
    manager.check_integrity()


def test_prefill_trace_extends_a_match_limited_by_sampler_last_token():
    manager = make_manager(granularity=32)
    history = torch.arange(256, dtype=torch.int32)
    req = new_req(manager, history)
    manager.allocate([req], phase="prefill")
    req.cached_len = len(history)
    manager.cache_req(req, finished=True)
    continuation = new_req(manager, history)
    assert continuation.cached_len == 224  # Matching excludes the sampler row.
    with patch.object(manager.tree, "trace", wraps=manager.tree.trace) as trace:
        traces = manager._prefill_traces([continuation])
    assert trace.call_count == 1
    assert traces[0].matched_len == len(history)


def test_prefill_trace_does_not_reuse_a_longer_pending_prompt_for_a_chunk():
    manager = make_manager()
    history = torch.arange(256, dtype=torch.int32)
    req = new_req(manager, history)
    manager.allocate([req], phase="prefill")
    req.cached_len = len(history)
    manager.cache_req(req, finished=True)
    continuation = new_req(manager, torch.cat((history, history[:64] + 1000)))
    continuation.input_ids = history[:128]
    with patch.object(manager.tree, "trace", wraps=manager.tree.trace) as trace:
        traces = manager._prefill_traces([continuation])
    assert trace.call_count == 1
    assert traces[0].matched_len == 128


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
        patch.object(manager, "_construction_plan", return_value=(
            (materialize, restore), ((restore,), (materialize,))
        )),
        patch(
            "logssms.kernel.linear_attention.AFFINE_LOG_STATE_KERNELS.restore",
            side_effect=lambda *_args: order.append("restore"),
        ),
        patch(
            "logssms.kernel.linear_attention.AFFINE_LOG_STATE_KERNELS.construct",
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
    chains, waves = manager._construction_plan((), plans)

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
        "logssms.kernel.linear_attention.AFFINE_LOG_STATE_KERNELS.construct",
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
    with patch.object(manager.analyzer, "_batch_branch_points", return_value=[]) as branches:
        manager.allocate([req], phase="decode")
        branches.assert_not_called()
        manager.allocate([req], phase="prefill")
        branches.assert_not_called()


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

    # Restore metadata must not pull a scalar from the device table. The CPU
    # ownership map already contains the same work slot.
    original_table = manager.resources.work_state_table
    manager.resources.work_state_table = SimpleNamespace(shape=original_table.shape)
    try:
        repeated = manager._make_restore_tasks(Batch(reqs=[req], phase="prefill"))
        assert repeated[0].output_slot == manager._work_slots[req.table_idx]
    finally:
        manager.resources.work_state_table = original_table

    chains, _ = manager._construction_plan(tasks, ())
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
    fused, _ = manager._construction_plan(tasks, (materialize,))
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
    source = next(node.merge_stats for node in manager.tree.path(manager.tree.trace(input_ids).node) if node.merge_stats is not None)
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


def test_manager_exports_online_merge_profile_and_cache_req_timing():
    with patch.dict(os.environ, {"LOGSSMS_PROFILE_SCHEDULER": "1"}):
        manager = make_manager(enable_merge_adjustment=True)
    req = new_req(manager, torch.arange(130, dtype=torch.int32))
    manager.allocate([req])
    req.cached_len = 130
    manager.cache_req(req, finished=True)

    pending = PendingRequest(
        uid=1, input_ids=torch.arange(150, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=16),
    )
    match = manager.match_req(pending)
    assert manager.prefill_acquire(PAGE_SIZE, match.cuda_handle)
    profile = manager.profile_result()
    online = profile["online_merge"]
    assert online["enabled"]
    assert online["min_sampling_len"] == 1024
    assert online["aggregate"]["owners_created"] == 1
    assert online["aggregate"]["hits_recorded"] == 1
    assert online["aggregate"]["advances_below_update_boundary"] == 1
    assert online["updates"] == []
    assert profile["cache_req_samples"][-1]["merge_stats_ms"] >= 0
    manager.unlock(match.cuda_handle)


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


def test_continuation_preserves_statistics_and_updates_across_turns():
    manager = make_manager(
        num_pages=32, num_states=32, enable_merge_adjustment=True,
        max_seq_len=2048,
    )
    stats = None
    previous_node = None
    for length in (576, 1152, 1728):
        # Each new prompt includes the previous request's actual cached tokens.
        req = new_req(manager, torch.arange(length, dtype=torch.int32))
        if stats is None:
            stats = req.cache_handle.merge_stats
            assert stats is not None
        else:
            assert req.cache_handle.merge_stats is stats
        manager.allocate([req])
        req.cached_len = length
        manager.cache_req(req, finished=True)
        node = manager.tree.trace(req.input_ids).node
        assert node.merge_stats is stats
        if previous_node is not None:
            assert previous_node.merge_stats is None
        previous_node = node
        assert stats.origin == 0
        assert stats.tail == length
    assert stats.intensity == 4
    assert stats.next_update_tail == 2048
    manager.check_integrity()


def test_internal_fork_keeps_original_statistics_and_starts_an_empty_owner():
    manager = make_manager(enable_merge_adjustment=True)
    req = new_req(manager, torch.arange(258, dtype=torch.int32))
    original = req.cache_handle.merge_stats
    manager.allocate([req])
    req.cached_len = 258
    manager.cache_req(req, finished=True)
    original.intensity = 6
    original.hit_positions.append(192)

    pending = PendingRequest(
        uid=1,
        input_ids=torch.cat((torch.arange(128, dtype=torch.int32),
                             torch.arange(128, dtype=torch.int32) + 10_000)),
        sampling_params=SamplingParams(max_tokens=16),
    )
    handle = manager.match_req(pending).cuda_handle
    assert handle.merge_stats is not original
    assert handle.merge_stats.origin == handle.merge_stats.tail == 128
    assert handle.merge_stats.intensity == 2
    assert not handle.merge_stats.hit_positions
    assert original.origin == 0 and original.tail == 256
    assert original.intensity == 6
    assert tuple(original.hit_positions) == (192,)
    assert manager.tree.trace(req.input_ids).node.merge_stats is original
    assert handle.source_merge_stats is original
    assert manager.prefill_acquire(PAGE_SIZE, handle)
    assert tuple(original.hit_positions) == (192, 128)
    assert not handle.merge_stats.hit_positions
    manager.unlock(handle)


def test_stale_continuation_retries_instead_of_sharing_a_writable_owner():
    manager = make_manager(enable_merge_adjustment=True)
    original = new_req(manager, torch.arange(130, dtype=torch.int32))
    manager.allocate([original])
    original.cached_len = 130
    manager.cache_req(original, finished=True)
    pending = PendingRequest(
        uid=1, input_ids=torch.arange(194, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=16),
    )
    first = manager.match_req(pending).cuda_handle
    second = manager.match_req(pending).cuda_handle
    assert first.merge_stats is second.merge_stats
    assert manager.prefill_acquire(PAGE_SIZE, first)
    assert not manager.prefill_acquire(PAGE_SIZE, second)
    retry = manager.match_req(pending).cuda_handle
    assert retry.merge_stats is not first.merge_stats
    assert retry.merge_stats.origin == 128
    assert manager.prefill_acquire(PAGE_SIZE, retry)
    manager.unlock(retry)
    manager.unlock(first)


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


def _gpu_tail_with_recycled_page(manager):
    history = torch.arange(256, dtype=torch.int32)
    frames, logs, states = manager.logstate_pool.reserve(kv=3, logs=1, works=1)
    manager.logstate_pool.commit_checkpoint(states[0])
    kv = torch.cat([frame * PAGE_SIZE + torch.arange(PAGE_SIZE, dtype=torch.int32)
                    for frame in frames])
    manager.tree.insert(history[:192], kv, (
        LogStateEntry(states[0], LogStateKind.CHECKPOINT, 0, 128, span_pages=2),
        LogStateEntry(logs[0], LogStateKind.LOG, 128, 192),
    ))
    manager._cached_page_tokens.update({frame * PAGE_SIZE: PAGE_SIZE for frame in frames})
    ghost = type(manager.tree)(PAGE_SIZE, torch.device("cpu"), GRANULARITY)
    node = ghost.insert(history, history + 1000, (
        LogStateEntry(40, LogStateKind.CHECKPOINT, 0, 128, span_pages=2),
        LogStateEntry(41, LogStateKind.LOG, 128, 192),
        LogStateEntry(42, LogStateKind.LOG, 192, 256),
    )).node
    manager.recycle_bin.remember(ghost, node)
    return logs[0]


def test_pending_match_reuses_layout_but_invalidates_on_resource_ownership_changes():
    manager = make_manager(num_pages=16, num_states=16, enable_merge_adjustment=True)
    slot = _gpu_tail_with_recycled_page(manager)
    pending = PendingRequest(7, torch.arange(320, dtype=torch.int32), SamplingParams(max_tokens=16))
    manager.prepare_prefill_requests([pending])
    with (
        patch.object(manager.tree, "trace", wraps=manager.tree.trace) as trace,
        patch.object(manager, "_plan_recycled_restore", wraps=manager._plan_recycled_restore) as plan,
    ):
        first = manager.match_req(pending)
        assert manager.match_req(pending) is first
        assert trace.call_count == plan.call_count == 1
        # Unrelated physical capacity changes need no new token comparisons.
        frames, _, _ = manager.logstate_pool.reserve(kv=1)
        assert manager.match_req(pending) is first
        manager.logstate_pool.release_kv(frames)
        owner = manager.tree.entry_owner(slot)
        manager.tree.lock(owner)
        second = manager.match_req(pending)
        assert second is not first
        assert trace.call_count == 1
        assert plan.call_count == 2
        assert not any(checkpoint.gpu_entries for checkpoint in second.cuda_handle.restore_plan.checkpoints)
        manager.tree.lock(owner, unlock=True)
        third = manager.match_req(pending)
        assert third is not second
        manager.tree.protect(owner)
        fourth = manager.match_req(pending)
        assert fourth is not third
        manager.recycle_bin.remember(manager.tree, owner)
        assert manager.match_req(pending) is not fourth
        assert trace.call_count == 2  # Only protection changed tree layout metadata.
        assert plan.call_count == 5
    manager.check_integrity()


def test_pending_branch_analysis_reuses_unchanged_inputs_and_refreshes_protection():
    manager = make_manager(num_pages=16, num_states=16)
    _gpu_tail_with_recycled_page(manager)
    pending = PendingRequest(7, torch.arange(320, dtype=torch.int32), SamplingParams(max_tokens=16))
    sibling = PendingRequest(8, torch.cat((pending.input_ids[:208], torch.arange(64) + 1000)).int(),
                             SamplingParams(max_tokens=16))
    with patch.object(manager.analyzer, "_batch_branch_points", wraps=manager.analyzer._batch_branch_points) as branches:
        manager.prepare_prefill_requests([pending, sibling])
        first = manager.match_req(pending)
        assert 208 in first.cuda_handle.restore_plan.protected_endpoints
        manager.prepare_prefill_requests([pending, sibling])
        assert manager.match_req(pending) is first
        assert branches.call_count == 1
        manager.prepare_prefill_requests([pending])
        second = manager.match_req(pending)
        assert second is not first
        assert 208 not in second.cuda_handle.restore_plan.protected_endpoints


def test_cached_recycle_match_preserves_lru_touch():
    manager = make_manager(num_pages=16, num_states=16)
    _gpu_tail_with_recycled_page(manager)
    pending = PendingRequest(7, torch.arange(320, dtype=torch.int32), SamplingParams(max_tokens=16))
    first = manager.match_req(pending)
    with patch.object(manager.recycle_bin, "touch", wraps=manager.recycle_bin.touch) as touch:
        assert manager.match_req(pending) is first
        touch.assert_called_once_with(first.cuda_handle.recycle_match.branch)


def test_match_reuse_profile_keeps_logical_recycle_hit_counts(monkeypatch):
    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", "1")
    manager = make_manager(num_pages=16, num_states=16)
    _gpu_tail_with_recycled_page(manager)
    pending = PendingRequest(7, torch.arange(320, dtype=torch.int32), SamplingParams(max_tokens=16))
    manager.prepare_prefill_requests([pending])
    first = manager.match_req(pending)
    assert manager.match_req(pending) is first
    assert manager._profile_stats["pending_match_reuses"] == 1
    assert manager._profile_stats["recycle_matches"] == 2
    manager.tree.lock(first.cuda_handle.lock_node)
    manager.match_req(pending)
    assert manager._profile_stats["prefix_match_reuses"] == 1
    assert manager._profile_stats["recycle_matches"] == 3
    manager.tree.lock(first.cuda_handle.lock_node, unlock=True)
    manager.prepare_prefill_requests([pending])
    assert manager._profile_stats["prefill_branch_reuses"] == 1


def test_pending_match_invalidates_when_input_tensor_is_replaced():
    manager = make_manager(num_pages=16, num_states=16)
    _gpu_tail_with_recycled_page(manager)
    pending = PendingRequest(7, torch.arange(320, dtype=torch.int32), SamplingParams(max_tokens=16))
    first = manager.match_req(pending)
    assert first.cuda_handle.cached_len == 192
    pending.input_ids = pending.input_ids + 10000
    second = manager.match_req(pending)
    assert second.cuda_handle.cached_len == 0
    assert second.cuda_handle.recycle_match is None


def test_pending_match_and_branch_analysis_invalidate_on_versioned_input_edits():
    manager = make_manager(num_pages=16, num_states=16)
    _gpu_tail_with_recycled_page(manager)
    pending = PendingRequest(7, torch.arange(320, dtype=torch.int32), SamplingParams(max_tokens=16))
    sibling = PendingRequest(8, pending.input_ids.clone(), SamplingParams(max_tokens=16))
    manager.prepare_prefill_requests([pending, sibling])
    assert manager.match_req(pending).cuda_handle.cached_len == 192
    pending.input_ids.add_(10000)
    with patch.object(manager.analyzer, "_batch_branch_points", wraps=manager.analyzer._batch_branch_points) as branches:
        manager.prepare_prefill_requests([pending, sibling])
        assert branches.call_count == 1
    assert manager.match_req(pending).cuda_handle.cached_len == 0


def test_recycle_merges_gpu_tail_with_missing_history_and_reuses_its_slot():
    manager = make_manager(num_pages=20, num_states=20)
    tail = _gpu_tail_with_recycled_page(manager)
    req = new_req(manager, torch.arange(321, dtype=torch.int32))
    assert req.cached_len == 192
    checkpoint, = req.cache_handle.restore_plan.checkpoints
    assert (checkpoint.begin, checkpoint.end) == (128, 256)
    assert checkpoint.gpu_entries[0].slot == tail
    before = manager._allocator.profile_usage()
    manager.allocate([req], phase="prefill")
    after = manager._allocator.profile_usage()
    assert after["work_slots"] == before["work_slots"]
    assert after["log_slots"] - before["log_slots"] == 2  # only new text, including its partial page
    assert manager._live_pages[req.table_idx][0].state_slot == tail
    assert torch.all(manager.state_table[req.table_idx, 192:256] == -1)
    batch = Batch([req], "prefill")
    with patch.object(manager, "_construct", return_value={}):
        manager.prepare_batch(batch)
    assert batch.logstate_checkpoint_writes == [(63, tail)]
    assert not manager.tree.has_slot(tail)
    assert manager.tree.match_prefix(req.input_ids[:192]).cuda_handle.cached_len == 128
    req.cached_len = 320
    manager.cache_req(req, finished=True)
    entries = manager.tree.iter_entries()
    assert [(entry.begin, entry.end, entry.kind) for entry in entries] == [
        (0, 128, LogStateKind.CHECKPOINT),
        (128, 256, LogStateKind.CHECKPOINT),
        (256, 320, LogStateKind.LOG),
    ]
    assert entries[1].slot == tail
    assert manager.tree.match_prefix(req.input_ids[:320]).cuda_handle.cached_len == 320
    manager.check_integrity()


@pytest.mark.parametrize("abort", [False, True])
def test_recycled_checkpoint_spans_prefill_chunks_without_intermediate_logs(abort):
    manager = make_manager(num_pages=20, num_states=20)
    history = torch.arange(256, dtype=torch.int32)
    ghost = type(manager.tree)(PAGE_SIZE, torch.device("cpu"), GRANULARITY)
    node = ghost.insert(history, history + 1000, (
        LogStateEntry(50, LogStateKind.CHECKPOINT, 0, 256, span_pages=4),
    )).node
    manager.recycle_bin.remember(ghost, node)
    tokens = torch.arange(257, dtype=torch.int32)
    req = new_req(manager, tokens)
    req.input_ids = tokens[:128]
    req.device_len = 128
    manager.allocate([req], phase="prefill")
    first = Batch([req], "prefill")
    with patch.object(manager, "_construct", return_value={}):
        manager.prepare_batch(first)
    assert first.logstate_checkpoint_writes == []
    assert first.logstate_checkpoint_ranges == [(0, 0, 256)]
    assert all(page.state_slot is None for page in manager._live_pages[0])
    req.cached_len = 128
    if abort:
        manager.free_to_cached(req)
        manager.cache_req(req, finished=True)
        assert manager.tree.size_info.total_size == 0
        assert manager.recycle_bin.match(history, start=0, granularity=16) is not None
    else:
        req.input_ids = tokens
        req.device_len = 257
        manager.allocate([req], phase="prefill")
        second = Batch([req], "prefill")
        with patch.object(manager, "_construct", return_value={}):
            manager.prepare_batch(second)
        slot = manager._recycled_restores[0].slots[256]
        assert second.logstate_checkpoint_writes == [(127, slot)]
        req.cached_len = 256
        manager.cache_req(req, finished=True)
        assert [(entry.begin, entry.end, entry.kind) for entry in manager.tree.iter_entries()] == [
            (0, 256, LogStateKind.CHECKPOINT),
        ]
    manager.check_integrity()


def test_abort_before_reused_checkpoint_endpoint_releases_detached_slot():
    manager = make_manager(num_pages=20, num_states=20)
    tail = _gpu_tail_with_recycled_page(manager)
    req = new_req(manager, torch.arange(321, dtype=torch.int32))
    # Cancellation after prefix construction, before any missing token was evaluated.
    req.device_len = 193
    req.input_ids = req.input_ids[:193]
    manager.allocate([req], phase="prefill")
    with patch.object(manager, "_construct", return_value={}):
        manager.prepare_batch(Batch([req], "prefill"))
    manager.free_to_cached(req)
    manager.cache_req(req, finished=True)
    assert tail not in manager._allocator.allocated_slots
    assert manager.tree.match_prefix(torch.arange(192, dtype=torch.int32)).cuda_handle.cached_len == 128
    manager.check_integrity()


def test_cancel_before_construction_keeps_the_original_gpu_tail():
    manager = make_manager(num_pages=20, num_states=20)
    tail = _gpu_tail_with_recycled_page(manager)
    req = new_req(manager, torch.arange(321, dtype=torch.int32))
    manager.allocate([req], phase="prefill")
    manager.free_to_cached(req)
    manager.cache_req(req, finished=True)
    assert tail in manager._allocator.allocated_slots
    assert manager.tree.entries_for_slot(tail)[0].kind is LogStateKind.LOG
    assert manager.tree.match_prefix(torch.arange(192, dtype=torch.int32)).cuda_handle.cached_len == 192
    manager.check_integrity()


def test_recycle_admission_counts_the_planned_shape_not_one_log_per_page():
    manager = make_manager(num_pages=20, num_states=20)
    _gpu_tail_with_recycled_page(manager)
    pending = PendingRequest(7, torch.arange(321, dtype=torch.int32), SamplingParams(max_tokens=16))
    handle = manager.match_req(pending).cuda_handle
    cost = manager.reservation_size(321 + 16 - 192, handle=handle)
    # Three new KV pages, two new Log pages, one request work state. The
    # historical checkpoint reuses a GPU slot and contributes no extra state.
    assert cost == manager._allocator.required_frames(3, 2, 1, free_decay_records=0)
    assert manager.prefill_acquire(cost, handle)
    manager.unlock(handle)


@pytest.mark.parametrize("abort_first", [False, True])
def test_shared_gpu_tail_has_only_one_recycled_slot_writer(abort_first):
    manager = make_manager(num_pages=20, num_states=20)
    tail = _gpu_tail_with_recycled_page(manager)
    tokens = torch.arange(321, dtype=torch.int32)
    first = new_req(manager, tokens, table_idx=0)
    second = new_req(manager, tokens, table_idx=1)
    assert first.cache_handle.restore_plan.checkpoints[0].gpu_entries[0].slot == tail
    assert all(not item.gpu_entries for item in second.cache_handle.restore_plan.checkpoints)
    manager.allocate([first, second], phase="prefill")
    batch = Batch([first, second], "prefill")
    with patch.object(manager, "_construct", return_value={}):
        manager.prepare_batch(batch)
    writes = [slot for _row, slot in batch.logstate_checkpoint_writes]
    assert len(writes) == len(set(writes)) == 2
    if abort_first:
        manager.free_to_cached(first)
    else:
        first.cached_len = 320
    manager.cache_req(first, finished=True)
    second.cached_len = 320
    manager.cache_req(second, finished=True)
    assert manager.tree.match_prefix(tokens[:320]).cuda_handle.cached_len == 320
    manager.check_integrity()


def test_recycled_batch_branch_preserves_the_split_physical_log_page():
    manager = make_manager(num_pages=20, num_states=20)
    history = torch.arange(256, dtype=torch.int32)
    ghost = type(manager.tree)(PAGE_SIZE, torch.device("cpu"), GRANULARITY)
    leaf = ghost.insert(history, history + 1000, (
        LogStateEntry(20, LogStateKind.CHECKPOINT, 0, 256, span_pages=4),
    )).node
    manager.recycle_bin.remember(ghost, leaf)
    first = PendingRequest(1, torch.arange(257, dtype=torch.int32), SamplingParams(max_tokens=16))
    second = PendingRequest(2, torch.cat((history[:96], history[96:] + 1000)), SamplingParams(max_tokens=16))
    manager.prepare_prefill_requests([first, second])
    plan = manager.match_req(first).cuda_handle.restore_plan
    assert 96 in plan.protected_endpoints
    assert [(entry.begin, entry.end) for entry in plan.checkpoints] == [(0, 64), (128, 256)]
    req = new_req(manager, first.input_ids, table_idx=1)
    manager.allocate([req], phase="prefill")
    assert manager._live_pages[1][1].state_slot is not None
    assert manager._live_pages[1][1].checkpoint_begin is None
    req.cached_len = 256
    manager.cache_req(req, finished=True)
    node = manager.tree.trace(history[:96]).node
    assert node.depth == 96 and node.is_protected
    manager.check_integrity()


def test_recycled_continuation_keeps_its_dynamic_merge_policy():
    from collections import deque
    from logssms.cache.prefix_cache.logstate import LogStateMergeStats

    manager = make_manager(num_pages=20, num_states=20, enable_merge_adjustment=True)
    ghost = type(manager.tree)(PAGE_SIZE, torch.device("cpu"), GRANULARITY)
    tokens = torch.arange(256, dtype=torch.int32)
    leaf = ghost.insert(tokens, tokens + 1000, tuple(
        LogStateEntry(index, LogStateKind.LOG, index * 64, (index + 1) * 64)
        for index in range(4)
    )).node
    leaf.merge_stats = LogStateMergeStats(0, 256, 6, deque([160, 192]), 1024)
    manager.recycle_bin.remember(ghost, leaf)
    pending = PendingRequest(1, torch.arange(257, dtype=torch.int32), SamplingParams(max_tokens=16))
    handle = manager.match_req(pending).cuda_handle
    assert handle.merge_stats.intensity == 6
    assert list(handle.merge_stats.hit_positions) == [160, 192]
    assert [(entry.begin, entry.end) for entry in handle.restore_plan.checkpoints] == [(0, 256)]
    assert handle.merge_stats is not leaf.merge_stats


def test_gpu_and_cpu_groups_use_one_merge_pass_and_one_intensity():
    from logssms.cache.prefix_cache.logstate import LogStateMergeStats

    manager = make_manager(num_pages=20, num_states=20, enable_merge_adjustment=True)
    history = torch.arange(320, dtype=torch.int32)
    frames, logs, _ = manager.logstate_pool.reserve(kv=3, logs=3)
    kv = torch.cat([frame * PAGE_SIZE + torch.arange(PAGE_SIZE, dtype=torch.int32) for frame in frames])
    manager.tree.insert(history[:192], kv, tuple(
        LogStateEntry(slot, LogStateKind.LOG, index * 64, (index + 1) * 64)
        for index, slot in enumerate(logs)
    ))
    manager._cached_page_tokens.update({frame * PAGE_SIZE: PAGE_SIZE for frame in frames})
    ghost = type(manager.tree)(PAGE_SIZE, torch.device("cpu"), GRANULARITY)
    leaf = ghost.insert(history, history + 1000, tuple(
        LogStateEntry(index, LogStateKind.LOG, index * 64, (index + 1) * 64)
        for index in range(5)
    )).node
    leaf.merge_stats = LogStateMergeStats(0, 320, 3, next_update_tail=1024)
    manager.recycle_bin.remember(ghost, leaf)
    req = new_req(manager, torch.arange(321, dtype=torch.int32))
    manager.allocate([req], phase="prefill")
    with patch.object(manager, "_construct", return_value={}):
        manager.prepare_batch(Batch([req], "prefill"))
    req.cached_len = 320
    manager.cache_req(req, finished=True)
    assert [(entry.begin, entry.end, entry.kind) for entry in manager.tree.iter_entries()] == [
        (0, 192, LogStateKind.CHECKPOINT), (192, 320, LogStateKind.CHECKPOINT),
    ]
    manager.check_integrity()


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
    cost = manager.reservation_size(128, handle=handle)
    assert manager.prefill_acquire(cost, handle, request_len=cost)
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
    assert manager.prefill_acquire(manager.admission_capacity, handle)
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
    assert manager.prefill_acquire(manager.reservation_size(2 * PAGE_SIZE, handle=handle), handle)
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
        reserved = sum(manager.reservation_size(req.remain_len, table_idx=req.table_idx) for req in reqs)
        cost = manager.reservation_size(2 * PAGE_SIZE, handle=handle)
        assert manager.prefill_acquire(reserved + cost, handle)
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
        "logssms.kernel.linear_attention.AFFINE_LOG_STATE_KERNELS.construct",
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
    with patch("logssms.kernel.linear_attention.AFFINE_LOG_STATE_KERNELS.construct", return_value=0):
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
    with patch("logssms.kernel.linear_attention.AFFINE_LOG_STATE_KERNELS.construct", return_value=0):
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
