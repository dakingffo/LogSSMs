from __future__ import annotations

import os
from unittest.mock import patch
from types import SimpleNamespace

import pytest
import torch

from logssms.cache import CacheResources
from logssms.cache.logstate_pool.mha_pool import MHALogStatePool
from logssms.cache.logstate import (
    LogStateCacheManager, LogStateCapacityError, LogStatePlan, PageMaterialize,
)
from logssms.cache.prefix_cache.logstate import LogStateCacheHandle
from logssms.cache.prefix_cache.logstate import LogStateEntry
from logssms.cache.prefix_cache.logstate import LogStateKind
from logssms.cache.logstate import LogStateRecycleBin, RecycledBranch, RecycledMatch
from logssms.core import Batch, Request, SamplingParams
from logssms.message import UserMsg
from logssms.scheduler.prefill import PendingRequest
from tests.logstate_helpers import remember_log_tail
from tests.logstate_helpers import matched_kv


PAGE_SIZE = 64
GRANULARITY = 16
MAX_RUNNING_REQ = 3
MAX_SEQ_LEN = 512


@pytest.fixture(autouse=True)
def cpu_cache_copy():
    # These tests check tree/allocator transitions, not CUDA kernel arithmetic.
    # Preserve actual snapshot-copy semantics without sending CPU pointers to Triton.
    def copy(pool, sources, source_granules, outputs, output_granules):
        for source, source_granule, output, output_granule in zip(
            sources.tolist(), source_granules.tolist(), outputs.tolist(), output_granules.tolist(), strict=True,
        ):
            pool.copy_conv_snapshot(source, source_granule, output, output_granule)

    from logssms.kernel.linear_attention.logstate import copy_states
    from logssms.kernel.linear_attention.logstate import restore_log_state_chains
    from tests.logstate_helpers import decay_view, head_view

    def restore(pool, chains):
        if pool.device.type == "cuda":
            return restore_log_state_chains(pool, chains)
        layout = pool.layout
        for base, logs, output in chains:
            for layer in range(layout.num_linear_layers):
                states = (torch.zeros(1, layout.num_value_heads, layout.head_k_dim, layout.head_v_dim)
                          if base is None else pool.read_states(torch.tensor([base]), layer).clone())
                for slot, begin, end in logs:
                    decay = decay_view(pool, slot)[layer]
                    keys, values = [], []
                    for head in range(layout.num_value_heads):
                        log = head_view(pool, slot, layer, head, state=False)
                        split = pool.page_size * layout.head_k_dim
                        keys.append(log[:split].view(pool.page_size, layout.head_k_dim))
                        values.append(log[split:].view(pool.page_size, layout.head_v_dim))
                    keys, values = torch.stack(keys, 1), torch.stack(values, 1)
                    for position in range(begin, end):
                        states *= decay[position].exp()[None, :, None, None]
                        states += keys[position, :, :, None] * values[position, :, None, :]
                pool.write_states(torch.tensor([output]), layer, states)

    def copy_state(pool, sources, destinations):
        if pool.device.type == "cuda":
            return copy_states(pool, sources, destinations)
        src = torch.tensor(sources, dtype=torch.int64)
        dst = torch.tensor(destinations, dtype=torch.int64)
        for layer in range(pool.layout.num_linear_layers):
            pool.write_states(dst, layer, pool.read_states(src, layer))

    with (
        patch("logssms.kernel.linear_attention.AFFINE_LOG_STATE_KERNELS.copy_conv", side_effect=copy),
        patch("logssms.kernel.linear_attention.logstate.copy_states", side_effect=copy_state),
        patch("logssms.kernel.linear_attention.AFFINE_LOG_STATE_KERNELS.restore", side_effect=restore),
    ):
        yield


def make_manager(
    num_pages             : int  = 8,
    num_states            : int  = 8,
    enable_merge_adjustment: bool = False,
    enable_memorized_evict: bool = True,
    max_seq_len           : int = MAX_SEQ_LEN,
    granularity           : int = GRANULARITY,
    device                : torch.device = torch.device("cpu"),
    value_head_ratio      : int = 1,
) -> LogStateCacheManager:
    num_work_states = MAX_RUNNING_REQ + 1
    cuda = device.type == "cuda"
    pool = MHALogStatePool(
        num_frames=(num_pages + 1) + (num_states + num_work_states + 1) * 7,
        page_size=PAGE_SIZE,
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
    manager.tree._checkpoint_timestamps[checkpoint.entries[0].slot] = 1
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
            side_effect=lambda *_args: order.append("materialize"),
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
        side_effect=lambda _pool, group: launches.append(group),
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
        (LogStateEntry(50, LogStateKind.LOG, 0, 64),
         LogStateEntry(51, LogStateKind.LOG, 64, 128)),
    ).node
    manager.recycle_bin = LogStateRecycleBin(16 * 1024)
    remember_log_tail(manager.recycle_bin, ghost_tree, ghost)
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
    frames, _, states = manager.logstate_pool.reserve(kv=4, works=1)
    manager.logstate_pool.commit_checkpoint(states[0])
    kv = torch.cat([frame * PAGE_SIZE + torch.arange(PAGE_SIZE, dtype=torch.int32)
                    for frame in frames])
    manager.tree.insert(history[:256], kv, (
        LogStateEntry(states[0], LogStateKind.CHECKPOINT, 0, 256, span_pages=4),
    ))
    manager._cached_page_tokens.update({frame * PAGE_SIZE: PAGE_SIZE for frame in frames})
    ghost_tree = type(manager.tree)(PAGE_SIZE, torch.device("cpu"), GRANULARITY)
    ghost_entries = (
        *(LogStateEntry(51 + i, LogStateKind.LOG, 256 + 64 * i, 320 + 64 * i)
          for i in range(4)),
    )
    ghost = ghost_tree.insert(history, history + 1000, ghost_entries).node
    manager.recycle_bin = LogStateRecycleBin(32 * 1024)
    remember_log_tail(manager.recycle_bin, ghost_tree, ghost)
    req = new_req(
        manager, torch.cat((history, torch.tensor([999], dtype=torch.int32)))
    )
    manager.allocate([req], phase="prefill")
    req.cached_len = 512
    with patch.object(manager, "_construct", wraps=manager._construct):
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
        *(LogStateEntry(50 + i, LogStateKind.LOG, 256 + 64 * i, 320 + 64 * i)
          for i in range(4)),
    )).node
    manager.recycle_bin = LogStateRecycleBin(32 * 1024)
    remember_log_tail(manager.recycle_bin, ghost_tree, ghost)
    req = new_req(
        manager, torch.cat((history, torch.tensor([999], dtype=torch.int32)))
    )
    assert req.cached_len == 16
    assert req.cache_handle.recycle_match.matched_tokens(16) == 256
    manager.allocate([req], phase="prefill")
    pages = manager._live_pages[req.table_idx]
    assert (pages[0].begin, pages[0].end) == (16, 64)
    assert (pages[4].begin, pages[4].end) == (256, 320)
    assert pages[4].state_slot is pages[6].state_slot is None
    assert pages[5].checkpoint_begin == 256
    assert (pages[7].begin, pages[7].end, pages[7].checkpoint_begin) == (
        448, 512, 384,
    )
    assert all(page.checkpoint_begin is None for page in pages[:4])


def test_recycled_log_tail_uses_current_gpu_shape_after_checkpoint_compression():
    from logssms.cache.logstate import CheckpointCoalesce

    manager = make_manager(num_pages=20, num_states=20, max_seq_len=640)
    history, _, (left, right) = cache_checkpoint_chain(manager, (2, 1))
    ghost = type(manager.tree)(PAGE_SIZE, torch.device("cpu"), GRANULARITY)
    tokens = torch.arange(320, dtype=torch.int32)
    node = ghost.insert(tokens, tokens + 1000, (
        LogStateEntry(70, LogStateKind.LOG, 192, 256),
        LogStateEntry(71, LogStateKind.LOG, 256, 320),
    )).node
    remember_log_tail(manager.recycle_bin, ghost, node)
    before = manager.recycle_bin.match(tokens, start=len(history), granularity=GRANULARITY)
    revision = manager.recycle_bin.revision
    released, _, skips = manager._commit_coalesce((
        CheckpointCoalesce((left.slot, right.slot), (2, 1), 0, 192, 3),
    ))
    assert released == 1 and skips == {}
    assert manager.recycle_bin.revision == revision
    after = manager.recycle_bin.match(tokens, start=192, granularity=GRANULARITY)
    assert after.branch is before.branch
    assert [(log.begin, log.end) for log in after.branch.logs] == [(192, 256), (256, 320)]

    req = new_req(manager, torch.arange(385, dtype=torch.int32))
    assert req.cached_len == 192
    assert [(entry.begin, entry.end, entry.slot) for entry in req.cache_handle.match_entries] == [
        (0, 192, right.slot),
    ]
    assert [(item.begin, item.end) for item in req.cache_handle.restore_plan.checkpoints] == [
        (192, 320),
    ]
    manager.allocate([req], phase="prefill")
    with patch.object(manager, "_construct", wraps=manager._construct):
        manager.prepare_batch(Batch([req], "prefill"))
    req.cached_len = 384
    manager.cache_req(req, finished=True)
    assert [(entry.begin, entry.end, entry.kind) for entry in manager.tree.iter_entries()] == [
        (0, 192, LogStateKind.CHECKPOINT),
        (192, 320, LogStateKind.CHECKPOINT),
        (320, 384, LogStateKind.LOG),
    ]
    manager.check_integrity()


def test_recycled_history_matches_unpruned_merge_layout():
    manager = make_manager(num_pages=20, num_states=20, max_seq_len=640)
    tokens = torch.arange(577, dtype=torch.int32)
    frames, logs, states = manager.logstate_pool.reserve(kv=4, logs=2, works=1)
    manager.logstate_pool.commit_checkpoint(states[0])
    kv = torch.cat([frame * PAGE_SIZE + torch.arange(PAGE_SIZE, dtype=torch.int32)
                    for frame in frames])
    manager.tree.insert(tokens[:256], kv, (
        LogStateEntry(states[0], LogStateKind.CHECKPOINT, 0, 128, span_pages=2),
        LogStateEntry(logs[0], LogStateKind.LOG, 128, 192),
        LogStateEntry(logs[1], LogStateKind.LOG, 192, 256),
    ))
    manager._cached_page_tokens.update({frame * PAGE_SIZE: PAGE_SIZE for frame in frames})
    ghost = type(manager.tree)(PAGE_SIZE, torch.device("cpu"), GRANULARITY)
    node = ghost.insert(tokens[:448], tokens[:448] + 1000, tuple(
        LogStateEntry(70 + i, LogStateKind.LOG, 256 + i * 64, 320 + i * 64)
        for i in range(3)
    )).node
    remember_log_tail(manager.recycle_bin, ghost, node)
    req = new_req(manager, tokens)
    assert req.cached_len == 256
    manager.allocate([req], phase="prefill")
    with patch.object(manager, "_construct", wraps=manager._construct):
        manager.prepare_batch(Batch([req], "prefill"))
    req.cached_len = 576
    manager.cache_req(req, finished=True)
    # The same historical round as [128, 64, 64, 64, 64, 64], then two new pages.
    assert [(entry.end - entry.begin, entry.kind) for entry in manager.tree.iter_entries()] == [
        (128, LogStateKind.CHECKPOINT),
        (128, LogStateKind.CHECKPOINT),
        (128, LogStateKind.CHECKPOINT),
        (64, LogStateKind.CHECKPOINT),
        (64, LogStateKind.LOG),
        (64, LogStateKind.LOG),
    ]
    manager.check_integrity()


def test_recycled_half_page_keeps_log_storage_and_restores_only_its_suffix():
    manager = make_manager(num_pages=12, num_states=12, granularity=32)
    tokens = torch.arange(129, dtype=torch.int32)
    frames, logs, _ = manager.logstate_pool.reserve(kv=1, logs=1)
    kv = frames[0] * PAGE_SIZE + torch.arange(32, dtype=torch.int32)
    manager.tree.insert(tokens[:32], kv, (
        LogStateEntry(logs[0], LogStateKind.LOG, 0, 32),
    ), cache_len=32)
    manager._cached_page_tokens[frames[0] * PAGE_SIZE] = 32
    ghost = type(manager.tree)(PAGE_SIZE, torch.device("cpu"), 32)
    node = ghost.insert(tokens[:64], tokens[:64] + 1000, (
        LogStateEntry(70, LogStateKind.LOG, 32, 64),
    )).node
    manager.recycle_bin.remember(ghost, node)
    req = new_req(manager, tokens)
    assert req.cached_len == 32
    assert req.cache_handle.recycle_match.matched_tokens(32) == 32
    assert req.cache_handle.restore_plan.checkpoints == ()
    manager.allocate([req], phase="prefill")
    pages = manager._live_pages[0]
    assert (pages[0].begin, pages[0].end) == (32, 64)
    assert pages[0].state_slot is not None and pages[0].checkpoint_begin is None
    with patch.object(manager, "_construct", wraps=manager._construct):
        manager.prepare_batch(Batch([req], "prefill"))
    req.cached_len = 128
    manager.cache_req(req, finished=True)
    assert [(entry.begin, entry.end, entry.kind) for entry in manager.tree.iter_entries()] == [
        (0, 32, LogStateKind.LOG),
        (32, 64, LogStateKind.LOG),
        (64, 128, LogStateKind.LOG),
    ]
    assert manager.tree.match_prefix(tokens[:128]).recoverable_len == 128
    manager.check_integrity()


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


@pytest.mark.parametrize("head_ratio", [1, 2, 4])
def test_recycle_merges_gpu_tail_with_missing_history_and_reuses_its_slot(head_ratio):
    manager = make_manager(num_pages=20, num_states=20, value_head_ratio=head_ratio)
    tail = _gpu_tail_with_recycled_page(manager)
    req = new_req(manager, torch.arange(321, dtype=torch.int32))
    assert req.cached_len == 192
    checkpoint, = req.cache_handle.restore_plan.checkpoints
    assert (checkpoint.begin, checkpoint.end) == (128, 256)
    assert checkpoint.gpu_entries[0].slot == tail
    before = manager._allocator.profile_usage()
    manager.allocate([req], phase="prefill")
    after = manager._allocator.profile_usage()
    assert after["work_slots"] - before["work_slots"] == int(head_ratio > 1)
    assert after["log_slots"] - before["log_slots"] == 2  # only new text, including its partial page
    destination = manager._live_pages[req.table_idx][0].state_slot
    assert (destination == tail) == (head_ratio == 1)
    assert len(manager._allocator.slot(destination).payload) == manager._allocator.payload_frames
    assert torch.all(manager.state_table[req.table_idx, 192:256] == -1)
    batch = Batch([req], "prefill")
    with (
        patch.object(manager, "_construct", wraps=manager._construct),
        patch.object(manager, "_construct_checkpoints", return_value=({}, (0, 0, {}))),
    ):
        manager.prepare_batch(batch)
    assert batch.logstate_checkpoint_writes == [(63, destination)]
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
    assert entries[1].slot == destination
    assert manager.tree.match_prefix(req.input_ids[:320]).cuda_handle.cached_len == 320
    manager.check_integrity()


@pytest.mark.parametrize("abort", [False, True])
def test_recycled_checkpoint_spans_prefill_chunks_without_intermediate_logs(abort):
    from logssms.cache.prefix_cache.logstate import LogStateMergeStats

    manager = make_manager(num_pages=20, num_states=20, enable_merge_adjustment=True)
    history = torch.arange(256, dtype=torch.int32)
    ghost = type(manager.tree)(PAGE_SIZE, torch.device("cpu"), GRANULARITY)
    node = ghost.insert(history, history + 1000, (
        *(LogStateEntry(50 + i, LogStateKind.LOG, 64 * i, 64 * (i + 1))
          for i in range(4)),
    )).node
    node.merge_stats = LogStateMergeStats(0, 256, 4, next_update_tail=1024)
    remember_log_tail(manager.recycle_bin, ghost, node)
    tokens = torch.arange(257, dtype=torch.int32)
    req = new_req(manager, tokens)
    req.input_ids = tokens[:128]
    req.device_len = 128
    manager.allocate([req], phase="prefill")
    first = Batch([req], "prefill")
    with patch.object(manager, "_construct", wraps=manager._construct):
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
        with patch.object(manager, "_construct", wraps=manager._construct):
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
    with patch.object(manager, "_construct", wraps=manager._construct):
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
    with patch.object(manager, "_construct", wraps=manager._construct):
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
        *(LogStateEntry(20 + i, LogStateKind.LOG, 64 * i, 64 * (i + 1))
          for i in range(4)),
    )).node
    remember_log_tail(manager.recycle_bin, ghost, leaf)
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
    remember_log_tail(manager.recycle_bin, ghost, leaf)
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
    remember_log_tail(manager.recycle_bin, ghost, leaf)
    req = new_req(manager, torch.arange(321, dtype=torch.int32))
    manager.allocate([req], phase="prefill")
    with patch.object(manager, "_construct", wraps=manager._construct):
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
        matched_kv(manager.tree, branch.cache_handle.node),
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


def cache_checkpoint_chain(manager, spans):
    tokens = torch.arange(sum(spans) * PAGE_SIZE, dtype=torch.int32)
    req = new_req(manager, tokens)
    manager.allocate([req])
    req.cached_len = len(tokens)
    manager.cache_req(req, finished=True)
    leaf = manager.tree.trace(tokens).node
    begin = 0
    checkpoints = []
    for span in spans:
        end = begin + span * PAGE_SIZE
        entries = [entry for entry in manager.tree.iter_entries()
                   if begin <= entry.begin and entry.end <= end]
        right = max(entries, key=lambda entry: entry.end)
        for entry in entries:
            manager.tree.detach_entry(entry.slot)
        manager.logstate_pool.release_slots([entry.slot for entry in entries if entry != right])
        manager.logstate_pool.materialize(right.slot)
        checkpoint = LogStateEntry(right.slot, LogStateKind.CHECKPOINT, begin, end, span_pages=span)
        manager.tree.attach_entry(leaf, checkpoint)
        checkpoints.append(checkpoint)
        begin = end
    manager.check_integrity()
    return tokens, leaf, checkpoints


@pytest.mark.parametrize("memorized", [False, True])
def test_pressure_drops_old_checkpoint_without_kv_eviction_or_prefill_restore(memorized):
    with patch.dict(os.environ, {"LOGSSMS_PROFILE_SCHEDULER": "1"}):
        manager = make_manager(enable_memorized_evict=memorized)
    tokens, leaf, (left, right) = cache_checkpoint_chain(manager, (4, 2))
    pool = manager.logstate_pool
    right_slot = torch.tensor([right.slot], dtype=torch.int64)
    values = torch.randn_like(pool.read_states(right_slot, 0))
    pool.write_states(right_slot, 0, values)
    kv = matched_kv(manager.tree, leaf).clone()
    record = pool.allocator.slot(left.slot)
    released_frames = len(record.payload) + sum(map(len, record.conv))

    pages, states, works = manager._allocate(pool.allocator.free_frames + released_frames, 0)

    assert not manager.tree.has_slot(left.slot)
    assert manager.tree.entries_for_slot(right.slot) == (
        LogStateEntry(right.slot, LogStateKind.CHECKPOINT, 0, 384, span_pages=6),
    )
    torch.testing.assert_close(pool.read_states(right_slot, 0), values)
    torch.testing.assert_close(matched_kv(manager.tree, leaf), kv)
    handle = manager.tree.match_prefix(tokens).cuda_handle
    assert handle.cached_len == len(tokens) and handle.recovery.logs == ()
    assert handle.recovery.base_checkpoint_slot == right.slot
    if memorized:
        assert len(manager.recycle_bin) == 0
    event = manager.profile_result()["lifecycle_events"][-1]
    assert event["nodes"] == [] and event["kv_tokens"] == 0
    assert event["dropped_checkpoints"][0]["slot"] == left.slot
    assert manager.profile_result()["aggregate"]["pressure_dropped_checkpoints"] == 1
    assert not len(works)
    manager._return_slots(pages, states)
    manager.check_integrity()


def test_heat_thinning_preserves_recovery_with_nonmonotonic_endpoint_timestamps():
    manager = make_manager()
    tokens, leaf, (left, middle, right) = cache_checkpoint_chain(manager, (2, 2, 2))
    manager.tree._checkpoint_timestamps[left.slot] = 999
    manager.tree._checkpoint_timestamps[middle.slot] = 1
    pool = manager.logstate_pool
    right_slot = torch.tensor([right.slot], dtype=torch.int64)
    values = torch.randn_like(pool.read_states(right_slot, 0))
    pool.write_states(right_slot, 0, values)
    released_frames = sum(
        len(record.payload) + sum(map(len, record.conv))
        for record in (pool.allocator.slot(left.slot), pool.allocator.slot(middle.slot))
    )
    needed_pages = pool.allocator.free_frames + released_frames
    assert manager._plan_eviction(needed_pages, 0, 0)[3] == [
        (left.slot, middle.slot), (middle.slot, right.slot),
    ]
    pages, states, _ = manager._allocate(needed_pages, 0)
    assert manager.tree.matched_entries(leaf) == (
        LogStateEntry(right.slot, LogStateKind.CHECKPOINT, 0, 384, span_pages=6),
    )
    torch.testing.assert_close(pool.read_states(right_slot, 0), values)
    handle = manager.tree.match_prefix(tokens).cuda_handle
    assert handle.cached_len == len(tokens)
    assert handle.recovery.base_checkpoint_slot == right.slot
    assert not len(manager.recycle_bin)
    manager._return_slots(pages, states)
    manager.check_integrity()


def test_active_kv_chain_can_thin_states_without_touching_its_work():
    manager = make_manager()
    tokens, leaf, entries = cache_checkpoint_chain(manager, (1, 1, 1, 1))
    kv = matched_kv(manager.tree, leaf).clone()
    # The active decoder only reads its private work; the historical path stays
    # KV-locked, but middle States remain valid pressure-eviction candidates.
    handle = manager.tree.make_handle(leaf)
    manager.lock(handle)
    _, _, works = manager._allocate(0, 0, 1)
    work = int(works[0])
    pool = manager.logstate_pool
    work_indices = torch.tensor([work], dtype=torch.int64)
    expected = torch.randn_like(pool.read_states(work_indices, 0))
    pool.write_states(work_indices, 0, expected)
    manager.tree._checkpoint_timestamps.update({entry.slot: i for i, entry in enumerate(entries)})
    middle = entries[1]
    record = manager._allocator.slot(middle.slot)
    frames = len(record.payload) + sum(map(len, record.conv))
    pages, logs, _ = manager._allocate(manager._allocator.free_frames + frames, 0)
    assert not manager.tree.has_slot(middle.slot)
    assert manager.tree.has_slot(entries[0].slot)
    assert leaf.is_locked and manager.tree.evictable_kv_frames() == 0
    torch.testing.assert_close(matched_kv(manager.tree, leaf), kv)
    torch.testing.assert_close(pool.read_states(work_indices, 0), expected)
    manager._return_slots(pages, logs)
    pool.release_slots([work])
    manager.unlock(handle)
    manager.check_integrity()


def test_recovery_inputs_are_pinned_only_until_work_is_constructed():
    manager = make_manager()
    tokens, _, entries = cache_checkpoint_chain(manager, (1, 1, 1))
    pending = PendingRequest(17, torch.cat((tokens[:128], torch.tensor([999], dtype=torch.int32))),
                             SamplingParams(max_tokens=16))
    handle = manager.match_req(pending).cuda_handle
    assert manager.prefill_acquire(0, handle)
    assert manager.tree._recovery_slot_refs == {entries[1].slot: 1}
    req = Request(input_ids=pending.input_ids, table_idx=0, cached_len=128,
                  output_len=16, uid=17, sampling_params=pending.sampling_params,
                  cache_handle=handle, max_device_len=145)
    manager.init_with_handle(0, handle)
    expected = torch.randn_like(manager.logstate_pool.read_states(
        torch.tensor([entries[1].slot]), 0,
    ))
    manager.logstate_pool.write_states(torch.tensor([entries[1].slot]), 0, expected)
    manager.allocate([req], phase="prefill")
    # Isolate ownership/copy semantics from materialization CUDA arithmetic.
    with patch.object(manager.analyzer, "analyze", return_value=LogStatePlan((), (), ())):
        manager.prepare_batch(Batch([req], "prefill"))
    assert manager.tree._recovery_slot_refs == {}
    assert manager._recovery_pins == {}
    torch.testing.assert_close(manager.logstate_pool.read_states(
        torch.tensor([manager._work_slots[0]]), 0,
    ), expected)
    assert handle.lock_node.is_locked
    assert any(pair[0] == entries[1].slot for pair in manager.tree.checkpoint_eviction_pairs())
    req.cached_len = 129
    manager.cache_req(req, finished=True)
    manager.check_integrity()


def test_checkpoint_coalesce_preserves_the_retained_endpoint_access_time():
    from logssms.cache.logstate import CheckpointCoalesce

    manager = make_manager()
    _, leaf, (left, right) = cache_checkpoint_chain(manager, (1, 1))
    manager.tree._checkpoint_timestamps.update({left.slot: 10, right.slot: 20})
    plan = CheckpointCoalesce((left.slot, right.slot), (1, 1), 0, 128, 2)
    with patch("logssms.cache.prefix_cache.logstate.time.monotonic_ns", return_value=100):
        released, _, skips = manager._commit_coalesce((plan,))
    assert released == 1 and skips == {}
    assert manager.tree._checkpoint_timestamps == {right.slot: 20}
    assert manager.tree.recovery_at(leaf, 128).base_checkpoint_slot == right.slot
    manager.check_integrity()


def test_pending_and_prefix_match_reuse_touch_only_recovery_checkpoint():
    manager = make_manager()
    tokens, _, entries = cache_checkpoint_chain(manager, (1, 1, 1))
    slots = [entry.slot for entry in entries]
    manager.tree._checkpoint_timestamps.update(dict(zip(slots, (10, 20, 30))))
    pending = PendingRequest(77, torch.cat((tokens, torch.tensor([999], dtype=torch.int32))),
                             SamplingParams(max_tokens=16))
    manager.prepare_prefill_requests([pending])
    with patch("logssms.cache.prefix_cache.logstate.time.monotonic_ns", return_value=100):
        first = manager.match_req(pending)
    with patch("logssms.cache.prefix_cache.logstate.time.monotonic_ns", return_value=200):
        assert manager.match_req(pending) is first
    assert manager.tree._checkpoint_timestamps == dict(zip(slots, (10, 20, 200)))
    # Rebuild the outer result while retaining the token-prefix match cache.
    manager._prefill_matches.clear()
    with patch("logssms.cache.prefix_cache.logstate.time.monotonic_ns", return_value=300):
        again = manager.match_req(pending)
    assert again.cuda_handle.recovery.base_checkpoint_slot == slots[-1]
    assert manager.tree._checkpoint_timestamps == dict(zip(slots, (10, 20, 300)))
    assert all(node.timestamp == 300 for node in again.cuda_handle.match_path)
    manager.check_integrity()


def test_last_checkpoint_is_pruned_without_creating_a_recycle_record():
    manager = make_manager()
    tokens, leaf, (checkpoint,) = cache_checkpoint_chain(manager, (2,))
    record = manager._allocator.slot(checkpoint.slot)
    frames = len(record.payload) + sum(map(len, record.conv))
    pages, states, _ = manager._allocate(manager._allocator.free_frames + frames, 0)
    assert not manager.tree.has_slot(checkpoint.slot)
    assert not len(manager.recycle_bin)
    assert manager.recycle_bin.match(tokens, start=0, granularity=GRANULARITY) is None
    manager._return_slots(pages, states)
    manager.check_integrity()


def test_unfulfillable_pressure_compression_plan_does_not_change_checkpoint_layout():
    manager = make_manager()
    tokens, leaf, entries = cache_checkpoint_chain(manager, (4, 2))
    revision = manager.tree.revision
    free_frames = manager._allocator.free_frames
    with pytest.raises(LogStateCapacityError):
        manager._allocate(1000, 1000)
    assert manager.tree.matched_entries(leaf) == tuple(entries)
    assert manager.tree.revision == revision
    assert manager._allocator.free_frames == free_frames
    assert not len(manager.recycle_bin)
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


def test_prepare_batch_preserves_log_at_protected_branch():
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
        "logssms.kernel.linear_attention.AFFINE_LOG_STATE_KERNELS.restore",
        wraps=manager.logstate_pool.kernels.restore,
    ) as restore:
        manager.prepare_batch(Batch(reqs=[branch], phase="prefill"))

    assert restore.call_count == 1
    assert len(restore.call_args.args[1][0][1]) == 2
    assert manager.last_plan is not None
    assert any(manager.tree.entry_is_protected(entry.slot) for entry in manager.tree.iter_entries())
    assert any(
        entry.kind is LogStateKind.LOG
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
    with patch("logssms.kernel.linear_attention.AFFINE_LOG_STATE_KERNELS.construct", return_value=None):
        manager.prepare_batch(batch)

    checkpoints = manager.tree.iter_entries()
    assert len(checkpoints) == 1
    assert checkpoints[0].kind is LogStateKind.CHECKPOINT
    assert checkpoints[0].begin == 0
    assert checkpoints[0].end == 128
    assert checkpoints[0].span_pages == 2

    manager.unlock(request.cache_handle)
    manager.check_integrity()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("pages,intensity,head_ratio,blocked", [
    (1, 2, 1, None), (2, 2, 1, None), (7, 3, 1, None),
    (1, 2, 4, None), (2, 2, 4, None), (4, 2, 4, None),
    (7, 3, 2, None), (7, 6, 3, None), (9, 8, 4, None), (17, 16, 3, None),
    (3, 2, 3, "gain"), (4, 3, 3, "gain"), (7, 6, 3, "gain"),
    (9, 8, 3, "gain"), (17, 16, 3, "gain"), (4, 3, 3, "capacity"),
])
def test_prepare_only_materializes_merge_outputs_and_recovers_work(pages, intensity, head_ratio, blocked):
    from logssms.kernel.linear_attention.logstate import store_log_states

    torch.manual_seed(401)
    manager = make_manager(num_pages=64, num_states=64, max_seq_len=2048,
                           device=torch.device("cuda"), value_head_ratio=head_ratio,
                           enable_merge_adjustment=True)
    pool = manager.logstate_pool
    heads = pool.layout.num_value_heads
    length = pages * PAGE_SIZE
    ids = torch.arange(length + 2, dtype=torch.int32)
    original = new_req(manager, ids)
    manager.allocate([original], phase="prefill")
    live = manager._live_pages[original.table_idx][:pages]
    slots = torch.tensor([page.state_slot for page in live], device="cuda", dtype=torch.int32).repeat_interleave(64)
    offsets = torch.arange(length, device="cuda", dtype=torch.int32) % 64
    keys = torch.randn(length, 2, 32, device="cuda", dtype=pool.dtype) * .01
    values = torch.randn(length, heads, 32, device="cuda", dtype=pool.dtype) * .01
    decays = -torch.rand(length, heads, device="cuda") * .01
    store_log_states(pool, slots, offsets, keys, values, decays, 0)
    for page in live:
        for granule in range(pool.layout.num_granules):
            pool.conv_view(page.state_slot, granule).zero_()
    expected = torch.zeros(heads, 32, 32, device="cuda")
    expanded = keys.repeat_interleave(head_ratio, 1).float()
    for position in range(length):
        expected *= decays[position].exp()[:, None, None]
        expected += expanded[position, :, :, None] * values[position, :, None, :].float()
    original.cached_len = len(ids)
    manager.cache_req(original, finished=True)
    leaf = manager.tree.leaves()[0]
    assert leaf.merge_stats is not None
    leaf.merge_stats.intensity = intensity
    leaf.merge_stats.next_update_tail = length + 4096
    request = new_req(manager, ids, table_idx=1)
    assert request.cached_len == length  # Exact real token prefix remains reusable.
    frames = [frame for page in live for frame in pool.allocator.slot(page.state_slot).payload]
    source = pool.storage[frames].clone()
    if blocked:
        field = "checkpoint_frame_savings" if blocked == "gain" else "checkpoint_frame_capacity"
        limit = 0 if blocked == "gain" else pool.allocator.payload_frames - 1
        with patch.object(pool.allocator, field, return_value=limit):
            manager.prepare_batch(Batch([request], "prefill"))
        # No persistent Log value or its allocation may be overwritten just
        # because a stronger merge was planned. Only work is reconstructed.
        torch.testing.assert_close(pool.storage[frames], source, atol=0, rtol=0)
    else:
        manager.prepare_batch(Batch([request], "prefill"))
    torch.testing.assert_close(
        pool.read_states(torch.tensor([manager._work_slots[1]], device="cuda"), 0)[0].float(),
        expected, atol=2e-4, rtol=5e-2,
    )
    entries = manager.tree.iter_entries()
    assert sum(entry.end - entry.begin for entry in entries) == length
    groups = [1] * pages if blocked else [intensity] * (pages // intensity)
    remainder = pages % intensity
    if not blocked and remainder:
        groups.append(remainder)
    assert [entry.end - entry.begin for entry in entries] == [span * PAGE_SIZE for span in groups]
    for entry, span in zip(entries, groups, strict=True):
        if span == 1:
            assert entry.kind is LogStateKind.LOG
        else:
            assert entry.kind is LogStateKind.CHECKPOINT and entry.span_pages == span
    manager.unlock(request.cache_handle)
    torch.cuda.synchronize()
    manager.check_integrity()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("branch", [False, True])
def test_shared_key_merge_preserves_multiple_request_recoveries(branch):
    from logssms.kernel.linear_attention.logstate import store_log_states

    torch.manual_seed(402)
    manager = make_manager(num_pages=32, num_states=32, device=torch.device("cuda"),
                           granularity=32, value_head_ratio=4)
    pool = manager.logstate_pool
    ids = torch.arange(258, dtype=torch.int32)
    original = new_req(manager, ids)
    manager.allocate([original], phase="prefill")
    pages = manager._live_pages[0][:4]
    slots = torch.tensor([page.state_slot for page in pages], device="cuda", dtype=torch.int32).repeat_interleave(64)
    positions = torch.arange(256, device="cuda", dtype=torch.int32) % 64
    keys = torch.randn(256, 2, 32, device="cuda", dtype=pool.dtype) * .01
    values = torch.randn(256, 8, 32, device="cuda", dtype=pool.dtype) * .01
    decay = -torch.rand(256, 8, device="cuda") * .01
    store_log_states(pool, slots, positions, keys, values, decay, 0)
    for page in pages:
        for granule in range(pool.layout.num_granules):
            pool.conv_view(page.state_slot, granule).zero_()
    original.cached_len = len(ids)
    manager.cache_req(original, finished=True)
    first = new_req(manager, ids, table_idx=1)
    second_ids = torch.cat([ids[:32], ids[32:] + 1000]) if branch else ids
    second = new_req(manager, second_ids, table_idx=2)
    assert (first.cached_len, second.cached_len) == (256, 32 if branch else 256)
    manager.prepare_batch(Batch([first, second], "prefill"))

    expected = torch.zeros(8, 32, 32, device="cuda")
    expanded = keys.repeat_interleave(4, 1).float()
    partial = None
    for position in range(256):
        expected *= decay[position].exp()[:, None, None]
        expected += expanded[position, :, :, None] * values[position, :, None, :].float()
        if position == 31:
            partial = expected.clone()
    for req, reference in ((first, expected), (second, partial if branch else expected)):
        actual = pool.read_states(torch.tensor([manager._work_slots[req.table_idx]], device="cuda"), 0)[0]
        torch.testing.assert_close(actual.float(), reference, atol=2e-4, rtol=5e-2)
        manager.unlock(req.cache_handle)
    assert any(entry.kind is LogStateKind.CHECKPOINT for entry in manager.tree.iter_entries())
    if branch:
        assert manager.tree.has_slot(pages[0].state_slot)
        assert pool.allocator.slot(pages[0].state_slot).kind == "log"
    torch.cuda.synchronize()
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
    with patch("logssms.kernel.linear_attention.AFFINE_LOG_STATE_KERNELS.construct", return_value=None):
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
