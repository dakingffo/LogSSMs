from collections import deque
import random
from unittest.mock import patch

import pytest
import torch

from logssms.cache.logstate import (
    OnlineAnalysisScheduler,
    _MergePathContext,
)
from logssms.cache.prefix_cache.logstate import (
    LogRadixTree,
    LogRadixTreeNode,
    LogStateEntry,
    LogStateKind,
    LogStateMergeStats,
)


PAGE_SIZE = 64
GRANULARITY = 16


def test_batch_branch_points_match_granule_trie_oracle():
    rng = random.Random(711)
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)
    for _ in range(100):
        paths = []
        for _ in range(rng.randint(1, 24)):
            parent = rng.choice(paths) if paths else ()
            shared = parent[:rng.randrange(len(parent) + 1)]
            path = shared + tuple(rng.randrange(4) for _ in range(rng.randrange(8)))
            paths.append(path)
        inputs = [torch.tensor(
            [value for value in path for _ in range(GRANULARITY)]
            + [9] * rng.randrange(GRANULARITY), dtype=torch.int32,
        ) for path in paths]
        expected = {}
        prefixes = {path[:end] for path in paths for end in range(1, len(path) + 1)}
        for prefix in prefixes:
            indices = {i for i, path in enumerate(paths) if path[:len(prefix)] == prefix}
            terminals = sum(paths[i] == prefix for i in indices)
            children = {paths[i][len(prefix)] for i in indices if len(paths[i]) > len(prefix)}
            if len(children) + int(terminals > 0) >= 2 or terminals >= 2:
                tokens = tuple(value for value in prefix for _ in range(GRANULARITY))
                expected[tokens] = indices
        assert scheduler._batch_branch_points(inputs) == expected


def ids(length: int, begin: int = 0) -> torch.Tensor:
    return torch.arange(begin, begin + length, dtype=torch.int32)


def make_history() -> tuple[LogRadixTree, torch.Tensor]:
    tree = LogRadixTree(
        PAGE_SIZE, torch.device("cpu"), granularity=GRANULARITY
    )
    input_ids = ids(256)
    entries = (
        LogStateEntry(
            0, LogStateKind.CHECKPOINT, 0, 64, span_pages=1
        ),
        LogStateEntry(1, LogStateKind.LOG, 64, 128),
        LogStateEntry(2, LogStateKind.LOG, 128, 192),
        LogStateEntry(3, LogStateKind.LOG, 192, 256),
    )
    tree.insert(input_ids, ids(256, 1000), entries)
    return tree, input_ids


def test_branch_inside_log_page_restores_work_state_and_protects_page():
    tree, history = make_history()
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)
    round_info = scheduler.begin_round(tree)
    request = torch.cat([history[:80], ids(32, 1000)])

    plan = scheduler.analyze(tree, [request], round_info)

    assert len(plan.protections) == 1
    assert plan.protections[0].position == 80
    assert plan.protections[0].log_slot == 1
    recovery = tree.match_prefix(request).cuda_handle.recovery
    assert recovery.position == 80
    assert recovery.base_checkpoint_slot == 0
    assert recovery.log_slots == (1,)
    assert plan.materialize == ()


def test_branch_at_checkpoint_uses_checkpoint_and_blocks_crossing_coalesce():
    tree, history = make_history()
    request = torch.cat([history[:64], ids(32, 1000)])

    handle = tree.match_prefix(request).cuda_handle
    assert handle.cached_len == 64
    assert handle.recovery.position == 64
    assert handle.recovery.base_checkpoint_slot == 0
    assert handle.recovery.logs == ()
    assert handle.recovery.conv_slot == 0
    assert handle.recovery.conv_granule == 3

    plan = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY).analyze(
        tree, [request]
    )
    assert len(plan.protections) == 1
    assert plan.protections[0].node is not None
    assert plan.protections[0].node.depth == 64
    assert not any(
        task.begin < 64 < task.end for task in plan.coalesce
    )


def test_protected_log_page_is_not_materialized_but_later_pages_are():
    tree, _history = make_history()
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)
    tree.protect(tree.entry_owner(1))

    plan = scheduler.analyze(tree, [_history])

    assert {task.log_slot for task in plan.materialize} == {2, 3}


def test_adjacent_page_checkpoints_coalesce_one_level_per_round():
    tree, _history = make_history()
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)
    leaf = tree.leaves()[0]
    tree.detach_entry(2)
    tree.detach_entry(3)
    tree.attach_entry(
        leaf,
        LogStateEntry(
            4, LogStateKind.CHECKPOINT, 128, 192, span_pages=1
        ),
    )
    tree.attach_entry(
        leaf,
        LogStateEntry(
            5, LogStateKind.CHECKPOINT, 192, 256, span_pages=1
        ),
    )

    plan = scheduler.analyze(tree, [_history])

    task = next(
        task
        for task in plan.coalesce
        if (task.left_slot, task.right_slot) == (4, 5)
    )
    assert (task.left_slot, task.right_slot) == (4, 5)
    assert (task.begin, task.end, task.next_span) == (128, 256, 2)


def test_materialized_checkpoints_coalesce_in_the_same_round():
    tree, _history = make_history()
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)

    plan = scheduler.analyze(tree, [_history])

    assert {task.log_slot for task in plan.materialize} == {1, 2, 3}
    assert {
        (task.left_slot, task.right_slot, task.next_span)
        for task in plan.coalesce
    } == {(0, 1, 2), (2, 3, 2)}


def test_internal_match_endpoint_plans_merges_only_on_the_hit_path():
    tree = LogRadixTree(
        PAGE_SIZE, torch.device("cpu"), granularity=GRANULARITY
    )
    shared = ids(4 * PAGE_SIZE)
    first = torch.cat([shared, ids(PAGE_SIZE, 10_000)])
    second = torch.cat([shared, ids(PAGE_SIZE, 20_000)])
    entries = tuple(
        LogStateEntry(
            slot,
            LogStateKind.CHECKPOINT,
            slot * PAGE_SIZE,
            (slot + 1) * PAGE_SIZE,
            span_pages=1,
        )
        for slot in range(5)
    )
    tree.insert(first, ids(len(first), 1000), entries)
    tree.insert(second, ids(len(second), 2000))
    endpoint = tree.trace(shared)
    assert endpoint.matched_len == len(shared)
    assert not endpoint.node.is_leaf()

    plan = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY).analyze(
        tree, [shared], collect_diagnostics=True
    )

    assert tuple(task.slots for task in plan.coalesce) == ((0, 1), (2, 3))
    diagnostics = dict(plan.merge_diagnostics)
    assert diagnostics["internal_endpoints"] == 1
    assert diagnostics["merge_anchors"] == 1
    assert diagnostics["planned_groups"] == 2
    assert diagnostics["planned_slot_releases"] == 2
    assert diagnostics["entries_beyond_match_endpoint"] == 0
    assert 4 not in {
        slot for task in plan.coalesce for slot in task.slots
    }


def test_adaptive_internal_endpoint_merges_without_an_owner():
    tree, leaf = make_checkpoint_history(4)
    leaf.merge_stats = LogStateMergeStats(origin=0, tail=256, intensity=4)
    shared = ids(128)
    branch = torch.cat([shared, ids(128, 10_000)])
    tree.insert(branch, ids(256, 20_000))
    endpoint = tree.trace(shared)
    assert not endpoint.node.is_leaf()
    assert endpoint.node.merge_stats is None

    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE, GRANULARITY, enable_merge_adjustment=True
    )
    plan = scheduler.analyze(tree, [shared])

    assert tuple(task.slots for task in plan.coalesce) == ((0, 1),)


def test_checkpoint_branch_remains_protected_in_later_batch():
    tree, _leaf = make_checkpoint_history(2)
    history = ids(2 * PAGE_SIZE)
    branch = torch.cat([history[:PAGE_SIZE], ids(PAGE_SIZE, 10_000)])
    tree.match_prefix(branch)
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)
    first = scheduler.analyze(tree, [branch])
    assert len(first.protections) == 1
    assert first.protections[0].node is not None
    tree.protect(first.protections[0].node)
    tree.insert(branch, ids(2 * PAGE_SIZE, 20_000))

    later = scheduler.analyze(tree, [history])

    assert not any(task.begin < PAGE_SIZE < task.end for task in later.coalesce)


def test_analysis_only_maintains_chains_touched_by_prefill_batch():
    tree = LogRadixTree(
        PAGE_SIZE, torch.device("cpu"), granularity=GRANULARITY
    )
    first = ids(128)
    second = ids(128, 1000)
    tree.insert(
        first,
        ids(128, 2000),
        (
            LogStateEntry(0, LogStateKind.LOG, 0, 64),
            LogStateEntry(1, LogStateKind.LOG, 64, 128),
        ),
    )
    tree.insert(
        second,
        ids(128, 3000),
        (
            LogStateEntry(2, LogStateKind.LOG, 0, 64),
            LogStateEntry(3, LogStateKind.LOG, 64, 128),
        ),
    )
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)

    plan = scheduler.analyze(tree, [first])

    assert {task.log_slot for task in plan.materialize} == {0, 1}
    assert all(
        slot not in {2, 3}
        for task in plan.coalesce
        for slot in task.slots
    )


def test_empty_prefill_batch_does_not_trigger_tree_maintenance():
    tree, _history = make_history()

    plan = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY).analyze(tree, [])

    assert plan.materialize == ()
    assert plan.coalesce == ()


def test_active_handle_boundary_blocks_checkpoint_coalesce():
    tree, _history = make_history()
    leaf = tree.leaves()[0]
    handle = tree.make_handle(leaf, PAGE_SIZE)
    tree.lock_handle(handle)

    plan = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY).analyze(
        tree, [_history]
    )

    assert not any(
        task.left_slot == 0 and task.right_slot == 1
        for task in plan.coalesce
    )
    assert all(task.begin >= PAGE_SIZE for task in plan.coalesce)


def test_branch_between_checkpoints_blocks_coalesce_without_covering_log():
    tree = LogRadixTree(
        PAGE_SIZE, torch.device("cpu"), granularity=GRANULARITY
    )
    history = ids(128)
    tree.insert(
        history,
        ids(128, 1000),
        (
            LogStateEntry(
                0, LogStateKind.CHECKPOINT, 0, 64, span_pages=1
            ),
            LogStateEntry(
                1, LogStateKind.CHECKPOINT, 64, 128, span_pages=1
            ),
        ),
    )
    request = torch.cat([history[:96], ids(32, 1000)])

    plan = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY).analyze(
        tree, [request], collect_diagnostics=True
    )

    assert len(plan.protections) == 0
    assert tree.match_prefix(request).cuda_handle.cached_len == 64
    assert len(plan.coalesce) == 0
    diagnostics = dict(plan.merge_diagnostics)
    assert diagnostics["entries_beyond_match_endpoint"] == 1
    assert diagnostics["segments_without_merge_pair"] == 1
    assert diagnostics["planned_groups"] == 0


def test_new_log_pages_are_not_touched_in_their_birth_round():
    tree, _history = make_history()
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)
    round_info = scheduler.begin_round(tree)
    leaf = tree.leaves()[0]
    tree.detach_entry(3)
    tree.attach_entry(leaf, LogStateEntry(6, LogStateKind.LOG, 192, 256))

    plan = scheduler.analyze(tree, [_history], round_info)

    assert {task.log_slot for task in plan.materialize} == {1, 2}


def test_partial_physical_log_page_is_replay_only():
    tree = LogRadixTree(
        PAGE_SIZE, torch.device("cpu"), granularity=GRANULARITY
    )
    tree.insert(
        ids(80),
        ids(80, 1000),
        (
            LogStateEntry(
                0, LogStateKind.CHECKPOINT, 0, 64, span_pages=1
            ),
            LogStateEntry(
                1,
                LogStateKind.LOG,
                64,
                80,
                storage_begin=64,
            ),
        ),
        cache_len=80,
    )

    plan = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY).analyze(
        tree, [ids(80)]
    )

    assert plan.materialize == ()


def make_checkpoint_history(
    count: int,
) -> tuple[LogRadixTree, LogRadixTreeNode]:
    tree = LogRadixTree(
        PAGE_SIZE, torch.device("cpu"), granularity=GRANULARITY
    )
    length = count * PAGE_SIZE
    tree.insert(
        ids(length),
        ids(length, 1000),
        tuple(
            LogStateEntry(
                slot,
                LogStateKind.CHECKPOINT,
                slot * PAGE_SIZE,
                (slot + 1) * PAGE_SIZE,
                span_pages=1,
            )
            for slot in range(count)
        ),
    )
    return tree, tree.leaves()[0]


def test_native_merge_path_metadata_matches_tree_checks():
    tree, leaf = make_checkpoint_history(6)
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)
    tree.protect_entry(2)
    branch = torch.cat([ids(80), ids(32, 10_000)])
    pending = scheduler._find_protections(tree, [branch])
    assert pending

    path = tree.path(leaf)
    context = _MergePathContext.build(
        tree, path, scheduler._entries_for_node(tree, leaf), pending,
    )
    assert context.protected_depths == tuple(sorted(
        node.depth for node in path if node.is_protected
    ))
    assert context.protected_entries == tuple(
        entry for entry in scheduler._entries_for_node(tree, leaf)
        if tree.entry_is_protected(entry.slot)
    )
    assert context.pending_positions == tuple(sorted(
        value.position for value in pending if value.trace.node in path
    ))
    for position in range(0, 6 * PAGE_SIZE + 1, GRANULARITY):
        assert (position in context.lock_boundaries) == tree.is_lock_boundary(
            leaf, position,
        )


def test_adaptive_merge_uses_intensity_as_a_maximum_group_size():
    expected = {
        2: ((0, 1),),
        7: (tuple(range(7)),),
        15: (tuple(range(8)), tuple(range(8, 15))),
        16: (tuple(range(8)), tuple(range(8, 16))),
    }
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE,
        GRANULARITY,
        enable_merge_adjustment=True,
    )

    for count, groups in expected.items():
        tree, leaf = make_checkpoint_history(count)
        leaf.merge_stats = LogStateMergeStats(
            origin=0,
            tail=count * PAGE_SIZE,
            intensity=8,
        )

        plan = scheduler.analyze(tree, [ids(count * PAGE_SIZE)])

        assert tuple(task.slots for task in plan.coalesce) == groups


def test_adaptive_merge_keeps_different_physical_spans_distinct():
    tree, leaf = make_checkpoint_history(10)
    for slot in range(10):
        tree.detach_entry(slot)
    tree.attach_entry(
        leaf, LogStateEntry(
            10, LogStateKind.CHECKPOINT, 0, 8 * PAGE_SIZE, span_pages=8
        )
    )
    tree.attach_entry(
        leaf, LogStateEntry(
            11, LogStateKind.CHECKPOINT,
            8 * PAGE_SIZE, 10 * PAGE_SIZE, span_pages=2,
        )
    )
    leaf.merge_stats = LogStateMergeStats(0, 10 * PAGE_SIZE, intensity=8)
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE, GRANULARITY, enable_merge_adjustment=True
    )

    assert scheduler.analyze(tree, [ids(10 * PAGE_SIZE)]).coalesce == ()


def test_adaptive_statistics_origin_does_not_block_safe_prefix_merges():
    tree, leaf = make_checkpoint_history(8)
    leaf.merge_stats = LogStateMergeStats(
        origin=4 * PAGE_SIZE,
        tail=8 * PAGE_SIZE,
        intensity=4,
    )
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE,
        GRANULARITY,
        enable_merge_adjustment=True,
    )

    plan = scheduler.analyze(tree, [ids(8 * PAGE_SIZE)])

    assert tuple(task.slots for task in plan.coalesce) == (
        (0, 1, 2, 3), (4, 5, 6, 7),
    )


def test_adaptive_merge_absorbs_older_lower_level_checkpoint():
    tree, leaf = make_checkpoint_history(6)
    for slot in range(6):
        tree.detach_entry(slot)
    tree.attach_entry(
        leaf,
        LogStateEntry(6, LogStateKind.CHECKPOINT, 0, 64, span_pages=1),
    )
    tree.attach_entry(
        leaf,
        LogStateEntry(7, LogStateKind.CHECKPOINT, 64, 192, span_pages=2),
    )
    tree.attach_entry(
        leaf,
        LogStateEntry(8, LogStateKind.CHECKPOINT, 192, 384, span_pages=3),
    )
    leaf.merge_stats = LogStateMergeStats(0, 384, intensity=8)
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE,
        GRANULARITY,
        enable_merge_adjustment=True,
    )

    plan = scheduler.analyze(tree, [ids(6 * PAGE_SIZE)])

    assert tuple(task.slots for task in plan.coalesce) == ((6, 7, 8),)
    assert plan.coalesce[0].next_span == 6


def test_merge_rounds_follow_nondecreasing_spans_without_remerging_outputs():
    tree = LogRadixTree(PAGE_SIZE, torch.device("cpu"), GRANULARITY)
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)
    next_slot = 0
    tail = 0

    def append_logs(count):
        nonlocal next_slot, tail
        end = tail + count * PAGE_SIZE
        values = tuple(
            LogStateEntry(next_slot + i, LogStateKind.LOG,
                          tail + i * PAGE_SIZE, tail + (i + 1) * PAGE_SIZE)
            for i in range(count)
        )
        tree.insert(ids(end), ids(end, 10_000), values)
        next_slot += count
        tail = end

    def maintain(intensity, appended):
        leaf = tree.trace(ids(tail)).node
        leaf.merge_stats = LogStateMergeStats(0, tail, intensity=intensity)
        plan = scheduler.analyze(tree, [ids(tail + appended * PAGE_SIZE)])
        for task in plan.materialize:
            entry = tree.detach_entry(task.log_slot)
            tree.attach_entry(leaf, LogStateEntry(
                entry.slot, LogStateKind.CHECKPOINT, entry.begin, entry.end,
                span_pages=1,
            ))
        for task in plan.coalesce:
            for slot in task.slots:
                tree.detach_entry(slot)
            tree.attach_entry(leaf, LogStateEntry(
                task.right_slot, LogStateKind.CHECKPOINT, task.begin, task.end,
                span_pages=task.next_span,
            ))
        append_logs(appended)
        return [entry.end - entry.begin for entry in tree.iter_entries()]

    append_logs(3)
    assert maintain(3, 4) == [192, 64, 64, 64, 64]
    assert maintain(6, 3) == [192, 256, 64, 64, 64]
    assert maintain(8, 1) == [448, 192, 64]


@pytest.mark.parametrize("adaptive", [False, True])
def test_two_nondecreasing_intervals_merge_even_below_the_intensity(adaptive):
    tree, leaf = make_checkpoint_history(3)
    tree.detach_entry(1)
    tree.detach_entry(2)
    tree.attach_entry(leaf, LogStateEntry(
        3, LogStateKind.CHECKPOINT, 64, 192, span_pages=2,
    ))
    leaf.merge_stats = LogStateMergeStats(0, 192, intensity=8)
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE, GRANULARITY, enable_merge_adjustment=adaptive,
    )
    plan = scheduler.analyze(tree, [ids(192)])
    assert tuple(task.slots for task in plan.coalesce) == ((0, 3),)
    assert plan.coalesce[0].next_span == 3


def test_statistics_on_ancestors_do_not_create_repeated_merge_anchors():
    tree, leaf = make_checkpoint_history(8)
    for node in tree.path(leaf):
        node.merge_stats = LogStateMergeStats(node.begin, node.depth, intensity=2)
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)
    plan = scheduler.analyze(tree, [ids(8 * PAGE_SIZE)], collect_diagnostics=True)
    assert dict(plan.merge_diagnostics)["merge_anchors"] == 1
    assert len(plan.coalesce) == 4


def test_merge_sampling_retires_completed_interval_hits():
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE,
        GRANULARITY,
        enable_merge_adjustment=True,
        min_sampling_len=1024,
    )
    stats = LogStateMergeStats(
        origin=0,
        tail=1023,
        hit_positions=deque((900, 100, 700)),
    )

    scheduler.advance_merge_stats(stats, 1024)

    assert tuple(stats.hit_positions) == ()


def test_merge_sampling_fits_log_log_slope_and_floors_intensity():
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE,
        GRANULARITY,
        enable_merge_adjustment=True,
        min_sampling_len=1024,
    )
    tail = 1024
    hits = []
    for index in range(8):
        distance = int((index + 0.5) * PAGE_SIZE)
        count = round(1e8 / distance ** 3.1)
        hits.extend([tail - distance] * count)
    stats = LogStateMergeStats(
        origin=0,
        tail=tail - 1,
        hit_positions=deque(hits),
    )

    scheduler.advance_merge_stats(stats, tail)

    assert stats.intensity == 3


def test_merge_sampling_multiplies_once_per_completed_interval():
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE,
        GRANULARITY,
        enable_merge_adjustment=True,
        min_sampling_len=128,
        max_merge_intensity=5,
        default_merge_growth_rate=2,
    )
    stats = LogStateMergeStats(origin=0, tail=0)

    scheduler.advance_merge_stats(stats, 64)
    assert stats.intensity == 2
    scheduler.advance_merge_stats(stats, 128)
    assert stats.intensity == 4
    scheduler.advance_merge_stats(stats, 192)
    assert stats.intensity == 4
    scheduler.advance_merge_stats(stats, 256)
    assert stats.intensity == 5


def test_merge_sampling_uses_complete_nonoverlapping_token_intervals():
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE, GRANULARITY,
        enable_merge_adjustment=True,
        min_sampling_len=1024,
        default_merge_growth_rate=2,
    )
    stats = scheduler.make_merge_stats(0)
    assert stats is not None
    scheduler.advance_merge_stats(stats, 1023)
    assert stats.intensity == 2
    scheduler.advance_merge_stats(stats, 1024)
    assert stats.intensity == 4
    scheduler.advance_merge_stats(stats, 1536)
    assert stats.intensity == 4
    scheduler.advance_merge_stats(stats, 2048)
    assert stats.intensity == 8


def test_online_merge_profile_reports_sampling_and_fallback_intervals():
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE, GRANULARITY, min_sampling_len=128, profile_enabled=True,
    )
    stats = scheduler.make_merge_stats(0)
    assert stats is not None
    scheduler.advance_merge_stats(stats, 64, owner_uuid=10)
    scheduler.record_hit(None, 64)
    scheduler.record_hit(stats, 65)
    scheduler.advance_merge_stats(stats, 128, owner_uuid=10)
    assert stats.intensity == 4
    scheduler.advance_merge_stats(stats, 192, owner_uuid=10)
    scheduler.record_hit(stats, 192)
    scheduler.advance_merge_stats(stats, 256, owner_uuid=10)
    assert stats.intensity == 8

    profile = scheduler.profile_result()
    assert profile is not None
    aggregate = profile["aggregate"]
    assert aggregate["owners_created"] == 1
    assert aggregate["hit_attempts"] == 3
    assert aggregate["hits_without_owner"] == 1
    assert aggregate["hits_after_tail"] == 1
    assert aggregate["hits_recorded"] == 1
    assert aggregate["interval_updates"] == aggregate["fallback_updates"] == 2
    assert aggregate["no_hits"] == aggregate["insufficient_bins"] == 1
    assert aggregate["advances_below_update_boundary"] == 2
    assert aggregate["sampled_hits"] == aggregate["retired_hits"] == 1
    assert [update["owner_uuid"] for update in profile["updates"]] == [10, 10]
    assert [update["window_end"] for update in profile["updates"]] == [128, 256]
    assert [update["intensity"] for update in profile["updates"]] == [4, 8]
    assert all(value >= 0 for value in profile["timing_ms"].values())


@pytest.mark.parametrize("near_hits, expected, clamp", [(1, 2, "min"), (100, 3, "max")])
def test_online_merge_profile_reports_fit_clamping(near_hits, expected, clamp):
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE, GRANULARITY, max_merge_intensity=3, profile_enabled=True,
    )
    stats = LogStateMergeStats(
        0, 1023, hit_positions=deque([992] * near_hits + [928]),
    )
    scheduler.advance_merge_stats(stats, 1024)

    profile = scheduler.profile_result()
    update = profile["updates"][0]
    assert stats.intensity == expected
    assert update["reason"] == "fitted"
    assert update["clamp"] == clamp
    if near_hits == 1:
        assert update["r_squared"] is None
    else:
        assert update["r_squared"] == pytest.approx(1)
    assert update["exponent"] == -update["slope"]
    assert update["sampled_hits"] == near_hits + 1
    assert update["occupied_bins"] == 2
    assert profile["aggregate"]["fit_updates"] == 1


def test_online_merge_profile_does_not_change_fits_or_queue_retirement():
    baseline = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY, min_sampling_len=128)
    profiled = OnlineAnalysisScheduler(
        PAGE_SIZE, GRANULARITY, min_sampling_len=128, profile_enabled=True,
    )
    original = LogStateMergeStats(0, 127, hit_positions=deque([96, 16, 80, 192]))
    measured = LogStateMergeStats(0, 127, hit_positions=deque(original.hit_positions))
    for tail in (128, 192, 256, 384):
        baseline.advance_merge_stats(original, tail)
        profiled.advance_merge_stats(measured, tail)
        assert original == measured
    assert baseline.profile_result() is None


def test_disabled_online_merge_profile_does_not_read_clock():
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)
    stats = scheduler.make_merge_stats(0)
    with patch("logssms.cache.logstate.perf_counter_ns", side_effect=AssertionError):
        scheduler.record_hit(stats, 0)
        scheduler.advance_merge_stats(stats, 1024)
    assert scheduler.profile_result() is None


def test_online_merge_profile_bounds_update_history_without_losing_totals():
    scheduler = OnlineAnalysisScheduler(
        PAGE_SIZE, GRANULARITY, min_sampling_len=128, profile_enabled=True,
    )
    scheduler._merge_profile["updates"] = deque(maxlen=1)
    stats = scheduler.make_merge_stats(0)
    scheduler.advance_merge_stats(stats, 384)
    profile = scheduler.profile_result()
    assert profile["aggregate"]["interval_updates"] == 3
    assert len(profile["updates"]) == 1
    assert profile["updates"][0]["window_end"] == 384
    assert profile["dropped_updates"] == 2


def test_online_merge_profile_reports_actual_plan_arity_and_spans():
    tree, leaf = make_checkpoint_history(7)
    leaf.merge_stats = LogStateMergeStats(0, 7 * PAGE_SIZE, intensity=8)
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)
    plan = scheduler.analyze(tree, [ids(7 * PAGE_SIZE)], collect_diagnostics=True)
    diagnostics = dict(plan.merge_diagnostics)
    assert diagnostics["anchors_intensity_8"] == 1
    assert diagnostics["planned_arity_7"] == 1
    assert diagnostics["same_span_groups"] == 1
    assert diagnostics["cross_span_groups"] == 0
    assert diagnostics["planned_covered_tokens"] == 7 * PAGE_SIZE


def test_analysis_matches_each_request_only_once():
    tree, history = make_history()
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)
    request = torch.cat([history, ids(64, 10000)])
    with patch.object(tree, "trace", wraps=tree.trace) as trace:
        plan = scheduler.analyze(tree, [request])
    assert trace.call_count == 1
    assert len(plan.materialize) == 3


def test_round_snapshots_unique_slots_without_scanning_entries():
    tree, history = make_history()
    # Splitting a log creates two logical entries with one physical slot.
    tree.match_prefix(torch.cat([history[:80], ids(32, 10000)]))
    expected = frozenset(entry.slot for entry in tree.iter_entries())
    assert tree.slot_is_fragmented(1)
    with patch.object(tree, "iter_entries", side_effect=AssertionError):
        actual = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY).begin_round(tree)
    assert actual.eligible_slots == expected


def test_analysis_reuses_owner_paths_within_batch_but_not_between_batches():
    tree, history = make_history()
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY)
    request = torch.cat([history, ids(64, 10000)])
    with patch.object(tree, "path", wraps=tree.path) as path:
        first = scheduler.analyze(tree, [request])
    # Matched path + one maximal materialize path + packing.
    # Coalescing shares that snapshot; prefixes are not packed individually.
    assert path.call_count == 3
    assert len(first.materialize) == 3
    tree.protect_entry(1)
    later = scheduler.analyze(tree, [request])
    assert 1 not in {task.log_slot for task in later.materialize}


def test_fused_sampling_preserves_unsorted_future_and_boundary_hits():
    scheduler = OnlineAnalysisScheduler(PAGE_SIZE, GRANULARITY, min_sampling_len=128)
    stats = LogStateMergeStats(
        0, 127, hit_positions=deque([192, 96, 256, 16, 128, 64]),
    )
    scheduler.advance_merge_stats(stats, 128)
    assert tuple(stats.hit_positions) == (192, 256, 128)
    scheduler.advance_merge_stats(stats, 256)
    assert tuple(stats.hit_positions) == (256,)
