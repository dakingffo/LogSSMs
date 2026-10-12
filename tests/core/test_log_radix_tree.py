import torch
import random
from unittest.mock import patch

from logssms.cache.prefix_cache.logstate import (
    LogRadixTree,
    LogStateEntry,
    LogStateKind,
)
from tests.logstate_helpers import matched_kv


PAGE_SIZE = 4


def make_tree() -> LogRadixTree:
    return LogRadixTree(PAGE_SIZE, torch.device("cpu"), granularity=1)


def ids(length: int, begin: int = 0) -> torch.Tensor:
    return torch.arange(begin, begin + length, dtype=torch.int32)


def test_incremental_reclamation_tracks_splits_locks_entries_eviction_and_reset():
    rng = random.Random(0)
    tree = make_tree()
    history = ids(128)
    leaf = tree.insert(history, ids(128, 1000), tuple(
        LogStateEntry(page, LogStateKind.LOG, page * 4, (page + 1) * 4)
        for page in range(32)
    )).node
    locks = []
    for _ in range(200):
        action = rng.randrange(4)
        if action == 0:
            handle = tree.match_prefix(history[:rng.randrange(1, 129)]).cuda_handle
            tree.lock_handle(handle)
            locks.append(handle)
        elif action == 1 and locks:
            tree.lock_handle(locks.pop(rng.randrange(len(locks))), unlock=True)
        elif action == 2:
            nodes = [node for node in tree._iter_nodes() if node.length > 1]
            if nodes:
                node = rng.choice(nodes)
                tree.split_at(node, rng.randrange(1, node.length))
        else:
            candidates = list(tree.iter_evict_candidates())
            if candidates:
                tree.evict_nodes(candidates[:1])
        tree.check_integrity()
    for handle in locks:
        tree.lock_handle(handle, unlock=True)
    tree.check_integrity()
    tree.reset()
    assert tree.evictable_kv_frames() == 0
    assert tree.evictable_slots() == ()
    tree.check_integrity()


def test_match_prefix_compares_tokens_only_once_and_preserves_raw_trace():
    tree = LogRadixTree(64, torch.device("cpu"), granularity=16)
    history = ids(128)
    tree.insert(history, ids(128, 1000), (
        LogStateEntry(0, LogStateKind.LOG, 0, 64),
    ))
    request = torch.cat((history[:96], ids(32, 10000)))
    with patch.object(tree, "_tree_walk", side_effect=AssertionError):
        result = tree.match_prefix(request)
    assert result.tree_match_len == 96
    assert result.cuda_handle.cached_len == 64
    assert result.cuda_handle.match_trace == tree.trace(request)
    assert result.cuda_handle.trace_revision == tree.revision
    tree.check_integrity()


def test_matched_kv_is_lazy_and_retains_parts_across_edge_splits():
    tree = make_tree()
    history, kv = ids(12), ids(12, 100)
    tree.insert(history, kv, (
        LogStateEntry(0, LogStateKind.CHECKPOINT, 0, 12, span_pages=3),
    ))
    with patch("torch.cat", wraps=torch.cat) as cat:
        handle = tree.match_prefix(history).cuda_handle
        assert cat.call_count == 0
        tree.lock_handle(handle)
        tree.split_at(handle.node, 4)
        torch.testing.assert_close(handle.kv_indices, kv)
        assert cat.call_count == 1
        assert handle.get_matched_indices() is handle.kv_indices
        assert cat.call_count == 1
        tree.lock_handle(handle, unlock=True)
    tree.check_integrity()


def test_empty_lazy_indices_keep_the_cache_device():
    tree = LogRadixTree(64, torch.device("meta"), granularity=32)
    for handle in (tree.match_prefix(ids(1)).cuda_handle, tree.make_handle(tree.root)):
        assert handle.kv_indices.device.type == "meta"
        assert handle.kv_indices.dtype == torch.int32
        assert handle.kv_indices.numel() == 0


def test_node_depth_survives_entry_splits_and_new_branches():
    tree = make_tree()
    history = ids(128)
    leaf = tree.insert(history, ids(128, 1000)).node
    tree.insert(torch.cat([history[:39], ids(13, 2000)]), ids(52, 3000))
    for end in range(4, 129, 4):
        tree.attach_entry(leaf, LogStateEntry(end, LogStateKind.LOG, end - 4, end))
    for node in tree._iter_nodes():
        depth = 0
        current = node
        while not current.is_root():
            depth += current.length
            current = current.parent
        assert node.depth == depth
    assert leaf.depth == 128
    tree.check_integrity()


def test_insert_match_and_collect_path_values():
    tree = make_tree()
    token_ids = ids(12)
    kv = ids(12, 100)
    entries = (
        LogStateEntry(0, LogStateKind.LOG, 0, 4),
        LogStateEntry(1, LogStateKind.LOG, 4, 8),
        LogStateEntry(2, LogStateKind.CHECKPOINT, 0, 12, span_pages=3),
    )

    result = tree.insert(token_ids, kv, entries)
    match = tree.trace(token_ids)

    assert result.previous_len == 0
    assert result.inserted_len == 12
    assert match.matched_len == 12
    assert torch.equal(matched_kv(tree, match.node), kv)
    assert tree.matched_entries(match.node) == entries
    assert tree.size_info.evictable_size == 12
    tree.check_integrity()


def test_split_uses_entry_end_as_ownership_boundary():
    tree = make_tree()
    token_ids = ids(12)
    entries = (
        LogStateEntry(0, LogStateKind.LOG, 0, 4),
        LogStateEntry(1, LogStateKind.LOG, 4, 8),
        LogStateEntry(2, LogStateKind.CHECKPOINT, 0, 12, span_pages=3),
    )
    leaf = tree.insert(token_ids, ids(12, 100), entries).node

    assert tree.entry_owner(0).depth == 4
    assert tree.entry_owner(1).depth == 8
    assert tree.entry_owner(2).depth == 12
    assert tree.entry_owner(0).entries == entries[:1]
    assert tree.entry_owner(1).entries == entries[1:2]
    assert leaf.entries == entries[2:]
    assert tree.matched_entries(leaf) == entries
    tree.check_integrity()


def test_branch_protection_and_live_lock_have_separate_lifetimes():
    tree = make_tree()
    node = tree.insert(
        ids(8), ids(8, 100),
        (LogStateEntry(0, LogStateKind.LOG, 0, 4),),
    ).node
    owner = tree.entry_owner(0)

    tree.protect(owner)
    assert owner.is_protected
    assert tree.size_info == (8, 0)

    tree.lock(node)
    assert tree.size_info == (0, 8)

    tree.lock(node, unlock=True)
    assert tree.size_info == (8, 0)

    tree.protect(owner, protected=False)
    assert not owner.is_protected
    tree.check_integrity()


def test_evict_returns_kv_and_linear_slots_without_freeing_them():
    tree = make_tree()
    left = ids(8)
    right = torch.cat([ids(4), ids(4, 20)])
    tree.insert(
        left,
        ids(8, 100),
        (
            LogStateEntry(3, LogStateKind.LOG, 0, 4),
            LogStateEntry(5, LogStateKind.LOG, 4, 8),
        ),
    )
    tree.insert(
        right,
        ids(8, 200),
        (LogStateEntry(4, LogStateKind.CHECKPOINT, 0, 8, span_pages=2),),
    )

    evicted_nodes = []
    evicted = tree.evict(4, evicted_nodes)
    assert evicted_nodes
    assert all(begin < end for _uuid, begin, end in evicted_nodes)

    assert len(evicted.kv_indices) >= 4
    assert len(evicted.entries) >= 1
    assert {entry.slot for entry in evicted.entries}.issubset({3, 4, 5})
    tree.check_integrity()


def test_evict_prefers_log_branch_over_older_checkpoint_branch():
    tree = make_tree()
    checkpoint = tree.insert(
        ids(4), ids(4, 100),
        (LogStateEntry(0, LogStateKind.CHECKPOINT, 0, 4, span_pages=1),),
    ).node
    log = tree.insert(
        ids(4, 20), ids(4, 200),
        (LogStateEntry(1, LogStateKind.LOG, 0, 4),),
    ).node
    tree._checkpoint_timestamps[0] = 1
    log.timestamp = 2
    assert list(tree.iter_evict_candidates()) == [log, checkpoint]


def test_matching_refreshes_kv_path_but_only_the_selected_checkpoint():
    tree = make_tree()
    tokens = ids(12)
    leaf = tree.insert(tokens, tokens + 100, tuple(
        LogStateEntry(slot, LogStateKind.CHECKPOINT, slot * 4, (slot + 1) * 4, span_pages=1)
        for slot in range(3)
    )).node
    tree._checkpoint_timestamps.update({0: 10, 1: 20, 2: 30})
    with patch("logssms.cache.prefix_cache.logstate.time.monotonic_ns", return_value=100):
        handle = tree.match_prefix(tokens).cuda_handle
    assert handle.recovery.base_checkpoint_slot == 2
    assert all(node.timestamp == 100 for node in tree.path(leaf))
    assert tree._checkpoint_timestamps == {0: 10, 1: 20, 2: 100}
    with patch("logssms.cache.prefix_cache.logstate.time.monotonic_ns", return_value=200):
        earlier = tree.match_prefix(tokens[:8]).cuda_handle
    assert earlier.recovery.base_checkpoint_slot == 1
    assert tree._checkpoint_timestamps == {0: 10, 1: 200, 2: 100}
    assert tree.entry_owner(2).timestamp == 100
    assert tree.checkpoint_eviction_pairs() == [(0, 1, 4, 10), (1, 2, 8, 200)]
    tree.check_integrity()


def test_log_endpoint_match_touches_its_base_checkpoint_not_earlier_states():
    tree = make_tree()
    tokens = ids(12)
    tree.insert(tokens, tokens + 100, (
        LogStateEntry(0, LogStateKind.CHECKPOINT, 0, 4, span_pages=1),
        LogStateEntry(1, LogStateKind.CHECKPOINT, 4, 8, span_pages=1),
        LogStateEntry(2, LogStateKind.LOG, 8, 12),
    ))
    tree._checkpoint_timestamps.update({0: 10, 1: 20})
    with patch("logssms.cache.prefix_cache.logstate.time.monotonic_ns", return_value=100):
        handle = tree.match_prefix(tokens[:10]).cuda_handle
    assert handle.cached_len == 10
    assert handle.recovery.base_checkpoint_slot == 1
    assert handle.recovery.log_slots == (2,)
    assert tree._checkpoint_timestamps == {0: 10, 1: 100}
    assert all(node.timestamp == 100 for node in handle.match_path)
    assert tree.entries_for_slot(2)[-1].end == 12  # split Log remains shared
    tree.check_integrity()


def test_insert_trace_split_and_handle_construction_do_not_heat_checkpoints():
    tree = make_tree()
    tokens = ids(8)
    leaf = tree.insert(tokens, tokens + 100, (
        LogStateEntry(0, LogStateKind.CHECKPOINT, 0, 8, span_pages=2),
    )).node
    tree._checkpoint_timestamps[0] = 10
    with patch("logssms.cache.prefix_cache.logstate.time.monotonic_ns", return_value=100):
        tree.insert(tokens, tokens + 100)
        assert leaf.timestamp == 100
        tree.trace(tokens)
        tree.recovery_at(leaf, 8)
        tree.make_handle(leaf)
        tree.split_at(leaf, 4)
    assert tree.checkpoint_timestamp(0) == 10
    tree.check_integrity()


def test_checkpoint_eviction_uses_state_heat_not_kv_heat_and_cleans_slot_reuse():
    tree = make_tree()
    cold = tree.insert(ids(4), ids(4, 100), (
        LogStateEntry(0, LogStateKind.CHECKPOINT, 0, 4, span_pages=1),
    )).node
    hot = tree.insert(ids(4, 20), ids(4, 200), (
        LogStateEntry(1, LogStateKind.CHECKPOINT, 0, 4, span_pages=1),
    )).node
    cold.timestamp, hot.timestamp = 999, 1
    tree._checkpoint_timestamps.update({0: 10, 1: 20})
    assert list(tree.iter_evict_candidates()) == [cold, hot]
    tree.evict_nodes([cold])
    assert 0 not in tree._checkpoint_timestamps
    with patch("logssms.cache.prefix_cache.logstate.time.monotonic_ns", return_value=100):
        tree.insert(ids(4, 40), ids(4, 300), (
            LogStateEntry(0, LogStateKind.CHECKPOINT, 0, 4, span_pages=1),
        ))
    assert tree.checkpoint_timestamp(0) == 100
    tree.reset()
    assert tree._checkpoint_timestamps == {}
    tree.check_integrity()


def test_small_evictions_follow_global_lru_without_refreshing_access_times():
    tree = make_tree()
    leaves = []
    for chain in range(5):
        leaf = tree.insert(ids(12, chain * 100), ids(12, 1000 + chain * 12), tuple(
            LogStateEntry(chain * 3 + page, LogStateKind.LOG, page * 4, (page + 1) * 4)
            for page in range(3)
        )).node
        for node in tree.path(leaf):
            node.timestamp = chain * 10
        leaves.append(leaf)
    timestamps = {node.uuid: node.timestamp for node in tree._iter_nodes()}
    for _ in range(3):
        candidate = next(tree.iter_evict_candidates())
        assert candidate is leaves[0]
        leaves[0] = candidate.parent
        tree.evict_nodes([candidate])
        tree.check_integrity()
        assert all(node.timestamp == timestamps[node.uuid] for node in tree._iter_nodes())
    assert next(tree.iter_evict_candidates()) is leaves[1]
    tree.reset()
    assert list(tree.iter_evict_candidates()) == []


def test_drop_old_checkpoint_preserves_kv_and_later_cumulative_state():
    tree = LogRadixTree(64, torch.device("cpu"), granularity=32)
    tokens = ids(384)
    leaf = tree.insert(tokens, tokens + 1000, (
        LogStateEntry(0, LogStateKind.CHECKPOINT, 0, 256, span_pages=4),
        LogStateEntry(1, LogStateKind.CHECKPOINT, 256, 384, span_pages=2),
    )).node
    assert tree.checkpoint_eviction_pairs() == [(0, 1, 256, tree.checkpoint_timestamp(0))]
    timestamp = tree.checkpoint_timestamp(1)
    kv = matched_kv(tree, leaf).clone()
    tree.drop_checkpoint(0, 1)
    assert tree.checkpoint_timestamp(1) == timestamp
    assert 0 not in tree._checkpoint_timestamps
    assert tree.matched_entries(leaf) == (
        LogStateEntry(1, LogStateKind.CHECKPOINT, 0, 384, span_pages=6),
    )
    torch.testing.assert_close(matched_kv(tree, leaf), kv)
    assert tree.recovery_at(leaf, 384).base_checkpoint_slot == 1
    assert tree.recovery_at(leaf, 256).position == 0
    assert not tree.has_slot(0)
    assert tree.size_info.evictable_size == 384
    tree.check_integrity()


def test_pressure_checkpoint_pairs_stop_at_logs_protections_and_branches_not_kv_locks():
    tree = make_tree()
    tokens = ids(16)
    leaf = tree.insert(tokens, tokens + 100, (
        LogStateEntry(0, LogStateKind.CHECKPOINT, 0, 4, span_pages=1),
        LogStateEntry(1, LogStateKind.CHECKPOINT, 4, 8, span_pages=1),
        LogStateEntry(2, LogStateKind.LOG, 8, 12),
        LogStateEntry(3, LogStateKind.CHECKPOINT, 12, 16, span_pages=1),
    )).node
    pairs = tree.checkpoint_eviction_pairs()
    assert [(left, right) for left, right, *_ in pairs] == [(0, 1)]
    first = tree.entry_owner(0)
    tree.protect(first)
    assert tree.checkpoint_eviction_pairs() == []
    tree.protect(first, False)
    tree.lock(tree.entry_owner(1))
    assert tree.checkpoint_eviction_pairs() == pairs
    assert 0 in tree.evictable_slots() and 1 not in tree.evictable_slots()
    tree.pin_recovery((1,))
    assert tree.checkpoint_eviction_pairs() == []
    assert 0 not in tree.evictable_slots()
    tree.pin_recovery((1,), release=True)
    assert tree.checkpoint_eviction_pairs() == pairs
    tree.lock(tree.entry_owner(1), unlock=True)
    tree.insert(torch.cat((tokens[:4], ids(4, 50))), ids(8, 200), (
        LogStateEntry(4, LogStateKind.CHECKPOINT, 4, 8, span_pages=1),
    ))
    assert len(first.children) == 2
    assert tree.checkpoint_eviction_pairs() == []
    assert tree.recovery_at(leaf, 16).base_checkpoint_slot == 3
    tree.check_integrity()


def test_locked_kv_path_allows_state_thinning_and_updates_capacity():
    tree = make_tree()
    tokens = ids(12)
    leaf = tree.insert(tokens, tokens + 100, tuple(
        LogStateEntry(slot, LogStateKind.CHECKPOINT, slot * 4, (slot + 1) * 4, span_pages=1)
        for slot in range(3)
    )).node
    tree.lock(leaf)
    assert tree.evictable_kv_frames() == 0
    assert set(tree.evictable_slots()) == {0, 1}
    tree.drop_checkpoint(1, 2)
    assert leaf.is_locked
    assert set(tree.evictable_slots()) == {0}
    torch.testing.assert_close(matched_kv(tree, leaf), tokens + 100)
    assert tree.recovery_at(leaf, 12).base_checkpoint_slot == 2
    tree.check_integrity()
    tree.drop_checkpoint(0, 2)
    assert tree.evictable_slots() == ()
    assert tree.locked_size == 12
    tree.check_integrity()
    tree.lock(leaf, unlock=True)
    assert tree.evictable_slots() == (2,)
    tree.check_integrity()


def test_locked_checkpoint_capacity_tracks_branch_and_protection_changes():
    tree = make_tree()
    tokens = ids(12)
    leaf = tree.insert(tokens, tokens + 100, tuple(
        LogStateEntry(slot, LogStateKind.CHECKPOINT, slot * 4, (slot + 1) * 4, span_pages=1)
        for slot in range(3)
    )).node
    tree.lock(leaf)
    first = tree.entry_owner(0)
    tree.protect(first)
    assert set(tree.evictable_slots()) == {1}
    tree.check_integrity()
    tree.protect(first, False)
    assert set(tree.evictable_slots()) == {0, 1}
    branch = tree.insert(torch.cat((tokens[:4], ids(4, 50))), ids(8, 200), (
        LogStateEntry(3, LogStateKind.CHECKPOINT, 4, 8, span_pages=1),
    )).node
    assert set(tree.evictable_slots()) == {1, 3}
    tree.check_integrity()
    tree.evict_nodes([branch])
    assert set(tree.evictable_slots()) == {0, 1}
    tree.check_integrity()


def test_new_tail_smaller_than_one_physical_page_is_not_admitted():
    tree = make_tree()
    history = ids(8)
    tree.insert(
        history,
        ids(8, 100),
        (
            LogStateEntry(0, LogStateKind.LOG, 0, 4),
            LogStateEntry(1, LogStateKind.LOG, 4, 8),
        ),
    )
    request = torch.cat([history[:2], ids(3, 20)])

    result = tree.insert(request, ids(5, 200))

    assert result.previous_len == 2
    assert result.inserted_len == 2
    assert tree.trace(request).matched_len == 2
    tree.check_integrity()


def test_kv_reclamation_requires_the_entire_physical_frame_to_be_evictable():
    tree = make_tree()
    token_ids = ids(8)
    kv = ids(8, 100)
    leaf = tree.insert(token_ids, kv).node
    assert tree.evictable_kv_frames() == 2

    branch = tree.split_at(leaf, 2)
    tree.lock(branch)
    assert tree.evictable_kv_frames() == 1
    tree.lock(branch, unlock=True)
    assert tree.evictable_kv_frames() == 2
    tree.check_integrity()


def test_zero_state_admission_returns_root_after_existing_kv_match():
    tree = make_tree()
    history = ids(8)
    tree.insert(
        history,
        ids(8, 100),
        (
            LogStateEntry(0, LogStateKind.LOG, 0, 4),
            LogStateEntry(1, LogStateKind.LOG, 4, 8),
        ),
    )

    result = tree.insert(
        history[:6],
        ids(6, 200),
        cache_len=0,
    )

    assert result.previous_len == 6
    assert result.inserted_len == 0
    assert result.node.is_root()
    tree.check_integrity()


def test_recovery_inside_log_page_stops_at_matched_granule():
    tree = LogRadixTree(64, torch.device("cpu"), granularity=16)
    history = ids(64)
    tree.insert(
        history,
        ids(64, 100),
        (LogStateEntry(7, LogStateKind.LOG, 0, 64),),
    )

    for matched_len in (16, 32, 48, 64):
        request = torch.cat([history[:matched_len], ids(16, 1000)])
        handle = tree.match_prefix(request).cuda_handle

        assert handle.cached_len == matched_len
        assert handle.recovery.position == matched_len
        assert handle.recovery.logs[0].begin == 0
        assert handle.recovery.logs[-1].end == matched_len
        assert all(entry.slot == 7 and entry.physical_begin == 0
                   for entry in handle.recovery.logs)
        assert all(left.end == right.begin for left, right in zip(
            handle.recovery.logs, handle.recovery.logs[1:]
        ))
        assert handle.recovery.conv_granule == matched_len // 16 - 1
    tree.check_integrity()
