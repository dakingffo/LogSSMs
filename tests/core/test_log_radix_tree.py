import torch

from logssms.cache.prefix_cache.logstate import (
    LogRadixTree,
    LogStateEntry,
    LogStateKind,
)


PAGE_SIZE = 4


def make_tree() -> LogRadixTree:
    return LogRadixTree(PAGE_SIZE, torch.device("cpu"), granularity=1)


def ids(length: int, begin: int = 0) -> torch.Tensor:
    return torch.arange(begin, begin + length, dtype=torch.int32)


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
    match = tree.match(token_ids)

    assert result.previous_len == 0
    assert result.inserted_len == 12
    assert match.matched_len == 12
    assert torch.equal(tree.matched_kv(match.node), kv)
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

    evicted = tree.evict(4)

    assert len(evicted.kv_indices) >= 4
    assert len(evicted.entries) >= 1
    assert {entry.slot for entry in evicted.entries}.issubset({3, 4, 5})
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
    cached_page_tokens = {100: 4, 104: 4}
    assert tree.evictable_kv_frames(cached_page_tokens) == 2

    branch = tree.split_at(leaf, 2)
    tree.lock(branch)
    assert tree.evictable_kv_frames(cached_page_tokens) == 1
    tree.lock(branch, unlock=True)
    assert tree.evictable_kv_frames(cached_page_tokens) == 2
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
