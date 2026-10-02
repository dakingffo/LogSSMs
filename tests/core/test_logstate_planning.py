import pytest
import torch

from logssms.cache.logstate import (
    LogStateCacheManager,
    PageMaterialize,
    _ConstructionChain,
)
from logssms.cache.prefix_cache.logstate import LogStateEntry, LogStateKind
from logssms.kernel import logstate as planning


@pytest.mark.parametrize("granularity", [16, 32, 64])
def test_native_branches_preserve_token_prefixes_and_ignore_partial_granules(granularity):
    common = torch.arange(granularity, dtype=torch.int64)
    inputs = [
        torch.cat((common, torch.full((granularity + 3,), value, dtype=torch.int64)))
        for value in (101, 102, 103)
    ]
    inputs[1] = inputs[1].to(torch.int32)
    assert planning.batch_branch_points(inputs, granularity) == [[granularity, 0, 1, 2]]
    assert planning.batch_branch_points([], granularity) == []
    assert planning.batch_branch_points(inputs[:1], granularity) == []


def test_native_log_coverage_reports_gap_and_original_entry_indices():
    # Unsorted intervals; checkpoint does not count as a replay log.
    path = [
        3, 128, 192, -1, 128, 1, 0, 0,
        1, 0, 64, -1, 0, 1, 0, 0,
        2, 64, 128, 1, 64, 1, 0, 0,
    ]
    assert planning.cover_logs(path, 0, 192) == [64, 1]
    assert planning.cover_logs(path, 128, 160) == [160, 0]


def test_native_materialize_respects_fragmented_and_protected_slots():
    path = [
        0, 0, 64, 1, 0, 1, 0, 0,
        1, 64, 128, -1, 64, 1, 0, 0,
        2, 128, 192, -1, 128, 1, 0, 1,
    ]
    assert planning.plan_materialize([path], [()], [(0, 1), (0, 2)], 64) == [[0, 0, 1]]
    # A pending branch inside the physical page protects the entire Log.
    assert planning.plan_materialize([path], [(96,)], [(0, 1)], 64) == []
    path[8 + 6] = 1
    assert planning.plan_materialize([path], [()], [(0, 1)], 64) == []


def test_native_dependency_plan_orders_old_log_reader_before_writer():
    full = LogStateEntry(4, LogStateKind.LOG, 0, 64)
    partial = full.slice(0, 32)
    chains = (
        _ConstructionChain(None, (full,), ((4,),), True),
        _ConstructionChain(None, (partial,), ((10,),), False),
    )
    plans = (PageMaterialize(4, 0, 64, None, (full,)),)
    assert LogStateCacheManager._schedule_construction_waves(chains, plans) == (
        (chains[1],), (chains[0],),
    )


def test_native_dependency_plan_rebases_only_without_an_earlier_output():
    first = LogStateEntry(1, LogStateKind.LOG, 0, 64)
    second = LogStateEntry(2, LogStateKind.LOG, 64, 128)
    writer = _ConstructionChain(None, (first,), ((1,),), True)
    reader = _ConstructionChain(None, (first, second), ((), (9,)), False)
    plans = (PageMaterialize(1, 0, 64, None, (first,)),)
    waves = LogStateCacheManager._schedule_construction_waves((writer, reader), plans)
    assert waves == ((writer,), (_ConstructionChain(1, (second,), ((9,),), False),))


def test_native_dependency_plan_rejects_duplicate_missing_and_cyclic_writers():
    a = LogStateEntry(1, LogStateKind.LOG, 0, 64)
    b = LogStateEntry(2, LogStateKind.LOG, 0, 64)
    plans = (PageMaterialize(1, 0, 64, None, (a,)),)
    writer = _ConstructionChain(None, (a,), ((1,),), True)
    with pytest.raises(RuntimeError, match="multiple writers"):
        LogStateCacheManager._schedule_construction_waves((writer, writer), plans)
    with pytest.raises(RuntimeError, match="Missing Checkpoint writer"):
        LogStateCacheManager._schedule_construction_waves(
            (_ConstructionChain(None, (a,), ((10,),), False),), plans,
        )
    with pytest.raises(RuntimeError, match="Cyclic"):
        LogStateCacheManager._schedule_construction_waves(
            (_ConstructionChain(2, (a,), ((1,),), True),
             _ConstructionChain(1, (b,), ((2,),), True)),
            plans + (PageMaterialize(2, 0, 64, None, (b,)),),
        )


def test_native_metadata_validation_precedes_pointer_access():
    module = planning._load_logstate_module()
    bad = torch.tensor([10, 1], dtype=torch.int64)
    empty = torch.empty(0, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="Invalid packed planning row"):
        module.plan_materialize(bad, empty, empty, 64)
    with pytest.raises(RuntimeError, match="CPU int64"):
        module.plan_materialize(bad.to(torch.int32), empty, empty, 64)
