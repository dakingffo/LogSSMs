from types import SimpleNamespace

import pytest

from logssms.cache.config import LogStateCacheConfig, StandardCacheConfig
from logssms.engine.mixed_graph import MixedGraphRunner
from logssms.scheduler.prefill import PrefillAdder
from logssms.utils.mixed import prefill_buckets, prefill_partitions


@pytest.mark.parametrize("granularity,expected", [
    (32, [(32,32,32,32), (64,32,32), (64,64), (128,)]),
    (16, [(16,)*8, (32,)+(16,)*6, (32,32)+(16,)*4,
          (32,32,32,16,16), (32,)*4, (64,32,32), (64,64), (128,)]),
    (64, [(64,64), (128,)]),
])
def test_partitions(granularity, expected):
    assert prefill_partitions(128, granularity) == tuple(expected)


@pytest.mark.parametrize("budget", [64, 96, 128, 192, 256])
@pytest.mark.parametrize("alignment", [16, 32, 64])
def test_partitions_preserve_budget_and_cover_every_count(budget, alignment):
    if budget % alignment:
        with pytest.raises(ValueError):
            prefill_partitions(budget, alignment)
        return
    parts = prefill_partitions(budget, alignment)
    assert [len(p) for p in parts] == list(range(budget // alignment, 0, -1))
    assert all(sum(p) == budget and all(n % alignment == 0 for n in p) for p in parts)
    assert all(all(a <= b for a,b in zip(old, new)) for old,new in zip(parts, parts[1:]))


@pytest.mark.parametrize("cache_config,expected", [
    (LogStateCacheConfig(granularity=16), 57),
    (LogStateCacheConfig(granularity=32), 30),
    (LogStateCacheConfig(granularity=64), 14),
    (StandardCacheConfig(page_size=64), 14),
    (StandardCacheConfig(page_size=128), 14),
    (StandardCacheConfig(page_size=256), 14),
])
def test_graph_enumeration(monkeypatch, cache_config, expected):
    from contextlib import nullcontext
    import torch
    made = []
    def capture(self, engine, size, partition):
        made.append((size, partition))
        return None
    monkeypatch.setattr(MixedGraphRunner, "_capture_batch", capture)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: SimpleNamespace(pool=lambda: None))
    monkeypatch.setattr(torch.cuda, "graph", lambda *a, **k: nullcontext())
    monkeypatch.setattr("logssms.engine.mixed_graph.logger.info_rank0", lambda _: None)
    engine = SimpleNamespace(
        config=SimpleNamespace(mixed_prefill_tokens=128, cache_config=cache_config,
                               max_running_req=32, scheduling_policy="mixed"),
        decode_graph_runner=SimpleNamespace(max_graph_bs=32),
        attn_backend=SimpleNamespace(make_mixed_capture=lambda *a: None),
        ctx=SimpleNamespace(forward_batch=lambda _: nullcontext()),
        model=SimpleNamespace(forward=lambda: None), max_seq_len=1024, stream=None,
    )
    runner = MixedGraphRunner(engine)
    assert len(runner.graphs) == expected
    assert all(size > len(p) for size, p in made)
    assert {p for s,p in made if s == 2} == {
        p for p in prefill_buckets(128, cache_config.prefill_alignment) if len(p) == 1
    }
    assert set(runner.graphs) == set(made)
    for _, partition in made:
        assert runner.partition(partition) == partition


@pytest.mark.parametrize("budget,alignment,capacities", [
    (128, 16, [16, 32, 64, 128]),
    (128, 32, [32, 64, 128]),
    (128, 64, [64, 128]),
    (32, 32, [32]),
    (96, 32, [32, 96]),
    (192, 32, [32, 96, 192]),
])
def test_prefill_bucket_capacities(budget, alignment, capacities):
    buckets = prefill_buckets(budget, alignment)
    assert sorted({sum(p) for p in buckets}) == capacities
    assert len(buckets) == len(set(buckets))
    for capacity in capacities:
        assert {p for p in buckets if sum(p) == capacity} == set(prefill_partitions(capacity, alignment))


@pytest.mark.parametrize("alignment,lengths,expected", [
    (32, [1], (32,)), (32, [32], (32,)), (32, [33], (64,)),
    (32, [65], (128,)), (32, [20,20], (32,32)),
    (32, [60,20], (64,64)), (32, [20,60], (64,64)),
    (32, [20,20,20], (64,32,32)), (32, [32]*4, (32,)*4),
    (16, [1], (16,)), (16, [1,1], (16,16)),
    (16, [30,10,10], (32,16,16)),
    (64, [1], (64,)), (64, [1,1], (64,64)),
])
def test_select_smallest_bucket_without_reordering(alignment, lengths, expected):
    runner = MixedGraphRunner.__new__(MixedGraphRunner)
    runner.buckets = prefill_buckets(128, alignment)
    runner.buckets_by_count = {
        count: tuple(p for p in runner.buckets if len(p) == count)
        for count in range(1, 128 // alignment + 1)
    }
    runner.sizes = [2,4,8,16,32]
    partition = runner.partition(lengths)
    assert partition == expected
    batch = SimpleNamespace(is_mixed=True, hidden_captor=None, size=len(lengths)+1,
                            prefill_partition=(),
                            prefill_reqs=[SimpleNamespace(extend_len=n) for n in lengths])
    key = runner._graph_key(batch)
    runner.graphs = {key: object()}
    assert runner.can_use_cuda_graph(batch)
    batch.prefill_partition = partition
    assert runner._graph_key(batch) == key
    assert all(sum(p) >= sum(partition) for p in runner.buckets_by_count[len(lengths)]
               if all(n <= cap for n,cap in zip(lengths,p)))


@pytest.mark.parametrize("lengths,expected", [
    ([32,32,32,32], [32,32,32,32]),
    ([64,32,32], [64,32,32]),
    ([32,96], [32,64]),
    ([100,1,1], [100]),
    ([1,100,1], [1,64]),
])
def test_fcfs_chunk_limits_do_not_skip_or_reorder(lengths, expected):
    adder = PrefillAdder(8192, 0, None, None, mixed_token_budget=128,
                        mixed_partitions=prefill_partitions(128,32))
    admitted = []
    for n in lengths:
        adder._select_budget(n)
        chunk = min(n, adder._chunk_budget(n))
        if not chunk:
            break
        admitted.append(chunk)
        adder._mixed_lengths.append(chunk)
        adder.token_budget -= chunk
    assert admitted == expected
