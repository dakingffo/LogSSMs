from logssms.cache.marconi.metadata import MarconiCacheMetadata
from types import SimpleNamespace

import pytest
import torch

from logssms.cache.marconi.layout import HybridBlockLayout
from logssms.cache.marconi.block_pool import HybridBlockPool
from logssms.cache.marconi.prefix_cache import MarconiCost, MarconiPrefixCache, tune_marconi
from logssms.cache.marconi.manager import MarconiCacheFactory
from logssms.cache.config import MarconiCacheConfig
from logssms.engine.config import EngineConfig
from logssms.distributed import DistributedInfo
from logssms.core import Request, SamplingParams, Batch


@pytest.mark.parametrize("block_size, width", [(0, 1), (64, 7), (128, 4), (256, 2), (448, 1), (512, 1)])
def test_block_size_layout_and_arguments(block_size, width):
    from logssms.cache.args import MarconiCacheArgumentParser

    parser = MarconiCacheArgumentParser()
    args = vars(parser.parser.parse_args(["--block-size", str(block_size)]))
    config = parser.make_config(args)
    assert config.block_size == block_size and config.page_size == block_size
    spec = SimpleNamespace(num_layers=48, state_shape=(24, 128, 128), conv_shape=(5120, 3))
    layout = HybridBlockLayout.make(16, spec, 2, 256, config.block_size)
    assert layout.block_size == (block_size or 448)
    assert layout.state_blocks_per_group == width
    assert layout.state_blocks == 3 * width
    assert layout.state_blocks * layout.block_bytes(2) == 3 * width * 16 * layout.block_size * 2048


def test_marconi_block_size_default_and_validation():
    from logssms.cache.args import MarconiCacheArgumentParser

    parser = MarconiCacheArgumentParser()
    assert parser.make_config(vars(parser.parser.parse_args([]))).block_size == 128
    with pytest.raises(SystemExit):
        parser.parser.parse_args(["--page-size", "64"])
    for size in (-64, 16, 32, 65):
        with pytest.raises(ValueError):
            MarconiCacheConfig(block_size=size)


def test_contiguous_allocator_fragmentation_and_coalescing():
    from logssms.cache.marconi.allocator import BlockAllocator

    allocator = BlockAllocator(16)
    blocks = allocator.allocate(16)
    allocator.release(blocks[::2])
    assert len(allocator.free) == 8
    assert not allocator.can_allocate_groups(1, 2)
    before = allocator.free.copy()
    with pytest.raises(RuntimeError):
        allocator.allocate_groups(2, 2)
    assert allocator.free == before
    allocator.release(blocks[1::2])
    assert allocator.intervals == [(0, 16)]
    state = allocator.allocate_groups(3, 3)
    kv = allocator.allocate(7)
    assert not set(kv).intersection(state)
    for offset in range(0, 9, 3):
        assert state[offset:offset + 3] == tuple(range(state[offset], state[offset] + 3))
    allocator.retain(state)
    allocator.release(state)
    assert not allocator.free
    allocator.release(state)
    allocator.release(kv)
    assert allocator.intervals == [(0, 16)]


def test_fragmented_state_admission_rejects_before_init():
    manager = small_manager(block_size=64)
    allocator = manager.pool.allocator
    held = allocator.allocate(len(allocator.free))
    allocator.release(held[::2])
    handle = manager.match_req(SimpleNamespace(input_ids=torch.arange(64))).cuda_handle
    needed = manager.reservation_size(64, handle=handle)
    assert manager.available_size >= needed
    assert not manager.prefill_acquire(needed, handle)
    assert not manager.requests
    allocator.release(held[1::2])
    assert manager.prefill_acquire(needed, handle)
    before = len(allocator.free)
    manager.init_with_handle(0, handle)
    assert before - len(allocator.free) == 3 * manager.plan.layout.state_blocks
    req = Request(torch.arange(64, dtype=torch.int32), 0, 0, 1, 0, SamplingParams(), handle, 1024)
    manager.allocate([req], phase="prefill")
    assert manager.requests[0].branch_state is None
    manager.cache_req(req, finished=True)
    manager.tree.reset()
    manager.check_integrity()


def test_admission_does_not_evict_for_future_kv():
    manager = small_manager(block_size=64)
    p = manager.pool
    pages = p.allocator.allocate(16)
    slot = p.allocate_state()
    tokens = tuple(range(1024))
    locations = tuple(page * 64 + offset for page in pages for offset in range(64))
    manager.tree.insert(tokens, locations, slot)
    p.allocator.release(pages)
    p.release_state(slot)
    handle = manager.match_req(SimpleNamespace(input_ids=torch.tensor([99999, 99998]))).cuda_handle
    assert handle.cached_len == 0
    total = len(p.allocator.free) + 1
    assert total < manager.available_size
    assert manager.prefill_acquire(total, handle)
    assert len(manager.tree.nodes) == 1
    assert not manager.tree.evicted
    manager.unlock(handle)
    manager.tree.reset()
    manager.check_integrity()


@pytest.mark.parametrize("block_size", [64, 128, 256])
def test_multiblock_state_views_do_not_overwrite_kv(block_size):
    spec = SimpleNamespace(num_layers=3, state_shape=(4, 32, 32), conv_shape=(256, 3))
    layout = HybridBlockLayout.make(2, spec, 1, 32, block_size)
    p = HybridBlockPool(layout, 32, "cpu", torch.float32)
    a, b = p.allocate_state(), p.allocate_state()
    kv = p.allocator.allocate(len(p.allocator.free))
    p.storage.fill_(9)
    for group, begin in enumerate(p.state_blocks[a][::layout.state_blocks_per_group]):
        p.recurrent_state[begin].fill_(group + 1)
        p.conv_state[begin].fill_(group + 3)
    p.copy_state(a, b)
    for src, dst in zip(p.state_blocks[a][::layout.state_blocks_per_group],
                        p.state_blocks[b][::layout.state_blocks_per_group]):
        torch.testing.assert_close(p.recurrent_state[src], p.recurrent_state[dst])
        torch.testing.assert_close(p.conv_state[src], p.conv_state[dst])
    p.zero_state(b)
    assert p.storage[:, list(kv)].eq(9).all()
    p.release_state(a)
    p.release_state(b)
    p.allocator.release(kv)
    assert p.allocator.intervals == [(0, 32)]


def pool(device="cpu", blocks=32):
    spec = SimpleNamespace(num_layers=3, state_shape=(2, 8, 8), conv_shape=(32, 3))
    layout = HybridBlockLayout.make(2, spec, 1, 8)
    return HybridBlockPool(layout, blocks, device, torch.float32)


def small_manager(device="cpu", kind="gdn", block_size=128):
    spec = SimpleNamespace(num_layers=3, state_shape=(4, 32, 32),
                           conv_shape=(256, 3), recurrent_type=kind)
    model = SimpleNamespace(is_hybrid=True, num_attention_layers=2,
                            make_linear_cache_spec=lambda tp: spec, num_kv_heads=1,
                            head_dim=32, hidden_size=128, num_qo_heads=4,
                            intermediate_size=256, num_layers=5, is_moe=False)
    config = SimpleNamespace(model_config=model, tp_info=DistributedInfo(0, 1),
                             dtype=torch.bfloat16, cache_config=MarconiCacheConfig(block_size=block_size),
                             max_seq_len=1024, max_running_req=2, num_page_override=64)
    factory = MarconiCacheFactory()
    plan = factory.plan(config, 64 << 20, None)
    resources = factory.allocate(config, plan, torch.device(device), config.dtype)
    return factory.make_manager(config, plan, resources)


@pytest.mark.parametrize("end", [1, 127, 128, 129, 255, 256, 257, 384, 385])
def test_double_state_aligned_tail_and_decode_swaps(end):
    manager = small_manager()
    p, ps = manager.pool, manager.page_size
    initial_free = len(p.allocator.free)
    ids = torch.arange(end, dtype=torch.int32)
    handle = manager.match_req(SimpleNamespace(input_ids=ids)).cuda_handle
    assert manager.prefill_acquire(manager.reservation_size(end, handle=handle), handle)
    manager.init_with_handle(0, handle)
    entry = manager.requests[0]
    assert len(entry.states) == 2

    def fill(slot, value):
        for block in p.state_blocks[slot][::p.layout.state_blocks_per_group]:
            p.recurrent_state[block].fill_(value)
            p.conv_state[block].fill_(value)

    def check(slot, value):
        for block in p.state_blocks[slot][::p.layout.state_blocks_per_group]:
            assert (p.recurrent_state[block] == value).all()
            assert (p.conv_state[block] == value).all()

    start = min(end, 140)
    req = Request(ids, 0, 0, 512, 0, SamplingParams(), handle, 1024)
    req.device_len = start
    manager.allocate([req], phase="prefill")
    batch = Batch([req], "prefill")
    manager.prepare_batch(batch)
    for position, slot in batch.cache_metadata.checkpoint_writes[0]:
        fill(slot, position)
    manager.after_forward(batch)
    for position in range(start + 1, end + 1):
        req.cached_len, req.device_len = position - 1, position
        manager.allocate([req], phase="decode")
        read, write = manager.state_table[0, position - 2:position].tolist()
        check(read, position - 1)
        assert (read != write) == ((position - 1) % ps == 0)
        fill(write, position)
        boundary = position // ps * ps
        if boundary:
            check(entry.state_at(boundary, ps), boundary)
    manager.cache_req(req, finished=True)
    manager.check_integrity()
    boundary = end // ps * ps
    match = manager.tree.match_prefix(tuple(range(end)))
    assert match.cuda_handle.cached_len == boundary
    if boundary:
        node = match.cuda_handle.node
        check(node.state, boundary)
        assert len(node.locations) == boundary
        assert len(manager.tree._blocks(node)) == boundary // ps
        manager.lock(match.cuda_handle)
        manager.init_with_handle(1, match.cuda_handle)
        resumed = manager.requests[1]
        check(resumed.state_at(boundary, ps), boundary)
        assert resumed.state_at(boundary + 1, ps) != resumed.state_at(boundary, ps)
        continuation = Request(torch.arange(boundary + 1, dtype=torch.int32), 1,
                               boundary, 1, 1, SamplingParams(), match.cuda_handle, 1024)
        manager.cache_req(continuation, finished=True)
    manager.tree.reset()
    assert len(p.allocator.free) == initial_free


def test_prefill_boundary_branch_snapshot_is_independent():
    manager = small_manager()
    p, ps = manager.pool, manager.page_size
    initial_free = len(p.allocator.free)
    pages = p.allocator.allocate(2)
    state = p.allocate_state()
    manager.tree.insert(tuple(range(2 * ps)),
                        tuple(page * ps + i for page in pages for i in range(ps)), state)
    p.allocator.release(pages)
    p.release_state(state)
    tokens = torch.tensor(list(range(ps)) + [999] * 12, dtype=torch.int32)
    handle = manager.match_req(SimpleNamespace(input_ids=tokens)).cuda_handle
    assert handle.cached_len == 0 and handle.branch_end == ps
    manager.lock(handle)
    manager.init_with_handle(0, handle)
    req = Request(tokens, 0, 0, 1, 0, SamplingParams(), handle, 1024)
    manager.allocate([req], phase="prefill")
    batch = Batch([req], "prefill")
    manager.prepare_batch(batch)
    writes = batch.cache_metadata.checkpoint_writes[0]
    assert [end for end, _ in writes] == [ps, ps + 12]
    for end, slot in writes:
        for block in p.state_blocks[slot][::p.layout.state_blocks_per_group]:
            p.recurrent_state[block].fill_(end)
            p.conv_state[block].fill_(end)
    manager.after_forward(batch)
    entry = manager.requests[0]
    branch = entry.checkpoints[ps]
    assert branch not in entry.states
    manager.cache_req(req, finished=False)
    for slot in entry.states:
        p.zero_state(slot)
    for block in p.state_blocks[branch][::p.layout.state_blocks_per_group]:
        assert (p.recurrent_state[block] == ps).all()
        assert (p.conv_state[block] == ps).all()
    manager.cache_req(req, finished=True)
    manager.check_integrity()
    manager.tree.reset()
    assert len(p.allocator.free) == initial_free


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", ["gdn", "mamba2"])
@pytest.mark.parametrize("block_size", [64, 128, 256])
def test_double_state_decode_graph_matches_single_work_state(kind, block_size):
    from logssms.cache.radix.state_pool import LinearStatePool
    from logssms.core import Context, set_global_ctx
    from logssms.layers.gated_delta import GatedDeltaConfig
    from logssms.layers.mamba2 import Mamba2Config
    from logssms.layers.linear_attention_backend import make_linear_attention_backend
    from tests.utils import clear_global_ctx

    torch.manual_seed(37)
    manager = small_manager("cuda", kind, block_size)
    shared, ps = manager.pool, manager.page_size
    cfg = (GatedDeltaConfig(2, 4, 32, 32, 256, 3, 0) if kind == "gdn"
           else Mamba2Config(4, 2, 32, 32, 3, 0))
    dense = LinearStatePool(2, 3, 256, 4, 4, 32, 32, shared.device, shared.dtype)
    weight = torch.randn(256, 1, 4, device="cuda", dtype=shared.dtype) * .1
    vector = torch.zeros(4, device="cuda")
    x = torch.randn(260, 256, device="cuda", dtype=shared.dtype) * .1
    gate, beta = [torch.randn(260, 4, device="cuda", dtype=shared.dtype) * .1 for _ in range(2)]
    dense_table = torch.zeros_like(manager.state_table)
    clear_global_ctx()
    ctx = Context(ps)
    set_global_ctx(ctx)
    backend = make_linear_attention_backend("marconi", recurrent_type=kind)
    handle = manager.match_req(SimpleNamespace(input_ids=torch.arange(140))).cuda_handle
    manager.lock(handle)
    manager.init_with_handle(0, handle)
    req = Request(torch.arange(140, dtype=torch.int32), 0, 0, 120, 0,
                  SamplingParams(), handle, 1024)
    snapshot = None

    def forward(batch, begin, end):
        if kind == "gdn":
            return backend.forward(x[begin:end], gate[begin:end], beta[begin:end],
                                   weight, vector, vector, cfg, batch)
        return backend.forward(x[begin:end], gate[begin:end], weight, None,
                               vector, vector, torch.ones_like(vector), cfg, batch)

    try:
        for begin, end in [(0, 64), (64, 140)] + [(i, i + 1) for i in range(140, 260)]:
            req.cached_len, req.device_len = begin, end
            req.input_ids = torch.arange(end, dtype=torch.int32)
            phase = "prefill" if end - begin > 1 else "decode"
            batch = Batch([req], phase)
            batch.padded_reqs = [req]
            manager.allocate([req], phase=phase)
            manager.prepare_batch(batch)
            reference = Batch([req], phase)
            reference.padded_reqs = [req]
            if phase == "prefill":
                writes = [(end - begin, 0)]
                boundary = end // ps * ps
                if begin < boundary < end:
                    writes.insert(0, (boundary - begin, 1))
                reference.cache_metadata = MarconiCacheMetadata(checkpoint_writes=[writes])
            ctx.linear_state, ctx.state_table = dense, dense_table
            backend.prepare_metadata(reference)
            expected = forward(reference, begin, end).clone()
            if phase == "prefill" and begin < end // ps * ps < end:
                snapshot = dense.recurrent_state[1, 0].clone(), dense.conv_state[1, 0].clone()
            elif end % ps == 0:
                snapshot = dense.recurrent_state[0, 0].clone(), dense.conv_state[0, 0].clone()
            ctx.linear_state, ctx.state_table = shared, manager.state_table
            backend.prepare_metadata(batch)
            if phase == "prefill":
                actual = forward(batch, begin, end)
            elif begin == 140:
                static_x, static_g, static_b = x[140:141], gate[140:141], beta[140:141]
                captured = batch.linear_attn_metadata
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = forward(batch, 140, 141)
                torch.cuda.current_stream().wait_stream(stream)
                graph.replay()
            else:
                static_x.copy_(x[begin:end])
                static_g.copy_(gate[begin:end])
                static_b.copy_(beta[begin:end])
                captured.slots.copy_(batch.linear_attn_metadata.slots)
                graph.replay()
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
            manager.after_forward(batch)
            if snapshot is not None:
                entry = manager.requests[0]
                slot = entry.state_at(end // ps * ps, ps)
                block = shared.state_blocks[slot][0]
                torch.testing.assert_close(shared.recurrent_state[block, 0], snapshot[0], rtol=2e-2, atol=2e-3)
                torch.testing.assert_close(shared.conv_state[block, 0], snapshot[1])
        manager.cache_req(req, finished=True)
        assert manager.tree.match_prefix(tuple(range(260))).cuda_handle.cached_len == 256
        manager.check_integrity()
    finally:
        clear_global_ctx()


def test_shared_blocks_and_group_mapping():
    p = pool()
    kv, = p.allocator.allocate(1)
    a, b = p.allocate_state(), p.allocate_state()
    assert not ({kv} & set(p.state_blocks[a] + p.state_blocks[b]))
    p.k_cache(0)[kv].fill_(9)
    for group, block in enumerate(p.state_blocks[a]):
        p.recurrent_state[block].fill_(group + 1)
        p.conv_state[block].fill_(group + 3)
    p.copy_state(a, b)
    for src, dst in zip(p.state_blocks[a], p.state_blocks[b]):
        torch.testing.assert_close(p.recurrent_state[src], p.recurrent_state[dst])
        torch.testing.assert_close(p.conv_state[src], p.conv_state[dst])
    assert (p.k_cache(0)[kv] == 9).all()
    p.release_state(a)
    p.release_state(b)
    p.allocator.release((kv,))
    assert len(p.allocator.free) == 32


def test_tree_partial_page_fork_owns_its_kv():
    p = pool()
    tree = MarconiPrefixCache("cpu", 64, MarconiCost(1, 1, 1024, 16), pool=p)
    pages = p.allocator.allocate(2)
    locations = tuple(page * 64 + t for page in pages for t in range(64))
    state = p.allocate_state()
    tree.insert(tuple(range(128)), locations, state)
    p.release_state(state)
    p.allocator.release(pages)
    assert tree.match_prefix(tuple(range(80))).cuda_handle.cached_len == 0
    assert tree.match_prefix(tuple(range(80))).cuda_handle.branch_end == 64
    # A different physical page backs the fork's shared partial KV page.
    fork_pages = p.allocator.allocate(2)
    fork_locs = tuple(page * 64 + t for page in fork_pages for t in range(64))
    fork_tokens = tuple(range(80)) + (999,) * 48
    fork_state = p.allocate_state()
    fork = tree.insert(fork_tokens, fork_locs, fork_state)
    p.allocator.release(fork_pages)
    p.release_state(fork_state)
    assert fork.locations[:64] == locations[:64]
    assert p.allocator.refs[fork_pages[0]] == 0
    assert p.allocator.refs[fork_pages[1]] > 0
    tree.check_integrity()
    handle = tree.match_prefix(fork_tokens).cuda_handle
    tree.lock_handle(handle)
    while tree.evict_one():
        pass
    assert tree.match_prefix(fork_tokens).cuda_handle.node is fork
    assert all(p.allocator.refs[b] > 0 for b in tree._blocks(fork))
    tree.lock_handle(handle, unlock=True)
    tree.reset()
    assert len(p.allocator.free) == 32


def test_internal_eviction_preserves_child_prefix():
    p = pool()
    tree = MarconiPrefixCache("cpu", 64, MarconiCost(1, 0, 1024, 16), pool=p)
    pages = p.allocator.allocate(3)
    locs = tuple(page * 64 + t for page in pages for t in range(64))
    for end in (64, 128, 192):
        slot = p.allocate_state()
        tree.insert(tuple(range(end)), locs[:end], slot)
        p.release_state(slot)
    p.allocator.release(pages)
    assert tree.evict_one()
    assert tree.match_prefix(tuple(range(192))).cuda_handle.cached_len == 192
    assert all(p.allocator.refs[b] for b in pages)
    assert tree.stats["intermediate_evictions"] == 1
    tree.reset()
    assert len(p.allocator.free) == 32


@pytest.mark.parametrize("model", ["Qwen3.5-0.8B", "Falcon-H1-0.5B-Instruct"])
def test_manager_tail_and_branch_lifecycle(model):
    config = EngineConfig(
        model_path=f"/home/daking/models/huggingface/{model}",
        tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
        max_running_req=2, max_seq_len_override=4096,
        cache_type="marconi", cache_config=MarconiCacheConfig(),
    )
    factory = MarconiCacheFactory()
    plan = factory.plan(config, 256 << 20, None)
    resources = factory.allocate(config, plan, torch.device("cpu"), config.dtype)
    manager = factory.make_manager(config, plan, resources)
    initial_free = len(manager.pool.allocator.free)

    def run(tokens, expected_hit):
        pending = SimpleNamespace(input_ids=torch.tensor(tokens, dtype=torch.int32))
        handle = manager.match_req(pending).cuda_handle
        assert handle.cached_len == expected_hit
        assert manager.prefill_acquire(manager.reservation_size(len(tokens), handle=handle), handle)
        manager.init_with_handle(0, handle)
        req = Request(input_ids=pending.input_ids, table_idx=0, cached_len=handle.cached_len,
                      output_len=1, uid=0, sampling_params=SamplingParams(), cache_handle=handle,
                      max_device_len=len(tokens) + 1)
        manager.allocate([req], phase="prefill")
        batch = Batch([req], "prefill")
        manager.prepare_batch(batch)
        manager.after_forward(batch)
        manager.cache_req(req, finished=True)
        manager.check_integrity()
        return batch.cache_metadata.checkpoint_writes

    ps = manager.page_size
    writes = run(list(range(2 * ps)), 0)
    assert len(writes[0]) == 1
    assert len([n for n in manager.tree.nodes if n.state is not None]) == 1
    run(list(range(2 * ps)) + [10000], 2 * ps)
    run(list(range(ps)) + [10001] * ps, 0)
    run(list(range(ps)) + [10002] * ps, ps)
    manager.tree.reset()
    assert len(manager.pool.allocator.free) == initial_free


def test_tuning_returns_grid_alpha():
    prefix = tuple(range(64))
    alpha, results = tune_marconi([(prefix, 1)], [(prefix + (99,), prefix + (99, 100))] * 3,
                                  64, MarconiCost(1, 1, 1024, 16), 8192)
    assert len(results) == 21
    assert 0 <= alpha <= 2


def test_tuning_worker_is_spawn_safe():
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    prefix = tuple(range(64))
    args = ([(prefix, 1)], [(prefix + (99,), prefix + (99, 100))],
            64, MarconiCost(1, 1, 1024, 16), 8192)
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
        assert executor.submit(tune_marconi, *args).result(timeout=60) == tune_marconi(*args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", ["gdn", "mamba2"])
def test_shared_pool_prefill_decode_and_graph_match_standard(kind):
    from dataclasses import replace
    from logssms.cache.radix.state_pool import LinearStatePool
    from logssms.core import Context, set_global_ctx
    from logssms.layers.gated_delta import GatedDeltaConfig
    from logssms.layers.mamba2 import Mamba2Config
    from logssms.layers.linear_attention_backend import make_linear_attention_backend
    from tests.utils import clear_global_ctx

    torch.manual_seed(17)
    device, dtype = torch.device("cuda"), torch.bfloat16
    cfg = (GatedDeltaConfig(2, 4, 32, 32, 256, 3, 0) if kind == "gdn"
           else Mamba2Config(4, 2, 32, 32, 3, 0))
    spec = SimpleNamespace(num_layers=3, state_shape=(4, 32, 32), conv_shape=(cfg.conv_dim, 3))
    shared = HybridBlockPool(HybridBlockLayout.make(2, spec, 1, 32), 16, device, dtype)
    slots = [shared.allocate_state() for _ in range(4)]
    dense = LinearStatePool(4, 3, cfg.conv_dim, 4, 4, 32, 32, device, dtype)
    weights = [torch.randn(cfg.conv_dim, 1, 4, device=device, dtype=dtype) * .1 for _ in range(3)]
    table = torch.tensor(slots[:2], device=device, dtype=torch.int32)[:, None].expand(2, 192).contiguous()
    clear_global_ctx()
    ctx = Context(64)
    ctx.state_table = table
    set_global_ctx(ctx)
    backend = make_linear_attention_backend("marconi", recurrent_type=kind)
    vector = torch.zeros(4, device=device, dtype=torch.float32)

    def forward(batch, x, gate, beta, layer):
        config = replace(cfg, layer_idx=layer)
        if kind == "gdn":
            return backend.forward(x, gate, beta, weights[layer], vector, vector, config, batch)
        return backend.forward(x, gate, weights[layer], None, vector, vector,
                               torch.ones_like(vector), config, batch)

    try:
        for phase, lengths in [("prefill", [70, 65]), ("decode", [1, 1])]:
            reqs = [Request(torch.arange(n if phase == "prefill" else 71, dtype=torch.int32),
                            i, 0 if phase == "prefill" else 70, 8, i,
                            SamplingParams(), None, 192) for i, n in enumerate(lengths)]
            batch = Batch(reqs, phase)
            batch.padded_reqs = reqs
            if phase == "prefill":
                batch.cache_metadata = MarconiCacheMetadata(checkpoint_writes=[[(64, slots[i + 2]), (n, slots[i])]
                                                 for i, n in enumerate(lengths)])
            ctx.linear_state = shared
            backend.prepare_metadata(batch)
            total = sum(lengths)
            x = torch.randn(total, cfg.conv_dim, device=device, dtype=dtype) * .1
            gate, beta = [torch.randn(total, 4, device=device, dtype=dtype) * .1 for _ in range(2)]
            for layer in range(3):
                ctx.linear_state = dense
                expected = forward(batch, x, gate, beta, layer).clone()
                ctx.linear_state = shared
                actual = forward(batch, x, gate, beta, layer)
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
                group, lane = divmod(layer, 2)
                for i, slot in enumerate(slots):
                    block = shared.state_blocks[slot][group * shared.layout.state_blocks_per_group]
                    torch.testing.assert_close(shared.conv_state[block, lane], dense.conv_state[i, layer])
                    torch.testing.assert_close(shared.recurrent_state[block, lane], dense.recurrent_state[i, layer],
                                               rtol=2e-2, atol=2e-3)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = forward(batch, x, gate, beta, 2)
        torch.cuda.current_stream().wait_stream(stream)
        ctx.linear_state = dense
        expected = forward(batch, x, gate, beta, 2)
        ctx.linear_state = shared
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-3)
    finally:
        clear_global_ctx()
