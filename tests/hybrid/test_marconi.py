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


def pool(device="cpu", blocks=32):
    spec = SimpleNamespace(num_layers=3, state_shape=(2, 8, 8), conv_shape=(32, 3))
    layout = HybridBlockLayout.make(2, spec, 1, 8)
    return HybridBlockPool(layout, blocks, device, torch.float32)


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
    tree.insert(tuple(range(96)), locations[:96], state)
    p.release_state(state)
    p.allocator.release(pages)
    assert tree.match_prefix(tuple(range(80))).cuda_handle.cached_len == 0
    assert tree.match_prefix(tuple(range(80))).cuda_handle.branch_end == 64
    # A different physical page backs the fork's shared partial KV page.
    fork_pages = p.allocator.allocate(2)
    fork_locs = tuple(page * 64 + t for page in fork_pages for t in range(64))
    fork_tokens = tuple(range(80)) + (999,) * 16
    fork_state = p.allocate_state()
    fork = tree.insert(fork_tokens, fork_locs[:96], fork_state)
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
        max_running_req=2, max_seq_len_override=256,
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

    writes = run(list(range(128)), 0)
    assert len(writes[0]) == 1
    assert len([n for n in manager.tree.nodes if n.state is not None]) == 1
    run(list(range(128)) + [400], 128)
    run(list(range(64)) + [500] * 64, 0)
    run(list(range(64)) + [600] * 64, 64)
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
                    block = shared.state_blocks[slot][group]
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
