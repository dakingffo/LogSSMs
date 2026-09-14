"""Hybrid RadixTreeNode / flat LinearStatePool / CacheManager state wiring.

CPU-only. Verifies the per-page linear-state caching design end to end:

  1. HybridRadixTreeNode: nodes carry token-indexed state mappings; split preserves
     state; evict returns both KV pages and unique state slots.
  2. CacheManager.allocate: allocates KV pages and state slots together.
  3. CacheManager.cache_req lifecycle: tree adopts a request's full pages + their
     slots (slots stay allocated, tree-owned), a finished request frees its tail page
     slot and wipes its state_table row, a cache-hit reuses the borrowed slot, and
     eviction returns KV pages + state slots. check_integrity holds for both.

Run: /home/daking/.conda/envs/daking/bin/python tests/hybrid/test_hybrid_radix_cache.py
"""
import os
import sys

_REPO = "/home/daking/PROJECT/pico-sglang"
sys.path.insert(0, os.path.join(_REPO, "python"))
sys.path.insert(0, _REPO)

import torch

from picosgl.core import Context, Request, SamplingParams, set_global_ctx
from picosgl.cache import (
    CacheResources,
    StandardCacheManager,
    StandardPrefixValue,
)
from picosgl.scheduler.prefill import PendingRequest
from picosgl.cache.linear_pool.state_pool import LinearStatePool
from picosgl.cache.prefix_cache.radix import HybridRadixTreeNode, RadixPrefixCache


def make_req(input_len, table_idx, output_len=10):
    return Request(
        input_ids=torch.arange(input_len, dtype=torch.int32),
        table_idx=table_idx,
        cached_len=0,
        output_len=output_len,
        uid=table_idx,
        sampling_params=SamplingParams(),
        cache_handle=None,
        max_device_len=input_len + output_len,
    )


def make_pool(num_slots=400, dtype=torch.float32):
    return LinearStatePool(
        num_slots=num_slots, num_linear_layers=1, conv_dim=8, kernel_size=4,
        num_v_heads=4, head_k_dim=4, head_v_dim=4,
        device=torch.device("cpu"), dtype=dtype,
    )


def make_cm(
    state_table, pool, page_table=None,
    num_pages=20, page_size=64, num_states=400,
):
    if page_table is None:
        page_table = torch.zeros((4, 512), dtype=torch.int32)
    resources = CacheResources(
        kv_pool=None,  # type: ignore[arg-type]
        page_table=page_table,
        linear_state=pool,
        state_table=state_table,
        work_state_table=None,
    )
    return StandardCacheManager(
        num_pages=num_pages, page_size=page_size,
        resources=resources,
        prefix_cache_type="hybrid_radix", num_states=num_states,
    )


# =====================================================================================
# 1. HybridRadixTreeNode: state-carrying nodes, split, evict
# =====================================================================================
def test1_hybrid_radix():
    hc = RadixPrefixCache(device=torch.device("cpu"), node_type=HybridRadixTreeNode)
    ids = torch.arange(192, dtype=torch.int32)       # 3 pages
    state = torch.tensor([10] * 64 + [11] * 64 + [12] * 64, dtype=torch.int32)
    res = hc.insert_prefix(ids, StandardPrefixValue(ids.clone(), state))
    assert res.handle.cached_len == 192
    matched_indices, matched_state = res.handle.get_matched_indices()
    assert matched_indices.tolist() == list(range(192))
    assert matched_state.tolist() == state.tolist()
    assert hc.size_info.evictable_size == 192
    # partial match -> split at page 2 (128 tokens): matched state = pages [0,2)
    m = hc.match_prefix(torch.arange(128, dtype=torch.int32))
    assert m.cuda_handle.cached_len == 128
    assert m.cuda_handle.get_matched_indices()[1].tolist() == state[:128].tolist()
    # extending with a new suffix creates a sibling node; state preserved through split
    ids2 = torch.arange(256, dtype=torch.int32)
    state2 = torch.tensor(
        [10] * 64 + [11] * 64 + [12] * 64 + [13] * 64,
        dtype=torch.int32,
    )
    hc.insert_prefix(ids2, StandardPrefixValue(ids2.clone(), state2))
    m2 = hc.match_prefix(ids2)
    assert m2.cuda_handle.get_matched_indices()[1].tolist() == state2.tolist()
    # evict 2 pages -> returns 2 state slots
    ev_idx, ev_state = hc.evict(128)
    assert ev_state is not None and len(ev_state) == 2, (ev_idx, ev_state)
    print("[1] Hybrid RadixPrefixCache OK")


# =====================================================================================
# 2. allocate: one KV page and state slot per page actually written
# =====================================================================================
def test2_allocate():
    pool = make_pool()
    st = torch.full((4, 512), -1, dtype=torch.int32)
    cm = make_cm(st, pool)
    req = make_req(100, table_idx=1)
    req.cached_len, req.device_len = 0, 100
    cm.allocate([req])
    # pages 0 (0-63) and 1 (64-99) written by a 100-token prefill; page 2 untouched
    assert st[1, 0] >= 0 and st[1, 64] >= 0 and st[1, 128] == -1, st[1]
    assert torch.all(st[1, :64] == st[1, 0])
    assert torch.all(st[1, 64:128] == st[1, 64])
    assert len(cm.free_states) == 400 - 2, len(cm.free_states)
    # decode step staying inside page 1 -> no new slot
    req.cached_len, req.device_len = 90, 91
    cm.allocate([req])
    assert st[1, 64] >= 0 and st[1, 128] == -1, st[1]
    assert len(cm.free_states) == 400 - 2, len(cm.free_states)
    # Rolling back within one page frees nothing, but must restore the live frontier.
    req.device_len = 95
    cm.free_to_cached(req)
    assert req.device_len == 91
    assert len(cm.free_pages) == 20 - 2
    assert len(cm.free_states) == 400 - 2
    # decode crossing into page 2 -> allocates page 2's slot
    req.cached_len, req.device_len = 128, 129
    cm.allocate([req])
    assert st[1, 128] >= 0, st[1]
    assert len(cm.free_states) == 400 - 3, len(cm.free_states)
    # Rejecting page 2 returns its KV page and state slot together.
    freed_page = int(cm.page_table[1, 128])
    freed_state = int(st[1, 128])
    with cm.lazy_free_region():
        cm.free_to_cached(req)
        assert len(cm.free_pages) == 20 - 3
        assert len(cm.free_states) == 400 - 3
    assert len(cm.free_pages) == 20 - 2
    assert len(cm.free_states) == 400 - 2
    assert int(cm.free_pages[-1]) == freed_page
    assert int(cm.free_states[-1]) == freed_state
    assert torch.all(st[1, 128:192] == -1)
    print("[2] allocate OK")


# =====================================================================================
# 3. cache_req lifecycle + check_integrity (idle)
# =====================================================================================
def test3_cache_req_lifecycle():
    N, PS = 20, 64
    pool = make_pool(num_slots=400)
    st = torch.full((4, 512), -1, dtype=torch.int32)
    pt = torch.zeros((4, 512), dtype=torch.int32)
    cm = make_cm(st, pool, pt)

    # ---- req 1: prefill 200 tokens (4 pages) ----
    r = make_req(200, table_idx=1)
    r.cached_len, r.device_len = 0, 200
    cm.allocate([r])
    s0, s1, s2, s3 = (int(st[1, i * PS]) for i in range(4))
    assert len(cm.free_states) == 400 - 4   # 4 slots allocated (no dummy slot)
    assert len(cm.free_pages) == N - 4

    # ---- cache_req(finished=False): tree adopts pages [0,3) + their slots ----
    match = cm.match_req(PendingRequest(r.uid, r.input_ids, r.sampling_params))
    r.cache_handle = match.cuda_handle
    r.cached_len = 200
    cm.cache_req(r, finished=False)
    assert r.cache_handle.cached_len == 192
    # slots are re-pointed to the tree's canonical slots (same ids); adopted slots stay
    # ALLOCATED (tree-owned, never freed) so the tree reference stays valid.
    assert st[1, 0] == s0 and st[1, 64] == s1 and st[1, 128] == s2, st[1]
    assert cm.prefix_cache.total_state_pages() == 3
    assert len(cm.free_states) == 400 - 4   # page-3 tail (s3) still this request's

    # ---- finish req 1: free the tail page-3 KV + its state slot; wipe the row ----
    cm.cache_req(r, finished=True)
    assert (st[1] == -1).all(), st[1]
    assert len(cm.free_states) == 400 - 3   # s3 freed; s0,s1,s2 live in the tree
    assert cm.prefix_cache.total_state_pages() == 3
    assert len(cm.free_pages) == N - 3      # 17 pages free

    # ---- req 2 reuses uid 1: cache hit on [0,64), extends to [64,100) ----
    r2 = make_req(100, table_idx=1)
    match2 = cm.match_req(PendingRequest(r2.uid, r2.input_ids, r2.sampling_params))
    assert match2.cuda_handle.cached_len == 64
    r2.cache_handle = match2.cuda_handle
    cm.lock(r2.cache_handle)
    matched_pages, matched_states = match2.cuda_handle.get_matched_indices()
    pt[1, :64] = matched_pages
    st[1, :64] = matched_states
    r2.cached_len, r2.device_len = 64, 100
    cm.allocate([r2])
    assert st[1, 0] == s0 and st[1, 64] >= 0 and st[1, 64] != s1, (st[1], s1)
    assert len(cm.free_states) == 400 - 4   # one fresh slot popped
    assert len(cm.free_pages) == N - 4      # 3 tree + 1 fresh (64-100)

    # ---- finish req 2: free page-1 tail slot; tree split at 64 -> [0,64)+[64,192) ----
    r2.cached_len, r2.device_len = 99, 100
    cm.cache_req(r2, finished=True)
    assert (st[1] == -1).all(), st[1]
    assert cm.prefix_cache.total_state_pages() == 3
    assert len(cm.free_states) == 400 - 3   # s1b freed; s0,s1,s2 in the tree
    assert len(cm.free_pages) == N - 3

    # ---- evict the [64,192) leaf: 2 KV pages + 2 state slots ----
    ev_idx, ev_state = cm.prefix_cache.evict(128)
    assert ev_state is not None and len(ev_state) == 2, ev_state
    cm.free_states = torch.cat([cm.free_states, ev_state])
    cm.free_pages = torch.cat([cm.free_pages, ev_idx[::PS]])
    assert cm.prefix_cache.total_state_pages() == 1
    assert len(cm.free_states) == 400 - 1
    assert len(cm.free_pages) == N - 1

    # ---- check_integrity: idle, both KV and state conserved ----
    assert (st == -1).all()
    cm.check_integrity()
    print("[3] cache_req lifecycle + check_integrity OK")


def test4_concurrent_insert_repoints_to_canonical_slots():
    pool = make_pool(num_slots=400)
    st = torch.full((4, 512), -1, dtype=torch.int32)
    pt = torch.zeros((4, 512), dtype=torch.int32)
    cm = make_cm(st, pool, pt)
    reqs = [make_req(200, table_idx=i) for i in (1, 2)]
    for req in reqs:
        handle = cm.match_req(PendingRequest(req.uid, req.input_ids, req.sampling_params))
        req.cache_handle = handle.cuda_handle
        cm.lock(req.cache_handle)
        req.cached_len, req.device_len = 0, 200

    cm.allocate(reqs)
    for req in reqs:
        req.cached_len = 200
    duplicate_pages = pt[2, :192:64].clone()
    duplicate_states = st[2, :192:64].clone()

    with cm.lazy_free_region():
        cm.cache_req(reqs[0], finished=False)
        cm.cache_req(reqs[1], finished=False)
        assert len(cm.free_pages) == 20 - 8
        assert len(cm.free_states) == 400 - 8

    assert torch.equal(pt[2, :192], pt[1, :192])
    assert torch.equal(st[2, :192], st[1, :192])
    assert torch.equal(cm.free_pages[-3:], duplicate_pages)
    assert torch.equal(cm.free_states[-3:], duplicate_states)
    print("[4] concurrent insert canonical re-point OK")


if __name__ == "__main__":
    set_global_ctx(Context(page_size=64))
    test1_hybrid_radix()
    test2_allocate()
    test3_cache_req_lifecycle()
    test4_concurrent_insert_repoints_to_canonical_slots()
    print("\nALL HYBRID RADIX CACHE TESTS PASSED")
