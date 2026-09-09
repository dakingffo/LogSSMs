"""Verify commit swaps candidate slots into token-indexed live page mappings.

Every token in one live page points to the same state slot. Verify writes candidate
snapshots to a separate reserve table. Commit swaps the last accepted snapshot for each
touched page with that page's live slot; no state data is copied.

Run: /home/daking/.conda/envs/daking/bin/python tests/hybrid/test_reserve_index_ops.py
"""
import os
import sys

_REPO = "/home/daking/PROJECT/pico-sglang"
sys.path.insert(0, os.path.join(_REPO, "python"))
sys.path.insert(0, _REPO)

import torch

import picosgl.core as core
from picosgl.core import Context, Request, SamplingParams, set_global_ctx
from picosgl.cache import CacheResources, StandardCacheManager
from picosgl.cache.linear_pool.state_pool import LinearStatePool


def setup_ctx() -> None:
    """Fresh 64-token context per test (pytest imports all modules up front, so the
    module level must stay side-effect free)."""
    core._GLOBAL_CTX = None
    set_global_ctx(Context(page_size=64))


def make_req(table_idx, cached_len):
    return Request(
        input_ids=torch.arange(200, dtype=torch.int32),
        table_idx=table_idx, cached_len=cached_len, output_len=10, uid=table_idx,
        sampling_params=SamplingParams(), cache_handle=None,
        max_device_len=210,
    )


def make_cm(st, reserve, nslots):
    pool = LinearStatePool(
        num_slots=nslots, num_linear_layers=1, conv_dim=8, kernel_size=4,
        num_v_heads=4, head_k_dim=4, head_v_dim=4,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    resources = CacheResources(
        kv_pool=None,  # type: ignore[arg-type]
        page_table=torch.zeros((8, 512), dtype=torch.int32),
        linear_state=pool,
        state_table=st,
        work_state_table=reserve,
    )
    cm = StandardCacheManager(
        num_pages=10, page_size=64,
        resources=resources,
        type="hybrid_radix", num_states=nslots,
        num_draft_states=reserve.numel(),
    )
    return cm, pool


def setup_pages(uid, page_slots, reserve_slots):
    st = torch.full((8, 512), -1, dtype=torch.int32)
    for page, slot in page_slots.items():
        st[uid, page * 64 : (page + 1) * 64] = slot
    reserve = torch.full((8, len(reserve_slots)), -1, dtype=torch.int32)
    reserve[uid] = torch.tensor(reserve_slots, dtype=torch.int32)
    return st, reserve


def mark_in_use(cm, slot_ids):
    """Take `slot_ids` out of the manager's free list (they are pre-populated)."""
    ids = torch.tensor(slot_ids, dtype=torch.int32)
    cm.free_states = cm.free_states[~torch.isin(cm.free_states, ids)]


# 4a. no page crossing: the accepted tail R[1] becomes page 1's live state.
def test_no_cross_swap():
    setup_ctx()
    st, reserve = setup_pages(2, {1: 40}, [50, 51, 52, 53])
    cm, pool = make_cm(st, reserve, nslots=200)
    mark_in_use(cm, [40, 50, 51, 52, 53])
    r = make_req(2, 120)
    snap = (pool.conv_state.clone(), pool.recurrent_state.clone())
    cm.commit_speculative(r, 120, 2)
    assert torch.all(st[2, 64:128] == 51)
    assert reserve[2].tolist() == [50, 40, 52, 53]
    assert len(cm.free_states) == 200 - 5
    assert torch.equal(pool.conv_state, snap[0]) and torch.equal(pool.recurrent_state, snap[1])
    print("[4a] commit no-cross swap OK")


# 4b. crossing: R[2] closes page 1 and R[3] becomes page 2's live state.
def test_cross_pin_refill_swap():
    setup_ctx()
    st, reserve = setup_pages(2, {1: 40, 2: 41}, [50, 51, 52, 53])
    cm, pool = make_cm(st, reserve, nslots=300)
    mark_in_use(cm, [40, 41, 50, 51, 52, 53])
    r = make_req(2, 125)
    snap = (pool.conv_state.clone(), pool.recurrent_state.clone())
    cm.commit_speculative(r, 125, 4)
    assert torch.all(st[2, 64:128] == 52)
    assert torch.all(st[2, 128:192] == 53)
    assert reserve[2].tolist() == [50, 51, 40, 41]
    assert len(cm.free_states) == 300 - 6
    assert torch.equal(pool.conv_state, snap[0]) and torch.equal(pool.recurrent_state, snap[1])
    print("[4b] commit crossing swaps OK")


# 4c. boundary token is the last accepted: R[1] closes page 1.
def test_cross_boundary_last():
    setup_ctx()
    st, reserve = setup_pages(2, {1: 40}, [50, 51, 52, 53])
    cm, pool = make_cm(st, reserve, nslots=300)
    mark_in_use(cm, [40, 50, 51, 52, 53])
    r = make_req(2, 126)
    snap = (pool.conv_state.clone(), pool.recurrent_state.clone())
    cm.commit_speculative(r, 126, 2)
    assert torch.all(st[2, 64:128] == 51)
    assert reserve[2].tolist() == [50, 40, 52, 53]
    assert len(cm.free_states) == 300 - 5
    assert torch.equal(pool.conv_state, snap[0]) and torch.equal(pool.recurrent_state, snap[1])
    print("[4c] commit boundary-last swap OK")


def test_stable_reserve_across_overlapping_batches():
    """Different in-flight cohorts keep disjoint table-owned reserve rows."""
    setup_ctx()
    reserve_width, max_running_req = 5, 4
    num_states = 100
    st = torch.full(
        (max_running_req + 1, 512), -1, dtype=torch.int32
    )
    reserve = torch.arange(
        num_states, num_states + max_running_req * reserve_width, dtype=torch.int32
    ).view(max_running_req, reserve_width)
    for uid in range(max_running_req):
        st[uid, :64] = uid

    pool = LinearStatePool(
        num_slots=num_states + max_running_req * reserve_width,
        num_linear_layers=1, conv_dim=8, kernel_size=4,
        num_v_heads=4, head_k_dim=4, head_v_dim=4,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    resources = CacheResources(
        kv_pool=None,  # type: ignore[arg-type]
        page_table=torch.zeros((max_running_req + 1, 512), dtype=torch.int32),
        linear_state=pool,
        state_table=st,
        work_state_table=reserve,
    )
    cm = StandardCacheManager(
        num_pages=10, page_size=64,
        resources=resources,
        type="hybrid_radix", num_states=num_states,
        num_draft_states=max_running_req * reserve_width,
    )
    original = reserve.clone()
    cohort_a = [make_req(0, 32), make_req(2, 32)]
    cohort_b = [make_req(1, 32), make_req(3, 32)]
    for req in cohort_a + cohort_b:
        req.device_len = req.cached_len + reserve_width

    cm.allocate(cohort_a)
    cm.allocate(cohort_b)
    assert torch.equal(reserve, original)
    assert all(st[req.table_idx, req.cached_len - 1] == req.table_idx for req in cohort_a + cohort_b)
    assert len(torch.unique(reserve)) == max_running_req * reserve_width
    print("[4d] stable overlapping-batch reserve OK")


if __name__ == "__main__":
    test_no_cross_swap()
    test_cross_pin_refill_swap()
    test_cross_boundary_last()
    test_stable_reserve_across_overlapping_batches()
    print("\nALL RESERVE INDEX OP TESTS PASSED")
