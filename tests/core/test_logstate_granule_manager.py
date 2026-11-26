from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from logssms.cache import CacheResources
from logssms.cache.logstate.frame_pool import MHALogStatePool
from logssms.cache.logstate.manager import LogStateCacheManager
from logssms.core import Batch
from tests.core.test_logstate_cache_manager import _cpu_copy_states as _cpu_copy_states
from tests.core.test_logstate_cache_manager import _stub_cpu_kernels, new_req


def make_manager(
    g=32,
    frames=96,
    device="cpu",
    max_seq_len=512,
    value_head_ratio=1,
    dtype=torch.bfloat16,
    enable_merge_adjustment=False,
    enable_hit_materialization=False,
):
    dev = torch.device(device)
    pool = MHALogStatePool(
        num_frames=frames,
        page_size=g,
        granularity=g,
        num_attention_layers=1,
        num_kv_heads=1,
        head_dim=64,
        num_linear_layers=1,
        num_key_heads=1,
        num_value_heads=value_head_ratio,
        head_k_dim=128,
        head_v_dim=128,
        conv_dim=(2 + value_head_ratio) * 128,
        state_len=3,
        device=dev,
        dtype=dtype,
    )
    _stub_cpu_kernels(pool)
    resources = CacheResources(
        pool,
        torch.zeros(5, max_seq_len, dtype=torch.int32, device=dev),
        pool,
        torch.full((5, max_seq_len), -1, dtype=torch.int32, device=dev),
        torch.full((5, 1), -1, dtype=torch.int32, device=dev),
    )
    return LogStateCacheManager(
        num_pages=frames - 1,
        num_states=frames // 8,
        num_work_states=5,
        page_size=g,
        resources=resources,
        granularity=g,
        enable_merge_adjustment=enable_merge_adjustment,
        enable_hit_materialization=enable_hit_materialization,
    )


def fill(manager, req):
    manager.allocate([req], phase="prefill")
    manager.prepare_batch(Batch([req], "prefill"))
    req.cached_len = req.device_len
    manager.cache_req(req, finished=True)


@pytest.mark.parametrize(
    "count,expected_ckpts,expected_logs",
    [
        (1, 0, 1),
        (2, 1, 0),
        (3, 1, 1),
        (4, 2, 0),
        (8, 4, 0),
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_prefill_materializes_complete_log_groups(count, expected_ckpts, expected_logs, enabled):
    manager = make_manager(enable_hit_materialization=enabled)
    assert manager.logstate_pool.allocator.checkpoint_log_count == 2
    tokens = torch.arange(count * 32, dtype=torch.int32)
    first = new_req(manager, tokens, uid=0, table_idx=0)
    fill(manager, first)
    second = new_req(manager, torch.cat((tokens, torch.tensor([999], dtype=torch.int32))), uid=1, table_idx=1)
    assert second.cached_len == len(tokens)
    manager.allocate([second], phase="prefill")
    with patch.object(manager._construction, "restore", wraps=manager._construction.restore) as restore:
        manager.prepare_batch(Batch([second], "prefill"))
    snap = manager.layout_snapshot()
    if enabled:
        assert snap["checkpoint"] == expected_ckpts
        assert snap["log"] == expected_logs
    else:
        assert snap["checkpoint"] == 0
        assert snap["log"] == count
    restore.assert_called_once()
    [task] = restore.call_args.args[0]
    assert task.output_slot == manager.work_state_table[second.table_idx, 0].item()
    if not enabled:
        assert len(task.logs) == count and sum(e.end - e.begin for e in task.logs) == len(tokens)
    manager.cache_req(second, finished=True)
    manager.check_integrity()


@pytest.mark.parametrize("g", [16, 32, 64])
def test_complete_base_groups_and_tail_are_model_sized(g):
    manager = make_manager(g=g, frames=192, enable_hit_materialization=True)
    n = manager._allocator.materialize_log_count
    count = 3 * n + 1
    tokens = torch.arange(count * g, dtype=torch.int32)
    fill(manager, new_req(manager, tokens, uid=0, table_idx=0))
    req = new_req(manager, torch.cat((tokens, torch.tensor([999], dtype=torch.int32))), uid=1, table_idx=1)
    manager.allocate([req], phase="prefill")
    manager.prepare_batch(Batch([req], "prefill"))
    snap = manager.layout_snapshot()
    assert snap["checkpoint"] == 3
    assert snap["log"] == 1
    manager.cache_req(req, finished=True)
    manager.check_integrity()
