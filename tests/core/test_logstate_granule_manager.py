from unittest.mock import patch

import pytest
import torch

from logssms.cache import CacheResources
from logssms.cache.logstate.frame_pool import MHALogStatePool
from logssms.cache.logstate.manager import LogStateCacheManager
from logssms.cache.logstate.prefix_cache import LogStateKind
from logssms.core import Batch
from tests.core.test_logstate_cache_manager import cpu_cache_copy as cpu_cache_copy
from tests.core.test_logstate_cache_manager import new_req
from tests.logstate_helpers import decay_view, write_head


def make_manager(g=32, frames=96, device='cpu', max_seq_len=512,
                 value_head_ratio=1, dtype=torch.bfloat16, enable_merge_adjustment=False,
                 enable_hit_materialization=False):
    device = torch.device(device)
    pool = MHALogStatePool(
        num_frames=frames, page_size=g, granularity=g,
        num_attention_layers=1, num_kv_heads=1, head_dim=64,
        num_linear_layers=1, num_key_heads=1, num_value_heads=value_head_ratio,
        head_k_dim=128, head_v_dim=128, conv_dim=(2+value_head_ratio)*128, state_len=3,
        device=device, dtype=dtype,
    )
    resources = CacheResources(
        pool, torch.zeros(5, max_seq_len, dtype=torch.int32, device=device), pool,
        torch.full((5, max_seq_len), -1, dtype=torch.int32, device=device),
        torch.full((5, 1), -1, dtype=torch.int32, device=device),
    )
    return LogStateCacheManager(num_pages=frames-1, num_states=frames//8,
                               num_work_states=5, page_size=g, resources=resources,
                               granularity=g, enable_merge_adjustment=enable_merge_adjustment,
                               enable_hit_materialization=enable_hit_materialization)


def fill(manager, req):
    manager.allocate([req], phase='prefill')
    manager.prepare_batch(Batch([req], 'prefill'))
    pool = manager.logstate_pool
    for page in manager._request_states[req.table_idx].live_granules:
        if page.state_slot is None or page.checkpoint_begin is not None:
            continue
        slot = page.state_slot
        decay_view(pool, slot).zero_()
        for head in range(pool.layout.num_value_heads):
            write_head(pool, slot, 0, head, torch.full((2 * pool.page_size * 128,), .01), state=False)
        pool.write_conv(torch.tensor([slot],device=pool.device), torch.tensor([0],device=pool.device),
                        0, torch.full((1,pool.layout.conv_dim,3), .1, device=pool.device))
    req.cached_len = req.device_len
    manager.cache_req(req, finished=True)


@pytest.mark.parametrize('count,expected', [(1, [(32,'log')]), (2,[(64,'checkpoint')]),
                                          (3,[(64,'checkpoint'),(32,'log')]),
                                          (4,[(64,'checkpoint')]*2),
                                          (8,[(64,'checkpoint')]*4)])
@pytest.mark.parametrize("enabled", [False, True])
def test_prefill_materializes_complete_log_groups(count, expected, enabled):
    manager = make_manager(enable_hit_materialization=enabled)
    assert manager.logstate_pool.allocator.checkpoint_log_count == 2
    tokens = torch.arange(count*32, dtype=torch.int32)
    first = new_req(manager, tokens)
    fill(manager, first)
    assert len(manager.tree._iter_nodes()) == 2
    second = new_req(manager, torch.cat((tokens, torch.tensor([999],dtype=torch.int32))))
    assert second.cached_len == len(tokens)
    manager.allocate([second],phase='prefill')
    with patch.object(manager._construction, "restore", wraps=manager._construction.restore) as restore:
        manager.prepare_batch(Batch([second], 'prefill'))
    entries = manager.tree.iter_entries()
    assert [(e.end-e.begin,e.kind.value) for e in entries] == (expected if enabled else [(32, "log")] * count)
    restore.assert_called_once()
    [task] = restore.call_args.args[0]
    assert task.output_slot == manager._request_states[second.table_idx].work_slot
    if not enabled:
        assert len(task.logs) == count and task.logs[-1].end == len(tokens)
    assert len(manager.tree._iter_nodes()) == 2
    manager.cache_req(second,finished=True)
    manager.check_integrity()


@pytest.mark.parametrize('g',[16,32,64])
def test_complete_base_groups_and_tail_are_model_sized(g):
    manager=make_manager(g=g,frames=192,enable_hit_materialization=True)
    n=manager._allocator.materialize_log_count
    count=3*n+1
    tokens=torch.arange(count*g,dtype=torch.int32)
    fill(manager,new_req(manager,tokens))
    req=new_req(manager,torch.cat((tokens,torch.tensor([999],dtype=torch.int32))))
    manager.allocate([req],phase='prefill');manager.prepare_batch(Batch([req],'prefill'))
    entries=manager.tree.iter_entries()
    assert all(e.span_granules%n==0 for e in entries if e.kind is LogStateKind.CHECKPOINT)
    assert sum(e.kind is LogStateKind.LOG for e in entries)==count%n
    assert sum(e.end-e.begin for e in entries)==count*g
    manager.cache_req(req,finished=True);manager.check_integrity()
