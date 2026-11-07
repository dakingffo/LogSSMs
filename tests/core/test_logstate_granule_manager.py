from tests.logstate_helpers import eviction_range
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
                 value_head_ratio=1, dtype=torch.bfloat16, enable_merge_adjustment=False):
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
                               granularity=g, enable_merge_adjustment=enable_merge_adjustment)


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
                                          (4,[(128,'checkpoint')]),
                                          (8,[(128,'checkpoint'),(128,'checkpoint')])])
def test_production_prepare_materializes_and_merges_once(count, expected):
    manager = make_manager()
    assert manager.logstate_pool.allocator.checkpoint_log_count == 2
    tokens = torch.arange(count*32, dtype=torch.int32)
    first = new_req(manager, tokens)
    fill(manager, first)
    assert len(manager.tree._iter_nodes()) == 2
    second = new_req(manager, torch.cat((tokens, torch.tensor([999],dtype=torch.int32))))
    assert second.cached_len == len(tokens)
    manager.allocate([second],phase='prefill')
    manager.prepare_batch(Batch([second],'prefill'))
    entries = manager.tree.iter_entries()
    assert [(e.end-e.begin,e.kind.value) for e in entries] == expected
    assert len(manager.tree._iter_nodes()) == 2
    manager.cache_req(second,finished=True)
    manager.check_integrity()


def test_tail_eviction_keeps_checkpoint_and_remembers_log():
    manager=make_manager()
    tokens=torch.arange(96,dtype=torch.int32)
    fill(manager,new_req(manager,tokens))
    second=new_req(manager,torch.cat((tokens,torch.tensor([1000],dtype=torch.int32))))
    manager.allocate([second],phase='prefill');manager.prepare_batch(Batch([second],'prefill'))
    manager.cache_req(second,finished=True)
    tree=manager.tree
    segments,_=tree.eviction_snapshot()
    tail=segments[-1]
    assert tail.entries[0].kind is LogStateKind.LOG
    manager.recycle_bin.remember(tree,tail)
    kv = tail._kv_frame_counts
    slots = tail.eviction_slots
    tree.evict([tail])
    manager.evictor._release_cached_kv(kv)
    manager.logstate_pool.retire_slots(slots)
    assert tree.match_prefix(tokens).cuda_handle.cached_len==64
    assert manager.recycle_bin.match(tokens,start=64,granularity=32) is not None
    assert len(tree._iter_nodes())==2
    tree.check_integrity()


def test_pressure_evicts_one_tail_granule_then_restores_with_current_gpu_shape():
    manager = make_manager()
    tokens = torch.arange(256, dtype=torch.int32)
    fill(manager, new_req(manager, tokens))
    pool = manager.logstate_pool
    raw = pool.reserve(kv=pool.allocator.free_frames)[0]
    new_kv, _, _ = manager.evictor.allocate(1, 0, recycle_bin=manager.recycle_bin)
    assert manager.tree.trace(tokens).matched_len == 224
    assert len(manager.tree._iter_nodes()) == 2
    pool.release_kv([value // 32 for value in new_kv])
    pool.release_kv(raw)
    manager.check_integrity()
    second = new_req(manager, torch.cat((tokens, torch.tensor([999], dtype=torch.int32))))
    assert second.cached_len == 224
    restore = manager._recycled_restores[second.table_idx].plan
    assert any(checkpoint.end == 256 for checkpoint in restore.checkpoints)
    fill(manager, second)
    assert manager.tree.match_prefix(tokens).cuda_handle.cached_len == 256
    assert all((entry.end - entry.begin) % 64 == 0 for entry in manager.tree.iter_entries()
               if entry.kind is LogStateKind.CHECKPOINT)
    manager.check_integrity()


def remember(manager, tokens, begin, end):
    from logssms.cache.logstate.prefix_cache import LogRadixTree, LogStateEntry
    ghost=LogRadixTree(manager.granularity,torch.device('cpu'))
    g=manager.granularity
    node=ghost.insert(tokens,tokens,tuple(LogStateEntry(1000+i,LogStateKind.LOG,left,left+g)
                                         for i,left in enumerate(range(begin,end,g)))).node
    manager.recycle_bin.remember(ghost,eviction_range(ghost,node))


def checkpoint_history(manager, spans):
    from logssms.cache.logstate.prefix_cache import LogStateEntry
    pool=manager.logstate_pool;g=manager.granularity
    frames,_,slots=pool.reserve(kv=sum(spans),works=len(spans))
    tokens=torch.arange(sum(spans)*g,dtype=torch.int32)
    mapping=(torch.tensor(frames,dtype=torch.int32)[:,None]*g+torch.arange(g,dtype=torch.int32)).flatten()
    entries=[];begin=0
    for slot,span in zip(slots,spans):
        pool.zero_work_slots([slot]);pool.commit_checkpoint(slot)
        entries.append(LogStateEntry(slot,LogStateKind.CHECKPOINT,begin,begin+span*g,span_granules=span))
        begin+=span*g
    manager.tree.insert(tokens,mapping,tuple(entries))
    manager.evictor.cached_page_tokens.update({frame*g:g for frame in frames})
    manager.check_integrity()
    return tokens,entries


@pytest.mark.parametrize('g',[16,32,64])
def test_complete_base_groups_and_tail_are_model_sized(g):
    manager=make_manager(g=g,frames=192)
    n=manager._allocator.checkpoint_log_count
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


def test_cpu_shape_can_materialize_after_a_missing_checkpoint_gap():
    manager=make_manager(frames=192)
    gpu,_=checkpoint_history(manager,[2])
    history=torch.arange(256,dtype=torch.int32)
    remember(manager,history,192,256)
    req=new_req(manager,torch.cat((history,torch.tensor([999],dtype=torch.int32))))
    assert req.cached_len==64
    plan=manager._recycled_restores[req.table_idx].plan
    assert [(p.begin,p.end) for p in plan.checkpoints]==[(192,256)]
    manager.allocate([req],phase='prefill');manager.prepare_batch(Batch([req],'prefill'))
    req.cached_len=len(req.input_ids);manager.cache_req(req,finished=True)
    assert manager.tree.match_prefix(history).recoverable_len==256
    manager.check_integrity()


def test_direct_checkpoint_survives_granule_sized_prefill_chunks():
    manager=make_manager(frames=192)
    history=torch.arange(128,dtype=torch.int32)
    remember(manager,history,0,128)
    req=new_req(manager,torch.cat((history,torch.tensor([999],dtype=torch.int32))))
    writes=[]
    for end in (32,64,96,128,129):
        req.device_len=end
        manager.allocate([req],phase='prefill')
        batch=Batch([req],'prefill');manager.prepare_batch(batch)
        writes.extend(batch.cache_metadata.checkpoint_writes)
        req.cached_len=end
        manager.cache_req(req,finished=end==129)
    assert len(writes)==1
    assert [(e.begin,e.end,e.kind.value) for e in manager.tree.iter_entries()]==[(0,128,'checkpoint')]
    manager.check_integrity()


def test_batch_branch_is_protected_before_recycled_checkpoint_allocation():
    manager=make_manager(frames=192)
    history=torch.arange(256,dtype=torch.int32)
    fill(manager,new_req(manager,history[:128]))
    remember(manager,history,128,256)
    long=new_req(manager,torch.cat((history,torch.tensor([999],dtype=torch.int32))))
    short=new_req(manager,torch.cat((history[:160],torch.tensor([888],dtype=torch.int32))),table_idx=1)
    manager.allocate([long,short],phase='prefill')
    assert 160 in manager._protected_prefixes[long.uid]
    for req in (long,short):
        restore=manager._recycled_restores[req.table_idx].plan
        assert all(not p.begin<160<p.end for p in restore.checkpoints)
    manager.prepare_batch(Batch([long,short],'prefill'))
    for req in (short,long):
        req.cached_len=req.device_len;manager.cache_req(req,finished=True)
    assert manager.tree.trace(history[:160]).node.is_protected
    assert manager.tree.match_prefix(history[:160]).recoverable_len==160
    manager.check_integrity()


def test_external_pending_recovery_prevents_recycled_source_rebinding():
    manager=make_manager(frames=192)
    history,checkpoints=checkpoint_history(manager,[2,2])
    full=torch.arange(192,dtype=torch.int32)
    remember(manager,full,128,192)
    first=new_req(manager,torch.cat((full,torch.tensor([999],dtype=torch.int32))))
    second=new_req(manager,torch.cat((full,torch.tensor([888],dtype=torch.int32))),table_idx=1)
    manager.allocate([first],phase='prefill')
    assert all(checkpoints[-1].slot not in {entry.slot for entry in cp.gpu_entries}
               for cp in manager._recycled_restores[0].plan.checkpoints)
    manager.prepare_batch(Batch([first],'prefill'))
    assert manager.tree.has_slot(checkpoints[-1].slot)
    manager.allocate([second],phase='prefill');manager.prepare_batch(Batch([second],'prefill'))
    for req in (first,second):
        req.cached_len=req.device_len;manager.cache_req(req,finished=True)
    manager.check_integrity()


@pytest.mark.parametrize('abort',[None,'first_chunk','before_commit'])
def test_recycled_prefill_reuses_full_checkpoint_and_releases_on_abort(abort):
    manager=make_manager(frames=192)
    _,checkpoints=checkpoint_history(manager,[2])
    history=torch.arange(128,dtype=torch.int32)
    fill(manager,new_req(manager,history[:97]))
    remember(manager,history,96,128)
    req=new_req(manager,torch.cat((history,torch.tensor([999],dtype=torch.int32))))
    assert req.cached_len==96
    restore=manager._recycled_restores[req.table_idx]
    assert restore.slots[128]==checkpoints[0].slot
    req.device_len=112
    manager.allocate([req],phase='prefill');manager.prepare_batch(Batch([req],'prefill'))
    assert not manager.tree.has_slot(checkpoints[0].slot)
    req.cached_len=112
    if abort=='first_chunk':
        manager.free_to_cached(req);manager.cache_req(req,finished=True)
    else:
        req.device_len=129
        manager.allocate([req],phase='prefill');manager.prepare_batch(Batch([req],'prefill'))
        req.cached_len=127 if abort else 129
        if abort:manager.free_to_cached(req)
        manager.cache_req(req,finished=True)
    manager.check_integrity()
    if abort is None:
        entries=manager.tree.iter_entries()
        assert [(e.begin,e.end) for e in entries]==[(0,128)]
        assert entries[0].slot==checkpoints[0].slot
