from types import SimpleNamespace
import pytest
import torch
from logssms.cache.logstate.frame_pool import MHALogStatePool
from logssms.kernel.linear_attention.logstate import store_log_states, restore_log_state_chains
from logssms.kernel.linear_attention.conv_copy import copy_conv_snapshots
from logssms.kernel.linear_attention.conv_decode import conv_decode
from logssms.kernel.linear_attention.packed_conv import PackedConvStateIO, packed_causal_conv1d

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')


def pool_for(g):
    pool=MHALogStatePool(num_frames=1024,page_size=g,granularity=g,
        num_attention_layers=2,num_kv_heads=1,head_dim=16,
        num_linear_layers=2,num_key_heads=1,num_value_heads=4,
        head_k_dim=128,head_v_dim=128,conv_dim=1536,state_len=3,
        dtype=torch.bfloat16,device=torch.device('cuda'))
    kv=pool.reserve(kv=512)[0];pool.release_kv(kv[::2])
    return pool


@pytest.mark.parametrize('g',[16,32,64])
def test_granule_logs_restore_fragmented_state_heads(g):
    torch.manual_seed(78)
    pool=pool_for(g)
    _,logs,works=pool.reserve(logs=2,works=2)
    source,output=works
    expected=[]
    for layer in range(2):
        state=torch.randn(1,4,128,128,device='cuda')*.01
        pool.write_states(torch.tensor([source],device='cuda'),layer,state)
        state=pool.read_states(torch.tensor([source],device='cuda'),layer).float()
        for slot in logs:
            keys=torch.nn.functional.normalize(torch.randn(g,1,128,device='cuda'),dim=-1).bfloat16()
            values=(torch.randn(g,4,128,device='cuda')*.01).bfloat16()
            decay=-torch.rand(g,4,device='cuda')*.01
            store_log_states(pool,torch.full((g,),slot,device='cuda',dtype=torch.int32),
                             torch.arange(g,device='cuda',dtype=torch.int32),keys,values,decay,layer)
            for pos in range(g):
                state*=decay[pos].exp()[None,:,None,None]
                state+=keys[pos,:, :,None].float().repeat_interleave(4,0)[None]*values[pos,:,None,:].float()[None]
        expected.append(state)
    restore_log_state_chains(pool,[(source,[(slot,0,g) for slot in logs],output)])
    for layer in range(2):
        torch.testing.assert_close(pool.read_states(torch.tensor([output],device='cuda'),layer).float(),
                                   expected[layer],atol=3e-4,rtol=.02)
    pool.allocator.check_integrity()


@pytest.mark.parametrize('g',[16,32,64])
def test_multiframe_conv_direct_decode_prefill_and_copy(g):
    torch.manual_seed(79)
    pool=pool_for(g)
    _,logs,works=pool.reserve(logs=1,works=2)
    source,work=works;output=logs[0]
    slots=lambda x:torch.tensor([x],device='cuda',dtype=torch.int32)
    zeros=slots(0)
    C=pool.layout.conv_dim
    weight=(torch.randn(C,1,4,device='cuda')*.05).bfloat16()
    initial=torch.randn(1,C,3,device='cuda').bfloat16()
    for layer in range(2):
        pool.write_conv(slots(source),zeros,layer,initial)
    copy_conv_snapshots(pool,slots(source),zeros,slots(work),zeros)
    for layer in range(2):
        torch.testing.assert_close(pool.read_conv(slots(work),zeros,layer),initial)
    config=SimpleNamespace(conv_dim=C,state_len=3,layer_idx=1)
    x=torch.randn(1,C,device='cuda').bfloat16()
    actual=conv_decode(x,weight,config,round_conv=True,pool=pool,
        read_slots=slots(source),write_slots=slots(output),work_slots=slots(work),
        read_granules=zeros,write_granules=zeros)
    reference=torch.nn.functional.silu(((torch.cat((initial,x[:,:,None]),2).float()
                  *weight[:,0].float()[None]).sum(-1)).bfloat16().float()).bfloat16()
    torch.testing.assert_close(actual,reference,atol=.002,rtol=.02)
    updated=torch.cat((initial[:,:,1:],x[:,:,None]),2)
    torch.testing.assert_close(pool.read_conv(slots(work),zeros,1),updated)
    torch.testing.assert_close(pool.read_conv(slots(output),zeros,1),updated)
    tokens=torch.randn(g,C,device='cuda').bfloat16()
    writes=torch.full((g,),-1,dtype=torch.int32,device='cuda');writes[-1]=output
    packed_causal_conv1d(tokens,weight,initial,torch.tensor([0,g],device='cuda',dtype=torch.int32),
        torch.tensor([[0,0]],device='cuda',dtype=torch.int32),
        state_io=PackedConvStateIO(pool.storage.view(pool.dtype),writes,torch.zeros_like(writes),slots(work),
                                  pool.conv_frame_table,pool.frame_bytes,1))
    expected=tokens[-3:].T[None]
    torch.testing.assert_close(pool.read_conv(slots(output),zeros,1),expected)
    torch.testing.assert_close(pool.read_conv(slots(work),zeros,1),expected)
    torch.testing.assert_close(pool.read_conv(slots(work),zeros,0),initial)
