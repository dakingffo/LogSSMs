import pytest
import torch

from logssms.cache.linear_pool.state_pool import LinearStatePool
from logssms.cache.logstate_pool.mha_pool import MHALogStatePool
from logssms.core import Batch, Context, Request, set_global_ctx
from logssms.layers.gated_delta import GatedDeltaConfig
from logssms.layers.mamba2 import Mamba2Config
from logssms.layers.linear_attention_backend import (
    GatedDeltaBackend, GatedDeltaLogStateBackend, Mamba2Backend, Mamba2LogStateBackend,
)
from logssms.layers.linear_attention_backend.mixed import (
    prepare_mixed_linear_metadata, MixedLinearCopyPlan,
)
from logssms.utils.mixed import prefill_buckets
from tests.utils import clear_global_ctx


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("granularity", [0, 16, 32, 64])
@pytest.mark.parametrize("mamba", [False, True])
def test_every_partition_replays_live_lengths_without_changing_other_states(granularity, mamba):
    clear_global_ctx()
    torch.manual_seed(824)
    device, dtype = torch.device("cuda"), torch.bfloat16
    partitions = prefill_buckets(128, granularity or 64)
    rows = max(map(len, partitions)) + 1
    ctx = Context(64)
    if granularity:
        pool = MHALogStatePool(
            num_frames=512, page_size=64, granularity=granularity,
            num_attention_layers=1, num_kv_heads=1, head_dim=64,
            num_linear_layers=1, num_key_heads=2, num_value_heads=2,
            head_k_dim=32, head_v_dim=32, conv_dim=192, state_len=3,
            device=device, dtype=dtype, recurrent_type="mamba2" if mamba else "gdn",
        )
        allocation = pool.reserve(logs=rows * 2, works=rows)
        assert allocation is not None
        _, logs, works = allocation
        ctx.state_table = torch.tensor(logs, device=device, dtype=torch.int32).view(rows,2).repeat_interleave(64,1)
        ctx.work_state_table = torch.tensor(works, device=device, dtype=torch.int32)[:,None]
        buffers = (pool.storage,)
        backend = Mamba2LogStateBackend() if mamba else GatedDeltaLogStateBackend()
    else:
        pool = LinearStatePool(rows * 2, 1, 192, 4, 2, 32, 32, device, dtype)
        ctx.state_table = torch.arange(rows*2, device=device, dtype=torch.int32).view(rows,2).repeat_interleave(64,1)
        buffers = (pool.conv_state, pool.recurrent_state)
        backend = Mamba2Backend() if mamba else GatedDeltaBackend()
    for tensor in buffers:
        tensor.zero_()
    ctx.linear_state = pool
    ctx.page_table = torch.arange(rows*128, device=device, dtype=torch.int32).view(rows,128)
    set_global_ctx(ctx)
    config = Mamba2Config(2,2,32,32,3,0) if mamba else GatedDeltaConfig(2,2,32,32,192,3,0)
    x = torch.randn(129,192,device=device,dtype=dtype) * .1
    gate = torch.randn(129,2,device=device,dtype=dtype) * .1
    beta = torch.randn_like(gate)
    weight = torch.randn(192,1,4,device=device,dtype=dtype) * .02
    A, bias, D = (torch.zeros(2,device=device) for _ in range(3))
    cb = torch.zeros(192,device=device,dtype=dtype)
    kwargs = dict(conv_weight=weight, A_log=A, dt_bias=bias, config=config)
    if mamba:
        kwargs.update(conv_bias=cb,D=D)
    full_inputs = dict(xbc=x,dt=gate) if mamba else dict(mixed_qkv=x,gate=gate,beta=beta)

    def reset():
        for tensor in buffers:
            tensor.zero_()

    def batch(lengths, capacity):
        reqs = [Request(torch.zeros(n,dtype=torch.int32),i,0,1,i,None,None,128)
                for i,n in enumerate([*lengths,1])]
        p,d = Batch(reqs[:-1],"prefill"),Batch(reqs[-1:],"decode")
        for part in (p,d):
            part.padded_reqs = part.reqs
            part.positions = torch.cat([torch.arange(r.device_len,device=device,dtype=torch.int32) for r in part.reqs])
            part.out_loc = torch.cat([ctx.page_table[r.table_idx,:r.device_len] for r in part.reqs])
            part.logstate_first_offsets = [0]*part.size
            backend.prepare_metadata(part)
        result = Batch(reqs,"mixed",num_prefill_reqs=len(lengths))
        result.mixed_parts = p,d
        result.prefill_token_capacity = capacity
        return result

    try:
        for partition in partitions:
            capacity = sum(partition)
            inputs = {k:v[:capacity+1] for k,v in full_inputs.items()}
            captured = batch(partition, capacity)
            captured.mixed_parts[0].prefill_token_capacity = capacity
            backend.prepare_metadata(captured.mixed_parts[0])
            target = prepare_mixed_linear_metadata(backend,*captured.mixed_parts,capacity)
            copy_plan = MixedLinearCopyPlan(target)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                backend.forward_mixed(captured,inputs,**kwargs)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = backend.forward_mixed(captured,inputs,**kwargs)
            for lengths in (partition, [1]*len(partition), [max(1,n-1-i) for i,n in enumerate(partition)]):
                live = batch(lengths, capacity)
                p,d = live.mixed_parts
                total = sum(lengths)
                reset()
                expected = torch.cat((
                    backend.forward(**{k:v[:total] for k,v in inputs.items()},batch=p,**kwargs),
                    backend.forward(**{k:v[capacity:] for k,v in inputs.items()},batch=d,**kwargs),
                )).clone()
                states = tuple(t.clone() for t in buffers)
                p.prefill_token_capacity = capacity
                backend.prepare_metadata(p)
                fresh = prepare_mixed_linear_metadata(backend,p,d,capacity)
                copy_plan.copy(fresh)
                reset()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(torch.cat((output[:total],output[capacity:])),expected,rtol=0,atol=0)
                for actual,want in zip(buffers,states):
                    torch.testing.assert_close(actual,want,rtol=0,atol=0)
            graph.reset()
    finally:
        clear_global_ctx()
