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
from tests.utils import clear_global_ctx


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("granularity", [16, 32, 64])
@pytest.mark.parametrize("mamba", [False, True])
def test_logstate_granule_chunks_match_single_prefill(granularity, mamba):
    clear_global_ctx()
    device, dtype = torch.device("cuda"), torch.bfloat16
    torch.manual_seed(419)
    pool = MHALogStatePool(
        num_frames=128, page_size=64, granularity=granularity,
        num_attention_layers=1, num_kv_heads=1, head_dim=64,
        num_linear_layers=1, num_key_heads=2, num_value_heads=2,
        head_k_dim=32, head_v_dim=32, conv_dim=192, state_len=3,
        device=device, dtype=dtype, recurrent_type="mamba2" if mamba else "gdn",
    )
    reserved = pool.reserve(logs=3, works=1)
    assert reserved is not None
    _, logs, works = reserved
    pool.storage.zero_()
    ctx = Context(64)
    ctx.linear_state = pool
    ctx.state_table = torch.tensor(logs, device=device, dtype=torch.int32).repeat_interleave(64)[None]
    ctx.page_table = torch.arange(192, device=device, dtype=torch.int32)[None]
    ctx.work_state_table = torch.tensor(works, device=device, dtype=torch.int32)[:, None]
    set_global_ctx(ctx)
    backend = Mamba2LogStateBackend() if mamba else GatedDeltaLogStateBackend()
    config = Mamba2Config(2, 2, 32, 32, 3, 0) if mamba else GatedDeltaConfig(2, 2, 32, 32, 192, 3, 0)
    x = torch.randn(129, 192, device=device, dtype=dtype) * .1
    gate = torch.randn(129, 2, device=device, dtype=dtype) * .1
    beta = torch.randn_like(gate)
    weight = torch.randn(192, 1, 4, device=device, dtype=dtype) * .02
    A, bias, D = (torch.zeros(2, device=device) for _ in range(3))
    conv_bias = torch.zeros(192, device=device, dtype=dtype)

    def run(step):
        pool.storage.zero_()
        outputs = []
        for begin in range(0, 129, step):
            end = min(begin + step, 129)
            req = Request(torch.zeros(end, dtype=torch.int32), 0, begin, 1, 0, None, None, 192)
            batch = Batch([req], "prefill")
            batch.positions = torch.arange(begin, end, device=device, dtype=torch.int32)
            batch.out_loc = ctx.page_table[0, begin:end]
            batch.logstate_first_offsets = [begin % 64]
            backend.prepare_metadata(batch)
            if mamba:
                output = backend.forward(x[begin:end], gate[begin:end], weight, conv_bias, A, bias, D, config, batch)
            else:
                output = backend.forward(x[begin:end], gate[begin:end], beta[begin:end], weight, A, bias, config, batch)
            outputs.append(output.clone())
        state = pool.read_states(torch.tensor(works, device=device), 0).clone()
        conv = pool.read_conv(torch.tensor(works, device=device), torch.zeros(1, device=device, dtype=torch.int32), 0).clone()
        return torch.cat(outputs), state, conv

    try:
        expected, actual = run(129), run(granularity)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=3e-2, atol=3e-3)
    finally:
        clear_global_ctx()


@pytest.mark.parametrize("mamba", [False, True])
@pytest.mark.parametrize("logstate", [False, True])
def test_mixed_variable_prefills_preserve_outputs_and_pool(mamba, logstate):
    clear_global_ctx()
    device, dtype = torch.device("cuda"), torch.bfloat16
    torch.manual_seed(311)
    config = Mamba2Config(2, 2, 32, 32, 3, 0) if mamba else GatedDeltaConfig(2, 2, 32, 32, 192, 3, 0)
    ctx = Context(64)
    if logstate:
        pool = MHALogStatePool(
            num_frames=128, page_size=64, granularity=32,
            num_attention_layers=1, num_kv_heads=1, head_dim=64,
            num_linear_layers=1, num_key_heads=2, num_value_heads=2,
            head_k_dim=32, head_v_dim=32, conv_dim=192, state_len=3,
            device=device, dtype=dtype, recurrent_type="mamba2" if mamba else "gdn",
        )
        reserved = pool.reserve(logs=3, works=3)
        assert reserved is not None
        _, logs, works = reserved
        pool.storage.zero_()
        ctx.state_table = torch.tensor(logs, device=device, dtype=torch.int32)[:, None].expand(3, 128).contiguous()
        ctx.work_state_table = torch.tensor(works, device=device, dtype=torch.int32)[:, None]
        buffers = (pool.storage,)
        backend = Mamba2LogStateBackend() if mamba else GatedDeltaLogStateBackend()
    else:
        pool = LinearStatePool(8, 1, 192, 4, 2, 32, 32, device, dtype)
        ctx.state_table = torch.arange(3, device=device, dtype=torch.int32)[:, None].expand(3, 128).contiguous()
        buffers = (pool.conv_state, pool.recurrent_state)
        for tensor in buffers:
            tensor.zero_()
        backend = Mamba2Backend() if mamba else GatedDeltaBackend()
    ctx.linear_state = pool
    ctx.page_table = torch.arange(128, dtype=torch.int32, device=device).repeat(3, 1)
    set_global_ctx(ctx)
    initial = tuple(t.clone() for t in buffers)
    x = torch.randn(128, 192, device=device, dtype=dtype) * .1
    gate = torch.randn(128, 2, device=device, dtype=dtype) * .1
    beta = torch.randn_like(gate)
    weight = torch.randn(192, 1, 4, device=device, dtype=dtype) * .02
    A, bias, D = (torch.zeros(2, device=device) for _ in range(3))
    conv_bias = torch.zeros(192, device=device, dtype=dtype)

    def forward(batch, a, b, c):
        if mamba:
            return backend.forward(a, b, weight, conv_bias, A, bias, D, config, batch)
        return backend.forward(a, b, c, weight, A, bias, config, batch)

    def reset():
        for dst, src in zip(buffers, initial):
            dst.copy_(src)

    try:
        for lengths in ([17, 64], [1, 31], [64, 63]):
            reqs = [Request(torch.zeros(n, dtype=torch.int32), i, 0, 1, i, None, None, n + 8)
                    for i, n in enumerate([*lengths, 1])]
            prefill, decode = Batch(reqs[:2], "prefill"), Batch(reqs[2:], "decode")
            for part in (prefill, decode):
                part.padded_reqs = part.reqs
                part.positions = torch.cat([torch.arange(r.device_len, device=device, dtype=torch.int32)
                                            for r in part.reqs])
                part.out_loc = part.positions
                backend.prepare_metadata(part)
            batch = Batch(reqs, "mixed", num_prefill_reqs=2)
            batch.mixed_parts = prefill, decode
            split = sum(lengths)
            reset()
            expected = torch.cat((
                forward(prefill, x[:split], gate[:split], beta[:split]),
                forward(decode, x[split:split + 1], gate[split:split + 1], beta[split:split + 1]),
            ))
            expected_buffers = tuple(t.clone() for t in buffers)
            reset()
            inputs = {"xbc": x[:split + 1], "dt": gate[:split + 1]} if mamba else {
                "mixed_qkv": x[:split + 1], "gate": gate[:split + 1], "beta": beta[:split + 1],
            }
            kwargs = dict(conv_weight=weight, A_log=A, dt_bias=bias, config=config)
            if mamba:
                kwargs.update(conv_bias=conv_bias, D=D)
            actual = backend.forward_mixed(batch, inputs, **kwargs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for actual, expected in zip(buffers, expected_buffers):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            from logssms.engine.warmup import record_kernel_variants

            reset()
            with record_kernel_variants(strict=True):
                backend.forward_mixed(batch, inputs, **kwargs)
    finally:
        clear_global_ctx()
