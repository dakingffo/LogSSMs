import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from logssms.cache.linear_pool.state_pool import LinearStatePool
from logssms.cache.logstate_pool.mha_pool import MHALogStatePool
from logssms.kernel.linear_attention.mamba2 import (
    prepare_mamba2, mamba2_prefill, mamba2_decode, mamba2_conv_decode,
)
from logssms.layers.mamba2 import Mamba2Config
from logssms.layers.linear_attention_backend.standard import PackedLinearIndices
from tests.utils import clear_global_ctx


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def reference(xbc, dt, A, bias, D, config, initial):
    x, b, c = xbc.float().split(
        [config.value_dim, config.num_groups * config.state_dim, config.num_groups * config.state_dim], -1,
    )
    b = b.reshape(-1, config.num_groups, config.state_dim).repeat_interleave(config.num_heads // config.num_groups, 1)
    c = c.reshape(-1, config.num_groups, config.state_dim).repeat_interleave(config.num_heads // config.num_groups, 1)
    x = x.reshape(-1, config.num_heads, config.head_dim)
    dt = F.softplus(dt.float() + bias.float())
    state = initial.float()
    outputs, states = [], []
    for t in range(len(xbc)):
        state = state * (-A.float().exp() * dt[t]).exp()[:, None, None]
        update = (x[t] * dt[t, :, None]).to(xbc.dtype).float()
        state = state + b[t, :, :, None] * update[:, None, :]
        outputs.append((state * c[t, :, :, None]).sum(1) + D.float()[:, None] * x[t])
        states.append(state.clone())
    return torch.stack(outputs), torch.stack(states)


@pytest.mark.parametrize("state_dim,groups,chunk", [(32, 1, 16), (128, 2, 32), (256, 1, 64)])
def test_packed_mamba2_prefill_matches_recurrence(state_dim, groups, chunk):
    torch.manual_seed(14)
    config = Mamba2Config(4, groups, 64, state_dim, 3, 0)
    lengths = [45, 71]
    xbc = torch.randn(sum(lengths), config.conv_dim, device="cuda", dtype=torch.bfloat16) * 0.1
    dt = torch.randn(sum(lengths), 4, device="cuda", dtype=torch.bfloat16) - 2
    A, bias, D = (torch.randn(4, device="cuda") * 0.1 for _ in range(3))
    initial = torch.randn(2, 4, state_dim, 64, device="cuda", dtype=xbc.dtype) * 0.1
    indices = PackedLinearIndices.from_lengths(lengths, xbc.device, chunk)
    keys, queries, values, decay = prepare_mamba2(xbc, dt, A, bias, config)
    actual, final, states = mamba2_prefill(xbc, keys, queries, values, decay, D, initial, indices, chunk)
    begin, ch = 0, 0
    for i, length in enumerate(lengths):
        expected, snapshots = reference(xbc[begin:begin+length], dt[begin:begin+length], A, bias, D, config, initial[i])
        torch.testing.assert_close(actual[begin:begin+length].float(), expected, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(final[i], snapshots[-1], atol=1e-5, rtol=1e-4)
        for end in list(range(chunk, length, chunk)) + [length]:
            torch.testing.assert_close(states[ch], snapshots[end-1], atol=1e-5, rtol=1e-4)
            ch += 1
        begin += length


def make_logstate(config):
    pool = MHALogStatePool(
        num_frames=256, page_size=64, granularity=32,
        num_attention_layers=1, num_kv_heads=2, head_dim=128,
        num_linear_layers=1, num_key_heads=config.num_groups,
        num_value_heads=config.num_heads, head_k_dim=config.state_dim,
        head_v_dim=config.head_dim, conv_dim=config.conv_dim, state_len=3,
        device=torch.device("cuda"), dtype=torch.bfloat16, recurrent_type="mamba2",
    )
    reservation = pool.reserve(logs=2, works=4)
    assert reservation is not None
    return pool, reservation


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("chunk_size", [64, 128])
def test_memorized_checkpoint_prefill_matches_log_state(reuse, chunk_size):
    from logssms.core import Batch, Context, Request, SamplingParams, set_global_ctx
    from logssms.layers.linear_attention_backend.mamba2_logstate import Mamba2LogStateBackend

    torch.manual_seed(79)
    config = Mamba2Config(4, 1, 64, 128, 3, 0)
    x = torch.randn(128, config.conv_dim, device="cuda", dtype=torch.bfloat16) * .1
    dt = torch.randn(128, 4, device="cuda", dtype=torch.bfloat16) - 2
    weight = torch.randn(config.conv_dim, 1, 4, device="cuda", dtype=x.dtype) * .02
    A, bias, D = (torch.randn(4, device="cuda") * .1 for _ in range(3))

    def run(direct):
        pool, (kv, logs, works) = make_logstate(config)
        ctx = Context(page_size=64)
        ctx.linear_state = pool
        ctx.page_table = torch.arange(128, device="cuda", dtype=torch.int32)[None]
        ctx.state_table = torch.tensor(logs, device="cuda", dtype=torch.int32).repeat_interleave(64)[None]
        ctx.work_state_table = torch.tensor([[works[0]]], device="cuda", dtype=torch.int32)
        target = logs[0] if reuse else works[1]
        if direct:
            ctx.state_table.fill_(-1)
        pool.zero_state(works[0])
        pool.zero_work_conv(works[0])
        backend = Mamba2LogStateBackend()
        set_global_ctx(ctx)
        try:
            outputs = []
            for begin in range(0, 128, chunk_size):
                end = begin + chunk_size
                req = Request(torch.arange(end, dtype=torch.int32), 0, begin, 1, 0,
                              SamplingParams(), None, 129)
                batch = Batch([req], "prefill")
                batch.positions = ctx.page_table[0, begin:end]
                batch.out_loc = batch.positions
                batch.logstate_first_offsets = [0]
                if direct:
                    batch.logstate_checkpoint_ranges = [(0, 0, 128)]
                    if end == 128:
                        batch.logstate_checkpoint_writes = [(chunk_size - 1, target)]
                backend.prepare_metadata(batch)
                outputs.append(backend.forward(x[begin:end], dt[begin:end], weight,
                                               None, A, bias, D, config, batch))
            state = pool.read_states(torch.tensor([target if direct else works[0]], device="cuda"), 0).clone()
            conv = pool.read_conv(
                torch.tensor([target if direct else logs[1]], device="cuda"),
                torch.tensor([pool.layout.num_granules - 1 if reuse or not direct else 0], device="cuda"), 0,
            ).clone()
            torch.cuda.synchronize()
            return torch.cat(outputs), state, conv
        finally:
            clear_global_ctx()

    expected, state, conv = run(False)
    actual, restored, restored_conv = run(True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(restored, state, atol=0, rtol=0)
    torch.testing.assert_close(restored_conv, conv, atol=0, rtol=0)


@pytest.mark.parametrize("eager", [False, True])
def test_mamba2_logstate_replay_and_graph_match_standard(eager):
    torch.manual_seed(31)
    config = Mamba2Config(4, 1, 64, 128, 3, 0)
    pool, reservation = make_logstate(config)
    log_slots = torch.tensor(reservation[1], device="cuda", dtype=torch.int32)
    work = torch.tensor(reservation[2][:2], device="cuda", dtype=torch.int32)
    bases = torch.tensor(reservation[2][2:], device="cuda", dtype=torch.int32)
    zeros = torch.zeros(2, device="cuda", dtype=torch.int32)
    pos, counts = zeros.clone(), zeros.clone()
    metadata = SimpleNamespace(
        work_slots=work, log_slots=log_slots, token_offsets=pos, replay_counts=counts,
    )
    initial = torch.randn(2, 4, 128, 64, device="cuda", dtype=pool.dtype) * 0.05
    pool.write_states(work, 0, initial)
    pool.write_states(bases, 0, initial)
    xbc = torch.randn(2, config.conv_dim, device="cuda", dtype=pool.dtype) * 0.1
    dt = torch.randn(2, 4, device="cuda", dtype=pool.dtype) - 2
    A, bias, D = (torch.randn(4, device="cuda") * 0.1 for _ in range(3))
    def forward():
        return mamba2_decode(xbc, dt, A, bias, D, config, metadata, pool, logstate=True, eager=eager)
    forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = forward()
    pool.write_states(work, 0, initial)
    expected_state = initial.float()
    for step in range(32):
        xbc.normal_(std=0.1)
        pos.fill_(step)
        counts.fill_(step % 16)
        graph.replay()
        for req in range(2):
            expected, states = reference(xbc[req:req+1], dt[req:req+1], A, bias, D, config, expected_state[req])
            expected_state[req] = states[-1]
            torch.testing.assert_close(actual[req].float(), expected[0], atol=4e-3, rtol=4e-2)
        if eager or (step + 1) % 16 == 0:
            saved = pool.read_states(work, 0)
            torch.testing.assert_close(saved.float(), expected_state, atol=2e-3, rtol=4e-2)
            expected_state = saved.float()
    # Generic affine reconstruction must also work for Mamba2 logs, not just GDN.
    saved = pool.read_states(work, 0).clone()
    pool.kernels.restore(pool, [(base, [(slot, 0, 32)], out)
                               for base, slot, out in zip(reservation[2][2:], reservation[1], reservation[2][:2])])
    torch.testing.assert_close(pool.read_states(work, 0), saved, atol=4e-3, rtol=5e-2)


def test_mamba2_decode_uses_different_read_and_write_slots():
    torch.manual_seed(8)
    config = Mamba2Config(4, 2, 64, 128, 3, 0)
    pool = LinearStatePool(4, 1, config.conv_dim, 4, 4, 128, 64, torch.device("cuda"), torch.bfloat16)
    pool.recurrent_state.normal_(std=0.1)
    original = pool.recurrent_state.clone()
    read = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    write = torch.tensor([2, 3], device="cuda", dtype=torch.int64)
    metadata = SimpleNamespace(read_slots=read, write_slots=write)
    xbc = torch.randn(2, config.conv_dim, device="cuda", dtype=pool.dtype) * 0.1
    dt = torch.randn(2, 4, device="cuda", dtype=pool.dtype) - 2
    A, bias, D = (torch.randn(4, device="cuda") * 0.1 for _ in range(3))
    actual = mamba2_decode(xbc, dt, A, bias, D, config, metadata, pool, logstate=False)
    for req in range(2):
        expected, states = reference(xbc[req:req+1], dt[req:req+1], A, bias, D, config, original[req, 0])
        torch.testing.assert_close(actual[req].float(), expected[0], atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(pool.recurrent_state[req+2, 0].float(), states[-1], atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(pool.recurrent_state[:2], original[:2])


def test_mamba2_conv_decode_bias_and_snapshot():
    config = Mamba2Config(4, 1, 64, 128, 3, 0)
    pool = LinearStatePool(4, 1, config.conv_dim, 4, 4, 128, 64, torch.device("cuda"), torch.bfloat16)
    pool.conv_state.normal_(std=0.1)
    read = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    write = torch.tensor([2, 3], device="cuda", dtype=torch.int64)
    metadata = SimpleNamespace(read_slots=read, write_slots=write)
    x = torch.randn(2, config.conv_dim, device="cuda", dtype=pool.dtype)
    weight = torch.randn(config.conv_dim, 1, 4, device="cuda", dtype=pool.dtype) * 0.1
    bias = torch.randn(config.conv_dim, device="cuda", dtype=pool.dtype) * 0.1
    conv_input = torch.cat((pool.conv_state[:2, 0], x.unsqueeze(-1)), -1)
    expected = F.silu(F.conv1d(conv_input, weight, bias, groups=config.conv_dim)).squeeze(-1)
    actual = mamba2_conv_decode(x, weight, bias, config, metadata, pool, logstate=False)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(pool.conv_state[2:, 0], conv_input[:, :, -3:])


def test_mamba2_asymmetric_logs_materialize_without_losing_key_rows():
    from logssms.kernel.linear_attention.logstate import store_log_states

    torch.manual_seed(52)
    pool = MHALogStatePool(
        num_frames=64, page_size=64, granularity=32,
        num_attention_layers=1, num_kv_heads=1, head_dim=256,
        num_linear_layers=2, num_key_heads=1, num_value_heads=2,
        head_k_dim=256, head_v_dim=64, conv_dim=640, state_len=3,
        device=torch.device("cuda"), dtype=torch.bfloat16, recurrent_type="mamba2",
    )
    _, logs, works = pool.reserve(logs=2, works=1)
    expected = torch.zeros(2, 2, 256, 64, device="cuda")
    slots = torch.tensor(logs, device="cuda", dtype=torch.int32).repeat_interleave(64)
    offsets = torch.arange(128, device="cuda", dtype=torch.int32) % 64
    for layer in range(2):
        key = torch.randn(128, 1, 256, device="cuda", dtype=pool.dtype).repeat_interleave(2, 1) * 0.1
        value = torch.randn(128, 2, 64, device="cuda", dtype=pool.dtype) * 0.1
        decay = torch.empty(128, 2, device="cuda").uniform_(-0.1, -0.01)
        store_log_states(pool, slots, offsets, key, value, decay, layer)
        for t in range(128):
            expected[layer] *= decay[t].exp()[:, None, None]
            expected[layer] += key[t].float()[:, :, None] * value[t].float()[:, None, :]
    pool.kernels.restore(pool, [(None, [(slot, 0, 64) for slot in logs], works[0])])
    pool.coalesce_groups([tuple(logs)], [works[0]])
    actual = torch.stack([pool.read_states(torch.tensor([logs[-1]], device="cuda"), layer)[0]
                          for layer in range(2)])
    torch.testing.assert_close(actual.float(), expected, atol=2e-3, rtol=3e-2)
