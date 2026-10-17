"""Small GPU correctness tests; no throughput measurements or model loading."""
import gc

import pytest
import torch

from logssms.cache.linear_pool.state_pool import LinearStatePool
from logssms.cache.logstate_pool.mha_pool import MHALogStatePool
from logssms.core import Batch, Context, set_global_ctx
from logssms.layers.gated_delta import GatedDeltaConfig
from logssms.layers.mamba2 import Mamba2Config
from logssms.layers.linear_attention_backend import (
    GatedDeltaBackend, GatedDeltaLogStateBackend, Mamba2Backend, Mamba2LogStateBackend,
)
from logssms.layers.linear_attention_backend.mixed import (
    copy_mixed_linear_metadata_, prepare_mixed_linear_metadata,
)
from tests.layers.test_logstate_linear_attention_backend import _make_context, _make_request
from tests.utils import clear_global_ctx


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
LENGTHS = [128, 1, 31, 32, 63, 64, 65, 127, 128]


@pytest.mark.parametrize("kind,checkpoints", [
    ("gdn_radix", False), ("gdn_logstate", False), ("gdn_logstate", True),
    ("mamba2_radix", False), ("mamba2_logstate", False), ("mamba2_logstate", True),
    ("mamba2_logstate_falcon05b", False),
])
def test_p128_graph_replay_matches_actual_length_eager(kind, checkpoints):
    clear_global_ctx()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(387)
    device, dtype = torch.device("cuda"), torch.bfloat16
    logstate, mamba = "logstate" in kind, kind.startswith("mamba2")
    backend = {
        "gdn_radix": GatedDeltaBackend, "gdn_logstate": GatedDeltaLogStateBackend,
        "mamba2_radix": Mamba2Backend, "mamba2_logstate": Mamba2LogStateBackend,
        "mamba2_logstate_falcon05b": Mamba2LogStateBackend,
    }[kind]()
    config = Mamba2Config(2, 2, 32, 32, 3, 0) if mamba else GatedDeltaConfig(2, 2, 32, 32, 192, 3, 0)
    falcon = kind == "mamba2_logstate_falcon05b"
    if falcon:
        config = Mamba2Config(24, 1, 64, 128, 3, 0)
        assert config.conv_dim == 1792
    heads = config.num_heads if mamba else config.num_v_heads
    channels = config.conv_dim
    if logstate:
        ctx = _make_context(device)
        if falcon:
            ctx.linear_state = MHALogStatePool(
                num_frames=128, page_size=64, granularity=32,
                num_attention_layers=1, num_kv_heads=2, head_dim=64,
                num_linear_layers=1, num_key_heads=config.num_groups,
                num_value_heads=heads, head_k_dim=config.state_dim,
                head_v_dim=config.head_dim, conv_dim=channels, state_len=3,
                device=device, dtype=dtype, recurrent_type="mamba2",
            )
            reservation = ctx.linear_state.reserve(logs=2, works=2)
            assert reservation is not None and reservation[1:] == ([0, 1], [2, 3])
        buffers = (ctx.linear_state.storage,)
        # Finite canary in every frame, including unallocated/shared log bytes.
        buffers[0].fill_(17)
        work = torch.tensor([2], device=device)
        ctx.linear_state.write_states(work, 0, torch.randn(
            1, heads, config.state_dim if mamba else 32, config.head_dim if mamba else 32,
            device=device, dtype=dtype,
        ) * .01)
        ctx.linear_state.write_conv(work, torch.zeros_like(work), 0,
                                    torch.randn(1, channels, 3, device=device, dtype=dtype) * .01)
    else:
        ctx = Context(page_size=64)
        ctx.linear_state = LinearStatePool(8, 1, 192, 4, 2, 32, 32, device, dtype)
        ctx.state_table = torch.tensor([[0] * 64 + [1] * 64, [6] * 128], dtype=torch.int32, device=device)
        buffers = (ctx.linear_state.conv_state, ctx.linear_state.recurrent_state)
        for tensor in buffers:
            tensor.normal_(0, .01)
    initial = tuple(t.clone() for t in buffers)
    set_global_ctx(ctx)

    def reset():
        for dst, src in zip(buffers, initial):
            dst.copy_(src)

    def batches(length):
        batch = Batch([_make_request(0, length)], "prefill")
        batch.positions = torch.arange(length, dtype=torch.int32, device=device)
        batch.out_loc = batch.positions
        batch.logstate_first_offsets = [0]
        if checkpoints and length >= 64:
            batch.logstate_checkpoint_ranges = [(0, 0, 64)]
            batch.logstate_checkpoint_writes = [(63, 3)]
        req = _make_request(5, 6)
        req.table_idx = 1
        decode = Batch([req], "decode")
        decode.padded_reqs = decode.reqs
        decode.positions = torch.tensor([5], dtype=torch.int32, device=device)
        decode.out_loc = decode.positions
        backend.prepare_metadata(batch)
        backend.prepare_metadata(decode)
        return batch, decode

    x = torch.randn(128, channels, dtype=dtype, device=device) * .1
    gate = torch.randn(128, heads, dtype=dtype, device=device) * .1
    beta = torch.randn_like(gate)
    weight = torch.randn(channels, 1, 4, dtype=dtype, device=device) * .02
    A, bias, D = (torch.randn(heads, device=device) * .1 for _ in range(3))
    conv_bias = torch.randn(channels, device=device, dtype=dtype) * .01
    static_x, static_gate, static_beta = x.clone(), gate.clone(), beta.clone()

    def forward(batch, inputs):
        a, b, c = inputs
        if mamba:
            return backend.forward(a, b, weight, conv_bias, A, bias, D, config, batch)
        return backend.forward(a, b, c, weight, A, bias, config, batch)

    graph = None
    try:
        capture_batch, decode = batches(128)
        captured = prepare_mixed_linear_metadata(backend, capture_batch, decode, 128)
        # Compile/autotune before capture on a side stream, using fixed shapes.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                reset()
                forward(capture_batch, (static_x, static_gate, static_beta))
        torch.cuda.current_stream().wait_stream(stream)
        reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = forward(capture_batch, (static_x, static_gate, static_beta))
        for length in ([31, 65] if falcon else LENGTHS):
            eager_batch, decode = batches(length)
            reset()
            expected = forward(eager_batch, (x[:length], gate[:length], beta[:length])).clone()
            expected_buffers = tuple(t.clone() for t in buffers)
            fresh = prepare_mixed_linear_metadata(backend, eager_batch, decode, 128)
            copy_mixed_linear_metadata_(captured, fresh)
            for dst, src in ((static_x, x), (static_gate, gate), (static_beta, beta)):
                dst.copy_(src)
                dst[length:].fill_(float("nan"))  # padding must never affect live state
            reset()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual[:length], expected, rtol=0, atol=0,
                                       msg=f"{kind}, checkpoints={checkpoints}, L={length}: output")
            for observed, reference, baseline in zip(buffers, expected_buffers, initial):
                # Whole-pool equality covers final states, conv/log snapshots,
                # inactive slots, unused granules and unwritten decay records.
                torch.testing.assert_close(observed, reference, rtol=0, atol=0,
                                           msg=f"{kind}, checkpoints={checkpoints}, L={length}: pool")
                untouched = reference == baseline
                assert torch.equal(observed[untouched], baseline[untouched])
        assert torch.cuda.max_memory_reserved() < 500 * 1024**2
    finally:
        if graph is not None:
            graph.reset()
        clear_global_ctx()
        gc.collect()
