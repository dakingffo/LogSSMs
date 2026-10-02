from types import SimpleNamespace

import pytest
import torch

from logssms.core import Batch, Context, clear_global_ctx, set_global_ctx
from logssms.engine.prefill_graph import PrefillGraphRunner, prefill_token_buckets
from logssms.distributed import DistributedInfo, tp_override
from logssms.layers.piecewise import prefill_piece


@pytest.fixture(autouse=True)
def single_rank():
    with tp_override(DistributedInfo(rank=0, size=1)):
        yield


def test_prefill_buckets_cover_budget_without_dense_capture():
    expected = ([1, 2, 4, 8, 16, 32]
                + list(range(64, 1025, 32))
                + list(range(1088, 4097, 64))
                + list(range(4224, 8193, 128)))
    assert prefill_token_buckets(8192) == expected
    assert len(expected) == 117
    assert prefill_token_buckets(0) == []
    assert prefill_token_buckets(1) == [1]
    assert prefill_token_buckets(97)[-2:] == [96, 97]
    assert prefill_token_buckets(600)[-2:] == [576, 600]


@pytest.mark.parametrize("budget", [1, 16, 31, 32, 33, 1023, 1024, 1025,
                                   4095, 4096, 4097, 8191, 8192, 8193])
def test_prefill_buckets_respect_boundaries_and_exact_budget(budget):
    buckets = prefill_token_buckets(budget)
    assert buckets == sorted(set(buckets))
    assert buckets[0] == 1 and buckets[-1] == budget
    for previous, current in zip(buckets, buckets[1:]):
        step = 32 if previous < 1024 else 64 if previous < 4096 else 128
        assert 0 < current - previous <= step
        if current == budget:
            continue
        assert current <= 32 or current % step == 0


class _Layer:
    def __init__(self, weight, index):
        self.weight, self.index = weight, index

    @prefill_piece
    def prepare(self, x, residual, positions):
        residual = x if residual is None else x + residual
        return residual @ self.weight.T + positions[:, None], residual

    @prefill_piece
    def finish(self, x, residual):
        return torch.tanh(x @ self.weight), residual

    def prefill_graph_pieces(self):
        sample = self.weight.new_empty((1, self.weight.shape[1]))
        positions = torch.empty(1, device=self.weight.device, dtype=torch.int32)
        return [
            (self.prepare, (sample, None if self.index == 0 else sample, positions),
             ("hidden", "residual", "positions"), ("projection", "residual")),
            (self.finish, (sample, sample), ("attention_output", "residual"),
             ("hidden", "residual")),
        ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("max_tokens", [128, 1024])
def test_piece_replay_handles_padding_shared_pool_and_changed_metadata(max_tokens):
    torch.manual_seed(77)
    layers = [_Layer(torch.randn(16, 16, device="cuda") * 0.03, i) for i in range(3)]
    model = SimpleNamespace(model=SimpleNamespace(layers=SimpleNamespace(op_list=layers)))
    ctx = Context(64)
    clear_global_ctx()
    set_global_ctx(ctx)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    runner = PrefillGraphRunner(model, max_tokens, stream)
    ctx.prefill_graph_runner = runner
    try:
        def forward(x, positions, phase, enabled):
            runner.enabled = enabled
            batch = Batch([], phase)
            batch.input_ids = torch.zeros(len(x), device="cuda", dtype=torch.int32)
            with ctx.forward_batch(batch):
                residual = None
                for layer in layers:
                    h, residual = layer.prepare(x, residual, positions)
                    # Dynamic attention stand-in uses only REAL token rows.
                    assert h.shape[0] == len(x)
                    h = h + torch.sin(h)
                    x, residual = layer.finish(h, residual)
                return x.clone(), residual.clone()

        counts = [65, 97, 2, 128, 67, 1]
        if max_tokens > 512:
            counts.extend([513, 577, 1023, 33])
        for tokens in counts:
            x = torch.randn(tokens, 16, device="cuda")
            positions = torch.arange(tokens, device="cuda", dtype=torch.int32) % 7
            expected = forward(x, positions, "prefill", False)
            actual = forward(x, positions, "prefill", True)
            for tensor, reference in zip(actual, expected):
                torch.testing.assert_close(tensor, reference, rtol=1e-4, atol=1e-5)
        replay_count = runner.replays
        forward(x, positions, "decode", True)
        assert runner.replays == replay_count
        batch = Batch([], "prefill")
        batch.input_ids = torch.empty(max_tokens + 1, device="cuda")
        assert not runner.can_use_cuda_graph(batch)
        batch.input_ids = batch.input_ids[:64]
        batch.hidden_captor = object()
        assert not runner.can_use_cuda_graph(batch)
        assert runner.padded_tokens > 0
    finally:
        runner.destroy_cuda_graphs()
        clear_global_ctx()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qwen_full_attention_piece_keeps_fused_kv_writes_outside_graph():
    from logssms.cache.kv_pool.mha_pool import MHAKVCachePool
    from logssms.models.qwen3_next import Qwen3NextDecoderLayer
    from logssms.utils import torch_dtype

    torch.manual_seed(43)
    rotary = SimpleNamespace(rotary_dim=32, max_position=256, base=10000.0, scaling=None)
    config = SimpleNamespace(
        layer_types=["full_attention"], hidden_size=128, head_dim=64,
        num_qo_heads=4, num_kv_heads=2, rotary_config=rotary, rms_norm_eps=1e-6,
        intermediate_size=256, hidden_act="silu", use_moe=lambda _: False,
    )
    with torch.device("cuda"), torch_dtype(torch.bfloat16):
        layer = Qwen3NextDecoderLayer(config, 0)
    for weight in layer.state_dict().values():
        weight.normal_(std=0.05)
    model = SimpleNamespace(model=SimpleNamespace(layers=SimpleNamespace(op_list=[layer])))
    ctx = Context(16)
    ctx.kv_cache = MHAKVCachePool(2, 1, 64, 8, 16, torch.bfloat16, torch.device("cuda"))
    ctx.kv_cache._kv_buffer.fill_(7.0)

    class Backend:
        def forward(self, q, layer_id, batch):
            return q

    ctx.attn_backend = Backend()
    clear_global_ctx()
    set_global_ctx(ctx)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    runner = PrefillGraphRunner(model, 64, stream)
    ctx.prefill_graph_runner = runner
    try:
        # Capturing pure projection/MLP pieces must not mutate persistent cache.
        assert (ctx.kv_cache._kv_buffer == 7.0).all()
        for tokens in (3, 17, 33, 1):
            hidden = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
            batch = Batch([], "prefill")
            batch.input_ids = torch.zeros(tokens, device="cuda", dtype=torch.int32)
            batch.positions = torch.arange(tokens, device="cuda", dtype=torch.int32) + 7
            batch.out_loc = torch.arange(tokens, device="cuda", dtype=torch.int32) + 19
            def forward(enabled):
                runner.enabled = enabled
                ctx.kv_cache._kv_buffer.fill_(7.0)
                with ctx.forward_batch(batch):
                    output = layer.forward(hidden.clone(), None, batch.positions)
                return tuple(x.clone() for x in output), ctx.kv_cache._kv_buffer.clone()
            expected, expected_cache = forward(False)
            actual, actual_cache = forward(True)
            for tensor, reference in zip(actual, expected):
                torch.testing.assert_close(tensor, reference, rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(actual_cache, expected_cache, rtol=2e-2, atol=2e-2)
        assert runner.replays == 8 and runner.padded_tokens > 0
    finally:
        runner.destroy_cuda_graphs()
        clear_global_ctx()
