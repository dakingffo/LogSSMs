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
    expected = [1, 2, 4, 8, 16, 32, 64] + list(range(96, 513, 32)) + list(range(576, 8193, 64))
    assert prefill_token_buckets(8192) == expected
    assert len(expected) == 141
    assert prefill_token_buckets(0) == []
    assert prefill_token_buckets(1) == [1]
    assert prefill_token_buckets(97)[-2:] == [96, 97]
    assert prefill_token_buckets(600)[-2:] == [576, 600]


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
