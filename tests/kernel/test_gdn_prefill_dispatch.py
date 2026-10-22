import importlib

import pytest
import torch

from logssms.kernel.linear_attention.vendor.fla import (
    ChunkStateIO, chunk_gated_delta_rule_to_cache,
)


class LaunchRecorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(**kwargs):
            meta = {"BV": 32, **kwargs}
            self.calls.append((grid(meta) if callable(grid) else grid, kwargs))
        return launch


def test_prefill_warmup_request_counts():
    from logssms.engine.prefill_warmup import prefill_warmup_cases
    from logssms.kernel.linear_attention.vendor.fla.config import prefill_config

    cases = prefill_warmup_cases(32, 8192, 32)
    for count in range(1, 5):
        selected = [lengths for lengths in cases if len(lengths) == count]
        assert selected
        assert all(min(lengths) > 0 and sum(lengths) <= 8192 for lengths in selected)
        for stage in ("output", "kkt", "solve", "wy"):
            variants = {tuple(prefill_config(stage, sum((n + 31) // 32 for n in lengths)).items())
                        for lengths in selected}
            expected = {tuple(prefill_config(stage, n).items()) for n in range(count, 257)}
            assert variants == expected


def test_prefill_warmup_covers_dispatch_configs():
    from logssms.engine.prefill_warmup import prefill_warmup_lengths
    from logssms.kernel.linear_attention.vendor.fla.config import prefill_config

    for budget in (64, 128, 256, 8192):
        lengths = prefill_warmup_lengths(32, budget)
        assert all(0 < n <= budget for n in lengths)
        for stage in ("output", "kkt", "solve", "wy"):
            variants = {tuple(prefill_config(stage, (n + 31) // 32).items()) for n in lengths}
            expected = {tuple(prefill_config(stage, n).items())
                        for n in range(1, (budget + 31) // 32 + 1)}
            assert variants == expected


@pytest.mark.parametrize("chunks", [1, 3, 4, 11, 12, 15, 16, 32])
@pytest.mark.parametrize("stage", ["output", "kkt", "solve", "wy"])
@pytest.mark.parametrize("chunk_size", [16, 32, 64])
@pytest.mark.parametrize("dim", [64, 128])
def test_prefill_chunk_dispatch(monkeypatch, stage, chunks, chunk_size, dim):
    modules = {
        "output": ("chunk_o", "chunk_fwd_kernel_o"),
        "kkt": ("chunk_scaled_dot_kkt", "chunk_scaled_dot_kkt_fwd_kernel"),
        "solve": ("solve_tril", "merge_16x16_to_32x32_inverse_kernel"),
        "wy": ("wy_fast", "recompute_w_u_fwd_kernel"),
    }
    name, kernel_name = modules[stage]
    module = importlib.import_module("logssms.kernel.linear_attention.vendor.fla." + name)
    fixed, fallback = LaunchRecorder(), LaunchRecorder()
    if stage == "solve":
        monkeypatch.setattr(module, kernel_name, fixed)
        monkeypatch.setattr(module, "solve_tril_16x16_kernel", fallback)
        monkeypatch.setattr(module, "merge_16x16_to_64x64_inverse_kernel", fallback)
    else:
        monkeypatch.setattr(module, kernel_name + "_configured", fixed)
        monkeypatch.setattr(module, kernel_name, fallback)
    tokens = chunks
    key = torch.empty(1, tokens, 2, dim, dtype=torch.bfloat16)
    value = torch.empty(1, tokens, 6, dim, dtype=key.dtype)
    gate = torch.empty(1, tokens, 6)
    indices = torch.empty(chunks, 2, dtype=torch.int32)
    cu = torch.arange(chunks + 1, dtype=torch.int32)
    matrix = torch.empty(1, tokens, 6, chunk_size)
    common = dict(cu_seqlens=cu, chunk_indices=indices)
    if stage == "output":
        module.chunk_fwd_o(key, key, value, torch.empty(0), g=gate,
                           chunk_size=chunk_size, **common)
    elif stage == "kkt":
        module.chunk_scaled_dot_kkt_fwd(key, g=gate, beta=gate,
                                       chunk_size=chunk_size, **common)
    elif stage == "solve":
        module.solve_tril(matrix, **common)
    else:
        module.recompute_w_u_fwd(key, value, gate, matrix, g=gate, **common)
    tuned = chunk_size == 32 and (stage == "solve" or dim == 128)
    assert len(fixed.calls) == int(tuned)
    assert len(fallback.calls) == int(not tuned)
    if not tuned:
        return
    grid, config = fixed.calls[0]
    if stage == "output":
        assert (config["BK"], config["BV"], config["num_warps"], config["num_stages"]) == (
            32, 32 if chunks < 4 else 64, 2 if chunks < 4 else 1, 3,
        )
        assert grid == (128 // config["BV"], chunks, 6)
    elif stage == "kkt":
        assert (config["BK"], config["num_warps"], config["num_stages"]) == (
            32, 4 if chunks < 12 else 1, 3,
        )
    elif stage == "solve":
        assert config["DOT_PRECISION"] == "ieee"
        assert (config["num_warps"], config["num_stages"]) == (
            4 if chunks < 16 else 1, 2 if chunks < 16 else 3,
        )
    else:
        assert (config["BK"], config["BV"], config["num_warps"], config["num_stages"]) == (
            64, 64, 4 if chunks < 16 else 1, 3,
        )


@pytest.mark.parametrize("requests", [1, 3, 4, 16])
@pytest.mark.parametrize("chunk_size,dim", [(32, 128), (16, 128), (64, 128), (32, 64)])
def test_prefill_state_dispatch(monkeypatch, requests, chunk_size, dim):
    from logssms.kernel.linear_attention.vendor.fla import chunk_delta_h

    recorder = LaunchRecorder()
    monkeypatch.setattr(chunk_delta_h, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", recorder)
    key = torch.empty(1, requests, 2, dim, dtype=torch.bfloat16)
    value = torch.empty(1, requests, 6, dim, dtype=key.dtype)
    slots = torch.arange(requests, dtype=torch.int32)
    io = ChunkStateIO(torch.empty(requests, 1, 6, dim, dim, dtype=key.dtype), slots, slots)
    chunk_delta_h.chunk_gated_delta_rule_fwd_h(
        key, value, value, torch.empty(1, requests, 6), io,
        torch.arange(requests + 1, dtype=torch.int32), chunk_size=chunk_size,
        chunk_indices=torch.empty(requests, 2, dtype=torch.int32),
        chunk_offsets=torch.arange(requests + 1, dtype=torch.int32),
    )
    grid, config = recorder.calls[0]
    tuned = chunk_size == 32 and dim == 128
    bv = 64 if tuned and requests >= 4 else 32
    assert grid == ((dim + bv - 1) // bv, requests * 6)
    assert (config["BV"], config["num_warps"], config["num_stages"]) == (
        bv, 4 if tuned else 2, 2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("lengths", [[31], [17, 31, 33, 65], [1] * 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_prefill_dispatch_matches_recurrence_and_graph(lengths, dtype):
    from logssms.layers.linear_attention_backend.standard import PackedLinearIndices

    torch.manual_seed(41)
    device = torch.device("cuda")
    indices = PackedLinearIndices.from_lengths(lengths, device, 32)
    n, tokens, heads = len(lengths), sum(lengths), 6
    q = torch.nn.functional.normalize(torch.randn(1, tokens, 2, 128, device=device), dim=-1).to(dtype)
    k = torch.nn.functional.normalize(torch.randn_like(q).float(), dim=-1).to(dtype)
    v = torch.randn(1, tokens, heads, 128, device=device, dtype=dtype) * .1
    g = -torch.rand(1, tokens, heads, device=device) * .1
    beta = torch.rand_like(g)
    nt = len(indices.chunk_indices)
    storage = torch.zeros(2 * n + nt, 1, heads, 128, 128, device=device, dtype=dtype)
    storage[:n].normal_(std=.01)
    reads = torch.arange(n, dtype=torch.int32, device=device)
    writes = torch.arange(2 * n, 2 * n + nt, dtype=torch.int32, device=device)
    io = ChunkStateIO(storage, reads, writes, final_write_slots=reads + n)
    expected, checkpoints = [], []
    start = 0
    for req, length in enumerate(lengths):
        state = storage[req, 0].float()
        for i in range(length):
            t = start + i
            kt = k[0, t].float().repeat_interleave(3, 0)
            qt = q[0, t].float().repeat_interleave(3, 0) * (128 ** -.5)
            state = state * g[0, t].exp()[:, None, None]
            residual = (v[0, t].float() - (state * kt[:, :, None]).sum(1)) * beta[0, t, :, None]
            state = state + kt[:, :, None] * residual[:, None, :]
            expected.append((state * qt[:, :, None]).sum(1))
            if (i + 1) % 32 == 0 or i == length - 1:
                checkpoints.append(state.clone())
        start += length
    expected = torch.stack(expected).unsqueeze(0).to(dtype)
    checkpoints = torch.stack(checkpoints).to(dtype)

    def run():
        return chunk_gated_delta_rule_to_cache(
            q, k, v, g, beta, io, indices.cu_seqlens, chunk_size=32, **indices.fla_kwargs(),
        )[0]

    output = run()
    torch.testing.assert_close(output, expected, rtol=.03, atol=.002)
    torch.testing.assert_close(storage[2 * n:, 0], checkpoints, rtol=.03, atol=.003)
    torch.testing.assert_close(storage[n:2 * n, 0], checkpoints[indices.endings], rtol=.03, atol=.003)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    storage[n:].fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(captured, output, rtol=0, atol=0)
    torch.testing.assert_close(storage[2 * n:, 0], checkpoints, rtol=.03, atol=.003)
