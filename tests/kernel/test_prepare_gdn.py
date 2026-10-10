import pytest
import torch
import torch.nn.functional as F

from logssms.kernel import prepare_gdn


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _reference(query, key, value, gate, beta, A_log, dt_bias):
    repeats = value.shape[2] // query.shape[2]
    def norm(x):
        values = x.float()
        return (values * torch.rsqrt(values.square().sum(-1, keepdim=True) + 1e-6)).to(x.dtype)
    query = norm(query.repeat_interleave(repeats, 2))
    key = norm(key.repeat_interleave(repeats, 2))
    return (query, key, value.contiguous(),
            -A_log.float().exp() * F.softplus(gate.float() + dt_bias),
            beta.float().sigmoid())


def _inputs(dtype, key_dim, value_dim, repeats, tokens=17):
    batch, key_heads = 2, 3
    value_heads = key_heads * repeats
    raw = torch.randn(batch, tokens, 2 * key_heads * key_dim + value_heads * value_dim + 8,
                      device="cuda", dtype=dtype)
    q, k, v, _ = raw.split([key_heads * key_dim, key_heads * key_dim,
                           value_heads * value_dim, 8], -1)
    q, k = [x.view(batch, tokens, key_heads, key_dim) for x in (q, k)]
    v = v.view(batch, tokens, value_heads, value_dim)
    beta, gate = torch.randn(batch, tokens, value_heads * 2, device="cuda", dtype=dtype).chunk(2, -1)
    a = torch.randn(value_heads * 2, device="cuda", dtype=torch.float32)[::2]
    a.mul_(.1)
    bias = torch.randn(value_heads * 2, device="cuda", dtype=dtype)[::2]
    if tokens:
        q[:, 0] = 0
        k[:, -1] = 0
        gate[:, 0] = -40
        gate[:, -1] = 40
        beta[:, 0] = -40
        beta[:, -1] = 40
    return q, k, v, gate, beta, a, bias


def _check(actual, expected):
    dtype = actual[0].dtype
    tolerance = 2e-3 if dtype == torch.bfloat16 else 5e-4 if dtype == torch.float16 else 1e-6
    for got, want in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(got, want, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    for got, want in zip(actual[3:], expected[3:]):
        torch.testing.assert_close(got, want, rtol=2e-6, atol=0)
        assert got.dtype == torch.float32
    assert all(x.is_contiguous() for x in actual)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dims", [(32, 64), (128, 128), (192, 96), (64, 256)])
@pytest.mark.parametrize("repeats", [1, 2])
def test_prepare_gdn_strided_views(dtype, dims, repeats):
    torch.manual_seed(11)
    args = _inputs(dtype, *dims, repeats)
    originals = [x.clone() for x in args]
    _check(prepare_gdn(*args), _reference(*args))
    for actual, original in zip(args, originals):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_prepare_gdn_graph_replay_uses_current_inputs(dtype):
    args = _inputs(dtype, 128, 128, 2)
    prepare_gdn(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = prepare_gdn(*args)
    for _ in range(3):
        for tensor in args:
            tensor.normal_(std=.1)
        graph.replay()
        _check(actual, _reference(*args))


def test_prepare_gdn_empty():
    args = _inputs(torch.bfloat16, 32, 64, 2, tokens=0)
    _check(prepare_gdn(*args), _reference(*args))


def test_prepare_gdn_softplus_threshold_and_negative_tail():
    args = _inputs(torch.float32, 128, 128, 2, tokens=7)
    gate, beta, a, bias = args[3:]
    edges = torch.tensor([-80, -20, 0, 19.999, 20, 20.001, 80], device="cuda")
    gate.copy_(edges[None, :, None])
    beta.copy_(edges[None, :, None])
    a.zero_()
    bias.zero_()
    _check(prepare_gdn(*args), _reference(*args))
