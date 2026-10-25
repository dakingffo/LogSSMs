import pytest
import torch
import torch.nn.functional as F

from logssms.kernel.norm import grouped_rms_norm_gated, rms_norm_branches
from logssms.kernel.fused import scaled_silu_and_mul, silu_and_mul


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("before", [False, True])
@pytest.mark.parametrize("groups,width", [(1, 192), (4, 64), (2, 2048)])
@pytest.mark.parametrize("distributed", [False, True])
def test_grouped_norm_gated(before, groups, width, distributed):
    torch.manual_seed(18)
    x = torch.randn(3, groups * width, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(3, groups * width + 16, device="cuda", dtype=x.dtype)[:, :groups * width]
    weight = torch.randn(groups * width, device="cuda", dtype=x.dtype)
    gate_scale = .7
    z = F.silu((gate * gate_scale).float())
    hidden = x.float() if before else x.float() * z
    variance = hidden.reshape(3, groups, width).square().mean(-1)
    remote = torch.rand_like(variance)
    reduce = (lambda local: (local + remote) / 2) if distributed else None
    if distributed:
        variance = (variance + remote) / 2
    expected = (hidden.reshape(3, groups, width) * torch.rsqrt(variance[..., None] + 1e-5)).flatten(1) * weight
    if before:
        expected *= z
    actual = grouped_rms_norm_gated(x, gate, weight, 1e-5, groups, before,
                                   reduce_mean=reduce, gate_scale=gate_scale)
    torch.testing.assert_close(actual, expected.to(x.dtype), atol=2e-2, rtol=1e-2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graphed = grouped_rms_norm_gated(x, gate, weight, 1e-5, groups, before,
                                        reduce_mean=reduce, gate_scale=gate_scale)
    graph.replay()
    torch.testing.assert_close(graphed, actual, atol=0, rtol=0)


@pytest.mark.parametrize("scale", [0.5, 1., 1.7])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_silu_mul(scale, dtype):
    torch.manual_seed(22)
    packed = torch.randn(7, 384, device="cuda", dtype=dtype)
    gate, up = packed.chunk(2, -1)
    expected = F.silu(gate * scale) * up
    actual = scaled_silu_and_mul(packed, scale)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-2)


@pytest.mark.parametrize("other", [False, True])
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_residual_norm_branches(other, residual, dtype):
    from flashinfer import fused_add_rmsnorm, rmsnorm

    torch.manual_seed(21)
    x = torch.randn(5, 768, device="cuda", dtype=dtype)
    y = torch.randn_like(x) if other else None
    r = torch.randn_like(x) if residual else None
    weight = torch.randn(768, device="cuda", dtype=dtype)
    hidden = x * .3 + y * .6 if other else x
    if r is None:
        expected = rmsnorm(hidden, weight, 1e-5)
        expected_residual = hidden
    else:
        expected_residual = r.clone()
        expected = hidden.clone()
        fused_add_rmsnorm(expected, expected_residual, weight, 1e-5)
    outputs, saved = rms_norm_branches(
        x, weight, 1e-5, r, other=y, input_scales=(.3, .6), output_scales=(.7, .8),
    )
    for got, scale in zip(outputs, (.7, .8)):
        torch.testing.assert_close(got, expected * scale, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(saved, expected_residual, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mamba_projection_scales_and_ungrouped_gate(dtype):
    from logssms.distributed import DistributedInfo, tp_override
    from logssms.layers.mamba2 import Mamba2
    from logssms.utils import torch_dtype

    torch.manual_seed(32)
    with tp_override(DistributedInfo(0, 1)), torch.device("cuda"), torch_dtype(dtype):
        layer = Mamba2(hidden_size=64, num_heads=4, num_groups=1, head_dim=32,
                       state_dim=32, conv_kernel_size=4, conv_bias=True, proj_bias=False,
                       output_bias=False, rms_norm=False, rms_norm_eps=1e-5,
                       norm_before_gate=False, multipliers=(.5, .7, .8, .9, .6), layer_idx=0)
        layer.in_proj.weight.normal_(std=.1)
        x = torch.randn(7, 64)
        expected = layer.in_proj.forward(x)
        xbc, gate, dt = layer.proj(x)
        torch.testing.assert_close(gate, expected[:, :128], atol=0, rtol=0)
        torch.testing.assert_close(xbc, expected[:, 128:-4], atol=0, rtol=0)
        torch.testing.assert_close(dt, expected[:, -4:], atol=0, rtol=0)
        assert xbc.untyped_storage().data_ptr() == gate.untyped_storage().data_ptr() == dt.untyped_storage().data_ptr()
        assert layer._config.conv_scales == (.7, .8, .9) and layer._config.dt_scale == .6
        value = torch.randn_like(gate)
        actual = silu_and_mul(value, gate, .5)
        torch.testing.assert_close(actual, value * F.silu(gate * .5), atol=2e-3, rtol=1e-2)
