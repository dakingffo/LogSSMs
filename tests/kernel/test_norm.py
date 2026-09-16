import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(7, 128), (2, 3, 96)])
def test_rms_norm_gated(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from logssms.kernel import rms_norm_gated

    generator = torch.Generator(device="cuda").manual_seed(17)
    hidden = torch.randn(shape, dtype=dtype, device="cuda", generator=generator)
    gate = torch.randn(shape, dtype=dtype, device="cuda", generator=generator)
    weight = torch.randn(shape[-1], dtype=torch.float32, device="cuda", generator=generator)
    eps = 1e-6

    hidden_float = hidden.float()
    expected = hidden_float * torch.rsqrt(
        hidden_float.pow(2).mean(-1, keepdim=True) + eps
    )
    expected = (expected * weight.float() * F.silu(gate.float())).to(dtype)
    actual = rms_norm_gated(hidden, gate, weight, eps)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
