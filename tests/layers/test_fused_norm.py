import pytest
import torch
from flashinfer import gemma_rmsnorm

from logssms.layers import RMSNormFused


@pytest.mark.parametrize("batch_size", [1, 32])
def test_zero_centered_fused_add_rmsnorm(batch_size: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(7)
    norm = RMSNormFused(128, eps=1e-6, zero_centered=True)
    norm.weight = torch.randn(128, device=device, dtype=dtype, generator=generator)
    hidden = torch.randn(batch_size, 128, device=device, dtype=dtype, generator=generator)
    residual = torch.randn(batch_size, 128, device=device, dtype=dtype, generator=generator)

    output, updated_residual = norm.forward(hidden.clone(), residual.clone())
    expected_residual = residual + hidden
    expected_output = gemma_rmsnorm(expected_residual, norm.weight, norm.eps)
    torch.testing.assert_close(updated_residual, expected_residual)
    torch.testing.assert_close(output, expected_output, rtol=2e-2, atol=2e-2)

    first_output, first_residual = norm.forward(hidden.clone())
    torch.testing.assert_close(first_residual, hidden)
    torch.testing.assert_close(first_output, gemma_rmsnorm(hidden, norm.weight, norm.eps))
