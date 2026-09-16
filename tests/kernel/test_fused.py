import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_sigmoid_and_mul(dtype: torch.dtype) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from logssms.kernel import sigmoid_and_mul

    generator = torch.Generator(device="cuda").manual_seed(17)
    input = torch.randn((7, 4, 128), dtype=dtype, device="cuda", generator=generator)
    qkv = torch.randn((7, 4 * 256 + 512), dtype=dtype, device="cuda", generator=generator)
    q_gate = qkv[:, : 4 * 256].view(7, 4, 256)
    _, gate = q_gate.chunk(2, dim=-1)
    assert not gate.is_contiguous()
    assert gate.stride(0) > gate.shape[1] * gate.stride(1)

    expected = input.float() * torch.sigmoid(gate.float())
    actual = sigmoid_and_mul(input, gate)
    torch.testing.assert_close(actual, expected.to(dtype), rtol=2e-2, atol=2e-2)

    inplace = input.clone()
    result = sigmoid_and_mul(inplace, gate, out=inplace)
    assert result.data_ptr() == inplace.data_ptr()
    torch.testing.assert_close(result, expected.to(dtype), rtol=2e-2, atol=2e-2)
