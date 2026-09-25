import pytest
import torch
import tvm_ffi

from logssms.tvm import call_cuda_ffi


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_ffi_uses_current_torch_stream() -> None:
    device = tvm_ffi.device("cuda:0")
    previous = tvm_ffi.get_raw_stream(device)
    stream = torch.cuda.Stream(device=0)

    with torch.cuda.stream(stream):
        observed = call_cuda_ffi(tvm_ffi.get_raw_stream, device)

    assert observed == stream.cuda_stream
    assert tvm_ffi.get_raw_stream(device) == previous
