import pytest
import torch

from logssms.utils.torch_utils import make_device_vectors


def test_vectors_share_packed_storage_and_keep_empty_shapes():
    vectors = make_device_vectors([[1, 2], [], [3, 4, 5]], "cpu")
    assert [v.tolist() for v in vectors] == [[1, 2], [], [3, 4, 5]]
    assert all(v.dtype == torch.int32 and v.is_contiguous() for v in vectors)
    assert vectors[0].untyped_storage().data_ptr() == vectors[2].untyped_storage().data_ptr()
    assert make_device_vectors([], "cpu") == ()
    assert make_device_vectors([[2**40]], "cpu", dtype=torch.int64)[0].item() == 2**40


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_async_vectors_own_storage_on_nondefault_stream():
    stream = torch.cuda.Stream()
    copied = []
    with torch.cuda.stream(stream):
        for index in range(64):
            first, second = make_device_vectors([[index] * 128, [index + 1] * 64], "cuda")
            copied.append((first.clone(), second.clone()))
            del first, second
    stream.synchronize()
    for index, (first, second) in enumerate(copied):
        assert torch.all(first == index)
        assert torch.all(second == index + 1)
