import pytest
import torch

from logssms.kernel.metadata import write_cache_mapping


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("page_size", [16, 32, 64])
def test_mapping_writes_only_allocated_ranges(device, page_size):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    pages = torch.full((4, 512), -7, dtype=torch.int32, device=device)[:, :256]
    states = torch.full((4, 640), -9, dtype=torch.int32, device=device)[:, :256]
    regions = [(2, 31, 31 + page_size, 1024, 8), (0, 64, 64 + page_size // 2, 2048, -1)]
    expected_pages, expected_states = pages.clone(), states.clone()
    for row, begin, end, page, state in regions:
        expected_pages[row, begin:end] = torch.arange(page, page + end - begin, device=device)
        expected_states[row, begin:end] = state
    write_cache_mapping(regions, pages, states, page_size)
    torch.testing.assert_close(pages, expected_pages)
    torch.testing.assert_close(states, expected_states)
    write_cache_mapping([], pages, states, page_size)
    torch.testing.assert_close(pages, expected_pages)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mapping_packets_remain_independent_across_streams():
    pages = torch.full((2, 128), -1, dtype=torch.int32, device="cuda")
    states = torch.full_like(pages, -1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for index in range(32):
            write_cache_mapping([(index % 2, index, index + 1, 1000 + index, index)], pages, states, 64)
    stream.synchronize()
    for index in range(32):
        assert pages[index % 2, index].item() == 1000 + index
        assert states[index % 2, index].item() == index
