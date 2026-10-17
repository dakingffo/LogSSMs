import pytest
import torch

from tests.kernel.test_gdn_prefill_direct import _pool


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["state", "conv"])
def test_pool_write_matches_indexed_scatter_and_graph_masks(dtype, kind):
    pool = _pool(16, 32, dtype, head_ratio=2)
    reservation = pool.reserve(logs=3, works=3)
    assert reservation is not None
    logs, works = reservation[1:]
    slots = torch.tensor([works[2], -1, works[0]], device=pool.device, dtype=torch.int64)
    granules = torch.tensor([0, -999, 0], device=pool.device, dtype=torch.int64)
    if kind == "state":
        values = torch.randn(3, 4, 32, 32, device=pool.device).transpose(-1, -2)
    else:
        slots.copy_(torch.tensor([logs[2], -1, logs[0]], device=pool.device))
        granules.copy_(torch.tensor([3, -999, 1], device=pool.device))
        values = torch.randn(3, 3, pool.layout.conv_dim, device=pool.device).transpose(-1, -2)
    pool.storage.fill_(19)
    initial = pool.storage.clone()

    def write():
        if kind == "state":
            pool.write_states(slots, 1, values)
        else:
            pool.write_conv(slots, granules, 1, values)

    def reference():
        pool.storage.copy_(initial)
        active = slots >= 0
        if kind == "state":
            indices = pool._state_indices(slots[active], 1)
            source = values[active].to(dtype).flatten(2)
        else:
            indices = pool._conv_indices(slots[active], granules[active], 1)
            source = values[active].to(dtype).flatten(1)
        pool.storage.view(dtype).flatten()[indices] = source
        return pool.storage.clone()

    expected = reference()
    pool.storage.copy_(initial)
    write()
    torch.testing.assert_close(pool.storage, expected, rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        write()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            write()
        for active_slots in ([-1, -1, -1], [works[1], -1, works[2]]):
            slots.copy_(torch.tensor(active_slots, device=pool.device))
            granules.copy_(torch.tensor([0, -999, 0], device=pool.device))
            values.normal_()
            expected = reference()
            pool.storage.copy_(initial)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(pool.storage, expected, rtol=0, atol=0)
    finally:
        graph.reset()
