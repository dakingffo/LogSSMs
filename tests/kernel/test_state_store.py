import pytest
import torch

from picosgl.kernel import store_state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_store_state(index_dtype: torch.dtype) -> None:
    pool = torch.zeros(7, 2, 4, 8, dtype=torch.bfloat16, device="cuda")
    indices = torch.tensor([5, 1, 6], dtype=index_dtype, device="cuda")
    states = torch.randn(3, 2, 4, 8, dtype=torch.bfloat16, device="cuda")

    store_state(pool, indices, states)

    torch.testing.assert_close(pool.index_select(0, indices.to(torch.int64)), states)
