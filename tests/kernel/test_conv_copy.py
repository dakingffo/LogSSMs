import pytest
import torch

from logssms.cache.logstate_pool.mha_pool import MHALogStatePool
from logssms.kernel.linear_attention.conv_copy import copy_conv_snapshots


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_copy_conv_snapshots_multiple_layers_and_granules():
    pool = MHALogStatePool(
        num_frames=64,
        page_size=64,
        granularity=16,
        num_attention_layers=1,
        num_kv_heads=1,
        head_dim=256,
        num_linear_layers=3,
        num_key_heads=1,
        num_value_heads=1,
        head_k_dim=128,
        head_v_dim=128,
        conv_dim=20,
        state_len=3,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    allocated = pool.reserve(logs=2, works=2)
    assert allocated is not None
    source_a, source_b = allocated[1]
    output_a, output_b = allocated[2]
    sources = [(source_a, 1), (source_b, 3)]
    outputs = [(output_a, 0), (output_b, 0)]
    expected = []
    for task, (slot, granule) in enumerate(sources):
        value = (torch.arange(3 * 20 * 3, device="cuda") + task * 1000)
        value = value.to(torch.bfloat16).view(3, 20, 3)
        pool.conv_view(slot, granule).copy_(value)
        expected.append(value.clone())
    for slot, granule in outputs:
        pool.conv_view(slot, granule).fill_(-1)

    def vector(values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    copy_conv_snapshots(
        pool,
        vector([slot for slot, _ in sources]),
        vector([granule for _, granule in sources]),
        vector([slot for slot, _ in outputs]),
        vector([granule for _, granule in outputs]),
    )
    for (slot, granule), value in zip(outputs, expected, strict=True):
        torch.testing.assert_close(pool.conv_view(slot, granule), value)
    for (slot, granule), value in zip(sources, expected, strict=True):
        torch.testing.assert_close(pool.conv_view(slot, granule), value)
