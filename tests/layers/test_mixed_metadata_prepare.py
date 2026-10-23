from logssms.cache.logstate.metadata import LogStateCacheMetadata
import pytest
import torch

from logssms.core import Batch, set_global_ctx
from logssms.layers.linear_attention_backend import GatedDeltaLogStateBackend
from tests.layers.test_logstate_linear_attention_backend import _make_context, _make_request
from tests.utils import clear_global_ctx
from logssms.layers.linear_attention_backend.mixed import MixedLinearCopyPlan


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mixed_copy_plan_preserves_strided_guards(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    storage = torch.full((64,), -777, device=device)
    destination = {"values": storage[::2], "phase": "decode"}
    plan = MixedLinearCopyPlan(destination)
    source = torch.arange(96, device=device)[::3]
    plan.copy({"values": source, "phase": "decode"})
    torch.testing.assert_close(storage[::2], source)
    assert (storage[1::2] == -777).all()


@pytest.mark.parametrize("invalid", ["shape", "constant", "structure"])
def test_mixed_copy_plan_validates_before_writing(invalid):
    first, second = torch.full((4,), -777), torch.full((8,), -777)
    plan = MixedLinearCopyPlan({"tensors": [first, second], "phase": "prefill"})
    source = {"tensors": [torch.ones_like(first), torch.ones_like(second)], "phase": "prefill"}
    if invalid == "shape":
        source["tensors"][1] = torch.ones(7, dtype=second.dtype)
    elif invalid == "constant":
        source["phase"] = "decode"
    else:
        source["tensors"] = tuple(source["tensors"])
    with pytest.raises(ValueError):
        plan.copy(source)
    assert (first == -777).all() and (second == -777).all()


@pytest.mark.parametrize("granularity", [16, 32, 64])
@pytest.mark.parametrize("begin,length", [(0, 1), (16, 31), (32, 65), (0, 128)])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_direct_mixed_prefill_maps_match_general_path(granularity, begin, length, checkpoint):
    clear_global_ctx()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = _make_context(device, granularity)
    set_global_ctx(ctx)
    batch = Batch([_make_request(begin, begin + length)], "prefill")
    batch.positions = torch.arange(begin, begin + length, device=device, dtype=torch.int32)
    batch.out_loc = batch.positions
    batch.cache_metadata = LogStateCacheMetadata(first_offsets=[begin % 64])
    if checkpoint:
        batch.cache_metadata.checkpoint_ranges = [(0, begin, begin + length)]
        batch.cache_metadata.checkpoint_writes = [(length - 1, 3)]
    backend = GatedDeltaLogStateBackend()
    try:
        backend.direct_prefill_metadata = False
        backend.prepare_metadata(batch)
        expected = batch.linear_attn_metadata
        backend.direct_prefill_metadata = True
        backend.prepare_metadata(batch)
        actual = batch.linear_attn_metadata
        for field in ("chunk_write_slots", "conv_token_slots", "conv_token_granules", "log_write_slots",
                      "work_slots", "conv_read_slots", "conv_read_granules"):
            torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
        assert actual.conv_snapshot_rows is None
        batch.prefill_token_capacity = 128
        backend.prepare_metadata(batch)
        for field in ("chunk_write_slots", "conv_token_slots", "conv_token_granules", "log_write_slots"):
            torch.testing.assert_close(getattr(actual, field), getattr(batch.linear_attn_metadata, field),
                                       rtol=0, atol=0)
        for field in ("cu_seqlens", "chunk_indices", "chunk_offsets", "endings", "conv_chunks"):
            torch.testing.assert_close(getattr(actual.prefill_indices, field),
                                       getattr(expected.prefill_indices, field), rtol=0, atol=0)
    finally:
        clear_global_ctx()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("granularity", [16, 32, 64])
def test_direct_prefill_variable_requests_and_checkpoint_ranges(device, granularity):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    clear_global_ctx()
    device = torch.device(device)
    ctx = _make_context(device, granularity)
    lengths, starts = [1, 127, 513, 2048], [0, 32, 16, 64]
    max_len = max(a + b for a, b in zip(lengths, starts))
    ctx.page_table = torch.arange(max_len, device=device, dtype=torch.int32).repeat(4, 1)
    ctx.state_table = torch.zeros((4, max_len), device=device, dtype=torch.int32)
    ctx.work_state_table = torch.full((4, 1), 2, device=device, dtype=torch.int32)
    reqs = [_make_request(start, start + length) for start, length in zip(starts, lengths)]
    for i, req in enumerate(reqs):
        req.table_idx = i
    batch = Batch(reqs, "prefill")
    batch.positions = torch.cat([torch.arange(start, start + length, device=device, dtype=torch.int32)
                                 for start, length in zip(starts, lengths)])
    batch.out_loc = batch.positions
    batch.cache_metadata = LogStateCacheMetadata(first_offsets=[start % 64 for start in starts])
    batch.cache_metadata.checkpoint_ranges = [(2, 16, 529), (3, 64, 1088), (3, 1600, 2112)]
    batch.cache_metadata.checkpoint_writes = [(640, 3), (1664, 3), (2688, 3)]
    set_global_ctx(ctx)
    backend = GatedDeltaLogStateBackend()
    try:
        backend.direct_prefill_metadata = False
        backend.prepare_metadata(batch)
        expected = batch.linear_attn_metadata
        backend.direct_prefill_metadata = True
        backend.prepare_metadata(batch)
        actual = batch.linear_attn_metadata
        for field in ("chunk_write_slots", "conv_token_slots", "conv_token_granules", "log_write_slots"):
            torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
    finally:
        clear_global_ctx()
