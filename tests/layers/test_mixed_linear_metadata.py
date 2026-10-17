"""CPU-only contracts for mixed graph preparation and replay."""
from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from logssms.cache.logstate_pool.mha_pool import MHALogStatePool
from logssms.core import Batch, Context, Request, set_global_ctx
from logssms.layers.linear_attention_backend import (
    GatedDeltaBackend, GatedDeltaLogStateBackend, Mamba2Backend, Mamba2LogStateBackend,
)
from logssms.layers.linear_attention_backend.mixed import (
    copy_mixed_linear_metadata_, prepare_mixed_linear_metadata,
)
from logssms.layers.linear_attention_backend.standard import PackedLinearIndices
from tests.utils import clear_global_ctx


def _make_request(table_idx, cached_len, device_len):
    return Request(
        input_ids=torch.zeros(device_len, dtype=torch.int32), table_idx=table_idx,
        cached_len=cached_len, output_len=1, uid=table_idx,
        sampling_params=None, cache_handle=None, max_device_len=device_len + 8,
    )


def _make_context(device):
    ctx = Context(page_size=64)
    ctx.linear_state = MHALogStatePool(
        num_frames=64, page_size=64, granularity=16,
        num_attention_layers=1, num_kv_heads=1, head_dim=64,
        num_linear_layers=1, num_key_heads=2, num_value_heads=2,
        head_k_dim=32, head_v_dim=32, conv_dim=192, state_len=3,
        device=device, dtype=torch.bfloat16,
    )
    assert ctx.linear_state.reserve(logs=2, works=2) is not None
    ctx.page_table = torch.arange(128, dtype=torch.int32).repeat(2, 1)
    ctx.state_table = torch.zeros((2, 128), dtype=torch.int32)
    ctx.state_table[:, 64:] = 1
    ctx.work_state_table = torch.tensor([[2], [3]], dtype=torch.int32)
    return ctx


def _tensors(tree):
    if isinstance(tree, torch.Tensor):
        yield tree
    elif is_dataclass(tree):
        for field in fields(tree):
            yield from _tensors(getattr(tree, field.name))
    elif isinstance(tree, tuple):
        for value in tree:
            yield from _tensors(value)


@pytest.fixture
def context():
    clear_global_ctx()
    yield
    clear_global_ctx()


def _batches(backend, lengths):
    prefill = Batch(
        reqs=[_make_request(i, 0, n) for i, n in enumerate(lengths)], phase="prefill",
    )
    prefill.positions = torch.cat([torch.arange(n, dtype=torch.int32) for n in lengths])
    prefill.out_loc = prefill.positions
    prefill.logstate_first_offsets = [0] * len(lengths)
    decode = Batch(reqs=[_make_request(1, 5, 6)], phase="decode")
    decode.padded_reqs = decode.reqs
    decode.positions = torch.tensor([5], dtype=torch.int32)
    decode.out_loc = decode.positions
    backend.prepare_metadata(prefill)
    backend.prepare_metadata(decode)
    return prefill, decode


@pytest.mark.parametrize("kind", ["gdn", "logstate", "mamba2", "mamba2_logstate"])
@pytest.mark.parametrize("lengths", [[1], [15], [16], [17], [63], [64], [65], [127], [128], [1, 65], [63, 65]])
def test_fixed_capacity_preserves_live_metadata(context, kind, lengths):
    if "logstate" in kind:
        ctx = _make_context(torch.device("cpu"))
        backend = Mamba2LogStateBackend() if kind == "mamba2_logstate" else GatedDeltaLogStateBackend()
        chunk_size = 16
    else:
        ctx = Context(page_size=64)
        ctx.state_table = torch.arange(256, dtype=torch.int32).view(2, 128)
        ctx.linear_state = SimpleNamespace(device=torch.device("cpu"))
        backend = Mamba2Backend() if kind == "mamba2" else GatedDeltaBackend()
        chunk_size = 64
    set_global_ctx(ctx)
    prefill, decode = _batches(backend, lengths)
    original = prefill.linear_attn_metadata
    before = [t.clone() for t in _tensors(original)]
    pm, dm = prepare_mixed_linear_metadata(backend, prefill, decode, 128)
    assert pm is prefill.linear_attn_metadata and dm is decode.linear_attn_metadata
    assert pm is not original
    for actual, expected in zip(_tensors(original), before):
        torch.testing.assert_close(actual, expected)
    indices = pm.prefill_indices
    assert indices.cu_seqlens.tolist() == [0] + list(torch.tensor(lengths).cumsum(0).tolist())
    expected = PackedLinearIndices.from_lengths(lengths, torch.device("cpu"), chunk_size)
    n = len(expected.chunk_indices)
    torch.testing.assert_close(indices.chunk_indices[:n], expected.chunk_indices)
    torch.testing.assert_close(indices.chunk_offsets, expected.chunk_offsets)
    torch.testing.assert_close(indices.endings, expected.endings)
    assert len(indices.chunk_indices) == len(lengths) + (128 - len(lengths)) // chunk_size
    for pairs, chunk in ((indices.chunk_indices, chunk_size), (indices.conv_chunks, 64)):
        live_rows = []
        for req, index in pairs.tolist():
            begin = sum(lengths[:req])
            live_rows.extend(begin + row for row in range(index * chunk, min((index + 1) * chunk, lengths[req])))
        assert live_rows == list(range(sum(lengths)))  # no duplicate/padded writes
    assert (pm.chunk_write_slots[n:] == -1).all()
    assert len(pm.conv_token_slots) == 128
    assert (pm.conv_token_slots[sum(lengths):] == -1).all()
    if "logstate" in kind:
        assert pm.cu_seqlens is indices.cu_seqlens
        assert (pm.log_write_slots[sum(lengths):] == -1).all()
        if kind == "logstate":
            assert pm.conv_snapshot_rows is None
        else:
            assert pm.conv_snapshot_rows.shape == (128,)
            assert pm.checkpoint_slots.shape == (128,)
            assert pm.log_slots is pm.log_write_slots
            torch.testing.assert_close(pm.conv_final_rows, original.conv_final_rows)
    else:
        assert (pm.checkpoint_slots[len(original.checkpoint_slots):] == -1).all()


@pytest.mark.parametrize("logstate", [False, True, "mamba2"])
def test_replay_changes_lengths_and_checkpoint_structure_without_rebinding(context, logstate):
    ctx = _make_context(torch.device("cpu"))
    set_global_ctx(ctx)
    backend = GatedDeltaLogStateBackend() if logstate else GatedDeltaBackend()
    if logstate == "mamba2":
        backend = Mamba2LogStateBackend()
    prefill, decode = _batches(backend, [128])
    captured = prepare_mixed_linear_metadata(backend, prefill, decode, 128)
    pointers = [t.data_ptr() for t in _tensors(captured)]
    for length in (1, 65, 16, 128):
        prefill, decode = _batches(backend, [length])
        if logstate and length == 65:
            prefill.logstate_checkpoint_ranges = [(0, 0, 64)]
            prefill.logstate_checkpoint_writes = [(63, 2)]
            backend.prepare_metadata(prefill)
        source = prepare_mixed_linear_metadata(backend, prefill, decode, 128)
        copy_mixed_linear_metadata_(captured, source)
        assert [t.data_ptr() for t in _tensors(captured)] == pointers
        for dst, src in zip(_tensors(captured), _tensors(source)):
            torch.testing.assert_close(dst, src)
        assert captured[0].prefill_indices.cu_seqlens.tolist() == [0, length]


@pytest.mark.parametrize("backend_type", [GatedDeltaLogStateBackend, Mamba2LogStateBackend])
def test_helper_never_reads_device_values_on_host(context, monkeypatch, backend_type):
    set_global_ctx(_make_context(torch.device("cpu")))
    backend = backend_type()
    prefill, decode = _batches(backend, [17])

    def forbidden(*args, **kwargs):
        raise AssertionError("host read of tensor")

    for name in ("item", "tolist", "cpu", "numpy", "__bool__"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    source = prepare_mixed_linear_metadata(backend, prefill, decode, 128)
    copy_mixed_linear_metadata_(source, source)


def test_rejects_wrong_bucket_before_copying(context):
    set_global_ctx(_make_context(torch.device("cpu")))
    backend = GatedDeltaBackend()
    prefill, decode = _batches(backend, [17])
    with pytest.raises(ValueError, match="capacity"):
        prepare_mixed_linear_metadata(backend, prefill, decode, 16)
    captured = prepare_mixed_linear_metadata(backend, prefill, decode, 128)
    before = [t.clone() for t in _tensors(captured)]
    prefill, decode = _batches(backend, [65])
    source = prepare_mixed_linear_metadata(backend, prefill, decode, 256)
    with pytest.raises(ValueError, match="shape"):
        copy_mixed_linear_metadata_(captured, source)
    for actual, expected in zip(_tensors(captured), before):
        torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="non-tensor"):
        copy_mixed_linear_metadata_(captured[1], replace(captured[1], phase="prefill"))


def test_mamba2_checkpoint_log_mask_and_snapshot_selection(context):
    set_global_ctx(_make_context(torch.device("cpu")))
    backend = Mamba2LogStateBackend()
    prefill, decode = _batches(backend, [65])
    prefill.logstate_checkpoint_ranges = [(0, 0, 64)]
    prefill.logstate_checkpoint_writes = [(63, 2)]
    backend.prepare_metadata(prefill)
    original = prefill.linear_attn_metadata
    pm, _ = prepare_mixed_linear_metadata(backend, prefill, decode, 128)
    assert pm.log_row_indices is None
    assert pm.log_slots[:64].tolist() == [-1] * 64
    assert pm.log_slots[64].item() == 1
    assert pm.log_slots[65:].tolist() == [-1] * 63
    assert pm.checkpoint_slots.tolist() == [2] + [-1] * 127
    # Every live store still gathers exactly the eager snapshot/state.
    for slots, selection, eager_slots, eager_selection in (
        (pm.checkpoint_slots, pm.checkpoint_snapshot_indices,
         original.checkpoint_slots, original.checkpoint_snapshot_indices),
        (pm.conv_write_slots, pm.conv_write_snapshot_indices,
         original.conv_write_slots, original.conv_write_snapshot_indices),
    ):
        live = slots >= 0
        torch.testing.assert_close(slots[live], eager_slots)
        torch.testing.assert_close(pm.conv_snapshot_rows[selection[live]],
                                   original.conv_snapshot_rows[eager_selection])


def test_standard_decode_slot_dtype_is_stable(context):
    set_global_ctx(_make_context(torch.device("cpu")))
    backend = GatedDeltaBackend()
    prefill, decode = _batches(backend, [17])
    captured = prepare_mixed_linear_metadata(backend, prefill, decode, 128)
    prefill, decode = _batches(backend, [65])
    md = decode.linear_attn_metadata
    slots = md.slots.to(torch.int32)
    decode.linear_attn_metadata = replace(md, slots=slots, read_slots=slots[:, 0], write_slots=slots[:, 1])
    source = prepare_mixed_linear_metadata(backend, prefill, decode, 128)
    assert source[1].slots.dtype == source[1].read_slots.dtype == source[1].write_slots.dtype == torch.int64
    copy_mixed_linear_metadata_(captured, source)


def test_masked_snapshot_stores_preserve_other_pool_bytes():
    pool = _make_context(torch.device("cpu")).linear_state
    pool.storage.fill_(19)
    slots = torch.tensor([-1, 2, -1], dtype=torch.int32)
    states = torch.arange(3 * 2 * 32 * 32, dtype=torch.float32).view(3, 2, 32, 32).to(pool.dtype)
    conv = torch.arange(3 * 192 * 3, dtype=torch.float32).view(3, 192, 3).to(pool.dtype)
    granules = torch.tensor([999, 0, -999], dtype=torch.int32)
    before = pool.storage.clone()
    pool.write_states(slots[1:2], 0, states[1:2])
    pool.write_conv(slots[1:2], granules[1:2], 0, conv[1:2])
    expected = pool.storage.clone()
    pool.storage.copy_(before)
    pool.write_states(slots, 0, states)
    pool.write_conv(slots, granules, 0, conv)
    torch.testing.assert_close(pool.storage, expected)


def test_masked_store_kernels_on_cpu_interpreter(monkeypatch):
    from triton.runtime.interpreter import InterpretedFunction
    from logssms.kernel.linear_attention import masked_store

    monkeypatch.setattr(masked_store, "state_pointer", InterpretedFunction(masked_store.state_pointer.fn))
    storage = torch.full((8, 64), 19.0)
    slots = torch.tensor([-1, 1, -1], dtype=torch.int32)
    values = torch.arange(3 * 2 * 8, dtype=torch.float32).view(3, 2, 8)
    # Each slot occupies one frame; layer 1 starts after 2 heads * 8 floats.
    frames = torch.tensor([[2], [3]], dtype=torch.int32)
    expected = storage.clone()
    expected[3, 16:32] = values[1].flatten()
    InterpretedFunction(masked_store._state_store.fn)[(3, 2, 1)](
        storage, frames, slots, values, 2, 8, 1, 32, 256, 1, 16,
    )
    torch.testing.assert_close(storage, expected)
    granules = torch.tensor([999, 1, -999], dtype=torch.int32)
    frames = torch.tensor([[[0], [1]], [[4], [5]]], dtype=torch.int32)
    conv = torch.arange(3 * 7, dtype=torch.float32).view(3, 7)
    expected[5, 7:14] = conv[1]
    InterpretedFunction(masked_store._conv_store.fn)[(3, 1)](
        storage, frames, slots, granules, conv, 7, 1, 2, 64, 16,
    )
    torch.testing.assert_close(storage, expected)


def test_fused_copy_groups_equal_sizes_deduplicates_aliases_and_handles_views(monkeypatch):
    from logssms.kernel import metadata

    class PretendCUDA(torch.Tensor):
        # Exercise dispatch with CPU storage; no CUDA context or device needed.
        @property
        def is_cuda(self):
            return True

    def tensor(values):
        return torch.tensor(values, dtype=torch.int64).as_subclass(PretendCUDA)

    source = [tensor([1, 2]), tensor([3, 4]), tensor([5, 6, 7])]
    target = [torch.zeros_like(value) for value in source]
    src_matrix = tensor([[8, 9], [10, 11]])
    dst_matrix = torch.zeros_like(src_matrix)
    # The same tensor appears twice, just as cu_seqlens/log_slots aliases do.
    src = dict(arrays=source, alias=source[0], matrix=src_matrix, view=src_matrix[:, 0])
    dst = dict(arrays=target, alias=target[0], matrix=dst_matrix, view=dst_matrix[:, 0])
    launches = []

    def copy(sources, outputs):
        launches.append((len(outputs), outputs[0].numel()))
        assert all(value.is_contiguous() for value in (*sources, *outputs))
        assert all(value.numel() == outputs[0].numel() for value in (*sources, *outputs))
        for a, b in zip(sources, outputs):
            b.copy_(a)

    monkeypatch.setattr(metadata, "copy_metadata", copy)
    copy_mixed_linear_metadata_(dst, src)
    assert sorted(launches) == [(1, 3), (1, 4), (2, 2)]
    for actual, expected in zip(target + [dst_matrix], source + [src_matrix]):
        torch.testing.assert_close(actual, expected)
