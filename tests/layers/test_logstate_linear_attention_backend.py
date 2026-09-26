import pytest
import torch

from logssms.cache.logstate_pool.frame_pool import FramedMHALogStatePool
from logssms.core import Batch, Context, Request, clear_global_ctx, set_global_ctx
from logssms.layers.gated_delta import GatedDeltaConfig
from logssms.layers.linear_attention_backend import LogStateLinearAttentionBackend
from logssms.kernel.linear_attention.framed_logstate import construct_framed_log_state_chains
from logssms.kernel.linear_attention.framed_logstate import restore_framed_log_state_chains
from logssms.kernel.linear_attention.vendor.fla import (
    fused_recurrent_gated_delta_rule_to_framed_logstate,
    fused_recurrent_gated_delta_rule_with_log,
)


def _make_request(cached_len: int, device_len: int) -> Request:
    return Request(
        input_ids=torch.zeros(device_len, dtype=torch.int32),
        table_idx=0,
        cached_len=cached_len,
        output_len=1,
        uid=0,
        sampling_params=None,  # type: ignore
        cache_handle=None,  # type: ignore
        max_device_len=device_len + 8,
    )


def _make_pool(device: torch.device) -> FramedMHALogStatePool:
    pool = FramedMHALogStatePool(
        num_frames=64,
        page_size=64,
        granularity=16,
        num_attention_layers=1,
        num_kv_heads=1,
        head_dim=64,
        num_linear_layers=1,
        num_key_heads=2,
        num_value_heads=2,
        head_k_dim=32,
        head_v_dim=32,
        conv_dim=192,
        state_len=3,
        device=device,
        dtype=torch.bfloat16,
    )
    allocated = pool.reserve(logs=2, works=2)
    assert allocated is not None
    assert allocated[1] == [0, 1] and allocated[2] == [2, 3]
    return pool


def _make_context(device: torch.device) -> Context:
    ctx = Context(page_size=64)
    ctx.linear_state = _make_pool(device)
    ctx.page_table = torch.arange(128, dtype=torch.int32, device=device).repeat(2, 1)
    ctx.state_table = torch.full(
        (2, 128), -1, dtype=torch.int32, device=device
    )
    ctx.state_table[0, :64].fill_(0)
    ctx.state_table[0, 64:].fill_(1)
    ctx.state_table[1].fill_(1)
    ctx.work_state_table = torch.tensor(
        [[2], [3]], dtype=torch.int32, device=device
    )
    ctx.linear_state.zero_state(2)
    ctx.linear_state.zero_work_conv(2)
    return ctx


def test_logstate_decode_metadata_uses_work_and_log_slots() -> None:
    clear_global_ctx()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = _make_context(device)
    backend = LogStateLinearAttentionBackend()
    req = _make_request(cached_len=20, device_len=21)
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = batch.reqs
    batch.positions = torch.tensor([20], dtype=torch.int32, device=device)
    batch.out_loc = batch.positions

    set_global_ctx(ctx)
    try:
        backend.prepare_metadata(batch)
    finally:
        clear_global_ctx()

    metadata = batch.linear_attn_metadata
    assert metadata.work_slots.tolist() == [2]
    assert metadata.log_slots.tolist() == [0]
    assert metadata.token_offsets.tolist() == [20]
    assert metadata.conv_read_slots.tolist() == [2]
    assert metadata.conv_read_granules.tolist() == [0]
    assert metadata.work_slots.dtype == torch.int32
    assert metadata.log_slots.dtype == torch.int32
    assert metadata.token_offsets.dtype == torch.int32
    assert metadata.conv_read_slots.dtype == torch.int32
    assert metadata.conv_read_granules.dtype == torch.int32


def test_logstate_new_request_reads_conv_from_work_slot() -> None:
    clear_global_ctx()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = _make_context(device)
    backend = LogStateLinearAttentionBackend()
    req = _make_request(cached_len=0, device_len=1)
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = batch.reqs
    batch.positions = torch.tensor([0], dtype=torch.int32, device=device)
    batch.out_loc = batch.positions

    set_global_ctx(ctx)
    try:
        backend.prepare_metadata(batch)
    finally:
        clear_global_ctx()

    metadata = batch.linear_attn_metadata
    assert metadata.conv_read_slots.tolist() == [2]
    assert metadata.conv_read_granules.tolist() == [0]


def test_logstate_uses_physical_offsets_after_unaligned_prefix_hit() -> None:
    clear_global_ctx()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = _make_context(device)
    ctx.page_table[0, 16:32] = 64 + torch.arange(
        16, dtype=torch.int32, device=device
    )
    backend = LogStateLinearAttentionBackend()
    req = _make_request(cached_len=16, device_len=32)
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.positions = torch.arange(16, 32, dtype=torch.int32, device=device)
    batch.out_loc = ctx.page_table[0, 16:32]

    set_global_ctx(ctx)
    try:
        backend.prepare_metadata(batch)
        metadata = batch.linear_attn_metadata
        assert metadata.token_offsets.tolist() == list(range(16))
        assert metadata.conv_snapshot_rows[0].tolist() == [15]
        assert metadata.conv_write_granules.tolist() == [0]

        decode = Batch(reqs=[_make_request(17, 18)], phase="decode")
        decode.padded_reqs = decode.reqs
        decode.positions = torch.tensor([17], dtype=torch.int32, device=device)
        decode.out_loc = ctx.page_table[0, 17:18]
        backend.prepare_metadata(decode)
        decoded = decode.linear_attn_metadata
        assert decoded.token_offsets.tolist() == [1]
        assert decoded.conv_read_granules.tolist() == [0]
    finally:
        clear_global_ctx()


def test_logstate_prefill_metadata_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", "1")
    clear_global_ctx()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = _make_context(device)
    backend = LogStateLinearAttentionBackend()
    req = _make_request(cached_len=16, device_len=32)
    batch = Batch(reqs=[req], phase="prefill")
    batch.positions = torch.arange(16, 32, dtype=torch.int32, device=device)
    batch.out_loc = ctx.page_table[0, 16:32]

    set_global_ctx(ctx)
    try:
        backend.prepare_metadata(batch)
        profile = backend.profile_result()
    finally:
        clear_global_ctx()

    assert profile is not None
    assert profile["aggregate"]["prefill_batches"] == 1
    assert profile["aggregate"]["item_calls"] == 1
    assert profile["aggregate"]["metadata_cpu_ms"] >= 0
    assert profile["aggregate"]["repeat_interleave_cpu_ms"] >= 0
    assert profile["aggregate"]["item_cpu_ms"] >= 0


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_logstate_unmapped_cached_prefix_reads_work_conv(phase: str) -> None:
    clear_global_ctx()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = _make_context(device)
    ctx.state_table[0, :16].fill_(-1)
    backend = LogStateLinearAttentionBackend()
    req = _make_request(cached_len=16, device_len=17)
    batch = Batch(reqs=[req], phase=phase)
    if phase == "decode":
        batch.padded_reqs = batch.reqs
    batch.positions = torch.tensor([16], dtype=torch.int32, device=device)
    batch.out_loc = batch.positions

    set_global_ctx(ctx)
    try:
        backend.prepare_metadata(batch)
    finally:
        clear_global_ctx()

    metadata = batch.linear_attn_metadata
    assert metadata.conv_read_slots.tolist() == [2]
    assert metadata.conv_read_granules.tolist() == [0]


def test_logstate_decode_graph_metadata_uses_int32_capture_buffers() -> None:
    clear_global_ctx()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = _make_context(device)
    backend = LogStateLinearAttentionBackend()
    real_req = _make_request(cached_len=20, device_len=21)
    dummy_req = _make_request(cached_len=0, device_len=1)
    dummy_req.table_idx = 1

    capture_batch = Batch(reqs=[dummy_req, dummy_req], phase="decode")
    capture_batch.padded_reqs = capture_batch.reqs
    capture_batch.positions = torch.zeros(2, dtype=torch.int32, device=device)
    capture_batch.out_loc = capture_batch.positions
    batch = Batch(reqs=[real_req], phase="decode")
    batch.padded_reqs = [real_req, dummy_req]
    batch.positions = torch.tensor([20, 0], dtype=torch.int32, device=device)
    batch.out_loc = batch.positions

    set_global_ctx(ctx)
    try:
        backend.init_capture_graph(max_seq_len=128, bs_list=[2])
        backend.prepare_for_capture(capture_batch)
        backend.prepare_metadata(batch)
        backend.prepare_for_replay(batch)
    finally:
        clear_global_ctx()

    assert backend.capture is not None
    assert backend.capture.work_slots.tolist() == [2, 3]
    assert backend.capture.log_slots.tolist() == [0, 1]
    assert backend.capture.token_offsets.tolist() == [20, 0]
    assert backend.capture.conv_read_slots.tolist() == [2, 3]
    assert backend.capture.conv_read_granules.tolist() == [0, 0]
    assert backend.capture.work_slots.dtype == torch.int32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_decode_direct_logstate_store_matches_temporary_outputs() -> None:
    device = torch.device("cuda")
    pool = FramedMHALogStatePool(
        num_frames=64,
        page_size=64,
        granularity=16,
        num_attention_layers=1,
        num_kv_heads=1,
        head_dim=64,
        num_linear_layers=2,
        num_key_heads=2,
        num_value_heads=2,
        head_k_dim=32,
        head_v_dim=32,
        conv_dim=192,
        state_len=3,
        device=device,
        dtype=torch.bfloat16,
    )
    allocated = pool.reserve(logs=3, works=3)
    assert allocated is not None
    torch.manual_seed(41)
    query = torch.randn(3, 1, 2, 32, device=device, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    gate = torch.randn(3, 1, 2, device=device, dtype=torch.bfloat16)
    beta = torch.randn_like(gate)
    A_log = torch.randn(2, device=device)
    dt_bias = torch.randn(2, device=device)
    work_slots = torch.tensor([3, 4, 5], dtype=torch.int32, device=device)
    log_slots = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    offsets = torch.tensor([0, 17, 63], dtype=torch.int32, device=device)
    initial = torch.randn(3, 2, 32, 32, device=device, dtype=torch.bfloat16)
    pool.write_states(work_slots, 1, initial)

    expected_output, expected_state, expected_key, expected_value, expected_decay = (
        fused_recurrent_gated_delta_rule_with_log(
            query, key, value, gate, beta, A_log, dt_bias, initial
        )
    )
    actual_output = fused_recurrent_gated_delta_rule_to_framed_logstate(
        query, key, value, gate, beta, A_log, dt_bias,
        pool, work_slots, log_slots, offsets, 1,
    )
    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(pool.read_states(work_slots, 1), expected_state[:, 0])
    for row, (slot, offset) in enumerate(zip(log_slots.tolist(), offsets.tolist())):
        for head in range(2):
            log = pool.head_view(slot, 1, head, state=False)
            torch.testing.assert_close(log[offset * 32:(offset + 1) * 32], expected_key[row, 0, head])
            value_offset = 64 * 32 + offset * 32
            torch.testing.assert_close(log[value_offset:value_offset + 32], expected_value[row, 0, head])
        torch.testing.assert_close(pool.decay_view(slot)[1, offset], expected_decay[row, 0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_logstate_prefill_then_decode_produces_lossless_update_log() -> None:
    clear_global_ctx()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    ctx = _make_context(device)
    pool = ctx.linear_state
    backend = LogStateLinearAttentionBackend()
    config = GatedDeltaConfig(
        num_k_heads=2,
        num_v_heads=2,
        head_k_dim=32,
        head_v_dim=32,
        conv_dim=192,
        state_len=3,
        layer_idx=0,
    )
    torch.manual_seed(3)
    conv_weight = (
        torch.randn(192, 1, 4, dtype=dtype, device=device) * 0.02
    )
    A_log = torch.zeros(2, device=device)
    dt_bias = torch.zeros(2, device=device)

    def run(req: Request, phase: str) -> torch.Tensor:
        batch = Batch(reqs=[req], phase=phase)  # type: ignore
        batch.padded_reqs = batch.reqs
        batch.positions = torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            device=device,
        )
        batch.out_loc = batch.positions
        length = req.extend_len
        mixed_qkv = torch.randn(length, 192, dtype=dtype, device=device)
        gate = torch.randn(length, 2, dtype=dtype, device=device)
        beta = torch.randn_like(gate)
        backend.prepare_metadata(batch)
        return backend.forward(
            mixed_qkv,
            gate,
            beta,
            conv_weight,
            A_log,
            dt_bias,
            config,
            batch,
        )

    set_global_ctx(ctx)
    try:
        assert run(_make_request(0, 20), "prefill").shape == (20, 2, 32)
        assert run(_make_request(20, 21), "decode").shape == (1, 2, 32)
    finally:
        clear_global_ctx()

    state = torch.zeros(2, 32, 32, device=device)
    for token in range(21):
        for head in range(2):
            log = pool.head_view(0, 0, head, state=False)
            key = log[token * 32:(token + 1) * 32].float()
            value_begin = 64 * 32 + token * 32
            value = log[value_begin:value_begin + 32].float()
            state[head] = state[head] * pool.decay_view(0)[0, token, head] + key[:, None] * value[None, :]
    torch.testing.assert_close(
        pool.read_states(torch.tensor([2], dtype=torch.int32, device=device), 0)[0].float(),
        state,
        rtol=2e-2,
        atol=8e-2,
    )
    torch.testing.assert_close(
        pool.read_conv(
            torch.tensor([2], dtype=torch.int32, device=device),
            torch.tensor([0], dtype=torch.int32, device=device),
            0,
        ),
        pool.read_conv(
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.tensor([1], dtype=torch.int32, device=device),
            0,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_logstate_multpage_decode_log_restores_work_state() -> None:
    clear_global_ctx()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    pool = FramedMHALogStatePool(
        num_frames=96,
        page_size=64,
        granularity=16,
        num_attention_layers=1,
        num_kv_heads=1,
        head_dim=64,
        num_linear_layers=1,
        num_key_heads=2,
        num_value_heads=2,
        head_k_dim=32,
        head_v_dim=32,
        conv_dim=192,
        state_len=3,
        device=device,
        dtype=dtype,
    )
    allocated = pool.reserve(logs=5, works=2)
    assert allocated is not None
    logs, works = allocated[1:]
    ctx = Context(page_size=64)
    ctx.linear_state = pool
    ctx.page_table = torch.arange(320, dtype=torch.int32, device=device).view(1, 320)
    ctx.state_table = torch.tensor(
        [slot for slot in logs for _ in range(64)],
        dtype=torch.int32, device=device,
    ).view(1, 320)
    ctx.work_state_table = torch.tensor(
        [[works[0]]], dtype=torch.int32, device=device
    )
    backend = LogStateLinearAttentionBackend()
    pool.zero_state(works[0])
    pool.zero_work_conv(works[0])
    config = GatedDeltaConfig(2, 2, 32, 32, 192, 3, 0)
    torch.manual_seed(73)
    mixed_qkv = torch.randn(291, 192, dtype=dtype, device=device)
    gate = torch.randn(291, 2, dtype=dtype, device=device)
    beta = torch.randn_like(gate)
    conv_weight = torch.randn(192, 1, 4, dtype=dtype, device=device) * 0.02
    A_log = torch.zeros(2, device=device)
    dt_bias = torch.zeros(2, device=device)

    def run(begin: int, end: int, phase: str) -> None:
        req = _make_request(begin, end)
        batch = Batch(reqs=[req], phase=phase)  # type: ignore
        batch.padded_reqs = batch.reqs
        batch.positions = torch.arange(begin, end, dtype=torch.int32, device=device)
        batch.out_loc = batch.positions
        backend.prepare_metadata(batch)
        backend.forward(
            mixed_qkv[begin:end], gate[begin:end], beta[begin:end],
            conv_weight, A_log, dt_bias, config, batch,
        )

    set_global_ctx(ctx)
    try:
        run(0, 35, "prefill")
        for token in range(35, 272):
            run(token, token + 1, "decode")
        expected = pool.read_states(
            torch.tensor([works[0]], dtype=torch.int32, device=device), 0
        ).clone()
        restored = torch.tensor([works[1]], dtype=torch.int32, device=device)
        restore_framed_log_state_chains(
            pool,
            [(None, [(slot, 0, 64) for slot in logs[:4]]
              + [(logs[4], 0, 16)], works[1])],
        )
        torch.testing.assert_close(
            pool.read_states(restored, 0), expected, rtol=2e-2, atol=4e-2,
        )
        torch.testing.assert_close(
            pool.read_conv(
                restored.new_tensor([logs[4]]), restored.new_tensor([0]), 0
            ),
            pool.read_conv(
                restored.new_tensor([works[0]]), restored.new_tensor([0]), 0
            ),
        )
    finally:
        clear_global_ctx()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_logstate_model_sized_prefill_log_restores_first_granule() -> None:
    clear_global_ctx()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    config = GatedDeltaConfig(16, 16, 128, 128, 6144, 3, 0)
    pool = FramedMHALogStatePool(
        num_frames=128,
        page_size=64,
        granularity=16,
        num_attention_layers=4,
        num_kv_heads=2,
        head_dim=128,
        num_linear_layers=1,
        num_key_heads=16,
        num_value_heads=16,
        head_k_dim=128,
        head_v_dim=128,
        conv_dim=6144,
        state_len=3,
        device=device,
        dtype=dtype,
    )
    allocated = pool.reserve(logs=2, works=3)
    assert allocated is not None
    log_slot, reference_log_slot = allocated[1]
    work_slot, restored_slot, reference_work_slot = allocated[2]
    ctx = Context(page_size=64)
    ctx.linear_state = pool
    ctx.page_table = torch.arange(64, dtype=torch.int32, device=device).view(1, 64)
    ctx.state_table = torch.full(
        (1, 64), log_slot, dtype=torch.int32, device=device
    )
    ctx.work_state_table = torch.tensor(
        [[work_slot]], dtype=torch.int32, device=device
    )
    pool.zero_state(work_slot)
    pool.zero_work_conv(work_slot)
    torch.manual_seed(127)
    mixed_qkv = torch.randn(35, 6144, device=device, dtype=dtype)
    gate = torch.randn(35, 16, device=device, dtype=dtype)
    beta = torch.randn_like(gate)
    conv_weight = torch.randn(6144, 1, 4, device=device, dtype=dtype) * 0.02
    A_log = torch.zeros(16, device=device)
    dt_bias = torch.zeros(16, device=device)
    req = _make_request(0, 35)
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.positions = torch.arange(35, dtype=torch.int32, device=device)
    batch.out_loc = batch.positions
    backend = LogStateLinearAttentionBackend()
    set_global_ctx(ctx)
    try:
        backend.prepare_metadata(batch)
        full_output = backend.forward(
            mixed_qkv, gate, beta, conv_weight, A_log, dt_bias, config, batch
        )
        full_state = pool.read_states(
            torch.tensor([work_slot], dtype=torch.int32, device=device), 0
        ).clone()
        restore_framed_log_state_chains(
            pool, [(None, [(log_slot, 0, 16)], restored_slot)]
        )
        actual = pool.read_states(
            torch.tensor([restored_slot], dtype=torch.int32, device=device), 0
        )
        ctx.state_table.fill_(reference_log_slot)
        ctx.work_state_table.fill_(reference_work_slot)
        pool.zero_state(reference_work_slot)
        pool.zero_work_conv(reference_work_slot)
        reference_req = _make_request(0, 16)
        reference_batch = Batch(reqs=[reference_req], phase="prefill")
        reference_batch.padded_reqs = reference_batch.reqs
        reference_batch.positions = torch.arange(
            16, dtype=torch.int32, device=device
        )
        reference_batch.out_loc = reference_batch.positions
        backend.prepare_metadata(reference_batch)
        backend.forward(
            mixed_qkv[:16], gate[:16], beta[:16], conv_weight,
            A_log, dt_bias, config, reference_batch,
        )
        expected = pool.read_states(
            torch.tensor([reference_work_slot], dtype=torch.int32, device=device), 0
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=4e-2)
        torch.testing.assert_close(
            pool.read_conv(
                torch.tensor([log_slot], dtype=torch.int32, device=device),
                torch.tensor([0], dtype=torch.int32, device=device), 0,
            ),
            pool.read_conv(
                torch.tensor([reference_work_slot], dtype=torch.int32, device=device),
                torch.tensor([0], dtype=torch.int32, device=device), 0,
            ),
        )
        pool.write_conv(
            torch.tensor([restored_slot], dtype=torch.int32, device=device),
            torch.tensor([0], dtype=torch.int32, device=device), 0,
            pool.read_conv(
                torch.tensor([log_slot], dtype=torch.int32, device=device),
                torch.tensor([0], dtype=torch.int32, device=device), 0,
            ),
        )
        ctx.state_table[:, :16].fill_(-1)
        ctx.work_state_table.fill_(restored_slot)
        suffix_req = _make_request(16, 35)
        suffix_batch = Batch(reqs=[suffix_req], phase="prefill")
        suffix_batch.padded_reqs = suffix_batch.reqs
        suffix_batch.positions = torch.arange(
            16, 35, dtype=torch.int32, device=device
        )
        suffix_batch.out_loc = suffix_batch.positions
        backend.prepare_metadata(suffix_batch)
        suffix_output = backend.forward(
            mixed_qkv[16:], gate[16:], beta[16:], conv_weight,
            A_log, dt_bias, config, suffix_batch,
        )
        torch.testing.assert_close(
            suffix_output, full_output[16:], rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            pool.read_states(
                torch.tensor([restored_slot], dtype=torch.int32, device=device), 0
            ),
            full_state, rtol=2e-2, atol=4e-2,
        )
    finally:
        clear_global_ctx()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("materialized", [False, True])
def test_logstate_prefix_restore_resumes_prefill_at_page_boundary(
    materialized: bool,
) -> None:
    clear_global_ctx()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    config = GatedDeltaConfig(2, 2, 32, 32, 192, 3, 0)
    torch.manual_seed(9)
    mixed_qkv = torch.randn(81, 192, dtype=dtype, device=device)
    gate = torch.randn(81, 2, dtype=dtype, device=device)
    beta = torch.randn_like(gate)
    conv_weight = (
        torch.randn(192, 1, 4, dtype=dtype, device=device) * 0.02
    )
    A_log = torch.zeros(2, device=device)
    dt_bias = torch.zeros(2, device=device)

    def run(
        ctx  : Context,
        begin: int,
        end  : int,
    ) -> torch.Tensor:
        req = _make_request(begin, end)
        batch = Batch(reqs=[req], phase="prefill")
        batch.padded_reqs = batch.reqs
        batch.positions = torch.arange(
            begin, end, dtype=torch.int32, device=device
        )
        batch.out_loc = batch.positions
        backend = ctx.linear_attn_backend
        backend.prepare_metadata(batch)
        return backend.forward(
            mixed_qkv[begin:end],
            gate[begin:end],
            beta[begin:end],
            conv_weight,
            A_log,
            dt_bias,
            config,
            batch,
        )

    full_ctx = _make_context(device)
    full_ctx.linear_attn_backend = LogStateLinearAttentionBackend()
    set_global_ctx(full_ctx)
    try:
        full = run(full_ctx, 0, 81).clone()
    finally:
        clear_global_ctx()

    resumed_ctx = _make_context(device)
    resumed_ctx.linear_attn_backend = LogStateLinearAttentionBackend()
    set_global_ctx(resumed_ctx)
    try:
        first = run(resumed_ctx, 0, 64)
        pool = resumed_ctx.linear_state
        expected_state = pool.read_states(torch.tensor([2], dtype=torch.int32, device=device), 0).clone()
        construct_framed_log_state_chains(
            pool,
            [(None, [(0, 0, 64)], [(
                0 if materialized else 2,
            )])],
        )
        if materialized:
            pool.copy_state(0, 2)
        source = torch.tensor([0], dtype=torch.int32, device=device)
        destination = torch.tensor([2], dtype=torch.int32, device=device)
        pool.write_conv(
            destination, torch.tensor([0], dtype=torch.int32, device=device), 0,
            pool.read_conv(source, torch.tensor([3], dtype=torch.int32, device=device), 0),
        )
        torch.testing.assert_close(
            pool.read_states(destination, 0), expected_state, rtol=2e-2, atol=4e-2
        )
        resumed_ctx.state_table[0, :64].fill_(-1)
        second = run(resumed_ctx, 64, 81)
        resumed = torch.cat([first, second])
    finally:
        clear_global_ctx()

    torch.testing.assert_close(
        full, resumed, rtol=2e-2, atol=2e-2
    )
