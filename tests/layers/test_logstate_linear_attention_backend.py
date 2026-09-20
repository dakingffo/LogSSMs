import pytest
import torch

from logssms.cache.linear_pool import LogStatePool
from logssms.core import Batch, Context, Request, clear_global_ctx, set_global_ctx
from logssms.layers.gated_delta import GatedDeltaConfig
from logssms.layers.linear_attention_backend import LogStateLinearAttentionBackend
from logssms.kernel.linear_attention import construct_log_states


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


def _make_pool(device: torch.device) -> LogStatePool:
    return LogStatePool(
        num_slots=4,
        page_size=64,
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


def _make_context(device: torch.device) -> Context:
    ctx = Context(page_size=64)
    ctx.linear_state = _make_pool(device)
    ctx.state_table = torch.full(
        (2, 128), -1, dtype=torch.int32, device=device
    )
    ctx.state_table[0, :64].fill_(0)
    ctx.state_table[0, 64:].fill_(1)
    ctx.state_table[1].fill_(1)
    ctx.work_state_table = torch.tensor(
        [[2], [3]], dtype=torch.int32, device=device
    )
    ctx.linear_state.recurrent_state[2].zero_()
    ctx.linear_state.conv_state[2].zero_()
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

    set_global_ctx(ctx)
    try:
        backend.prepare_metadata(batch)
    finally:
        clear_global_ctx()

    metadata = batch.linear_attn_metadata
    assert metadata.work_slots.tolist() == [2]
    assert metadata.log_slots.tolist() == [0]
    assert metadata.token_offsets.tolist() == [20]
    assert metadata.conv_read_slots.tolist() == [0]
    assert metadata.conv_read_granules.tolist() == [1]
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
    batch = Batch(reqs=[real_req], phase="decode")
    batch.padded_reqs = [real_req, dummy_req]
    batch.positions = torch.tensor([20, 0], dtype=torch.int32, device=device)

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
    assert backend.capture.conv_read_slots.tolist() == [0, 3]
    assert backend.capture.conv_read_granules.tolist() == [1, 0]
    assert backend.capture.work_slots.dtype == torch.int32


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
        state = (
            state * pool.decay_log[0, 0, token, :, None, None]
            + pool.key_log[0, 0, token, :, :, None].float()
            * pool.residual_value_log[0, 0, token, :, None, :].float()
        )
    torch.testing.assert_close(
        pool.recurrent_state[2, 0].float(),
        state,
        rtol=2e-2,
        atol=8e-2,
    )


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
        expected_state = pool.recurrent_state[2].clone()
        construct_log_states(
            key_log=pool.key_log,
            residual_value_log=pool.residual_value_log,
            decay_log=pool.decay_log,
            recurrent_state=pool.recurrent_state,
            base_slots=torch.tensor([-1], dtype=torch.int32, device=device),
            task_segment_offsets=torch.tensor(
                [0, 1], dtype=torch.int32, device=device
            ),
            log_slots=torch.tensor([0], dtype=torch.int32, device=device),
            log_begins=torch.tensor([0], dtype=torch.int32, device=device),
            log_ends=torch.tensor([64], dtype=torch.int32, device=device),
            output_offsets=torch.tensor([0, 1], dtype=torch.int32, device=device),
            output_slots=torch.tensor(
                [0 if materialized else 2], dtype=torch.int32, device=device
            ),
        )
        if materialized:
            pool.recurrent_state[2].copy_(pool.recurrent_state[0])
        pool.conv_state[2, :, 0].copy_(pool.conv_state[0, :, 3])
        torch.testing.assert_close(
            pool.recurrent_state[2], expected_state, rtol=2e-2, atol=4e-2
        )
        resumed_ctx.state_table[0, :64].fill_(-1)
        second = run(resumed_ctx, 64, 81)
        resumed = torch.cat([first, second])
    finally:
        clear_global_ctx()

    torch.testing.assert_close(
        full, resumed, rtol=2e-2, atol=2e-2
    )
