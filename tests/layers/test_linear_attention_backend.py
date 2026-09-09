from types import SimpleNamespace

import pytest
import torch

from picosgl.cache.linear_pool.state_pool import LinearStatePool
from picosgl.core import Batch, Context, Request, clear_global_ctx, set_global_ctx
from picosgl.layers.gated_delta import GatedDeltaConfig
from picosgl.layers.linear_attention_backend import LinearAttentionBackend


def _make_inputs(seq_len: int) -> dict:
    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size, num_k_heads, num_v_heads = 2, 2, 4
    head_k_dim = head_v_dim = 64
    return dict(
        query=torch.randn(
            batch_size, seq_len, num_k_heads, head_k_dim,
            device=device, dtype=dtype,
        ),
        key=torch.randn(
            batch_size, seq_len, num_k_heads, head_k_dim,
            device=device, dtype=dtype,
        ),
        value=torch.randn(
            batch_size, seq_len, num_v_heads, head_v_dim,
            device=device, dtype=dtype,
        ),
        gate=torch.randn(
            batch_size, seq_len, num_v_heads,
            device=device, dtype=dtype,
        ),
        beta=torch.randn(
            batch_size, seq_len, num_v_heads,
            device=device, dtype=dtype,
        ),
        A_log=torch.randn(num_v_heads, device=device, dtype=torch.float32),
        dt_bias=torch.randn(num_v_heads, device=device, dtype=torch.float32),
        initial_state=torch.randn(
            batch_size, num_v_heads, head_k_dim, head_v_dim,
            device=device, dtype=dtype,
        ),
    )


def _make_request(table_idx: int, cached_len: int, device_len: int) -> Request:
    return Request(
        input_ids=torch.zeros(device_len, dtype=torch.int32),
        table_idx=table_idx,
        cached_len=cached_len,
        output_len=1,
        uid=table_idx,
        sampling_params=None,  # type: ignore
        cache_handle=None,  # type: ignore
        max_device_len=device_len + 1,
    )


def test_decode_graph_metadata_uses_capture_buffers() -> None:
    clear_global_ctx()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = LinearAttentionBackend()
    ctx = Context(page_size=64)
    ctx.linear_state = SimpleNamespace(device=device)  # type: ignore
    ctx.state_table = torch.tensor(
        [[3] * 64 + [4] * 64, [5] * 64 + [6] * 64, [9] * 128],
        dtype=torch.int32,
        device=device,
    )
    real_req = _make_request(0, cached_len=64, device_len=65)
    dummy_req = _make_request(2, cached_len=0, device_len=1)
    batch = Batch(reqs=[real_req], phase="decode")
    batch.padded_reqs = [real_req, dummy_req]
    capture_batch = Batch(reqs=[dummy_req, dummy_req], phase="decode")
    capture_batch.padded_reqs = capture_batch.reqs

    set_global_ctx(ctx)
    try:
        backend.init_capture_graph(max_seq_len=128, bs_list=[2])
        backend.prepare_for_capture(capture_batch)
        backend.prepare_metadata(batch)
        backend.prepare_for_replay(batch)
    finally:
        clear_global_ctx()

    assert backend.capture is not None
    assert backend.capture.read_slots.tolist() == [3, 9]
    assert backend.capture.write_slots.tolist() == [4, 9]


def test_verify_metadata_uses_live_baseline_and_reserve() -> None:
    clear_global_ctx()
    backend = LinearAttentionBackend()
    ctx = Context(page_size=64)
    ctx.state_table = torch.tensor(
        [
            [11] * 64 + [12] * 64 + [13] * 64,
            [20] * 64 + [21] * 64 + [22] * 64,
        ],
        dtype=torch.int32,
    )
    ctx.work_state_table = torch.tensor(
        [[100, 101, 102], [110, 111, 112]], dtype=torch.int32
    )
    req0 = _make_request(0, cached_len=62, device_len=65)
    req1 = _make_request(1, cached_len=128, device_len=130)
    batch = Batch(reqs=[req0, req1], phase="verify")
    batch.padded_reqs = batch.reqs

    set_global_ctx(ctx)
    try:
        backend.prepare_metadata(batch)
    finally:
        clear_global_ctx()

    groups = batch.linear_attn_metadata.verify_groups
    assert groups is not None
    rows3, reads3, writes3 = groups[3]
    rows2, reads2, writes2 = groups[2]
    assert rows3.tolist() == [0, 1, 2]
    assert reads3.tolist() == [11]
    assert writes3.tolist() == [[100, 101, 102]]
    assert rows2.tolist() == [3, 4]
    assert reads2.tolist() == [21]
    assert writes2.tolist() == [[110, 111]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_chunk_gated_delta_returns_page_checkpoints() -> None:
    from picosgl.kernel.linear_attention.vendor.fla import chunk_gated_delta_rule

    inputs = _make_inputs(seq_len=96)
    query = torch.nn.functional.normalize(inputs["query"], dim=-1)
    key = torch.nn.functional.normalize(inputs["key"], dim=-1)
    value = inputs["value"]
    gate = -inputs["gate"].float().abs()
    beta = inputs["beta"].sigmoid()
    initial_state = inputs["initial_state"]

    query, key, value, gate, beta = [
        tensor.flatten(0, 1).unsqueeze(0)
        for tensor in (query, key, value, gate, beta)
    ]
    cu_seqlens = torch.tensor([0, 96, 192], dtype=torch.int64, device=query.device)
    output, final_state, states = chunk_gated_delta_rule(
        query, key, value, gate, beta, initial_state, cu_seqlens
    )

    assert output.shape == value.shape
    assert final_state.shape == initial_state.shape
    assert states.shape == (4, *initial_state.shape[1:])
    _, first_page_state, _ = chunk_gated_delta_rule(
        query[:, :64],
        key[:, :64],
        value[:, :64],
        gate[:, :64],
        beta[:, :64],
        initial_state[:1],
        torch.tensor([0, 64], dtype=torch.int64, device=query.device),
    )
    torch.testing.assert_close(states[0], first_page_state[0].to(states), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        states[[1, 3]], final_state.to(states), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_packed_prefill_matches_individual_requests() -> None:
    clear_global_ctx()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    config = GatedDeltaConfig(
        num_k_heads=2,
        num_v_heads=2,
        head_k_dim=32,
        head_v_dim=32,
        conv_dim=192,
        state_len=3,
        layer_idx=0,
    )
    lengths = [96, 64]
    torch.manual_seed(11)
    mixed_qkv = torch.randn(sum(lengths), config.conv_dim, device=device, dtype=dtype)
    gate = torch.randn(sum(lengths), config.num_v_heads, device=device, dtype=dtype)
    beta = torch.randn_like(gate)
    conv_weight = torch.randn(
        config.conv_dim, 1, config.state_len + 1, device=device, dtype=dtype
    ) * 0.02
    A_log = torch.zeros(config.num_v_heads, device=device)
    dt_bias = torch.zeros(config.num_v_heads, device=device)
    state_table = torch.tensor(
        [[0] * 64 + [1] * 64, [2] * 64 + [-1] * 64],
        dtype=torch.int32,
        device=device,
    )

    def make_pool() -> LinearStatePool:
        return LinearStatePool(
            num_slots=4,
            num_linear_layers=1,
            conv_dim=config.conv_dim,
            kernel_size=config.state_len + 1,
            num_v_heads=config.num_v_heads,
            head_k_dim=config.head_k_dim,
            head_v_dim=config.head_v_dim,
            device=device,
            dtype=dtype,
        )

    reqs = [_make_request(i, cached_len=0, device_len=length) for i, length in enumerate(lengths)]
    batch = Batch(reqs=reqs, phase="prefill")
    batch.padded_reqs = reqs
    packed_pool = make_pool()
    packed_ctx = Context(page_size=64)
    packed_ctx.linear_state = packed_pool
    packed_ctx.state_table = state_table
    set_global_ctx(packed_ctx)
    try:
        packed_ctx.linear_attn_backend.prepare_metadata(batch)
        packed_output = packed_ctx.linear_attn_backend.forward(
            mixed_qkv, gate, beta, conv_weight, A_log, dt_bias, config, batch
        ).clone()
    finally:
        clear_global_ctx()

    reference_pool = make_pool()
    reference_output = torch.empty_like(packed_output)
    reference_ctx = Context(page_size=64)
    reference_ctx.linear_state = reference_pool
    reference_ctx.state_table = state_table
    set_global_ctx(reference_ctx)
    try:
        offset = 0
        for req, length in zip(reqs, lengths):
            one_batch = Batch(reqs=[req], phase="prefill")
            one_batch.padded_reqs = one_batch.reqs
            reference_ctx.linear_attn_backend.prepare_metadata(one_batch)
            end = offset + length
            reference_output[offset:end] = reference_ctx.linear_attn_backend.forward(
                mixed_qkv[offset:end],
                gate[offset:end],
                beta[offset:end],
                conv_weight,
                A_log,
                dt_bias,
                config,
                one_batch,
            )
            offset = end
    finally:
        clear_global_ctx()

    torch.testing.assert_close(packed_output, reference_output, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(packed_pool.conv_state[:3], reference_pool.conv_state[:3])
    torch.testing.assert_close(
        packed_pool.recurrent_state[:3], reference_pool.recurrent_state[:3],
        rtol=1e-2, atol=5e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_chunked_prefill_resumes_from_stored_checkpoint() -> None:
    clear_global_ctx()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    config = GatedDeltaConfig(
        num_k_heads=2,
        num_v_heads=2,
        head_k_dim=32,
        head_v_dim=32,
        conv_dim=192,
        state_len=3,
        layer_idx=0,
    )
    seq_len = 96
    split = 64
    torch.manual_seed(17)
    mixed_qkv = torch.randn(seq_len, config.conv_dim, device=device, dtype=dtype)
    gate = torch.randn(seq_len, config.num_v_heads, device=device, dtype=dtype)
    beta = torch.randn_like(gate)
    conv_weight = torch.randn(
        config.conv_dim, 1, config.state_len + 1, device=device, dtype=dtype
    ) * 0.02
    A_log = torch.zeros(config.num_v_heads, device=device)
    dt_bias = torch.zeros(config.num_v_heads, device=device)
    state_table = torch.tensor(
        [[0] * 64 + [1] * 64], dtype=torch.int32, device=device
    )

    def make_pool() -> LinearStatePool:
        return LinearStatePool(
            num_slots=2,
            num_linear_layers=1,
            conv_dim=config.conv_dim,
            kernel_size=config.state_len + 1,
            num_v_heads=config.num_v_heads,
            head_k_dim=config.head_k_dim,
            head_v_dim=config.head_v_dim,
            device=device,
            dtype=dtype,
        )

    def run(
        ctx  : Context,
        req  : Request,
        start: int,
        end  : int,
    ) -> torch.Tensor:
        batch = Batch(reqs=[req], phase="prefill")
        batch.padded_reqs = batch.reqs
        ctx.linear_attn_backend.prepare_metadata(batch)
        return ctx.linear_attn_backend.forward(
            mixed_qkv[start:end],
            gate[start:end],
            beta[start:end],
            conv_weight,
            A_log,
            dt_bias,
            config,
            batch,
        )

    full_pool = make_pool()
    full_ctx = Context(page_size=64)
    full_ctx.linear_state = full_pool
    full_ctx.state_table = state_table
    set_global_ctx(full_ctx)
    try:
        full_output = run(full_ctx, _make_request(0, 0, seq_len), 0, seq_len).clone()
    finally:
        clear_global_ctx()

    resumed_pool = make_pool()
    resumed_ctx = Context(page_size=64)
    resumed_ctx.linear_state = resumed_pool
    resumed_ctx.state_table = state_table
    set_global_ctx(resumed_ctx)
    try:
        first_output = run(resumed_ctx, _make_request(0, 0, split), 0, split)
        second_output = run(
            resumed_ctx,
            _make_request(0, split, seq_len),
            split,
            seq_len,
        )
        resumed_output = torch.cat([first_output, second_output]).clone()
    finally:
        clear_global_ctx()

    torch.testing.assert_close(full_output, resumed_output, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(full_pool.conv_state[1], resumed_pool.conv_state[1])
    torch.testing.assert_close(
        full_pool.recurrent_state[1], resumed_pool.recurrent_state[1],
        rtol=1e-2, atol=5e-2,
    )
