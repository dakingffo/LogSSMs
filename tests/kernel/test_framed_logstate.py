import pytest
import torch
import triton
import triton.language as tl

from logssms.cache.logstate_pool.frame_pool import FramedMHALogStatePool
from logssms.kernel.linear_attention.framed_logstate import (
    construct_framed_log_state_chains,
    restore_framed_log_state_chains,
)
from logssms.kernel.linear_attention.vendor.fla.fused_recurrent import (
    _frame_byte_offset,
    fused_recurrent_gated_delta_rule_to_framed_logstate,
    fused_recurrent_gated_delta_rule_with_log,
)


@triton.jit
def _frame_byte_offset_test_kernel(frames, offsets, FRAME_BYTES: tl.constexpr):
    row = tl.program_id(0)
    frame = tl.load(frames + row)
    tl.store(offsets + row, _frame_byte_offset(frame, FRAME_BYTES))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_framed_decode_frame_address_uses_64_bit_arithmetic():
    frame_bytes = 786432  # Qwen3.5-2B LogState frame size
    frames = torch.tensor([0, 2730, 2731, 2988, 3563], dtype=torch.int32, device="cuda")
    offsets = torch.empty(len(frames), dtype=torch.int64, device="cuda")
    _frame_byte_offset_test_kernel[(len(frames),)](frames, offsets, frame_bytes)
    torch.testing.assert_close(offsets, frames.to(torch.int64) * frame_bytes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_framed_reconstruction_matches_serial_recurrence():
    torch.manual_seed(7)
    pool = FramedMHALogStatePool(
        num_frames=64,
        page_size=64,
        granularity=16,
        num_attention_layers=1,
        num_kv_heads=1,
        head_dim=256,
        num_linear_layers=2,
        num_key_heads=2,
        num_value_heads=2,
        head_k_dim=128,
        head_v_dim=128,
        conv_dim=20,
        state_len=3,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    allocated = pool.reserve(logs=2, works=2)
    assert allocated is not None
    logs = allocated[1]
    base, output = allocated[2]
    expected = torch.empty((2, 2, 128, 128), dtype=torch.float32, device="cuda")
    for layer in range(2):
        for head in range(2):
            initial = torch.randn((128, 128), device="cuda") * 0.01
            pool.head_view(base, layer, head, state=True).copy_(initial.to(torch.bfloat16))
            expected[layer, head] = initial.to(torch.bfloat16).float()
    for slot in logs:
        for layer in range(2):
            for head in range(2):
                log = pool.head_view(slot, layer, head, state=False)
                keys = (torch.randn((64, 128), device="cuda") * 0.01).to(torch.bfloat16)
                values = (torch.randn((64, 128), device="cuda") * 0.01).to(torch.bfloat16)
                log[:64 * 128].copy_(keys.flatten())
                log[64 * 128:].copy_(values.flatten())
                decay = torch.empty((64,), device="cuda").uniform_(0.98, 0.999)
                pool.decay_view(slot)[layer, :, head].copy_(decay)
                for token in range(64):
                    expected[layer, head] *= decay[token]
                    expected[layer, head] += (
                        keys[token].float()[:, None]
                        * values[token].float()[None, :]
                    )

    restore_framed_log_state_chains(
        pool, [(base, [(logs[0], 0, 64), (logs[1], 0, 64)], output)]
    )
    for layer in range(2):
        for head in range(2):
            torch.testing.assert_close(
                pool.head_view(output, layer, head, state=True),
                expected[layer, head].to(torch.bfloat16),
                rtol=3e-2, atol=3e-2,
            )

    construct_framed_log_state_chains(
        pool, [(base, [(logs[0], 0, 64), (logs[1], 0, 64)],
                [(logs[0],), (logs[1],)])]
    )
    for layer in range(2):
        for head in range(2):
            torch.testing.assert_close(
                pool.head_view(logs[1], layer, head, state=True),
                expected[layer, head].to(torch.bfloat16),
                rtol=3e-2, atol=3e-2,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_framed_restore_keeps_concurrent_requests_separate():
    torch.manual_seed(31)
    pool = FramedMHALogStatePool(
        num_frames=256,
        page_size=64,
        granularity=16,
        num_attention_layers=1,
        num_kv_heads=2,
        head_dim=256,
        num_linear_layers=2,
        num_key_heads=2,
        num_value_heads=2,
        head_k_dim=128,
        head_v_dim=128,
        conv_dim=20,
        state_len=3,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    allocated = pool.reserve(logs=8, works=8)
    assert allocated is not None
    logs, works = allocated[1:]
    expected = torch.zeros((8, 2, 2, 128, 128), device="cuda")
    for row, slot in enumerate(logs):
        for layer in range(2):
            for head in range(2):
                log = pool.head_view(slot, layer, head, state=False)
                keys = (torch.randn((64, 128), device="cuda") * 0.01).to(torch.bfloat16)
                values = (torch.randn((64, 128), device="cuda") * 0.01).to(torch.bfloat16)
                log[:64 * 128].copy_(keys.flatten())
                log[64 * 128:].copy_(values.flatten())
                decay = torch.empty((64,), device="cuda").uniform_(0.98, 0.999)
                pool.decay_view(slot)[layer, :, head].copy_(decay)
                for token in range(64):
                    expected[row, layer, head] *= decay[token]
                    expected[row, layer, head] += (
                        keys[token].float()[:, None]
                        * values[token].float()[None, :]
                    )

    restore_framed_log_state_chains(
        pool,
        [(None, [(log, 0, 64)], work) for log, work in zip(logs, works)],
    )
    for row, slot in enumerate(works):
        for layer in range(2):
            for head in range(2):
                torch.testing.assert_close(
                    pool.head_view(slot, layer, head, state=True),
                    expected[row, layer, head].to(torch.bfloat16),
                    rtol=3e-2, atol=3e-2,
                )

    construct_framed_log_state_chains(
        pool,
        [(None, [(log, 0, 64)], [(work,)]) for log, work in zip(logs, works)],
    )
    for row, slot in enumerate(works):
        for layer in range(2):
            for head in range(2):
                torch.testing.assert_close(
                    pool.head_view(slot, layer, head, state=True),
                    expected[row, layer, head].to(torch.bfloat16),
                    rtol=3e-2, atol=3e-2,
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_framed_construct_keeps_carry_across_waves_per_chain():
    pool = FramedMHALogStatePool(
        num_frames=256,
        page_size=64,
        granularity=16,
        num_attention_layers=1,
        num_kv_heads=1,
        head_dim=64,
        num_linear_layers=1,
        num_key_heads=1,
        num_value_heads=1,
        head_k_dim=32,
        head_v_dim=32,
        conv_dim=64,
        state_len=3,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    allocated = pool.reserve(logs=11, works=2)
    assert allocated is not None
    logs, works = allocated[1:]
    chains = []
    for row, row_logs in enumerate((logs[:9], logs[9:])):
        for slot in row_logs:
            view = pool.head_view(slot, 0, 0, state=False)
            view[:64 * 32].fill_(0.01 * (row + 1))
            view[64 * 32:].fill_(0.01)
            pool.decay_view(slot)[0, :, 0].fill_(1)
        chains.append((
            None,
            [(slot, 0, 64) for slot in row_logs],
            [()] * (len(row_logs) - 1) + [(works[row],)],
        ))

    construct_framed_log_state_chains(pool, chains)
    for row, count in enumerate((9, 2)):
        expected = 64 * count * 0.01 * (row + 1) * 0.01
        torch.testing.assert_close(
            pool.head_view(works[row], 0, 0, state=True),
            torch.full((32, 32), expected, device="cuda", dtype=torch.bfloat16),
            rtol=5e-2, atol=1e-3,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_framed_decode_matches_recurrent_reference():
    torch.manual_seed(12)
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
        conv_dim=128,
        state_len=3,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    allocated = pool.reserve(logs=2, works=2)
    assert allocated is not None
    logs, works = allocated[1:]
    initial = torch.randn((2, 2, 32, 32), dtype=torch.bfloat16, device="cuda") * 0.01
    for row, slot in enumerate(works):
        for head in range(2):
            pool.head_view(slot, 0, head, state=True).copy_(initial[row, head])
    query = torch.randn((2, 1, 2, 32), dtype=torch.bfloat16, device="cuda")
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    gate = torch.randn((2, 1, 2), dtype=torch.bfloat16, device="cuda")
    beta = torch.randn_like(gate)
    A_log = torch.randn((2,), dtype=torch.float32, device="cuda")
    dt_bias = torch.randn_like(A_log)
    expected, states, prepared_key, residual_value, decay = (
        fused_recurrent_gated_delta_rule_with_log(
            query, key, value, gate, beta, A_log, dt_bias, initial
        )
    )
    actual = fused_recurrent_gated_delta_rule_to_framed_logstate(
        query, key, value, gate, beta, A_log, dt_bias, pool,
        torch.tensor(works, dtype=torch.int32, device="cuda"),
        torch.tensor(logs, dtype=torch.int32, device="cuda"),
        torch.tensor([15, 31], dtype=torch.int32, device="cuda"),
        0,
        torch.zeros(2, dtype=torch.int32, device="cuda"),
    )
    torch.testing.assert_close(actual, expected)
    for row, slot in enumerate(logs):
        for head in range(2):
            torch.testing.assert_close(
                pool.head_view(works[row], 0, head, state=True),
                states[row, 0, head],
            )
            log = pool.head_view(slot, 0, head, state=False)
            position = [15, 31][row]
            torch.testing.assert_close(
                log[position * 32:(position + 1) * 32],
                prepared_key[row, 0, head],
            )
            torch.testing.assert_close(
                log[64 * 32 + position * 32:64 * 32 + (position + 1) * 32],
                residual_value[row, 0, head],
            )
            torch.testing.assert_close(
                pool.decay_view(slot)[0, position, head],
                decay[row, 0, head],
            )


@pytest.mark.parametrize("granularity", [32, 64])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_framed_decode_replays_at_fixed_sixteen_token_intervals(granularity):
    torch.manual_seed(39)
    pool = FramedMHALogStatePool(
        num_frames=64,
        page_size=64,
        granularity=granularity,
        num_attention_layers=1,
        num_kv_heads=1,
        head_dim=128,
        num_linear_layers=1,
        num_key_heads=1,
        num_value_heads=1,
        head_k_dim=128,
        head_v_dim=128,
        conv_dim=128,
        state_len=3,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    allocated = pool.reserve(logs=2, works=2)
    assert allocated is not None
    logs, works = allocated[1:]
    for slot in works:
        pool.zero_state(slot)
    eager_log = torch.tensor([logs[0]], dtype=torch.int32, device="cuda")
    eager_work = torch.tensor([works[0]], dtype=torch.int32, device="cuda")
    replay_log = torch.tensor([logs[1]], dtype=torch.int32, device="cuda")
    replay_work = torch.tensor([works[1]], dtype=torch.int32, device="cuda")
    A_log = torch.zeros(1, device="cuda")
    dt_bias = torch.zeros_like(A_log)
    committed = pool.head_view(works[1], 0, 0, state=True).clone()

    for offset in range(granularity):
        query = torch.randn(1, 1, 1, 128, dtype=pool.dtype, device="cuda")
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        gate = torch.zeros(1, 1, 1, dtype=pool.dtype, device="cuda")
        beta = torch.zeros_like(gate)
        position = torch.tensor([offset], dtype=torch.int32, device="cuda")
        count = torch.tensor([offset % 16], dtype=torch.int32, device="cuda")
        eager = fused_recurrent_gated_delta_rule_to_framed_logstate(
            query, key, value, gate, beta, A_log, dt_bias, pool,
            eager_work, eager_log, position, 0, count,
            enable_eager_state=True,
        )
        replay = fused_recurrent_gated_delta_rule_to_framed_logstate(
            query, key, value, gate, beta, A_log, dt_bias, pool,
            replay_work, replay_log, position, 0, count,
        )
        torch.testing.assert_close(replay, eager, atol=0.04, rtol=0.04)
        if (offset + 1) % 16 == 0:
            torch.testing.assert_close(
                pool.head_view(works[1], 0, 0, state=True),
                pool.head_view(works[0], 0, 0, state=True),
                atol=0.04, rtol=0.04,
            )
            committed = pool.head_view(works[1], 0, 0, state=True).clone()
        else:
            assert torch.equal(
                pool.head_view(works[1], 0, 0, state=True), committed
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_framed_construct_restores_internal_branch_before_inplace_checkpoint():
    torch.manual_seed(37)
    pool = FramedMHALogStatePool(
        num_frames=64,
        page_size=64,
        granularity=16,
        num_attention_layers=1,
        num_kv_heads=2,
        head_dim=128,
        num_linear_layers=1,
        num_key_heads=2,
        num_value_heads=2,
        head_k_dim=128,
        head_v_dim=128,
        conv_dim=20,
        state_len=3,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    allocated = pool.reserve(logs=1, works=1)
    assert allocated is not None
    log_slot, work_slot = allocated[1][0], allocated[2][0]
    expected_at_16 = torch.zeros((2, 128, 128), device="cuda")
    expected_at_64 = torch.zeros_like(expected_at_16)
    for head in range(2):
        log = pool.head_view(log_slot, 0, head, state=False)
        keys = (torch.randn(64, 128, device="cuda") * 0.01).to(torch.bfloat16)
        values = (torch.randn(64, 128, device="cuda") * 0.01).to(torch.bfloat16)
        decays = torch.empty(64, device="cuda").uniform_(0.98, 0.999)
        log[:64 * 128].copy_(keys.flatten())
        log[64 * 128:].copy_(values.flatten())
        pool.decay_view(log_slot)[0, :, head].copy_(decays)
        for token in range(64):
            expected_at_64[head] *= decays[token]
            expected_at_64[head] += (
                keys[token].float()[:, None] * values[token].float()[None, :]
            )
            if token == 15:
                expected_at_16[head].copy_(expected_at_64[head])

    construct_framed_log_state_chains(
        pool,
        [(None, [(log_slot, 0, 16), (log_slot, 16, 64)],
          [(work_slot,), (log_slot,)])],
    )
    for head in range(2):
        torch.testing.assert_close(
            pool.head_view(work_slot, 0, head, state=True),
            expected_at_16[head].to(torch.bfloat16),
            rtol=3e-2, atol=3e-2,
        )
        torch.testing.assert_close(
            pool.head_view(log_slot, 0, head, state=True),
            expected_at_64[head].to(torch.bfloat16),
            rtol=3e-2, atol=3e-2,
        )
