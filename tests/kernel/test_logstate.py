import pytest
import torch

from logssms.cache.linear_pool.logstate_pool import LogStatePool
from logssms.kernel.linear_attention import (
    construct_log_states,
    store_log_states,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_construct_log_states_emits_work_state_and_checkpoints_once() -> None:
    torch.manual_seed(7)
    pool = LogStatePool(
        num_slots=5,
        page_size=64,
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
    pool.key_log[:2].normal_(std=0.01)
    pool.residual_value_log[:2].normal_(std=0.01)
    pool.decay_log[:2].uniform_(0.98, 0.999)
    pool.recurrent_state[2].normal_(std=0.01)
    key = pool.key_log[:2].float().clone()
    value = pool.residual_value_log[:2].float().clone()
    decay = pool.decay_log[:2].clone()
    checkpoint = pool.recurrent_state[2].float().clone()
    expected = []
    for slot in range(2):
        for token in range(64):
            checkpoint.mul_(decay[slot, :, token, :, None, None]).add_(
                key[slot, :, token, :, :, None]
                * value[slot, :, token, :, None, :]
            )
        expected.append(checkpoint.clone())

    device = pool.device
    construct_log_states(
        key_log=pool.key_log,
        residual_value_log=pool.residual_value_log,
        decay_log=pool.decay_log,
        recurrent_state=pool.recurrent_state,
        base_slots=torch.tensor([2], dtype=torch.int32, device=device),
        task_segment_offsets=torch.tensor([0, 2], dtype=torch.int32, device=device),
        log_slots=torch.tensor([0, 1], dtype=torch.int32, device=device),
        log_begins=torch.tensor([0, 0], dtype=torch.int32, device=device),
        log_ends=torch.tensor([64, 64], dtype=torch.int32, device=device),
        output_offsets=torch.tensor([0, 1, 3], dtype=torch.int32, device=device),
        output_slots=torch.tensor([0, 1, 4], dtype=torch.int32, device=device),
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(
        pool.recurrent_state[0], expected[0].to(torch.bfloat16),
        rtol=2e-2, atol=2e-2,
    )
    torch.testing.assert_close(
        pool.recurrent_state[1], expected[1].to(torch.bfloat16),
        rtol=2e-2, atol=2e-2,
    )
    torch.testing.assert_close(
        pool.recurrent_state[4], expected[1].to(torch.bfloat16),
        rtol=2e-2, atol=2e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_store_log_states_uses_slot_and_token_mapping() -> None:
    pool = LogStatePool(
        num_slots=4,
        page_size=64,
        num_linear_layers=2,
        num_key_heads=2,
        num_value_heads=2,
        head_k_dim=32,
        head_v_dim=32,
        conv_dim=20,
        state_len=3,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    slots = torch.tensor([1, 1, 3], dtype=torch.int64, device="cuda")
    token_offsets = torch.tensor([5, 6, 19], dtype=torch.int64, device="cuda")
    keys = torch.randn(3, 2, 32, dtype=torch.bfloat16, device="cuda")
    values = torch.randn_like(keys)
    decays = torch.rand(3, 2, dtype=torch.float32, device="cuda")

    store_log_states(
        key_log=pool.key_log,
        residual_value_log=pool.residual_value_log,
        decay_log=pool.decay_log,
        recurrent_state=pool.recurrent_state,
        slots=slots,
        token_offsets=token_offsets,
        keys=keys,
        residual_values=values,
        decays=decays,
        layer=1,
    )

    torch.testing.assert_close(pool.key_log[slots, 1, token_offsets], keys)
    torch.testing.assert_close(
        pool.residual_value_log[slots, 1, token_offsets], values
    )
    torch.testing.assert_close(pool.decay_log[slots, 1, token_offsets], decays)
