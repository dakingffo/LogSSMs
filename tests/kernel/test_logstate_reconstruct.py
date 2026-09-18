import pytest
import torch

from logssms.cache.linear_pool.logstate_pool import LogStatePool
from logssms.kernel.linear_attention import (
    materialize_log_states_inplace,
    reconstruct_log_states,
    store_log_states,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("head_dims", [(8, 6), (128, 128)])
def test_reconstruct_log_states_matches_recurrent_reference(
    head_dims: tuple[int, int],
) -> None:
    torch.manual_seed(0)
    head_k_dim, head_v_dim = head_dims
    pool = LogStatePool(
        num_slots=4,
        page_size=64,
        num_linear_layers=2,
        num_key_heads=2,
        num_value_heads=2,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_dim=20,
        state_len=3,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    pool.key_log[:2].normal_(std=0.1)
    pool.residual_value_log[:2].normal_(std=0.1)
    pool.decay_log[:2].uniform_(0.8, 0.99)
    pool.recurrent_state[2].normal_(std=0.1)
    pool.conv_state[1, :, 0].normal_()

    expected = pool.recurrent_state[2].float().clone()
    for slot, begin, end in ((0, 16, 32), (1, 0, 16)):
        for token in range(begin, end):
            for layer in range(2):
                for value_head in range(2):
                    decay = pool.decay_log[slot, layer, token, value_head]
                    key = pool.key_log[
                        slot, layer, token, value_head
                    ].float()
                    value = pool.residual_value_log[
                        slot, layer, token, value_head
                    ].float()
                    expected[layer, value_head].mul_(decay).add_(
                        key[:, None] * value[None, :]
                    )

    device = pool.device
    reconstruct_log_states(
        key_log=pool.key_log,
        residual_value_log=pool.residual_value_log,
        decay_log=pool.decay_log,
        recurrent_state=pool.recurrent_state,
        conv_state=pool.conv_state,
        output_slots=torch.tensor([3], dtype=torch.int32, device=device),
        base_slots=torch.tensor([2], dtype=torch.int32, device=device),
        task_log_offsets=torch.tensor([0, 2], dtype=torch.int32, device=device),
        log_slots=torch.tensor([0, 1], dtype=torch.int32, device=device),
        log_begins=torch.tensor([16, 0], dtype=torch.int32, device=device),
        log_ends=torch.tensor([32, 16], dtype=torch.int32, device=device),
        source_conv_slots=torch.tensor([1], dtype=torch.int32, device=device),
        source_conv_granules=torch.tensor([0], dtype=torch.int32, device=device),
        output_conv_granules=torch.tensor([0], dtype=torch.int32, device=device),
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(
        pool.recurrent_state[3],
        expected.to(torch.bfloat16),
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        pool.conv_state[3, :, 0],
        pool.conv_state[1, :, 0],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_materialize_log_states_inplace_handles_concurrent_private_chains() -> None:
    torch.manual_seed(1)
    pool = LogStatePool(
        num_slots=6,
        page_size=64,
        num_linear_layers=2,
        num_key_heads=2,
        num_value_heads=2,
        head_k_dim=8,
        head_v_dim=6,
        conv_dim=20,
        state_len=3,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    pool.key_log[:4].normal_(std=0.1)
    pool.residual_value_log[:4].normal_(std=0.1)
    pool.decay_log[:4].uniform_(0.8, 0.99)
    pool.recurrent_state[5].normal_(std=0.1)

    key = pool.key_log[:4].float().clone()
    value = pool.residual_value_log[:4].float().clone()
    decay = pool.decay_log[:4].clone()
    protected_key = pool.key_log[2].clone()
    protected_value = pool.residual_value_log[2].clone()
    protected_decay = pool.decay_log[2].clone()

    def consume(
        state: torch.Tensor,
        slot : int,
        begin: int,
        end  : int,
    ) -> torch.Tensor:
        for token in range(begin, end):
            for layer in range(2):
                for head in range(2):
                    state[layer, head].mul_(decay[slot, layer, token, head]).add_(
                        key[slot, layer, token, head, :, None]
                        * value[slot, layer, token, head, None, :]
                    )
        return state

    state0 = consume(torch.zeros_like(pool.recurrent_state[0].float()), 0, 0, 64)
    state1 = consume(state0.clone(), 1, 0, 64)
    state3 = consume(pool.recurrent_state[5].float().clone(), 2, 16, 32)
    state3 = consume(state3, 3, 0, 64)

    device = pool.device
    materialize_log_states_inplace(
        key_log=pool.key_log,
        residual_value_log=pool.residual_value_log,
        decay_log=pool.decay_log,
        recurrent_state=pool.recurrent_state,
        base_slots=torch.tensor([-1, 5], dtype=torch.int32, device=device),
        chain_log_offsets=torch.tensor([0, 2, 4], dtype=torch.int32, device=device),
        log_slots=torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device),
        log_begins=torch.tensor([0, 0, 16, 0], dtype=torch.int32, device=device),
        log_ends=torch.tensor([64, 64, 32, 64], dtype=torch.int32, device=device),
        checkpoint_flags=torch.tensor([1, 1, 0, 1], dtype=torch.int32, device=device),
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(
        pool.recurrent_state[0], state0.to(torch.bfloat16), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        pool.recurrent_state[1], state1.to(torch.bfloat16), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        pool.recurrent_state[3], state3.to(torch.bfloat16), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(pool.key_log[2], protected_key)
    torch.testing.assert_close(pool.residual_value_log[2], protected_value)
    torch.testing.assert_close(pool.decay_log[2], protected_decay)
