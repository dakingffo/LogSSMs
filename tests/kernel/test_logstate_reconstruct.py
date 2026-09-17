import pytest
import torch

from logssms.cache.linear_pool.logstate_pool import LogStatePool
from logssms.kernel.linear_attention import reconstruct_log_states


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
        num_key_heads=1,
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
                    key = pool.key_log[slot, layer, token, 0].float()
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
