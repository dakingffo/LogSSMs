import pytest
import torch

from logssms.cache.linear_pool.logstate_pool import LogStatePool


def make_pool(page_size: int = 64) -> LogStatePool:
    return LogStatePool(
        num_slots=3,
        page_size=page_size,
        num_linear_layers=2,
        num_key_heads=2,
        num_value_heads=2,
        head_k_dim=8,
        head_v_dim=6,
        conv_dim=32,
        state_len=3,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )


def test_logstate_pool_is_composed_of_page_sized_slots():
    pool = make_pool()

    assert pool.num_slots == 3
    assert pool.page_size == 64
    assert pool.num_granules == 4
    assert pool.key_log.shape == (3, 2, 64, 2, 8)
    assert pool.residual_value_log.shape == (3, 2, 64, 2, 6)
    assert pool.decay_log.shape == (3, 2, 64, 2)
    assert pool.recurrent_state.shape == (3, 2, 2, 8, 6)
    assert pool.conv_state.shape == (3, 2, 4, 32, 3)
    assert pool._storage.shape == (3, pool.slot_bytes)


def test_log_and_checkpoint_are_two_views_of_one_slot_payload():
    pool = make_pool()
    pool._storage.zero_()

    slot = pool.slot(1)
    slot.recurrent_state[0, 0, 0, 0] = 7
    assert slot.log.key[0, 0, 0, 0] == 7

    slot.log.key[0, 0, 0, 0] = 11
    assert slot.recurrent_state[0, 0, 0, 0] == 11
    assert pool.slot(0).recurrent_state[0, 0, 0, 0] == 0


def test_each_granule_has_an_independent_conv_snapshot():
    pool = make_pool()
    slot = pool.slot(2)

    slot.conv_state.zero_()
    slot.conv_state[:, 1].fill_(3)
    assert torch.all(slot.conv_state[:, 0] == 0)
    assert torch.all(slot.conv_state[:, 1] == 3)
    assert torch.all(slot.conv_state[:, 2:] == 0)


def test_granule_returns_sixteen_token_log_and_shared_conv_slot():
    pool = make_pool()
    granule = pool.granule(1, 2)

    assert granule.log.key.shape == (2, 16, 2, 8)
    assert granule.log.residual_value.shape == (2, 16, 2, 6)
    assert granule.log.decay.shape == (2, 16, 2)
    assert granule.conv_state.shape == (2, 32, 3)
    assert granule.log.key.data_ptr() == pool.key_log[1, :, 32:48].data_ptr()
    assert granule.conv_state.data_ptr() == pool.conv_state[1, :, 2].data_ptr()


def test_recovery_position_maps_to_the_preceding_granule():
    pool = make_pool()

    assert [
        pool.recovery_granule_index(position)
        for position in (16, 32, 48, 64)
    ] == [0, 1, 2, 3]
    assert [
        pool.token_granule_index(position)
        for position in (0, 15, 16, 31, 32, 47, 48, 63)
    ] == [0, 0, 1, 1, 2, 2, 3, 3]
    with pytest.raises(ValueError, match="not aligned"):
        pool.recovery_granule_index(15)


def test_page_size_must_be_a_multiple_of_log_granularity():
    with pytest.raises(AssertionError, match="must be divisible"):
        make_pool(page_size=63)
