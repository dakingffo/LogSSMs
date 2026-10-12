import torch

from logssms.core import SamplingParams
from logssms.distributed import DistributedInfo
from logssms.message import BaseSpeculatorMsg, make_handshake_message, make_init_message
from logssms.message.speculator.dflash import DFlashHandshakeMsg, DFlashInitMsg
from logssms.scheduler import SchedulerConfig
from logssms.speculator import DataPlaneSizes, SpeculatorReserve, make_data_plane_sizes
from logssms.speculator.drafters import (
    make_hidden_captor,
    make_speculator_argument_parser,
)
from logssms.speculator.drafters.attention import DraftAttentionBackend
from logssms.speculator.drafters.dflash import (
    DFlashHiddenCaptor,
    DFlashSpeculatorConfig,
    DFlashState,
)
from logssms.speculator.drafters.pool import DraftKVPool
from logssms.speculator.hidden_captor import HiddenCapturePoint


def test_dflash_config_and_arguments() -> None:
    parser = make_speculator_argument_parser("DFLASH")
    kwargs = vars(parser.parser.parse_args([
        "--speculative-num-draft-tokens", "8",
        "--speculator-window-size", "64",
        "--dflash-target-layer-ids", "1,3,5",
    ]))
    config = parser.make_config(kwargs)

    assert config == DFlashSpeculatorConfig(
        block_size=8,
        window_size=64,
        target_layer_ids=(1, 3, 5),
    )
    assert config.num_draft_tokens == 7
    assert config.make_reserve(4) == SpeculatorReserve(
        num_draft_slots=32,
        draft_slots_per_request=8,
    )
    assert kwargs == {}


def test_dflash_hidden_capture_and_messages() -> None:
    config = DFlashSpeculatorConfig(
        block_size=4,
        window_size=3,
        target_layer_ids=(1, 3),
    )
    captor = make_hidden_captor("DFLASH", config)
    assert isinstance(captor, DFlashHiddenCaptor)
    layer_1 = torch.arange(10).view(5, 2)
    layer_3 = layer_1 + 20
    captor.capture(HiddenCapturePoint.DECODER_INPUT, 1, layer_1 - 1)
    positions = torch.arange(len(layer_1))
    captor.capture(HiddenCapturePoint.DECODER_INPUT, 2, layer_1, None, positions)
    captor.capture(
        HiddenCapturePoint.DECODER_INPUT,
        4,
        layer_3 - 2,
        torch.full_like(layer_3, 2),
        positions,
    )
    expected = torch.cat([layer_1, layer_3], dim=-1)
    torch.testing.assert_close(captor.full_hidden, expected)

    handshake = make_handshake_message("DFLASH", config, b"id", 8, 4, 6, 32)
    assert isinstance(handshake, DFlashHandshakeMsg)
    assert handshake.block_size == 4
    assert BaseSpeculatorMsg.decoder(handshake.encoder()) == handshake

    msg, hidden = make_init_message(
        "DFLASH",
        config,
        7,
        2,
        5,
        torch.tensor([10, 11, 12, 13, 14, 15]),
        captor,
        SamplingParams(),
    )
    assert isinstance(msg, DFlashInitMsg)
    assert msg.context_positions == [2, 3, 4]
    assert msg.anchor_position == 5
    assert msg.anchor_token == 15
    torch.testing.assert_close(hidden, expected[-3:])
    assert BaseSpeculatorMsg.decoder(msg.encoder()) == msg


def test_dflash_data_plane_and_state_alignment() -> None:
    speculator_config = DFlashSpeculatorConfig()
    config = SchedulerConfig(
        model_path="unused",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_running_req=8,
        max_decode_tokens=30,
        speculative_algorithm="DFLASH",
        speculative_draft_model_path="unused",
        speculator_config=speculator_config,
    )
    assert make_data_plane_sizes(config, 2048, 100) == DataPlaneSizes(
        max_hidden_rows=128,
        hidden_size=10240,
        max_prob_rows=30,
        vocab_size=100,
    )

    state = DFlashState(
        table_idx=0,
        sampling_params=SamplingParams(),
        context_positions=[0, 1],
        context_hidden=torch.zeros(2, 4),
        anchor_position=2,
        anchor_token=12,
    )
    state.clear_pending()
    hidden = torch.arange(8, dtype=torch.float32).view(2, 4)
    state.update([3, 4], [13, 14], hidden)
    assert state.pending_positions == [2, 3]
    assert state.anchor_position == 4
    assert state.anchor_token == 14
    torch.testing.assert_close(state.pending_hidden, hidden)


def test_dflash_block_attention_is_non_causal() -> None:
    pool = DraftKVPool(
        max_running_req=1,
        window_size=2,
        max_batch_size=1,
        num_spec_tokens=3,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    persistent = pool.append_persistent_batch([0], [1])
    pool.store(
        persistent,
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[1.0, 0.0]]]),
    )
    scratch = pool.scratch_block(torch.tensor([0]), 2)
    pool.store(
        scratch,
        torch.tensor([[[0.0, 1.0]], [[1.0, 1.0]]]),
        torch.tensor([[[0.0, 1.0]], [[2.0, 2.0]]]),
    )
    indices, valid = pool.batch_indices([0], torch.tensor([0]), scratch_depth=2)
    query = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    backend = DraftAttentionBackend("native", 1, 1, 2, torch.float32, torch.device("cpu"))

    output = backend.forward_block(query, pool, indices, valid, [3])

    keys = pool.k[indices.long()].transpose(1, 2)
    values = pool.v[indices.long()].transpose(1, 2)
    scores = torch.matmul(
        query.transpose(1, 2), keys.transpose(-1, -2)
    ) * (2**-0.5)
    expected = torch.matmul(scores.softmax(-1), values).transpose(1, 2)
    torch.testing.assert_close(output, expected)


def test_dflash_block_attention_supports_ragged_batch() -> None:
    pool = DraftKVPool(
        max_running_req=2,
        window_size=3,
        max_batch_size=2,
        num_spec_tokens=3,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    persistent = pool.append_persistent_batch([0, 1], [1, 2])
    pool.store(
        persistent,
        torch.tensor([[[1., 0.]], [[0., 1.]], [[1., 1.]]]),
        torch.tensor([[[1., 2.]], [[3., 4.]], [[5., 6.]]]),
    )
    batch_rows = torch.tensor([0, 1])
    scratch = pool.scratch_block(batch_rows, 2)
    pool.store(
        scratch,
        torch.tensor([[[2., 0.]], [[0., 2.]], [[2., 1.]], [[1., 2.]]]),
        torch.tensor([[[7., 8.]], [[9., 10.]], [[11., 12.]], [[13., 14.]]]),
    )
    indices, valid = pool.batch_indices([0, 1], batch_rows, scratch_depth=2)
    query = torch.tensor(
        [
            [[[1., 0.], [0., 1.]], [[1., 1.], [1., -1.]]],
            [[[0., 1.], [1., 0.]], [[2., 1.], [-1., 1.]]],
        ]
    )
    backend = DraftAttentionBackend("native", 2, 1, 2, torch.float32, torch.device("cpu"))

    output = backend.forward_block(query, pool, indices, valid, [3, 4])

    expected_rows = []
    for row, cache_len in enumerate((3, 4)):
        key = pool.k[indices[row, :cache_len].long()].repeat_interleave(2, dim=1)
        value = pool.v[indices[row, :cache_len].long()].repeat_interleave(2, dim=1)
        scores = torch.einsum("qhd,khd->hqk", query[row], key) * (2**-0.5)
        expected_rows.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), value))
    torch.testing.assert_close(output, torch.stack(expected_rows))
