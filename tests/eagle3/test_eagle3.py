import torch

from picosgl.core import SamplingParams
from picosgl.distributed import DistributedInfo
from picosgl.message import BaseSpeculatorMsg, make_handshake_message, make_init_message
from picosgl.message.speculator.eagle3 import Eagle3HandshakeMsg, Eagle3InitMsg
from picosgl.models.llama import LlamaConfig
from picosgl.models.config import RotaryConfig
from picosgl.scheduler import SchedulerConfig
from picosgl.speculator import DataPlaneSizes, make_data_plane_sizes
from picosgl.speculator.drafters import (
    make_hidden_captor,
    make_speculator_argument_parser,
)
from picosgl.speculator.drafters.eagle3 import (
    Eagle3HiddenCaptor,
    Eagle3SpeculatorConfig,
)
from picosgl.speculator.hidden_captor import HiddenCapturePoint


def test_eagle3_config_and_hidden_capture() -> None:
    speculator_config = Eagle3SpeculatorConfig(
        num_draft_tokens=4,
        window_size=96,
        target_layer_ids=(2, 14, 25),
    )
    config = SchedulerConfig(
        model_path="unused-target",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_running_req=8,
        max_decode_tokens=16,
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path="unused-draft",
        speculator_config=speculator_config,
    )
    assert make_data_plane_sizes(config, 2048, 151936) == DataPlaneSizes(
        max_hidden_rows=96,
        hidden_size=6144,
        max_prob_rows=16,
        vocab_size=151936,
    )

    captor = make_hidden_captor("EAGLE3", speculator_config)
    assert isinstance(captor, Eagle3HiddenCaptor)
    layers = {
        layer_id: torch.full((5, 2), layer_id)
        for layer_id in speculator_config.target_layer_ids
    }
    captor.capture(HiddenCapturePoint.DECODER_INPUT, 1, torch.empty(5, 2))
    for layer_id, hidden in layers.items():
        captor.capture(HiddenCapturePoint.DECODER_INPUT, layer_id, hidden)
    expected = torch.cat(list(layers.values()), dim=-1)
    torch.testing.assert_close(captor.full_hidden, expected)
    torch.testing.assert_close(captor.select(slice(1, 3)).full_hidden, expected[1:3])


def test_eagle3_default_layers_follow_target_depth() -> None:
    target_config = LlamaConfig(
        num_layers=16,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=64,
        hidden_size=2048,
        vocab_size=128256,
        rms_norm_eps=1e-5,
        rotary_config=RotaryConfig(64, 131072, 500000.0, None),
        tie_word_embeddings=True,
        architectures=["LlamaForCausalLM"],
        intermediate_size=8192,
        hidden_act="silu",
    )
    config = Eagle3SpeculatorConfig().resolve(target_config)
    assert config.target_layer_ids == (2, 8, 13)


def test_eagle3_messages_and_arguments() -> None:
    parser = make_speculator_argument_parser("EAGLE3")
    kwargs = vars(parser.parser.parse_args([
        "--speculative-num-draft-tokens", "4",
        "--speculator-window-size", "3",
        "--eagle3-target-layer-ids", "2,14,25",
    ]))
    config = parser.make_config(kwargs)
    assert config == Eagle3SpeculatorConfig(4, 3, (2, 14, 25))
    assert kwargs == {}

    handshake = make_handshake_message("EAGLE3", config, b"id", 8, 6, 16, 32)
    assert isinstance(handshake, Eagle3HandshakeMsg)
    assert BaseSpeculatorMsg.decoder(handshake.encoder()) == handshake

    full_hidden = torch.arange(30).reshape(5, 6)
    captor = Eagle3HiddenCaptor(config, full_hidden)
    msg, hidden = make_init_message(
        "EAGLE3",
        config,
        7,
        2,
        4,
        torch.tensor([10, 11, 12, 13, 14]),
        captor,
        SamplingParams(),
    )
    assert isinstance(msg, Eagle3InitMsg)
    assert msg.carry_positions == [2, 3, 4]
    assert msg.carry_tokens == [12, 13, 14]
    torch.testing.assert_close(hidden, full_hidden[-3:])
    assert BaseSpeculatorMsg.decoder(msg.encoder()) == msg
