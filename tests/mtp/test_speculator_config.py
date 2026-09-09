import torch

from picosgl.distributed import DistributedInfo
from picosgl.scheduler import SchedulerConfig
from picosgl.server.args import parse_args
from picosgl.speculator import (
    DataPlaneSizes,
    SpeculatorReserve,
    make_data_plane_sizes,
)
from picosgl.speculator.drafters import make_hidden_captor, make_speculator_argument_parser
from picosgl.speculator.drafters.mtp import MTPHiddenCaptor, MTPSpeculatorConfig
from picosgl.speculator.hidden_captor import HiddenCapturePoint


def test_mtp_speculator_config() -> None:
    speculator_config = MTPSpeculatorConfig(
        num_draft_tokens=3,
        window_size=96,
    )
    config = SchedulerConfig(
        model_path="unused",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_running_req=8,
        speculative_algorithm="MTP",
        speculative_draft_model_path="unused",
        speculator_config=speculator_config,
    )

    assert config.decode_batch_budget == 12
    assert speculator_config.make_reserve(config.max_running_req) == SpeculatorReserve(
        num_draft_slots=32,
        draft_slots_per_request=4,
    )
    assert make_data_plane_sizes(config, 1024, 100) == DataPlaneSizes(
        max_hidden_rows=96,
        hidden_size=1024,
        max_prob_rows=12,
        vocab_size=100,
    )

    hidden = torch.arange(12).view(4, 3)
    hidden_captor = make_hidden_captor("MTP", speculator_config)
    hidden_captor.capture(HiddenCapturePoint.DECODER_INPUT, 0, hidden + 1)
    hidden_captor.capture(HiddenCapturePoint.LM_HEAD_INPUT, None, hidden)
    assert isinstance(hidden_captor, MTPHiddenCaptor)
    selected = hidden_captor.select(slice(1, 3))
    assert torch.equal(selected.full_hidden, hidden[1:3])
    assert torch.equal(
        hidden_captor.get_carry_hidden(slice(1, 3)),
        hidden[1:3],
    )


def test_mtp_argument_parser_is_independent(tmp_path) -> None:
    parser = make_speculator_argument_parser("MTP")
    kwargs = vars(parser.parser.parse_args([
        "--speculative-num-draft-tokens", "6",
        "--speculator-window-size", "192",
    ]))
    assert parser.make_config(kwargs) == MTPSpeculatorConfig(6, 192)
    assert kwargs == {}

    model_path = str(tmp_path)
    server_args = parse_args([
        "--model-path", model_path,
        "--dtype", "bfloat16",
        "--dummy-weight",
        "--max-running-requests", "16",
        "--speculative-algorithm", "MTP",
        "--speculative-draft-model-path", model_path,
        "--speculative-num-draft-tokens", "6",
        "--speculator-window-size", "192",
    ])
    assert server_args.speculator_config == MTPSpeculatorConfig(6, 192)
    assert server_args.max_decode_tokens == 48
