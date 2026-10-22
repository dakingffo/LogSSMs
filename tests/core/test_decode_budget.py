from dataclasses import replace

import pytest

from logssms.server.args import parse_args


@pytest.mark.parametrize("running", [1, 3, 32, 64, 128])
def test_ar_decode_budget_defaults_to_max_running_requests(tmp_path, running):
    args = parse_args([
        "--model", str(tmp_path), "--dummy-weight", "--dtype", "bfloat16",
        "--max-running-requests", str(running),
    ])
    assert args.max_decode_tokens == running
    assert args.decode_batch_budget == running
    assert replace(args, max_decode_tokens=None).decode_batch_budget == running


def test_explicit_decode_budget_is_preserved(tmp_path):
    args = parse_args([
        "--model", str(tmp_path), "--dummy-weight", "--dtype", "bfloat16",
        "--max-running-requests", "64", "--max-decode-tokens", "17",
    ])
    assert args.max_decode_tokens == args.decode_batch_budget == 17


@pytest.mark.parametrize("running", [8, 17, 32, 64])
def test_speculative_decode_budget_defaults_to_half_running_requests(tmp_path, running):
    args = parse_args([
        "--model", str(tmp_path), "--dummy-weight", "--dtype", "bfloat16",
        "--max-running-requests", str(running),
        "--speculative-algorithm", "MTP",
        "--speculative-draft-model-path", str(tmp_path),
    ])
    assert args.max_decode_tokens == args.decode_batch_budget == running // 2
    assert replace(args, max_decode_tokens=None).decode_batch_budget == running // 2
    assert replace(args, max_decode_tokens=23).decode_batch_budget == 23
