import pytest

from logssms.cache import (
    LogStateCacheConfig,
    StandardCacheConfig,
    make_cache_argument_parser,
)
from logssms.server.args import parse_args


def test_standard_cache_arguments_only_build_page_size_config() -> None:
    parser = make_cache_argument_parser("radix")
    kwargs = vars(parser.parser.parse_args(["--page-size", "32"]))

    assert parser.make_config(kwargs) == StandardCacheConfig(page_size=32)
    assert kwargs == {}


def test_logstate_arguments_build_algorithm_specific_config() -> None:
    parser = make_cache_argument_parser("logstate")
    kwargs = vars(parser.parser.parse_args([
        "--granularity", "32",
        "--enable-merge-adjustment",
        "--min-sampling-len", "512",
        "--max-merge-intensity", "12",
        "--base-intensity-increment", "3",
    ]))

    config = parser.make_config(kwargs)
    assert config == LogStateCacheConfig(
        granularity=32,
        enable_merge_adjustment=True,
        min_sampling_len=512,
        max_merge_intensity=12,
        base_intensity_increment=3,
    )
    assert config.page_size == 64
    assert kwargs == {}


def test_server_rejects_cache_arguments_from_another_cache_type(tmp_path) -> None:
    common = [
        "--model", str(tmp_path),
        "--dtype", "bfloat16",
        "--dummy-weight",
    ]

    with pytest.raises(SystemExit):
        parse_args(common + ["--cache-type", "logstate", "--page-size", "32"])
    with pytest.raises(SystemExit):
        parse_args(common + ["--cache-type", "radix", "--granularity", "16"])


def test_server_uses_logstate_defaults(tmp_path) -> None:
    args = parse_args([
        "--model", str(tmp_path),
        "--dtype", "bfloat16",
        "--dummy-weight",
        "--cache-type", "logstate",
    ])

    assert args.cache_config == LogStateCacheConfig()
    assert args.page_size == 64


def test_logstate_rejects_sub_chunk_granularity() -> None:
    parser = make_cache_argument_parser("logstate")

    with pytest.raises(SystemExit):
        parser.parser.parse_args(["--granularity", "8"])
