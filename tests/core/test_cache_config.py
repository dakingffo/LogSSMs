import pytest

from logssms.cache import (
    LogStateCacheConfig,
    StandardCacheConfig,
    make_cache_argument_parser,
)
from logssms.cache.logstate_pool.layout import LOG_STATE_GRANULARITY
from logssms.layers.linear_attention_backend import (
    GatedDeltaLogStateBackend, make_linear_attention_backend,
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
        "--eager-state", "true",
        "--memorized-evict", "false",
        "--merge-adjustment", "true",
        "--min-sampling-len", "512",
        "--max-merge-intensity", "12",
        "--default-merge-growth-rate", "3",
    ]))

    config = parser.make_config(kwargs)
    assert config == LogStateCacheConfig(
        granularity=32,
        enable_eager_state=True,
        enable_memorized_evict=False,
        enable_merge_adjustment=True,
        min_sampling_len=512,
        max_merge_intensity=12,
        default_merge_growth_rate=3,
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
    with pytest.raises(SystemExit):
        parse_args(common + ["--cache-type", "radix", "--eager-state", "true"])
    with pytest.raises(SystemExit):
        parse_args(common + ["--cache-type", "radix", "--memorized-evict", "false"])


def test_server_uses_logstate_defaults(tmp_path) -> None:
    args = parse_args([
        "--model", str(tmp_path),
        "--dtype", "bfloat16",
        "--dummy-weight",
        "--cache-type", "logstate",
    ])

    assert args.cache_config == LogStateCacheConfig()
    assert not args.cache_config.enable_eager_state
    assert args.cache_config.enable_memorized_evict
    assert args.cache_config.enable_merge_adjustment
    assert args.cache_config.max_merge_intensity == 16
    assert args.cache_config.granularity == LOG_STATE_GRANULARITY == 32
    assert args.page_size == 64


def test_server_can_explicitly_disable_merge_adjustment(tmp_path) -> None:
    args = parse_args([
        "--model", str(tmp_path), "--dtype", "bfloat16", "--dummy-weight",
        "--cache-type", "logstate", "--merge-adjustment", "false",
    ])
    assert not args.cache_config.enable_merge_adjustment


@pytest.mark.parametrize("flag", ["--graph", "--cuda-graph-max-bs"])
@pytest.mark.parametrize("max_bs", [0, 4])
def test_server_uses_common_cuda_graph_configuration(tmp_path, flag, max_bs):
    args = parse_args([
        "--model", str(tmp_path), "--dtype", "bfloat16", "--dummy-weight",
        flag, str(max_bs),
    ])
    assert args.cuda_graph_max_bs == max_bs
    assert not hasattr(args, "prefill_cuda_graph")


def test_server_rejects_removed_prefill_graph_switch(tmp_path):
    with pytest.raises(SystemExit):
        parse_args([
            "--model", str(tmp_path), "--dtype", "bfloat16", "--dummy-weight",
            "--prefill-cuda-graph", "true",
        ])


def test_logstate_rejects_sub_chunk_granularity() -> None:
    parser = make_cache_argument_parser("logstate")

    with pytest.raises(SystemExit):
        parser.parser.parse_args(["--granularity", "8"])


def test_logstate_backend_receives_eager_state() -> None:
    backend = make_linear_attention_backend("logstate", enable_eager_state=True)
    assert isinstance(backend, GatedDeltaLogStateBackend)
    assert backend.enable_eager_state


@pytest.mark.parametrize("flag", [
    "--eager-state",
    "--memorized-evict",
    "--merge-adjustment",
])
def test_logstate_boolean_flags_accept_explicit_value(flag: str) -> None:
    parser = make_cache_argument_parser("logstate")
    name = "enable_" + flag.removeprefix("--").replace("-", "_")
    assert getattr(parser.parser.parse_args([flag, "true"]), name)
    assert not getattr(parser.parser.parse_args([flag, "false"]), name)
    with pytest.raises(SystemExit):
        parser.parser.parse_args([flag, "not-a-bool"])


def test_server_accepts_explicitly_disabled_memorized_evict(tmp_path) -> None:
    args = parse_args([
        "--model", str(tmp_path),
        "--dtype", "bfloat16",
        "--dummy-weight",
        "--cache-type", "logstate",
        "--memorized-evict", "false",
    ])
    assert not args.cache_config.enable_memorized_evict


def test_server_accepts_explicit_dt_boolean(tmp_path) -> None:
    common = [
        "--model", str(tmp_path),
        "--dtype", "bfloat16",
        "--dummy-weight",
    ]
    assert parse_args(common + ["--dt-separation", "true"]).dt_separation
    assert not parse_args(common + ["--dt-separation", "false"]).dt_separation
    with pytest.raises(SystemExit):
        parse_args(common + ["--dt-separation", "invalid"])
