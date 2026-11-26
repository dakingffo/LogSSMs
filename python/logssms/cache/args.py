from __future__ import annotations

import argparse
from abc import ABC, abstractmethod

from logssms.utils import Registry
from logssms.utils.args import parse_bool

from logssms.cache.config import (
    BaseCacheConfig,
    LOG_STATE_GRANULARITIES,
    LogStateCacheConfig,
    NaiveCacheConfig,
    RadixCacheConfig,
    MarconiCacheConfig,
)


class CacheArgumentParserBase(ABC):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(add_help=False)
        self.add_arguments(self.parser)

    @staticmethod
    @abstractmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None: ...

    @staticmethod
    @abstractmethod
    def make_config(kwargs: dict) -> BaseCacheConfig: ...


class NaiveCacheArgumentParser(CacheArgumentParserBase):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--page-size",
            type=int,
            default=NaiveCacheConfig.page_size,
            help="Physical cache page size in tokens.",
        )

    @staticmethod
    def make_config(kwargs: dict) -> NaiveCacheConfig:
        return NaiveCacheConfig(page_size=kwargs.pop("page_size"))


class RadixCacheArgumentParser(CacheArgumentParserBase):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--page-size",
            type=int,
            default=RadixCacheConfig.page_size,
            help="Physical cache page size in tokens.",
        )

    @staticmethod
    def make_config(kwargs: dict) -> RadixCacheConfig:
        return RadixCacheConfig(page_size=kwargs.pop("page_size"))


class MarconiCacheArgumentParser(CacheArgumentParserBase):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--block-size", type=int, default=128,
                            help="Marconi block size in tokens, a multiple of 64; zero selects auto sizing.")
        parser.add_argument("--marconi-bootstrap-multiplier", type=int, default=5)

    @staticmethod
    def make_config(kwargs):
        return MarconiCacheConfig(kwargs.pop("block_size"), kwargs.pop("marconi_bootstrap_multiplier"))


class LogStateCacheArgumentParser(CacheArgumentParserBase):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--granularity",
            type=int,
            choices=LOG_STATE_GRANULARITIES,
            default=LogStateCacheConfig.granularity,
            help=(
                "LogState storage, prefix-match, recovery and intermediate prefill-chunk "
                "alignment in tokens (default: %(default)s)."
            ),
        )
        parser.add_argument(
            "--merge-adjustment",
            type=parse_bool,
            dest="enable_merge_adjustment",
            default=LogStateCacheConfig.enable_merge_adjustment,
            metavar="{true,false}",
            help="Use online hit density for unified LogState eviction (default: true). Set false for Log-first, binary Checkpoint thinning.",
        )
        parser.add_argument(
            "--eager-state",
            type=parse_bool,
            dest="enable_eager_state",
            default=LogStateCacheConfig.enable_eager_state,
            metavar="{true,false}",
            help="Write the LogState decode work state after every token instead of replaying logs within a granule.",
        )
        parser.add_argument(
            "--hit-materialization",
            type=parse_bool,
            dest="enable_hit_materialization",
            default=LogStateCacheConfig.enable_hit_materialization,
            metavar="{true,false}",
            help="Materialize reusable Log groups on prefill hits (default: %(default)s).",
        )
        parser.add_argument(
            "--min-sampling-len",
            type=int,
            default=LogStateCacheConfig.min_sampling_len,
            help="Token window for hit-density samples.",
        )
        parser.add_argument(
            "--max-density-exponent", type=float,
            default=LogStateCacheConfig.max_density_exponent,
            help="Upper bound of the fitted hit-density exponent.",
        )
        parser.add_argument(
            "--hit-half-life", type=int,
            default=LogStateCacheConfig.hit_half_life,
            help="Hit-statistics half-life in admitted requests.",
        )

    @staticmethod
    def make_config(kwargs: dict) -> LogStateCacheConfig:
        return LogStateCacheConfig(
            granularity=kwargs.pop("granularity"),
            enable_eager_state=kwargs.pop("enable_eager_state"),
            enable_merge_adjustment=kwargs.pop("enable_merge_adjustment"),
            min_sampling_len=kwargs.pop("min_sampling_len"),
            max_density_exponent=kwargs.pop("max_density_exponent"),
            hit_half_life=kwargs.pop("hit_half_life"),
            enable_hit_materialization=kwargs.pop("enable_hit_materialization"),
        )


SUPPORTED_CACHE_ARGUMENT_PARSERS = Registry[
    type[CacheArgumentParserBase]
]("Cache Argument Parser")
SUPPORTED_CACHE_ARGUMENT_PARSERS.register("naive")(NaiveCacheArgumentParser)
SUPPORTED_CACHE_ARGUMENT_PARSERS.register("radix")(RadixCacheArgumentParser)
SUPPORTED_CACHE_ARGUMENT_PARSERS.register("logstate")(LogStateCacheArgumentParser)
SUPPORTED_CACHE_ARGUMENT_PARSERS.register("marconi")(MarconiCacheArgumentParser)


def make_cache_argument_parser(cache_type: str) -> CacheArgumentParserBase:
    return SUPPORTED_CACHE_ARGUMENT_PARSERS[cache_type]()


__all__ = [
    "CacheArgumentParserBase",
    "LogStateCacheArgumentParser",
    "SUPPORTED_CACHE_ARGUMENT_PARSERS",
    "NaiveCacheArgumentParser",
    "RadixCacheArgumentParser",
    "make_cache_argument_parser",
]
