from __future__ import annotations

import argparse
from abc import ABC, abstractmethod

from logssms.utils import Registry
from logssms.utils.args import parse_bool

from .config import (
    BaseCacheConfig,
    LOG_STATE_GRANULARITIES,
    LogStateCacheConfig,
    StandardCacheConfig,
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


class StandardCacheArgumentParser(CacheArgumentParserBase):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--page-size",
            type=int,
            default=StandardCacheConfig.page_size,
            help="Physical cache page size in tokens.",
        )

    @staticmethod
    def make_config(kwargs: dict) -> StandardCacheConfig:
        return StandardCacheConfig(page_size=kwargs.pop("page_size"))


class LogStateCacheArgumentParser(CacheArgumentParserBase):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--granularity",
            type=int,
            choices=LOG_STATE_GRANULARITIES,
            default=LogStateCacheConfig.granularity,
            help=(
                "LogState prefix-match, recovery and intermediate prefill-chunk alignment in tokens. "
                "The physical page size remains fixed at 64 tokens."
            ),
        )
        parser.add_argument(
            "--merge-adjustment",
            type=parse_bool,
            dest="enable_merge_adjustment",
            default=LogStateCacheConfig.enable_merge_adjustment,
            metavar="{true,false}",
            help="Adapt LogState merge intensity from online branch-hit samples (default: true). Set false for binary merging.",
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
            "--memorized-evict",
            type=parse_bool,
            dest="enable_memorized_evict",
            default=LogStateCacheConfig.enable_memorized_evict,
            metavar="{true,false}",
            help="Remember evicted cache layout on CPU (default: true). Set false to disable memorized eviction.",
        )
        parser.add_argument(
            "--min-sampling-len",
            type=int,
            default=LogStateCacheConfig.min_sampling_len,
            help="Minimum observed chain length before fitting hit density.",
        )
        parser.add_argument(
            "--max-merge-intensity",
            type=int,
            default=LogStateCacheConfig.max_merge_intensity,
            help="Maximum fitted LogState merge intensity.",
        )
        parser.add_argument(
            "--default-merge-growth-rate",
            type=int,
            default=LogStateCacheConfig.default_merge_growth_rate,
            help="Merge-intensity multiplier when hit density cannot be fitted.",
        )

    @staticmethod
    def make_config(kwargs: dict) -> LogStateCacheConfig:
        return LogStateCacheConfig(
            granularity=kwargs.pop("granularity"),
            enable_eager_state=kwargs.pop("enable_eager_state"),
            enable_memorized_evict=kwargs.pop("enable_memorized_evict"),
            enable_merge_adjustment=kwargs.pop("enable_merge_adjustment"),
            min_sampling_len=kwargs.pop("min_sampling_len"),
            max_merge_intensity=kwargs.pop("max_merge_intensity"),
            default_merge_growth_rate=kwargs.pop("default_merge_growth_rate"),
        )


SUPPORTED_CACHE_ARGUMENT_PARSERS = Registry[
    type[CacheArgumentParserBase]
]("Cache Argument Parser")
SUPPORTED_CACHE_ARGUMENT_PARSERS.register("naive")(StandardCacheArgumentParser)
SUPPORTED_CACHE_ARGUMENT_PARSERS.register("radix")(StandardCacheArgumentParser)
SUPPORTED_CACHE_ARGUMENT_PARSERS.register("logstate")(LogStateCacheArgumentParser)


def make_cache_argument_parser(cache_type: str) -> CacheArgumentParserBase:
    return SUPPORTED_CACHE_ARGUMENT_PARSERS[cache_type]()


__all__ = [
    "CacheArgumentParserBase",
    "LogStateCacheArgumentParser",
    "SUPPORTED_CACHE_ARGUMENT_PARSERS",
    "StandardCacheArgumentParser",
    "make_cache_argument_parser",
]
