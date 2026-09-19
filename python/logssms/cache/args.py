from __future__ import annotations

import argparse
from abc import ABC, abstractmethod

from logssms.utils import Registry

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
                "Minimum LogState prefix-match and recovery granularity in tokens. "
                "The physical page size remains fixed at 64 tokens."
            ),
        )
        parser.add_argument(
            "--enable-merge-adjustment",
            action="store_true",
            help="Adapt LogState merge intensity from online branch-hit samples.",
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
            "--base-intensity-increment",
            type=int,
            default=LogStateCacheConfig.base_intensity_increment,
            help="Merge-intensity increment when hit density cannot be fitted.",
        )

    @staticmethod
    def make_config(kwargs: dict) -> LogStateCacheConfig:
        return LogStateCacheConfig(
            granularity=kwargs.pop("granularity"),
            enable_merge_adjustment=kwargs.pop("enable_merge_adjustment"),
            min_sampling_len=kwargs.pop("min_sampling_len"),
            max_merge_intensity=kwargs.pop("max_merge_intensity"),
            base_intensity_increment=kwargs.pop("base_intensity_increment"),
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
