from __future__ import annotations

import argparse

from ...args import SpeculatorArgumentParserBase
from .config import DFlashSpeculatorConfig


class DFlashArgumentParser(SpeculatorArgumentParserBase):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--speculative-dflash-block-size",
            "--speculative-num-draft-tokens",
            type=int,
            dest="dflash_block_size",
            default=DFlashSpeculatorConfig.block_size,
            help="DFlash verify block width, including its anchor token.",
        )
        parser.add_argument(
            "--speculative-dflash-draft-window-size",
            "--speculator-window-size",
            type=int,
            dest="dflash_window_size",
            default=DFlashSpeculatorConfig.window_size,
            help="Number of committed target-feature rows retained by the DFlash drafter.",
        )
        parser.add_argument(
            "--dflash-target-layer-ids",
            type=lambda value: tuple(int(item) for item in value.split(",")),
            default=DFlashSpeculatorConfig.target_layer_ids,
            metavar="LAYER,...",
        )

    @staticmethod
    def make_config(kwargs: dict) -> DFlashSpeculatorConfig:
        return DFlashSpeculatorConfig(
            block_size=kwargs.pop("dflash_block_size"),
            window_size=kwargs.pop("dflash_window_size"),
            target_layer_ids=kwargs.pop("dflash_target_layer_ids"),
        )


__all__ = ["DFlashArgumentParser"]
