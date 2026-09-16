from __future__ import annotations

import argparse

from ...args import SpeculatorArgumentParserBase
from .config import MTPSpeculatorConfig


class MTPArgumentParser(SpeculatorArgumentParserBase):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--speculative-num-draft-tokens",
            type=int,
            default=MTPSpeculatorConfig.num_draft_tokens,
            help="Number of MTP draft tokens per verify round.",
        )
        parser.add_argument(
            "--speculator-window-size",
            type=int,
            default=MTPSpeculatorConfig.window_size,
            help="MTP attention window size.",
        )

    @staticmethod
    def make_config(kwargs: dict) -> MTPSpeculatorConfig:
        return MTPSpeculatorConfig(
            num_draft_tokens=kwargs.pop("speculative_num_draft_tokens"),
            window_size=kwargs.pop("speculator_window_size"),
        )
