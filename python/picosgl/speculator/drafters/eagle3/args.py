from __future__ import annotations

import argparse

from ...args import SpeculatorArgumentParserBase
from .config import Eagle3SpeculatorConfig


class Eagle3ArgumentParser(SpeculatorArgumentParserBase):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--speculative-num-draft-tokens",
            type=int,
            default=Eagle3SpeculatorConfig.num_draft_tokens,
        )
        parser.add_argument(
            "--speculator-window-size",
            type=int,
            default=Eagle3SpeculatorConfig.window_size,
        )
        parser.add_argument(
            "--eagle3-target-layer-ids",
            type=lambda value: tuple(int(item) for item in value.split(",")),
            default=Eagle3SpeculatorConfig.target_layer_ids,
            metavar="LOW,MID,HIGH",
        )

    @staticmethod
    def make_config(kwargs: dict) -> Eagle3SpeculatorConfig:
        target_layer_ids = kwargs.pop("eagle3_target_layer_ids")
        if target_layer_ids is not None and len(target_layer_ids) != 3:
            raise ValueError("--eagle3-target-layer-ids requires exactly three layers")
        return Eagle3SpeculatorConfig(
            num_draft_tokens=kwargs.pop("speculative_num_draft_tokens"),
            window_size=kwargs.pop("speculator_window_size"),
            target_layer_ids=target_layer_ids,
        )


__all__ = ["Eagle3ArgumentParser"]
