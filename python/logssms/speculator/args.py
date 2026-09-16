from __future__ import annotations

import argparse
from abc import ABC, abstractmethod

from .base import BaseSpeculatorConfig


class SpeculatorArgumentParserBase(ABC):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(add_help=False)
        self.add_arguments(self.parser)

    @staticmethod
    @abstractmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None: ...

    @staticmethod
    @abstractmethod
    def make_config(kwargs: dict) -> BaseSpeculatorConfig: ...
