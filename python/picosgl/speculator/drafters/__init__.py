from picosgl.utils import Registry

from ..args import SpeculatorArgumentParserBase
from ..base import BaseSpeculatorConfig
from ..hidden_captor import HiddenCaptorBase
from .dflash.args import DFlashArgumentParser as _DFlashArgumentParser
from .dflash.hidden_captor import DFlashHiddenCaptor as _DFlashHiddenCaptor
from .eagle3.args import Eagle3ArgumentParser as _Eagle3ArgumentParser
from .eagle3.hidden_captor import Eagle3HiddenCaptor as _Eagle3HiddenCaptor
from .mtp.args import MTPArgumentParser as _MTPArgumentParser
from .mtp.hidden_captor import MTPHiddenCaptor as _MTPHiddenCaptor


SUPPORTED_SPECULATOR_ARGUMENT_PARSERS = Registry[
    type[SpeculatorArgumentParserBase]
]("Speculator Argument Parser")
SUPPORTED_SPECULATOR_ARGUMENT_PARSERS.register("MTP")(_MTPArgumentParser)
SUPPORTED_SPECULATOR_ARGUMENT_PARSERS.register("EAGLE3")(_Eagle3ArgumentParser)
SUPPORTED_SPECULATOR_ARGUMENT_PARSERS.register("DFLASH")(_DFlashArgumentParser)
SUPPORTED_HIDDEN_CAPTORS = Registry[type[HiddenCaptorBase]]("Hidden Captor")
SUPPORTED_HIDDEN_CAPTORS.register("MTP")(_MTPHiddenCaptor)
SUPPORTED_HIDDEN_CAPTORS.register("EAGLE3")(_Eagle3HiddenCaptor)
SUPPORTED_HIDDEN_CAPTORS.register("DFLASH")(_DFlashHiddenCaptor)


def make_speculator_argument_parser(
    algorithm: str,
) -> SpeculatorArgumentParserBase:
    return SUPPORTED_SPECULATOR_ARGUMENT_PARSERS[algorithm]()


def make_hidden_captor(
    algorithm: str,
    config   : BaseSpeculatorConfig,
) -> HiddenCaptorBase:
    return SUPPORTED_HIDDEN_CAPTORS[algorithm](config)


__all__ = [
    "SUPPORTED_SPECULATOR_ARGUMENT_PARSERS",
    "SUPPORTED_HIDDEN_CAPTORS",
    "make_hidden_captor",
    "make_speculator_argument_parser",
]
