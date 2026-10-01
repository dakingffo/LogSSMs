from dataclasses import dataclass
from typing import Callable

from logssms.utils import Registry

from .conv_copy import copy_framed_conv_snapshots
from .framed_logstate import (
    construct_framed_log_state_chains,
    restore_framed_log_state_chains,
    store_framed_log_states,
)


@dataclass
class LogStateKernels:
    """Operator-specific launches used by cache maintenance, resolved once."""

    restore  : Callable
    construct: Callable
    copy_conv: Callable


SUPPORTED_LOG_STATE_KERNELS = Registry[LogStateKernels]("LogState kernels")
GATED_DELTA_LOG_STATE_KERNELS = LogStateKernels(
    restore=restore_framed_log_state_chains,
    construct=construct_framed_log_state_chains,
    copy_conv=copy_framed_conv_snapshots,
)
SUPPORTED_LOG_STATE_KERNELS.register("gdn")(GATED_DELTA_LOG_STATE_KERNELS)


__all__ = [
    "construct_framed_log_state_chains",
    "restore_framed_log_state_chains",
    "store_framed_log_states",
    "LogStateKernels",
    "SUPPORTED_LOG_STATE_KERNELS",
]
