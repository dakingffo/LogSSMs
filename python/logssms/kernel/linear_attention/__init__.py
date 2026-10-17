from dataclasses import dataclass
from typing import Callable

from logssms.utils import Registry

from .conv_copy import copy_conv_snapshots
from .logstate import (
    restore_log_state_chains,
    store_log_states,
)


@dataclass
class LogStateKernels:

    restore  : Callable
    copy_conv: Callable


SUPPORTED_LOG_STATE_KERNELS = Registry[LogStateKernels]("LogState kernels")
AFFINE_LOG_STATE_KERNELS = LogStateKernels(
    restore=restore_log_state_chains,
    copy_conv=copy_conv_snapshots,
)
for recurrent_type in ("gdn", "mamba2"):
    SUPPORTED_LOG_STATE_KERNELS.register(recurrent_type)(AFFINE_LOG_STATE_KERNELS)


__all__ = [
    "restore_log_state_chains",
    "store_log_states",
    "LogStateKernels",
    "SUPPORTED_LOG_STATE_KERNELS",
]
