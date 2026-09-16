from importlib import import_module
from typing import TYPE_CHECKING, Any

from .base import (
    BaseSpeculatorConfig,
    DraftManagerBase,
    DraftState,
    EngineBase,
    SpeculatorReserve,
)
from .hidden_captor import HiddenCaptorBase, HiddenCapturePoint, with_speculator

if TYPE_CHECKING:
    from .client import (
        BroadcastSpeculatorClient,
        MainSpeculatorClient,
        SpeculatorClientBase,
    )
    from .data_plane import (
        CUDAIPCDataPlane,
        DataPlane,
        DataPlaneSizes,
        NCCLDataPlane,
        SharedMemoryDataPlane,
        make_data_plane_sizes,
    )
    from .runner import SpeculatorRunner
    from .server import (
        make_drafter_engine,
        make_draft_manager,
        make_local_data_plane_pair,
        make_speculator_client,
        speculator_worker,
    )


_LAZY_IMPORTS = {
    "BroadcastSpeculatorClient": (".client", "BroadcastSpeculatorClient"),
    "MainSpeculatorClient"     : (".client", "MainSpeculatorClient"),
    "SpeculatorClientBase"     : (".client", "SpeculatorClientBase"),
    "CUDAIPCDataPlane"         : (".data_plane", "CUDAIPCDataPlane"),
    "DataPlane"                : (".data_plane", "DataPlane"),
    "DataPlaneSizes"           : (".data_plane", "DataPlaneSizes"),
    "NCCLDataPlane"            : (".data_plane", "NCCLDataPlane"),
    "SharedMemoryDataPlane"    : (".data_plane", "SharedMemoryDataPlane"),
    "make_data_plane_sizes"    : (".data_plane", "make_data_plane_sizes"),
    "SpeculatorRunner"         : (".runner", "SpeculatorRunner"),
    "make_drafter_engine"      : (".server", "make_drafter_engine"),
    "make_draft_manager"       : (".server", "make_draft_manager"),
    "make_local_data_plane_pair": (".server", "make_local_data_plane_pair"),
    "make_speculator_client"   : (".server", "make_speculator_client"),
    "speculator_worker"        : (".server", "speculator_worker"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value

__all__ = [
    "DraftState",
    "DraftManagerBase",
    "EngineBase",
    "BaseSpeculatorConfig",
    "SpeculatorReserve",
    "HiddenCaptorBase",
    "HiddenCapturePoint",
    "with_speculator",
    "SpeculatorClientBase",
    "MainSpeculatorClient",
    "BroadcastSpeculatorClient",
    "DataPlane",
    "DataPlaneSizes",
    "NCCLDataPlane",
    "CUDAIPCDataPlane",
    "SharedMemoryDataPlane",
    "make_data_plane_sizes",
    "SpeculatorRunner",
    "make_drafter_engine",
    "make_draft_manager",
    "make_local_data_plane_pair",
    "speculator_worker",
    "make_speculator_client",
]
