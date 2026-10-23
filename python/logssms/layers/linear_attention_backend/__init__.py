from logssms.utils import Registry

from .base import BaseGatedDeltaBackend, BaseMamba2Backend, BaseLinearAttnBackend, BaseLinearAttnMetadata
from .standard import (
    LinearAttnCaptureData,
    LinearAttnMetadata,
)
from .logstate import (
    LogStateLinearAttnCaptureData,
    LogStateLinearAttnMetadata,
)
from .gdn import GatedDeltaBackend
from .gdn_logstate import GatedDeltaLogStateBackend
from .mamba2 import Mamba2Backend
from .mamba2_logstate import Mamba2LogStateBackend


SUPPORTED_LINEAR_ATTENTION_BACKENDS = Registry[type[BaseLinearAttnBackend]]("linear attention backend")
for cache_type in ("naive", "radix", "marconi"):
    SUPPORTED_LINEAR_ATTENTION_BACKENDS.register(f"gdn/{cache_type}")(GatedDeltaBackend)
SUPPORTED_LINEAR_ATTENTION_BACKENDS.register("gdn/logstate")(GatedDeltaLogStateBackend)
for cache_type in ("naive", "radix", "marconi"):
    SUPPORTED_LINEAR_ATTENTION_BACKENDS.register(f"mamba2/{cache_type}")(Mamba2Backend)
SUPPORTED_LINEAR_ATTENTION_BACKENDS.register("mamba2/logstate")(Mamba2LogStateBackend)


def make_linear_attention_backend(
    cache_type: str, *, recurrent_type: str = "gdn", enable_eager_state: bool = False,
) -> BaseLinearAttnBackend:
    backend = SUPPORTED_LINEAR_ATTENTION_BACKENDS[f"{recurrent_type}/{cache_type}"]
    return backend(enable_eager_state=enable_eager_state) if cache_type == "logstate" else backend()

__all__ = [
    "BaseLinearAttnBackend",
    "BaseGatedDeltaBackend",
    "BaseMamba2Backend",
    "SUPPORTED_LINEAR_ATTENTION_BACKENDS",
    "BaseLinearAttnMetadata",
    "GatedDeltaBackend",
    "LinearAttnCaptureData",
    "LinearAttnMetadata",
    "GatedDeltaLogStateBackend",
    "Mamba2Backend",
    "Mamba2LogStateBackend",
    "LogStateLinearAttnCaptureData",
    "LogStateLinearAttnMetadata",
    "make_linear_attention_backend",
]
