from logssms.utils import Registry

from .base import BaseGatedDeltaBackend, BaseLinearAttentionBackend, BaseLinearAttentionMetadata
from .native import (
    LinearAttentionBackend,
    LinearAttentionCaptureData,
    LinearAttentionMetadata,
)
from .logstate import (
    LogStateLinearAttentionBackend,
    LogStateLinearAttentionCaptureData,
    LogStateLinearAttentionMetadata,
)


SUPPORTED_LINEAR_ATTENTION_BACKENDS = Registry[type[BaseLinearAttentionBackend]]("linear attention backend")
for cache_type in ("naive", "radix"):
    SUPPORTED_LINEAR_ATTENTION_BACKENDS.register(f"gdn/{cache_type}")(LinearAttentionBackend)
SUPPORTED_LINEAR_ATTENTION_BACKENDS.register("gdn/logstate")(LogStateLinearAttentionBackend)


def make_linear_attention_backend(
    cache_type: str, *, recurrent_type: str = "gdn", enable_eager_state: bool = False,
) -> BaseLinearAttentionBackend:
    backend = SUPPORTED_LINEAR_ATTENTION_BACKENDS[f"{recurrent_type}/{cache_type}"]
    return backend(enable_eager_state=enable_eager_state) if cache_type == "logstate" else backend()

__all__ = [
    "BaseLinearAttentionBackend",
    "BaseGatedDeltaBackend",
    "SUPPORTED_LINEAR_ATTENTION_BACKENDS",
    "BaseLinearAttentionMetadata",
    "LinearAttentionBackend",
    "LinearAttentionCaptureData",
    "LinearAttentionMetadata",
    "LogStateLinearAttentionBackend",
    "LogStateLinearAttentionCaptureData",
    "LogStateLinearAttentionMetadata",
    "make_linear_attention_backend",
]
