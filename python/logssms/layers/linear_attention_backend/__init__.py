from .base import BaseLinearAttentionBackend, BaseLinearAttentionMetadata
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


def make_linear_attention_backend(
    cache_type: str, *, enable_eager_state: bool = False,
) -> BaseLinearAttentionBackend:
    if cache_type == "logstate":
        return LogStateLinearAttentionBackend(enable_eager_state=enable_eager_state)
    return LinearAttentionBackend()

__all__ = [
    "BaseLinearAttentionBackend",
    "BaseLinearAttentionMetadata",
    "LinearAttentionBackend",
    "LinearAttentionCaptureData",
    "LinearAttentionMetadata",
    "LogStateLinearAttentionBackend",
    "LogStateLinearAttentionCaptureData",
    "LogStateLinearAttentionMetadata",
    "make_linear_attention_backend",
]
