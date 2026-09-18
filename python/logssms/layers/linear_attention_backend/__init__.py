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


def make_linear_attention_backend(cache_type: str) -> BaseLinearAttentionBackend:
    if cache_type == "logstate":
        return LogStateLinearAttentionBackend()
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
