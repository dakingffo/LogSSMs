from __future__ import annotations

from typing import Protocol

from logssms.utils import Registry, init_logger

from .base import BaseMoeBackend

logger = init_logger(__name__)


class MoeBackendCreator(Protocol):
    def __call__(self) -> BaseMoeBackend: ...


SUPPORTED_MOE_BACKENDS = Registry[MoeBackendCreator]("MoE Backend")


@SUPPORTED_MOE_BACKENDS.register("fused")
def make_fused_moe_backend():
    from .fused import FusedMoe

    return FusedMoe()


def make_moe_backend(backend: str) -> BaseMoeBackend:
    return SUPPORTED_MOE_BACKENDS[backend]()


__all__ = [
    "BaseMoeBackend",
    "make_moe_backend",
    "SUPPORTED_MOE_BACKENDS",
]
