from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Callable, ParamSpec, Self, TypeVar

import torch


P = ParamSpec("P")
R = TypeVar("R")


class HiddenCapturePoint(Enum):
    DECODER_INPUT = auto()
    LM_HEAD_INPUT = auto()


class HiddenCaptorBase(ABC):
    @abstractmethod
    def capture(
        self,
        point   : HiddenCapturePoint,
        layer_id: int | None,
        *args   : Any,
        **kwargs: Any,
    ) -> None: ...

    @abstractmethod
    def select(self, index) -> Self: ...

    @abstractmethod
    def get_carry_hidden(self, index) -> torch.Tensor: ...


def with_speculator(
    point         : HiddenCapturePoint,
    layer_id_field: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    if point is HiddenCapturePoint.DECODER_INPUT:
        assert layer_id_field is not None
    else:
        assert layer_id_field is None

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> R:
            from logssms.core import get_global_ctx

            batch = get_global_ctx().batch
            captor = batch.hidden_captor

            if captor is not None:
                layer_id = (
                    getattr(self, layer_id_field)
                    if layer_id_field is not None else None
                )
                captor.capture(point, layer_id, *args, **kwargs)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


__all__ = [
    "HiddenCapturePoint",
    "HiddenCaptorBase",
    "with_speculator",
]
