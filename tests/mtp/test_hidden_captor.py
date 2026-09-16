from __future__ import annotations

from types import SimpleNamespace

import torch

from logssms.core import Batch, clear_global_ctx, set_global_ctx
from logssms.speculator.hidden_captor import (
    HiddenCaptorBase,
    HiddenCapturePoint,
    with_speculator,
)


class RecordingCaptor(HiddenCaptorBase):
    def __init__(self) -> None:
        self.events: list[tuple[HiddenCapturePoint, int | None, tuple, dict]] = []

    def capture(
        self,
        point   : HiddenCapturePoint,
        layer_id: int | None,
        *args,
        **kwargs,
    ) -> None:
        self.events.append((point, layer_id, args, kwargs))

    def select(self, index) -> RecordingCaptor:
        raise NotImplementedError

    def get_carry_hidden(self, index) -> torch.Tensor:
        raise NotImplementedError


class Decoder:
    _layer_id = 7

    @with_speculator(
        HiddenCapturePoint.DECODER_INPUT,
        layer_id_field="_layer_id",
    )
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x + 1, x + 2


class LMHead:
    @with_speculator(HiddenCapturePoint.LM_HEAD_INPUT)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=-1)


def test_decorators_route_through_batch_hidden_captor() -> None:
    batch = Batch([], "verify")
    captor = RecordingCaptor()
    batch.hidden_captor = captor
    set_global_ctx(SimpleNamespace(batch=batch))  # type: ignore[arg-type]
    try:
        decoder_output = Decoder().forward(torch.zeros(2, 3))
        LMHead().forward(decoder_output[0])
    finally:
        clear_global_ctx()

    assert len(captor.events) == 2
    assert captor.events[0][0] is HiddenCapturePoint.DECODER_INPUT
    assert captor.events[0][1] == 7
    assert captor.events[0][2][0].shape == (2, 3)
    assert captor.events[1][0] is HiddenCapturePoint.LM_HEAD_INPUT
    assert captor.events[1][1] is None
    assert captor.events[1][2][0] is decoder_output[0]
