from __future__ import annotations

from dataclasses import dataclass

import torch

from logssms.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    def encoder(self) -> dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: list[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    uid            : int
    input_ids      : torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int
