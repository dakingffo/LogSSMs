from __future__ import annotations

from dataclasses import dataclass

import torch

from logssms.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg:
    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: list[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    uid              : int
    next_token       : int | list[int]
    finished         : bool
    return_token_ids: bool = False
    cached_tokens   : int | None = None


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid            : int
    text           : str | list[dict[str, str]] | None
    sampling_params: SamplingParams
    profile_tag     : str | None       = None
    input_ids       : torch.Tensor | None = None
    return_token_ids: bool             = False


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int
