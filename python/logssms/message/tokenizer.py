from __future__ import annotations

from dataclasses import dataclass

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
    uid       : int
    next_token: int | list[int] # decode or verify 
    finished  : bool


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid            : int
    text           : str | list[dict[str, str]]
    sampling_params: SamplingParams


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int
