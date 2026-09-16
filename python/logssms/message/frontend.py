from __future__ import annotations

from dataclasses import dataclass

from .utils import deserialize_type, serialize_type


@dataclass
class BaseFrontendMsg:
    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: dict) -> BaseFrontendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    data: list[BaseFrontendMsg]


@dataclass
class UserReply(BaseFrontendMsg):
    uid               : int
    incremental_output: str
    finished          : bool
