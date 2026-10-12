from __future__ import annotations

from dataclasses import dataclass

from ..utils import serialize_type


class BaseSpeculatorMsg:

    def encoder(self) -> dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: dict) -> BaseSpeculatorMsg:
        from . import decode_speculator_message

        return decode_speculator_message(json)


@dataclass
class SpeculatorHandshakeMsg(BaseSpeculatorMsg):
    connection_id  : bytes
    max_hidden_rows: int
    hidden_size    : int
    max_prob_rows  : int
    vocab_size     : int


@dataclass
class SpeculatorHandshakeAckMsg(BaseSpeculatorMsg):
    pass


@dataclass
class SpeculatorInitMsg(BaseSpeculatorMsg):
    input_rows: int


@dataclass
class SpeculatorStepReq(BaseSpeculatorMsg):
    uid             : int
    n_drafts        : int
    append_positions: list[int]
    append_tokens   : list[int]
    sampling        : bool


@dataclass
class SpeculatorStepMsg(BaseSpeculatorMsg):
    reqs       : list[SpeculatorStepReq]
    input_rows : int
    output_rows: int


@dataclass
class SpeculatorReply(BaseSpeculatorMsg):
    uid         : int
    draft_tokens: list[int]


@dataclass
class SpeculatorReplyMsg(BaseSpeculatorMsg):
    reqs: list[SpeculatorReply]


@dataclass
class SpeculatorRemoveMsg(BaseSpeculatorMsg):
    uid: int
