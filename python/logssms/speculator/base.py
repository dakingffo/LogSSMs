from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

if TYPE_CHECKING:
    from logssms.engine.config import EngineConfig
    from logssms.message import (
        SpeculatorHandshakeAckMsg,
        SpeculatorHandshakeMsg,
        SpeculatorInitMsg,
        SpeculatorReplyMsg,
        SpeculatorStepMsg,
    )
    from logssms.scheduler.config import SchedulerConfig
    from logssms.models import ModelConfig
    from logssms.models.drafters import BaseDrafterModel


@dataclass(frozen=True)
class SpeculatorReserve:
    num_draft_slots        : int = 0
    draft_slots_per_request: int = 0


class BaseSpeculatorConfig(ABC):
    algorithm       : ClassVar[str]
    num_draft_tokens: int

    @property
    def hidden_size_multiplier(self) -> int:
        return 1

    @property
    def requires_model_resolution(self) -> bool:
        return False

    @property
    @abstractmethod
    def max_init_hidden_rows(self) -> int: ...

    def resolve(self, model_config: ModelConfig) -> BaseSpeculatorConfig:
        return self

    @abstractmethod
    def make_reserve(self, max_running_req: int) -> SpeculatorReserve: ...

    @abstractmethod
    def validate(self, config: EngineConfig) -> None: ...


@dataclass
class DraftState:

    draft_tokens: list[int]           = field(default_factory=list)
    draft_probs : torch.Tensor | None = None


class EngineBase(ABC):

    device          : torch.device
    stream          : torch.cuda.Stream
    num_spec_tokens : int
    vocab_size      : int

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        device: torch.device,
        config: SchedulerConfig,
    ) -> EngineBase:
        pass

    @staticmethod
    @abstractmethod
    def load_drafter(
        device: torch.device, config: EngineConfig
    ) -> BaseDrafterModel:
        pass

    @abstractmethod
    def draft(self, states: list[DraftState]) -> None:
        pass

    def destroy(self) -> None:
        pass


class DraftManagerBase(ABC):
    def __init__(self, engine: EngineBase) -> None:
        self.engine = engine

    @abstractmethod
    def handshake(
        self, msg: SpeculatorHandshakeMsg
    ) -> SpeculatorHandshakeAckMsg: ...

    @abstractmethod
    def init(self, msg: SpeculatorInitMsg, tensor: torch.Tensor) -> None: ...

    @abstractmethod
    def step(
        self,
        msg   : SpeculatorStepMsg,
        tensor: torch.Tensor | None,
    ) -> tuple[SpeculatorReplyMsg, torch.Tensor | None]: ...

    @abstractmethod
    def remove(self, uid: int) -> None: ...
