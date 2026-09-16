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
    """Polymorphic per-request spec-decode state held by the target-side VerifyManager.

    Only these two fields are visible across the target/drafter boundary; drafter engines
    keep their richer per-request state in subclasses (e.g. ``MTPState``). The VerifyManager
    allocates a ``DraftState`` per table_idx, the drafter fills it in via ``client.step``.
    """

    draft_tokens: list[int]           = field(default_factory=list)
    draft_probs : torch.Tensor | None = None


class EngineBase(ABC):
    """Drafter-side engine: fills ``DraftState.draft_tokens`` / ``draft_probs`` for a batch.

    Lives in the speculator process, drives the standalone drafter model on its own device /
    stream. Drafter runs are blocking on the target side (VerifyManager waits for the reply),
    so ``stream`` is for the engine's own internal overlap, not target-side pipelining.
    """

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
        """Build an algorithm engine and load its standalone drafter model."""

    @staticmethod
    @abstractmethod
    def load_drafter(
        device: torch.device, config: EngineConfig
    ) -> BaseDrafterModel:
        """Load the algorithm's standalone drafter model."""

    @abstractmethod
    def draft(self, states: list[DraftState]) -> None:
        """Fill each state's draft_tokens / draft_probs for this round."""

    def destroy(self) -> None:
        """Release engine-side resources (CUDA graph pools etc.) on shutdown."""
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
