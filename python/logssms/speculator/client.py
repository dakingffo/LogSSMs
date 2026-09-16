from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from logssms.message import (
    BaseSpeculatorMsg,
    SpeculatorHandshakeAckMsg,
    SpeculatorHandshakeMsg,
    SpeculatorInitMsg,
    SpeculatorRemoveMsg,
    SpeculatorReplyMsg,
    SpeculatorStepMsg,
)
from logssms.message.queue import ZmqPullQueue, ZmqPushQueue

from .data_plane import DataPlane, DataPlaneSizes

if TYPE_CHECKING:
    from logssms.scheduler.config import SchedulerConfig
    from logssms.scheduler.io import SchedulerIOMixin


class SpeculatorClientBase(ABC):
    def __init__(
        self,
        config      : SchedulerConfig,
        device      : torch.device,
        vocab_size  : int,
        hidden_size : int,
    ):
        self.config = config
        self.device = device
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    @abstractmethod
    def init(
        self,
        msg   : SpeculatorInitMsg,
        hidden: torch.Tensor,
    ) -> None:
        """Seed the drafter's per-request state with a prefill terminal window."""
        ...

    @abstractmethod
    def step(
        self,
        msg   : SpeculatorStepMsg,
        tensor: torch.Tensor | None,
    ) -> tuple[SpeculatorReplyMsg, torch.Tensor | None]:
        """Blocking draft round. Returns the reply and the drafter's draft_probs rows
        (None if no request sampled this round)."""
        ...

    @abstractmethod
    def remove(self, uid: int) -> None:
        """Drop the drafter's per-request state (finish / abort)."""
        ...

    @abstractmethod
    def destroy(self) -> None:
        ...


class MainSpeculatorClient(SpeculatorClientBase):
    def __init__(
        self,
        config      : SchedulerConfig,
        device      : torch.device,
        vocab_size  : int,
        hidden_size : int,
        handshake   : SpeculatorHandshakeMsg,
        scheduler_io: SchedulerIOMixin | None = None,
        data_plane   : DataPlane | None        = None,
    ):
        super().__init__(config, device, vocab_size, hidden_size)
        self._io = scheduler_io
        assert data_plane is not None
        self.data_plane = data_plane
        self.sender = ZmqPushQueue(
            config.zmq_drafter_addr, create=True, encoder=BaseSpeculatorMsg.encoder
        )
        self.receiver = ZmqPullQueue(
            config.zmq_drafter_reply_addr,
            create=True,
            decoder=BaseSpeculatorMsg.decoder,
        )

        sizes = DataPlaneSizes(
            handshake.max_hidden_rows,
            handshake.hidden_size,
            handshake.max_prob_rows,
            handshake.vocab_size,
        )
        # Connection metadata must go out before transport initialization, which blocks
        # until the worker consumes the same handshake.
        self.sender.put(handshake)
        self.data_plane.init_rank0(handshake.connection_id, sizes)
        ack = self.receiver.get()
        assert isinstance(ack, SpeculatorHandshakeAckMsg), (
            f"expected SpeculatorHandshakeAckMsg, got {ack!r}"
        )

    def init(
        self,
        msg   : SpeculatorInitMsg,
        hidden: torch.Tensor,
    ) -> None:
        self.sender.put(msg)
        self.data_plane.send_hidden(hidden)

    def step(
        self,
        msg   : SpeculatorStepMsg,
        tensor: torch.Tensor | None,
    ) -> tuple[SpeculatorReplyMsg, torch.Tensor | None]:
        self.sender.put(msg)
        if tensor is not None:
            assert tensor.shape[0] == msg.input_rows
            self.data_plane.send_hidden(tensor)
        else:
            assert msg.input_rows == 0
        probs: torch.Tensor | None = None
        if msg.output_rows > 0:
            probs = self.data_plane.recv_probs(msg.output_rows)
        reply = self.receiver.get()
        if self._io is not None and self.config.tp_info.size > 1:
            self._io._send_draft_to_ranks(reply, probs)
        return reply, probs

    def remove(self, uid: int) -> None:
        self.sender.put(SpeculatorRemoveMsg(uid=uid))

    def destroy(self) -> None:
        self.data_plane.destroy()
        self.sender.stop()
        self.receiver.stop()


class BroadcastSpeculatorClient(SpeculatorClientBase):
    """Non-primary-rank stand-in for ``SpeculatorClientBase``.

    Only rank0 has the ZMQ/data-plane client; other TP ranks receive the draft results rank0
    broadcasts (``scheduler.io``) and present the same ``step`` interface so
    ``VerifyManager`` is rank-agnostic. ``init``/``remove`` are no-ops — no drafter state
    lives on non-primary ranks.
    """

    def __init__(
        self,
        config      : SchedulerConfig,
        device      : torch.device,
        vocab_size  : int,
        hidden_size : int,
        scheduler_io: SchedulerIOMixin | None = None
    ):
        super().__init__(config, device, vocab_size, hidden_size)
        self._io = scheduler_io

    def step(
        self,
        msg   : SpeculatorStepMsg,
        tensor: torch.Tensor | None,
    ) -> tuple[SpeculatorReplyMsg, torch.Tensor | None]:
        return self._io._recv_draft_from_rank0(self.vocab_size)

    def init(self, *args, **kwargs) -> None:
        pass

    def remove(self, uid: int) -> None:
        pass

    def destroy(self) -> None:
        pass
