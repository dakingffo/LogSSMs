from __future__ import annotations

from collections.abc import Callable

from logssms.message import (
    SpeculatorHandshakeMsg,
    SpeculatorInitMsg,
    SpeculatorRemoveMsg,
    SpeculatorStepMsg,
)
from logssms.message.queue import ZmqPullQueue, ZmqPushQueue
from logssms.utils import init_logger

from .base import DraftManagerBase, EngineBase
from .data_plane import DataPlane, DataPlaneSizes

logger = init_logger(__name__)


class SpeculatorRunner:
    def __init__(
        self,
        engine_factory  : Callable[[], EngineBase],
        manager_factory : Callable[[EngineBase], DraftManagerBase],
        data_plane      : DataPlane,
        recv            : ZmqPullQueue,
        reply           : ZmqPushQueue,
        on_engine_ready : Callable[[], None] | None = None,
        data_plane_sizes: DataPlaneSizes | None     = None,
    ) -> None:
        self.engine_factory = engine_factory
        self.manager_factory = manager_factory
        self.engine: EngineBase | None = None
        self.manager: DraftManagerBase | None = None
        self.data_plane = data_plane
        self.recv = recv
        self.reply = reply
        self.on_engine_ready = on_engine_ready
        self.data_plane_sizes = data_plane_sizes

    def run_forever(self) -> None:
        self.engine = self.engine_factory()
        self.manager = self.manager_factory(self.engine)
        if self.data_plane_sizes is not None:
            self.data_plane.prepare_rank1(self.data_plane_sizes)
        if self.on_engine_ready is not None:
            self.on_engine_ready()

        handshake = self.recv.get()
        assert isinstance(handshake, SpeculatorHandshakeMsg), (
            f"expected SpeculatorHandshakeMsg, got {handshake!r}"
        )
        sizes = DataPlaneSizes(
            handshake.max_hidden_rows,
            handshake.hidden_size,
            handshake.max_prob_rows,
            handshake.vocab_size,
        )
        if self.data_plane_sizes is not None:
            assert sizes == self.data_plane_sizes
        self.data_plane.init_rank1(handshake.connection_id, sizes)
        self.reply.put(self.manager.handshake(handshake))
        logger.info(
            "Speculator ready: device=%s", self.engine.device
        )

        try:
            while True:
                msg = self.recv.get()
                if isinstance(msg, SpeculatorInitMsg):
                    tensor = self.data_plane.recv_hidden(msg.input_rows)
                    self.manager.init(msg, tensor)
                elif isinstance(msg, SpeculatorStepMsg):
                    tensor = (
                        self.data_plane.recv_hidden(msg.input_rows)
                        if msg.input_rows > 0 else None
                    )
                    reply, output = self.manager.step(msg, tensor)
                    if output is not None:
                        assert output.shape[0] == msg.output_rows
                        self.data_plane.send_probs(output)
                    else:
                        assert msg.output_rows == 0
                    self.reply.put(reply)
                elif isinstance(msg, SpeculatorRemoveMsg):
                    self.manager.remove(msg.uid)
                else:
                    raise NotImplementedError(
                        f"unknown speculator message: {msg!r}"
                    )
        except KeyboardInterrupt:
            pass
