from __future__ import annotations

import multiprocessing as mp
import platform
from typing import TYPE_CHECKING

import torch

from picosgl.distributed import DistributedInfo, tp_override
from picosgl.message import (
    BaseSpeculatorMsg,
    make_handshake_message,
)
from picosgl.message.queue import ZmqPullQueue, ZmqPushQueue
from picosgl.utils import Registry, init_logger

from .base import DraftManagerBase, EngineBase
from .data_plane import (
    CUDAIPCDataPlane,
    DataPlane,
    NCCLDataPlane,
    SharedMemoryDataPlane,
    make_data_plane_sizes,
)
from .drafters.mtp import MTPDraftManager, MTPEngine
from .drafters.eagle3 import Eagle3DraftManager, Eagle3Engine
from .drafters.dflash import DFlashDraftManager, DFlashEngine
from .runner import SpeculatorRunner
from .client import (
    BroadcastSpeculatorClient,
    MainSpeculatorClient,
    SpeculatorClientBase,
)

if TYPE_CHECKING:
    from picosgl.scheduler.config import SchedulerConfig
    from picosgl.scheduler.scheduler import Scheduler


SUPPORTED_DRAFT_MANAGER = Registry[type[DraftManagerBase]]("Draft Manager")
SUPPORTED_DRAFTER_ENGINE = Registry[type[EngineBase]]("Drafter Engine")
SUPPORTED_DRAFT_MANAGER.register("MTP")(MTPDraftManager)
SUPPORTED_DRAFTER_ENGINE.register("MTP")(MTPEngine)
SUPPORTED_DRAFT_MANAGER.register("EAGLE3")(Eagle3DraftManager)
SUPPORTED_DRAFTER_ENGINE.register("EAGLE3")(Eagle3Engine)
SUPPORTED_DRAFT_MANAGER.register("DFLASH")(DFlashDraftManager)
SUPPORTED_DRAFTER_ENGINE.register("DFLASH")(DFlashEngine)
logger = init_logger(__name__)


def make_drafter_engine(
    device: torch.device,
    config: SchedulerConfig,
) -> EngineBase:
    algorithm = config.speculative_algorithm
    assert algorithm is not None
    return SUPPORTED_DRAFTER_ENGINE[algorithm].from_config(device, config)


def make_draft_manager(
    engine: EngineBase,
    config: SchedulerConfig,
) -> DraftManagerBase:
    algorithm = config.speculative_algorithm
    assert algorithm is not None
    return SUPPORTED_DRAFT_MANAGER[algorithm](engine)


def make_speculator_data_plane(
    config          : SchedulerConfig,
    device          : torch.device,
    rank            : int,
    local_data_plane: DataPlane | None,
) -> NCCLDataPlane | CUDAIPCDataPlane | SharedMemoryDataPlane:
    if config.dt_separation:
        return NCCLDataPlane(device, rank=rank, dtype=config.dtype)
    else:
        assert local_data_plane is not None
        assert local_data_plane.rank == rank
        assert local_data_plane.device == device
        return local_data_plane


def make_local_data_plane_pair(
    config    : SchedulerConfig,
    mp_context: mp.context.BaseContext | None = None,
) -> tuple[CUDAIPCDataPlane, CUDAIPCDataPlane] | tuple[
    SharedMemoryDataPlane, SharedMemoryDataPlane
]:
    data_plane_cls = (
        SharedMemoryDataPlane
        if "microsoft" in platform.release().lower()
        else CUDAIPCDataPlane
    )
    if data_plane_cls is SharedMemoryDataPlane:
        logger.warning("Using the shared-memory compatibility data plane under WSL")
    return data_plane_cls.make_pair(
        torch.device("cuda:0"), config.dtype, mp_context
    )


# The speculator always owns an independent process. DT separation only selects whether
# that process shares Target rank 0's device or runs on the next device.
@torch.inference_mode()
def speculator_worker(
    *,
    args                  : SchedulerConfig,
    speculator_start_event: mp.synchronize.Event | None = None,
    speculator_ready_event: mp.synchronize.Event | None = None,
    speculator_data_plane : DataPlane | None = None,
    ack_queue             : mp.Queue[str] | None = None,
) -> None:
    # In shared-device mode Target rank 0 must capture its initial free-memory baseline
    # before this process creates a CUDA context or allocates the drafter.
    if not args.dt_separation:
        assert speculator_start_event is not None
        speculator_start_event.wait()

    device = torch.device(
        f"cuda:{args.tp_info.size}" if args.dt_separation else "cuda:0"
    )
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(stream)

    # the drafter is always tp=1 (standalone rank 0): the split controls the DEVICE it
    # runs on (--enable-dt-separation), not the drafter's TP sharding. Layer builders
    # read the global TP info, which the main engine sets for itself — this process must
    # set it before constructing the model.
    with tp_override(DistributedInfo(0, 1)):
        speculator_config = args.speculator_config
        assert speculator_config is not None

        def on_engine_ready() -> None:
            if not args.dt_separation:
                assert speculator_ready_event is not None
                speculator_ready_event.set()
            if ack_queue is not None:
                ack_queue.put("Speculator is ready")

        runner = SpeculatorRunner(
            engine_factory=lambda: make_drafter_engine(device, args),
            manager_factory=lambda engine: make_draft_manager(engine, args),
            data_plane=make_speculator_data_plane(
                args, device, rank=1,
                local_data_plane=speculator_data_plane,
            ),
            recv=ZmqPullQueue(
                args.zmq_drafter_addr,
                create=False,
                decoder=BaseSpeculatorMsg.decoder,
            ),
            reply=ZmqPushQueue(
                args.zmq_drafter_reply_addr,
                create=False,
                encoder=BaseSpeculatorMsg.encoder,
            ),
            on_engine_ready=on_engine_ready,
            data_plane_sizes=make_data_plane_sizes(
                args,
                args.model_config.hidden_size,
                args.model_config.vocab_size,
            ),
        )
        runner.run_forever()


# called by scheduler
def make_speculator_client(
    sche: Scheduler, config: SchedulerConfig
) -> SpeculatorClientBase:
    mc = config.model_config
    if not config.tp_info.is_primary():
        return BroadcastSpeculatorClient(
            config, sche.device, mc.vocab_size, mc.hidden_size,
            scheduler_io=sche
        )
    speculator_config = config.speculator_config
    algorithm = config.speculative_algorithm
    assert speculator_config is not None
    assert algorithm is not None
    data_plane = make_speculator_data_plane(
        config, sche.device, rank=0,
        local_data_plane=sche.speculator_data_plane,
    )
    sizes = make_data_plane_sizes(
        config,
        mc.hidden_size,
        mc.vocab_size,
    )
    connection_id = data_plane.make_connection_id()
    handshake = make_handshake_message(
        algorithm,
        speculator_config,
        connection_id,
        sizes.max_hidden_rows,
        sizes.hidden_size,
        sizes.max_prob_rows,
        sizes.vocab_size,
    )
    return MainSpeculatorClient(
        config, sche.device, mc.vocab_size, mc.hidden_size,
        handshake=handshake,
        scheduler_io=sche,
        data_plane=data_plane,
    )
