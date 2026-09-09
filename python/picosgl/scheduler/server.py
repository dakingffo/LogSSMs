from __future__ import annotations

import logging
import multiprocessing as mp
import signal
from typing import TYPE_CHECKING

import torch

from picosgl.utils import init_logger

from .config import SchedulerConfig
from .scheduler import Scheduler

if TYPE_CHECKING:
    from picosgl.speculator.data_plane import DataPlane


@torch.inference_mode()
def schedule_worker(
    args                  : SchedulerConfig,
    ack_queue             : mp.Queue[str],
    speculator_start_event: mp.synchronize.Event | None = None,
    speculator_ready_event: mp.synchronize.Event | None = None,
    speculator_data_plane : DataPlane | None = None,
    shutdown_event         : mp.synchronize.Event | None = None,
) -> None:
    if shutdown_event is not None:
        signal.signal(signal.SIGINT, lambda *_: shutdown_event.set())

    scheduler = Scheduler(
        args, speculator_start_event, speculator_ready_event,
        speculator_data_plane, shutdown_event,
    )
    scheduler.sync_all_ranks()

    if args.tp_info.is_primary():
        ack_queue.put("Scheduler is ready")

    if args.silent_output:
        logging.disable(logging.INFO)

    try:
        scheduler.run_forever()
    except KeyboardInterrupt:
        logger = init_logger(__name__)
        if args.tp_info.is_primary():
            logger.info("\nScheduler exiting gracefully...")
        scheduler.shutdown()
