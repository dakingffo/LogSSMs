from __future__ import annotations

import sys
import os
import signal
import time
import multiprocessing as mp
from dataclasses import replace

from logssms.distributed import DistributedInfo
from logssms.utils import init_logger
from logssms.scheduler import schedule_worker
from logssms.speculator import make_local_data_plane_pair, speculator_worker
from logssms.tokenizer import tokenize_worker, detokenize_worker


def _join_workers(workers: list[mp.Process], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    for worker in workers:
        worker.join(timeout=max(0.0, deadline - time.monotonic()))


def _stop_workers(workers, shutdown_event, logger, timeout: float = 60.0) -> None:
    workers = [worker for worker in workers if worker.pid is not None]
    if shutdown_event is not None:
        shutdown_event.set()

    # Keep the speculator alive while schedulers finish collectives and dump profiles.
    schedulers = [worker for worker in workers if worker.name.endswith("-scheduler")]
    _join_workers(schedulers, timeout)

    for worker in workers:
        if worker.is_alive() and worker not in schedulers:
            try:
                os.kill(worker.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    _join_workers(workers, 5.0)

    for worker in workers:
        if worker.is_alive():
            logger.warning("Worker %s did not stop; terminating pid=%s", worker.name, worker.pid)
            worker.terminate()
    _join_workers(workers, 5.0)

    for worker in workers:
        if worker.is_alive():
            logger.warning("Worker %s ignored termination; killing pid=%s", worker.name, worker.pid)
            worker.kill()
    _join_workers(workers, 5.0)

    for worker in workers:
        logger.info("Worker %s exited with code %s", worker.name, worker.exitcode)


def launch_server() -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args = parse_args(sys.argv[1:])
    logger = init_logger(__name__, "initializer")
    workers: list[mp.Process] = []
    ack_queue = None
    scheduler_shutdown_event = None

    def start_worker(**kwargs) -> None:
        worker = mp.Process(**kwargs)
        workers.append(worker)
        worker.start()

    def start_subprocess() -> None:
        nonlocal ack_queue, scheduler_shutdown_event
        mp.set_start_method("spawn", force=True)

        world_size = server_args.tp_info.size
        ack_queue = mp.Queue()
        scheduler_shutdown_event = mp.Event()
        enable_speculator_worker = server_args.speculative_algorithm is not None
        speculator_start_event = (
            mp.Event()
            if enable_speculator_worker and not server_args.dt_separation else None
        )
        speculator_ready_event = (
            mp.Event()
            if enable_speculator_worker and not server_args.dt_separation else None
        )
        if enable_speculator_worker and not server_args.dt_separation:
            target_data_plane, speculator_data_plane = make_local_data_plane_pair(
                server_args
            )
        else:
            target_data_plane, speculator_data_plane = None, None

        for i in range(world_size):
            args = replace(
                server_args,
                tp_info=DistributedInfo(i, world_size),
            )
            kwargs: dict = {
                "args": args,
                "ack_queue": ack_queue,
                "speculator_start_event": speculator_start_event,
                "speculator_ready_event": speculator_ready_event,
                "speculator_data_plane": target_data_plane if i == 0 else None,
                "shutdown_event": scheduler_shutdown_event,
            }
            start_worker(
                target=schedule_worker,
                kwargs=kwargs,
                daemon=False,
                name=f"logssms-TP{i}-scheduler",
            )

        if enable_speculator_worker:
            start_worker(
                target=speculator_worker,
                kwargs={
                    "args": server_args,
                    "speculator_start_event": speculator_start_event,
                    "speculator_ready_event": speculator_ready_event,
                    "speculator_data_plane": speculator_data_plane,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name="logssms-speculator",
            )

        for i in range(server_args.num_tokenizer):
            start_worker(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "tokenizer_addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "local_bs": 1,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"logssms-tokenizer-{i}",
            )

        start_worker(
            target=detokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "detokenizer_addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "detokenizer_id": 0,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="logssms-detokenizer-0",
        )

        num_workers = server_args.num_tokenizer + 2 + int(enable_speculator_worker)
        for _ in range(num_workers):
            logger.info(ack_queue.get())

    try:
        run_api_server(server_args, start_subprocess)
    finally:
        # A second terminal Ctrl-C must not interrupt child reaping halfway through.
        previous_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            _stop_workers(workers, scheduler_shutdown_event, logger)
            if ack_queue is not None:
                ack_queue.close()
                ack_queue.join_thread()
        finally:
            signal.signal(signal.SIGINT, previous_handler)


if __name__ == "__main__":
    launch_server()
