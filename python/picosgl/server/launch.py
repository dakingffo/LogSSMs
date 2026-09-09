from __future__ import annotations

import sys
import multiprocessing as mp
from dataclasses import replace

from picosgl.distributed import DistributedInfo
from picosgl.utils import init_logger
from picosgl.scheduler import schedule_worker
from picosgl.speculator import make_local_data_plane_pair, speculator_worker
from picosgl.tokenizer import tokenize_worker, detokenize_worker


def launch_server() -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args = parse_args(sys.argv[1:])
    logger = init_logger(__name__, "initializer")

    def start_subprocess() -> None:
        mp.set_start_method("spawn", force=True)

        world_size = server_args.tp_info.size
        ack_queue: mp.Queue[str] = mp.Queue()
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
            mp.Process(
                target=schedule_worker,
                kwargs=kwargs,
                daemon=False,
                name=f"picosgl-TP{i}-scheduler",
            ).start()

        if enable_speculator_worker:
            mp.Process(
                target=speculator_worker,
                kwargs={
                    "args": server_args,
                    "speculator_start_event": speculator_start_event,
                    "speculator_ready_event": speculator_ready_event,
                    "speculator_data_plane": speculator_data_plane,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name="picosgl-speculator",
            ).start()

        for i in range(server_args.num_tokenizer):
            mp.Process(
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
                name=f"picosgl-tokenizer-{i}",
            ).start()

        mp.Process(
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
            name="picosgl-detokenizer-0",
        ).start()

        num_workers = server_args.num_tokenizer + 2 + int(enable_speculator_worker)
        for _ in range(num_workers):
            logger.info(ack_queue.get())

    run_api_server(server_args, start_subprocess)


if __name__ == "__main__":
    launch_server()
