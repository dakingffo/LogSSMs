from __future__ import annotations

import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NamedTuple, Self

import torch

from logssms.distributed import (
    init_pynccl_drafter_target_separation,
    make_nccl_uid_bytes,
)

if TYPE_CHECKING:
    from logssms.scheduler.config import SchedulerConfig


class DataPlaneSizes(NamedTuple):
    max_hidden_rows: int
    hidden_size    : int
    max_prob_rows  : int
    vocab_size     : int


def make_data_plane_sizes(
    config     : SchedulerConfig,
    hidden_size: int,
    vocab_size : int,
) -> DataPlaneSizes:
    speculator_config = config.speculator_config
    assert speculator_config is not None
    K = speculator_config.num_draft_tokens
    max_batch_size = min(
        config.max_running_req,
        config.decode_batch_budget // K,
    )
    return DataPlaneSizes(
        max(speculator_config.max_init_hidden_rows, max_batch_size * (K + 1)),
        hidden_size * speculator_config.hidden_size_multiplier,
        max_batch_size * K, vocab_size,
    )


class DataPlane(ABC):
    device: torch.device
    rank  : int

    @abstractmethod
    def make_connection_id(self) -> bytes:
        """Create transport-specific connection metadata for the handshake."""

    def prepare_rank1(self, sizes: DataPlaneSizes) -> None:
        """Allocate worker-owned transport storage before signaling engine readiness."""
        pass

    @abstractmethod
    def init_rank0(self, connection_id: bytes, sizes: DataPlaneSizes) -> None:
        """Target side: initialize the communicator (blocks until rank1 inits)."""

    @abstractmethod
    def init_rank1(self, connection_id: bytes, sizes: DataPlaneSizes) -> None:
        """Drafter side: initialize the communicator (blocks until rank0 inits)."""

    @abstractmethod
    def send_hidden(self, hidden: torch.Tensor) -> None:
        """Target → drafter: the appended carry-hidden rows (or a request's init window)."""

    @abstractmethod
    def recv_hidden(self, num_rows: int) -> torch.Tensor:
        """Drafter side of the transfer; returns the target's rows on its device."""

    @abstractmethod
    def send_probs(self, probs: torch.Tensor) -> None:
        """Drafter → target: the draft_probs rows for this step's sampling requests."""

    @abstractmethod
    def recv_probs(self, num_rows: int) -> torch.Tensor:
        """Target side of the transfer; returns the drafter's rows on its device."""

    def destroy(self) -> None:
        pass


class NCCLDataPlane(DataPlane):
    def __init__(
        self,
        device     : torch.device,
        rank       : int,
        world_size : int = 2,
        dtype      : torch.dtype = torch.bfloat16,
    ) -> None:
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.dtype = dtype
        self._nccld = None
        self._hidden_in: torch.Tensor | None = None
        self._hidden_out: torch.Tensor | None = None
        self._probs_in: torch.Tensor | None = None
        self._probs_out: torch.Tensor | None = None

    def make_connection_id(self) -> bytes:
        return make_nccl_uid_bytes()

    def init_rank0(self, connection_id: bytes, sizes: DataPlaneSizes) -> None:
        self._nccld = init_pynccl_drafter_target_separation(
            rank=0, world_size=self.world_size, nccl_uid=connection_id
        )
        self._alloc_bufs(sizes)

    def init_rank1(self, connection_id: bytes, sizes: DataPlaneSizes) -> None:
        self._nccld = init_pynccl_drafter_target_separation(
            rank=1, world_size=self.world_size, nccl_uid=connection_id
        )
        self._alloc_bufs(sizes)

    def _alloc_bufs(self, sizes: DataPlaneSizes) -> None:
        max_hidden, hidden_size, max_prob, vocab_size = sizes
        self._hidden_in = torch.zeros(
            max_hidden, hidden_size, dtype=self.dtype, device=self.device
        )
        self._hidden_out = torch.zeros(
            2 * max_hidden, hidden_size, dtype=self.dtype, device=self.device
        )
        self._probs_in = torch.zeros(
            max_prob, vocab_size, dtype=torch.float32, device=self.device
        )
        self._probs_out = torch.zeros(
            2 * max_prob, vocab_size, dtype=torch.float32, device=self.device
        )

    def send_hidden(self, hidden: torch.Tensor) -> None:
        rows = hidden.shape[0]
        assert self._hidden_in is not None and rows <= self._hidden_in.shape[0]
        self._hidden_in[:rows].copy_(hidden)  # rest stale; receiver only reads [:rows]
        self._nccld.all_gather(self._hidden_out, self._hidden_in)

    def recv_hidden(self, num_rows: int) -> torch.Tensor:
        self._nccld.all_gather(self._hidden_out, self._hidden_in)
        return self._hidden_out[:num_rows].clone()

    def send_probs(self, probs: torch.Tensor) -> None:
        rows = probs.shape[0]
        assert self._probs_in is not None and rows <= self._probs_in.shape[0]
        self._probs_in[:rows].copy_(probs)
        self._nccld.all_gather(self._probs_out, self._probs_in)

    def recv_probs(self, num_rows: int) -> torch.Tensor:
        S = self._probs_out.shape[0] // 2
        self._nccld.all_gather(self._probs_out, self._probs_in)
        return self._probs_out[S : S + num_rows].clone()

    def destroy(self) -> None:
        if self._nccld is not None:
            torch.cuda.synchronize(self.device)


class _QueueDataPlane(DataPlane):
    def __init__(
        self,
        device     : torch.device,
        rank       : int,
        ipc_queues : tuple[mp.Queue, mp.Queue],
        dtype      : torch.dtype,
    ) -> None:
        assert rank in (0, 1)
        self.device = device
        self.rank = rank
        self.dtype = dtype
        target_to_speculator, speculator_to_target = ipc_queues
        if rank == 0:
            self._send_queue = target_to_speculator
            self._recv_queue = speculator_to_target
        else:
            self._send_queue = speculator_to_target
            self._recv_queue = target_to_speculator

        self._hidden: torch.Tensor | None = None
        self._probs : torch.Tensor | None = None

    @classmethod
    def make_pair(
        cls,
        device    : torch.device,
        dtype     : torch.dtype,
        mp_context: mp.context.BaseContext | None = None,
    ) -> tuple[Self, Self]:
        queue_factory = mp.Queue if mp_context is None else mp_context.Queue
        queues = (queue_factory(), queue_factory())
        return (
            cls(device, rank=0, ipc_queues=queues, dtype=dtype),
            cls(device, rank=1, ipc_queues=queues, dtype=dtype),
        )

    def make_connection_id(self) -> bytes:
        return b""

    @abstractmethod
    def _publish(self, buffer: torch.Tensor, value: torch.Tensor) -> None: ...

    @abstractmethod
    def _consume(self, buffer: torch.Tensor, num_rows: int) -> torch.Tensor: ...

    def send_hidden(self, hidden: torch.Tensor) -> None:
        assert self.rank == 0 and self._hidden is not None
        self._publish(self._hidden, hidden)

    def recv_hidden(self, num_rows: int) -> torch.Tensor:
        assert self.rank == 1 and self._hidden is not None
        return self._consume(self._hidden, num_rows)

    def send_probs(self, probs: torch.Tensor) -> None:
        assert self.rank == 1 and self._probs is not None
        self._publish(self._probs, probs)

    def recv_probs(self, num_rows: int) -> torch.Tensor:
        assert self.rank == 0 and self._probs is not None
        return self._consume(self._probs, num_rows)

    def destroy(self) -> None:
        torch.cuda.synchronize(self.device)
        for queue in (self._send_queue, self._recv_queue):
            queue.close()
            queue.join_thread()


class CUDAIPCDataPlane(_QueueDataPlane):
    """Same-device data plane backed by CUDA IPC memory and interprocess events.

    The worker owns both shared buffers so their memory is visible before Target cache
    sizing. Each direction has one producer-owned CUDA event and one multiprocessing
    queue used only as a generation fence. Tensor payloads remain on the GPU; the queues
    carry setup handles and row counts, never hidden states or probabilities.
    """

    def __init__(
        self,
        device     : torch.device,
        rank       : int,
        ipc_queues : tuple[mp.Queue, mp.Queue],
        dtype      : torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(device, rank, ipc_queues, dtype)
        self._send_event: torch.cuda.Event | None = None
        self._recv_event: torch.cuda.Event | None = None

    def init_rank0(self, connection_id: bytes, sizes: DataPlaneSizes) -> None:
        assert self.rank == 0 and not connection_id
        hidden, probs, peer_event_handle = self._recv_queue.get()
        assert isinstance(hidden, torch.Tensor) and isinstance(probs, torch.Tensor)
        self._hidden = hidden
        self._probs = probs
        self._verify_bufs(sizes)
        self._recv_event = torch.cuda.Event.from_ipc_handle(
            self.device, peer_event_handle
        )
        self._send_event = torch.cuda.Event(interprocess=True)
        self._send_queue.put(self._send_event.ipc_handle())

    def prepare_rank1(self, sizes: DataPlaneSizes) -> None:
        assert self.rank == 1
        self._alloc_bufs(sizes)
        self._send_event = torch.cuda.Event(interprocess=True)
        self._send_queue.put(
            (self._hidden, self._probs, self._send_event.ipc_handle())
        )

    def init_rank1(self, connection_id: bytes, sizes: DataPlaneSizes) -> None:
        assert self.rank == 1 and not connection_id
        self._verify_bufs(sizes)
        peer_event_handle = self._recv_queue.get()
        self._recv_event = torch.cuda.Event.from_ipc_handle(
            self.device, peer_event_handle
        )

    def _alloc_bufs(self, sizes: DataPlaneSizes) -> None:
        max_hidden, hidden_size, max_prob, vocab_size = sizes
        self._hidden = torch.empty(
            max_hidden, hidden_size, dtype=self.dtype, device=self.device
        )
        self._probs = torch.empty(
            max_prob, vocab_size, dtype=torch.float32, device=self.device
        )

    def _verify_bufs(self, sizes: DataPlaneSizes) -> None:
        max_hidden, hidden_size, max_prob, vocab_size = sizes
        assert self._hidden is not None and self._probs is not None
        assert self._hidden.shape == (max_hidden, hidden_size)
        assert self._hidden.dtype == self.dtype and self._hidden.device == self.device
        assert self._probs.shape == (max_prob, vocab_size)
        assert self._probs.dtype == torch.float32 and self._probs.device == self.device

    def _publish(self, buffer: torch.Tensor, value: torch.Tensor) -> None:
        rows = value.shape[0]
        assert rows <= buffer.shape[0]
        buffer[:rows].copy_(value)
        assert self._send_event is not None
        self._send_event.record(torch.cuda.current_stream(self.device))
        self._send_queue.put(rows)

    def _consume(self, buffer: torch.Tensor, num_rows: int) -> torch.Tensor:
        rows = self._recv_queue.get()
        assert rows == num_rows, f"CUDA IPC row mismatch: expected {num_rows}, got {rows}"
        assert num_rows <= buffer.shape[0]
        assert self._recv_event is not None
        torch.cuda.current_stream(self.device).wait_event(self._recv_event)
        return buffer[:num_rows].clone()


class SharedMemoryDataPlane(_QueueDataPlane):
    """Compatibility data plane for platforms without CUDA IPC support.

    The worker owns two shared CPU tensors. Transfers synchronize through the same two
    queues as ``CUDAIPCDataPlane`` and stage through host memory. This is a correctness
    fallback, not a production-performance path.
    """

    def __init__(
        self,
        device     : torch.device,
        rank       : int,
        ipc_queues : tuple[mp.Queue, mp.Queue],
        dtype      : torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(device, rank, ipc_queues, dtype)

    def init_rank0(self, connection_id: bytes, sizes: DataPlaneSizes) -> None:
        assert self.rank == 0 and not connection_id
        hidden, probs = self._recv_queue.get()
        assert isinstance(hidden, torch.Tensor) and isinstance(probs, torch.Tensor)
        self._hidden = hidden
        self._probs = probs
        self._verify_bufs(sizes)
        self._send_queue.put("ready")

    def prepare_rank1(self, sizes: DataPlaneSizes) -> None:
        assert self.rank == 1
        max_hidden, hidden_size, max_prob, vocab_size = sizes
        self._hidden = torch.empty(
            max_hidden, hidden_size, dtype=self.dtype
        ).share_memory_()
        self._probs = torch.empty(
            max_prob, vocab_size, dtype=torch.float32
        ).share_memory_()
        self._send_queue.put((self._hidden, self._probs))

    def init_rank1(self, connection_id: bytes, sizes: DataPlaneSizes) -> None:
        assert self.rank == 1 and not connection_id
        self._verify_bufs(sizes)
        assert self._recv_queue.get() == "ready"

    def _verify_bufs(self, sizes: DataPlaneSizes) -> None:
        max_hidden, hidden_size, max_prob, vocab_size = sizes
        assert self._hidden is not None and self._probs is not None
        assert self._hidden.shape == (max_hidden, hidden_size)
        assert self._hidden.dtype == self.dtype and self._hidden.device.type == "cpu"
        assert self._probs.shape == (max_prob, vocab_size)
        assert self._probs.dtype == torch.float32 and self._probs.device.type == "cpu"

    def _publish(self, buffer: torch.Tensor, value: torch.Tensor) -> None:
        rows = value.shape[0]
        assert rows <= buffer.shape[0]
        buffer[:rows].copy_(value)
        torch.cuda.current_stream(self.device).synchronize()
        self._send_queue.put(rows)

    def _consume(self, buffer: torch.Tensor, num_rows: int) -> torch.Tensor:
        rows = self._recv_queue.get()
        assert rows == num_rows, (
            f"shared-memory row mismatch: expected {num_rows}, got {rows}"
        )
        assert num_rows <= buffer.shape[0]
        return buffer[:num_rows].to(self.device)
