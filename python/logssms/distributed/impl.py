from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Protocol
from ctypes import c_void_p
from dataclasses import dataclass

import torch
import torch.distributed as dist

from logssms.tvm import call_cuda_ffi

if TYPE_CHECKING:
    from logssms.distributed import DistributedInfo


@dataclass
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass
class NoopDistributedImpl(DistributedImpl):
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return x


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    class NCCLWrapper(Protocol):
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...
        def get_buffer(self) -> c_void_p: ...

    comm: NCCLWrapper

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        call_cuda_ffi(self.comm.all_reduce, x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        from .info import get_tp_info

        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        call_cuda_ffi(self.comm.all_gather, result, x)
        return result


class DistributedCommunicator:
    impl: DistributedImpl = NoopDistributedImpl()

    @classmethod
    def all_reduce(cls, x: torch.Tensor) -> torch.Tensor:
        return cls.impl.all_reduce(x)

    @classmethod
    def all_gather(cls, x: torch.Tensor) -> torch.Tensor:
        return cls.impl.all_gather(x)


def enable_pynccl_distributed(
    tp_info     : DistributedInfo,
    tp_cpu_group: dist.ProcessGroup,
    max_bytes   : int
) -> None:
    from .pynccl import init_pynccl

    # Wrap the raw PyNCCL FFI object so its all_reduce(x, "sum") signature matches the
    # DistributedImpl protocol (callers invoke impl.all_reduce(x)).
    wrapper = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )
    DistributedCommunicator.impl = PyNCCLDistributedImpl(comm=wrapper)


def destroy_distributed() -> None:
    DistributedCommunicator.impl = NoopDistributedImpl()
