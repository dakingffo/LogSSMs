from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from picosgl.env import ENV
from picosgl.tvm import load_aot

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@functools.cache
def _load_nccl_module() -> Module:
    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=["-lnccl"])


@functools.cache
def _get_pynccl_wrapper_cls():
    import tvm_ffi

    @tvm_ffi.register_object("picosgl.NCCLWrapper")
    class NCCLWrapper(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return NCCLWrapper


def init_pynccl(
    *,
    tp_rank       : int,
    tp_size       : int,
    tp_cpu_group  : torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
):
    import torch

    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    module = _load_nccl_module()
    cls = _get_pynccl_wrapper_cls()

    if tp_rank == 0:
        id_list = [module.create_nccl_uid()]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )
    else:
        id_list = [None]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )

    nccl_id = id_list[0]
    assert nccl_id is not None, f"Failed to get NCCL unique ID on {tp_rank = }"

    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)


def make_nccl_uid_bytes() -> bytes:
    """Create a fresh ncclUniqueId and return its 128 raw bytes (for the zmq handshake)."""
    return bytes([v & 0xFF for v in _load_nccl_module().create_nccl_uid()])


def _nccl_uid_to_ffi(nccl_uid: bytes) -> list[int]:
    """Decode the 128 raw uid bytes into the signed-char ``Array<char>`` the FFI expects."""
    assert len(nccl_uid) == 128, f"ncclUniqueId must be 128 bytes, got {len(nccl_uid)}"
    return [v if v < 128 else v - 256 for v in nccl_uid]


def init_pynccl_drafter_target_separation(
    *,
    rank          : int,
    world_size    : int,
    nccl_uid      : bytes,
    max_size_bytes: int = 0,
):
    """Init a standalone NCCL communicator from a uid shipped over the control plane.

    Used for the 2-rank target↔drafter data plane, which has no gloo group to broadcast
    the uid through (unlike the TP group's ``init_pynccl``). ``nccl_uid`` is the 128 raw
    bytes of the ncclUniqueId created on rank 0 and carried by SpeculatorHandshakeMsg.
    """
    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    module = _load_nccl_module()
    cls = _get_pynccl_wrapper_cls()
    return cls(rank, world_size, max_size_bytes, _nccl_uid_to_ffi(nccl_uid)),
    
