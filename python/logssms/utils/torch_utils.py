from __future__ import annotations

import functools
import inspect
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def make_device_vectors(vectors, device, *, dtype=None):
    """Pack small integer metadata into one transfer on the current stream.

    Each call owns fresh pinned storage; it is never overwritten while its
    asynchronous copy is pending (PyTorch tracks the pinned allocation).
    """
    import torch

    device = torch.device(device)
    offsets = [0]
    flat = []
    for values in vectors:
        flat.extend(values)
        offsets.append(len(flat))
    host = torch.tensor(
        flat, dtype=torch.int32 if dtype is None else dtype,
        device="cpu", pin_memory=device.type == "cuda",
    )
    packed = host.to(device, non_blocking=True)
    return tuple(packed[begin:end] for begin, end in zip(offsets, offsets[1:]))


@contextmanager
def torch_dtype(dtype: torch.dtype):
    import torch  # real import when used

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    import torch.cuda.nvtx as nvtx

    def decorator(fn):
        if inspect.isgeneratorfunction(fn):
            @functools.wraps(fn)
            def generator_wrapper(self, *args, **kwargs):
                display_name = name
                if layer_id_field and hasattr(self, layer_id_field):
                    display_name = name.format(getattr(self, layer_id_field))
                with nvtx.range(display_name):
                    yield from fn(self, *args, **kwargs)
            return generator_wrapper
        else:
            @functools.wraps(fn)
            def wrapper(self, *args, **kwargs):
                display_name = name
                if layer_id_field and hasattr(self, layer_id_field):
                    display_name = name.format(getattr(self, layer_id_field))
                with nvtx.range(display_name):
                    return fn(self, *args, **kwargs)
            return wrapper

    return decorator
