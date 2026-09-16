from __future__ import annotations

import functools


@functools.cache
def _get_torch_cuda_version() -> tuple[int, int] | None:
    import torch
    import torch.version

    if not torch.cuda.is_available() or not torch.version.cuda:
        return None
    return torch.cuda.get_device_capability()


def is_arch_supported(major: int, minor: int = 0) -> bool:
    arch = _get_torch_cuda_version()
    return arch >= (major, minor) if arch is not None else False


def is_sm90_supported() -> bool:
    return is_arch_supported(9, 0)


def is_sm100_supported() -> bool:
    return is_arch_supported(10, 0)
