# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import contextlib
import logging
import os
import platform
import sys
import warnings
from enum import Enum
from functools import cache, lru_cache

import torch
import triton
from packaging import version as package_version

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def check_environments():

    if sys.platform == 'win32':

        try:
            from importlib.metadata import PackageNotFoundError, metadata
            metadata('triton-windows')

        except PackageNotFoundError:
            logger.warning(
                "Detected Windows operating system. Consider installing triton-windows "
                "(https://github.com/triton-lang/triton-windows) for better compatibility. "
                "Without it, some features may not work correctly.",
            )

    triton_version = package_version.parse(triton.__version__)
    required_triton_version = package_version.parse("3.3.0")

    if triton_version < required_triton_version:
        logger.warning(
            f"Current Triton version {triton_version} is below the recommended 3.3.0 version. "
            "Errors may occur and these issues will not be fixed. "
            "Please consider upgrading Triton.",
        )


    py_version = package_version.parse(f"{sys.version_info.major}.{sys.version_info.minor}")
    required_py_version = package_version.parse("3.11")

    if py_version < required_py_version:
        logger.warning(
            f"Current Python version {py_version} is below the recommended 3.11 version. "
            "It is recommended to upgrade to Python 3.11 or higher for the best experience.",
        )

    return None


check_environments()


def _cpu_device_warning():
    warnings.warn(('Triton is not supported on current platform, roll back to CPU.'), stacklevel=2)


@cache
def check_pytorch_version(version_s: str = '2.4') -> bool:
    return package_version.parse(torch.__version__) >= package_version.parse(version_s)


@cache
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except Exception:
        _cpu_device_warning()
        return 'cpu'


def map_triton_backend_to_torch_device() -> str:
    backend = get_available_device()
    return {'cuda': 'cuda', 'hip': 'cuda', 'xpu': 'xpu'}.get(backend, backend)


device = get_available_device() if get_available_device() != 'hip' else 'cuda'
device_torch_lib = getattr(torch, device)
device_platform = get_available_device()
device_name = map_triton_backend_to_torch_device()

IS_AMD = (device_platform == 'hip')

IS_ARM = platform.machine().lower() in ('aarch64', 'arm64')

IS_INTEL = (device_platform == 'xpu')
IS_INTEL_ALCHEMIST = (IS_INTEL and 'Intel(R) Arc(TM) A' in torch.xpu.get_device_name(0))

IS_NPU = (device_platform == 'npu')

IS_NVIDIA = (device_platform == 'cuda')
IS_NVIDIA_HOPPER = (
    IS_NVIDIA and (
        'NVIDIA H' in torch.cuda.get_device_name(0)
        or torch.cuda.get_device_capability()[0] == 9
    )
)
IS_NVIDIA_SM100 = (IS_NVIDIA and torch.cuda.get_device_capability()[0] == 10)
IS_NVIDIA_BLACKWELL = (IS_NVIDIA and torch.cuda.get_device_capability()[0] in (10, 12))


IS_TF32_SUPPORTED = (IS_NVIDIA and torch.cuda.get_device_capability(0)[0] >= 8)
IS_TMA_SUPPORTED = (
    IS_NVIDIA
    and torch.cuda.get_device_capability(0)[0] >= 9
    and os.environ.get('FLA_USE_TMA', '0') == '1'
    and (
        hasattr(triton.language, '_experimental_make_tensor_descriptor')
        or hasattr(triton.language, 'make_tensor_descriptor')
    )
)

if IS_NVIDIA and not IS_TF32_SUPPORTED:


    os.environ['TRITON_F32_DEFAULT'] = 'ieee'


def _default_alloc_fn(size: int, alignment: int, stream: int | None):
    return torch.empty(size, device=torch.device(device_name, device_torch_lib.current_device()), dtype=torch.int8)


if IS_TMA_SUPPORTED:
    logger.info('TMA is supported, using TMA by default.')
    triton.set_allocator(_default_alloc_fn)
elif IS_NVIDIA_BLACKWELL:


    logger.info('Blackwell detected: registering default global_scratch allocator.')
    triton.set_allocator(_default_alloc_fn)


def get_all_max_shared_mem():
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)['max_shared_mem']
            for i in range(device_torch_lib.device_count())
        ]
    except Exception:
        _cpu_device_warning()
        return [-1]


class Backend(Enum):
    ADA = 101376
    AMPERE = 166912
    HOPPER = 232448
    DEFAULT = 102400

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value


@cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False


if check_pytorch_version('2.4'):
    if device == 'cpu':
        device = 'cuda'
        device_torch_lib = getattr(torch, device)

    def custom_device_ctx(index: int):
        if index is None:
            return contextlib.nullcontext()
        try:
            return device_torch_lib.device(index)
        except (AttributeError, AssertionError, RuntimeError):
            return contextlib.nullcontext()
else:
    assert device == 'cuda', 'Only cuda device is supported for PyTorch version < 2.4.0.'

    def custom_device_ctx(index: int):
        if index is None:
            return contextlib.nullcontext()
        try:
            return torch.cuda.device(index)
        except (AttributeError, AssertionError, RuntimeError):
            return contextlib.nullcontext()
