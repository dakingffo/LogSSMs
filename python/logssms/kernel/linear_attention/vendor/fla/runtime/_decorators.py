# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import contextlib
import functools
from collections import deque
from collections.abc import Callable
from typing import Any

import torch

from ._config import FLA_DISABLE_TENSOR_CACHE, FLA_TENSOR_CACHE_SIZE
from ._device import custom_device_ctx


def tensor_cache(
    fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    cached: deque = deque(maxlen=FLA_TENSOR_CACHE_SIZE)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if FLA_DISABLE_TENSOR_CACHE:
            return fn(*args, **kwargs)

        for cached_args, cached_kwargs, cached_result in cached:
            if len(args) != len(cached_args) or len(kwargs) != len(cached_kwargs):
                continue
            if all(a is b for a, b in zip(args, cached_args, strict=False)) and \
                    all(k in cached_kwargs and v is cached_kwargs[k] for k, v in kwargs.items()):
                return cached_result

        result = fn(*args, **kwargs)
        cached.append((args, kwargs, result))
        return result

    return wrapper


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        processed_args = tuple(arg.contiguous() if isinstance(arg, torch.Tensor) else arg for arg in args)
        processed_kwargs = {
            key: value.contiguous() if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        }
        tensor = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
        if tensor is None:
            tensor = next((value for value in kwargs.values() if isinstance(value, torch.Tensor)), None)
        ctx = custom_device_ctx(tensor.device.index) if tensor is not None else contextlib.nullcontext()
        with ctx:
            return fn(*processed_args, **processed_kwargs)

    return wrapper
