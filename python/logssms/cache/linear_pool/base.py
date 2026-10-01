from typing import Protocol

import torch


class LinearCachePool(Protocol):
    """Common resource identity; tensor access belongs to each operator backend."""

    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...
