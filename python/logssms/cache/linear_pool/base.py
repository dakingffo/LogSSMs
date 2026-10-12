from typing import Protocol

import torch


class LinearCachePool(Protocol):

    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...
