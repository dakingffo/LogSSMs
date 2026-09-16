from __future__ import annotations

import torch
import torch.nn.functional as F
from logssms.distributed import DistributedCommunicator, get_tp_info
from logssms.utils import div_even

from .base import BaseOP


class LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism."""

    def __init__(
        self,
        input_size : int,
        output_size: int,
        has_bias   : bool,
    ):
        self.weight = torch.empty(output_size, input_size)
        self.bias = torch.empty(output_size) if has_bias else None

    def forward(
        self, 
        x: torch.Tensor # [N, D]
    ) -> torch.Tensor: # [N, D']
        return F.linear(x, self.weight, self.bias) # [N, D] @ [D, D']


class LinearReplicated(LinearTPImpl):
    """
    Linear layer where weights are replicated (not sharded) across all TP ranks.
    Each GPU holds the full weight matrix.
    """

    def __init__(
        self,
        input_size : int,
        output_size: int,
        has_bias   : bool,
    ):
        super().__init__(input_size, output_size, has_bias)


class LinearColParallelMerged(LinearTPImpl):
    def __init__(
        self,
        input_size  : int,
        output_sizes: list[int],
        has_bias    : bool,
    ):
        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, tp_output_size, has_bias)


class LinearColParallelPartitioned(LinearTPImpl):
    def __init__(
        self,
        input_size    : int,
        partition_size: int,
        partitions    : list[tuple[int, bool]],
        has_bias      : bool,
    ):
        tp_info = get_tp_info()
        tp_partitions = [
            div_even(partition, tp_info.size, allow_replicate)
            for partition, allow_replicate in partitions
        ]
        tp_output_size = sum(tp_partitions) * partition_size
        super().__init__(input_size, tp_output_size, has_bias)


class LinearRowParallel(LinearTPImpl):
    def __init__(
        self,
        input_size : int,
        output_size: int,
        has_bias   : bool,
    ):
        tp_info = get_tp_info()
        tp_input_size = div_even(input_size, tp_info.size)
        tp_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(tp_input_size, tp_output_size, has_bias)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearColumnParallel(LinearTPImpl):
    def __init__(
        self,
        input_size : int,
        output_size: int,
        has_bias   : bool,
    ):
        tp_info = get_tp_info()
        tp_output_size = div_even(output_size, tp_info.size)
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, tp_output_size, has_bias)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            gathered = self._comm.all_gather(y)
            gathered = gathered.view((self._tp_size,) + y.shape)
            gathered = gathered.permute(1, 0, 2).contiguous()
            y = gathered.reshape((y.shape[0], self._tp_size * y.shape[1]))
        return y
