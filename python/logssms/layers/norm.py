import torch

from logssms.kernel import rms_norm_gated
from logssms.utils import nvtx_annotate

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(
        self,
        size         : int,
        eps          : float,
        zero_centered: bool = False,
    ) -> None:
        from flashinfer import gemma_rmsnorm, rmsnorm

        self.eps = eps
        self.zero_centered = zero_centered
        self.weight = torch.zeros(size) if zero_centered else torch.empty(size)
        self.rmsnorm = gemma_rmsnorm if zero_centered else rmsnorm

    @nvtx_annotate("RMSNorm")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.zero_centered and x.ndim > 2:
            shape = x.shape
            return self.rmsnorm(
                x.reshape(-1, shape[-1]), self.weight, self.eps
            ).reshape(shape)
        else:
            return self.rmsnorm(x, self.weight, self.eps)

    @nvtx_annotate("RMSNorm")
    def forward_inplace(self, x: torch.Tensor) -> None:
        if self.zero_centered and x.ndim > 2:
            x_flat = x.reshape(-1, x.shape[-1])
            self.rmsnorm(x_flat, self.weight, self.eps, out=x_flat)
        else:
            self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormGated(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.ones(size, dtype=torch.float32)

    @nvtx_annotate("RMSNormGated")
    def forward(
        self,
        hidden_states: torch.Tensor,
        gate         : torch.Tensor,
    ) -> torch.Tensor:
        return rms_norm_gated(hidden_states, gate, self.weight, self.eps)


class RMSNormFused(BaseOP):
    def __init__(
        self,
        size         : int,
        eps          : float,
        zero_centered: bool = False,
    ) -> None:
        from flashinfer import (
            fused_add_rmsnorm,
            gemma_fused_add_rmsnorm,
            gemma_rmsnorm,
            rmsnorm,
        )

        self.eps = eps
        self.weight = torch.zeros(size) if zero_centered else torch.empty(size)
        self.rmsnorm = gemma_rmsnorm if zero_centered else rmsnorm
        self.fused_add_rmsnorm = (
            gemma_fused_add_rmsnorm if zero_centered else fused_add_rmsnorm
        )

    @nvtx_annotate("RMSNormFused")
    def forward(
        self,
        x       : torch.Tensor,
        residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        else:
            self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
            return x, residual
