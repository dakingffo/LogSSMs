from __future__ import annotations

from typing import NamedTuple


class KernelConfig(NamedTuple):
    num_threads  : int
    max_occupancy: int
    use_pdl      : bool

    @property
    def template_args(self) -> str:
        pdl = "true" if self.use_pdl else "false"
        return f"{self.num_threads},{self.max_occupancy},{pdl}"
