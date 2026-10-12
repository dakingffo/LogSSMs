from __future__ import annotations

from typing import NamedTuple


class KernelConfig(NamedTuple):
    num_threads  : int
    max_occupancy: int
    use_pdl      : bool
