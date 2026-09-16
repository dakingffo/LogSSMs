from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    rank: int
    size: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        return self.rank == 0


_TP_INFO: DistributedInfo | None = None


def set_tp_info(tp_info: DistributedInfo) -> None:
    global _TP_INFO
    assert _TP_INFO is None, "TP info has been set"
    _TP_INFO = tp_info


def get_tp_info() -> DistributedInfo:
    global _TP_INFO
    assert _TP_INFO is not None, "TP info has not been set"
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO


@contextmanager
def tp_override(tp_info: DistributedInfo):
    global _TP_INFO
    saved = _TP_INFO
    _TP_INFO = tp_info
    try:
        yield
    finally:
        _TP_INFO = saved
