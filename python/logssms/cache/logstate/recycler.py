from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from weakref import WeakValueDictionary

import torch

from .prefix_cache import (
    LogRadixEvictionNode,
    LogRadixTree,
    LogStateEntry,
    LogStateKind,
    LogStateHitStats,
)

if TYPE_CHECKING:
    from .online_analysis_scheduler import MaterializationPlan


@dataclass(frozen=True)
class RecycledLog:
    begin: int
    end  : int
    analysis_owner: LogStateHitStats | None = None


@dataclass(frozen=True)
class RecycledBranch:
    tokens            : tuple[int, ...]
    logs              : tuple[RecycledLog, ...]
    protected_endpoints: frozenset[int]
    _serial           : int = field(default=0, compare=False, repr=False)

    @property
    def analysis_owner(self):
        return self.logs[-1].analysis_owner if self.logs else None


@dataclass(frozen=True)
class RecycledMatch:
    branch     : RecycledBranch
    matched_len: int

    @property
    def analysis_owner(self):
        return next(
            (
                log.analysis_owner
                for log in reversed(self.branch.logs)
                if log.begin < self.matched_len <= log.end
            ),
            None,
        )

    def matched_tokens(self, start: int) -> int:

        return sum(
            max(0, min(log.end, self.matched_len) - max(log.begin, start))
            for log in self.branch.logs
        )


@dataclass(frozen=True)
class RecycledCheckpoint:
    begin: int
    end  : int
    gpu_entries: tuple[LogStateEntry, ...] = ()


@dataclass(frozen=True)
class RecycledRestorePlan:
    cached_len         : int
    matched_len: int
    checkpoints        : tuple[RecycledCheckpoint, ...]
    protected_endpoints: frozenset[int]
    gpu_materialize    : tuple[MaterializationPlan, ...] = ()


@dataclass
class _RecycledRestore:
    plan     : RecycledRestorePlan
    slots    : dict[int, int] = field(default_factory=dict)
    detached : set[int] = field(default_factory=set)
    committed: set[int] = field(default_factory=set)
    prepared : bool = False


class LogStateRecycleBin:
    def __init__(self, capacity_bytes: int) -> None:
        from logssms.kernel import logstate as planning

        assert capacity_bytes >= 0
        self.capacity_bytes = capacity_bytes
        self._native = planning.make_recycle_bin(capacity_bytes)
        self._matches: WeakValueDictionary[int, RecycledBranch] = WeakValueDictionary()
        self.revision = 0
        self._owners: dict[int, LogStateHitStats] = {}

    def remember(
        self,
        tree: LogRadixTree,
        snapshot: LogRadixEvictionNode,
    ) -> None:
        from logssms.kernel import logstate as planning

        logs = [
            (entry.begin, entry.end)
            for entry in snapshot.entries if entry.kind is LogStateKind.LOG
        ]
        if self.capacity_bytes == 0 or not logs:
            return
        self.revision += 1
        source = snapshot.node
        path = tree.path(source)
        keys = [part.key for part in path]
        stats = source.analysis_owner
        if stats is not None:
            self._owners[stats.uuid] = stats
        self._native["remember"](
            keys,
            planning._pack(logs),
            [
                part.depth
                for part in path
                if part.is_protected and any(begin <= part.depth <= end for begin, end in logs)
            ],
            [] if stats is None else [v for begin, end in logs for v in (begin, end, stats.uuid)],
        )
        self._release_owners()

    def _release_owners(self):
        for owner_id in self._native["retired_owners"]():
            self._owners.pop(owner_id, None)

    def match(
        self, tokens: torch.Tensor, *, start: int, granularity: int, touch: bool = True,
    ) -> RecycledMatch | None:
        from logssms.kernel import logstate as planning

        assert tokens.device.type == "cpu"
        result = self._native["match"](tokens, start, granularity, touch)
        if len(result) == 0:
            return None
        serial, matched_len = result
        branch = self._matches.get(serial)
        if branch is None:
            rows = planning._unpack(self._native["get"](serial, tokens))
            owner_ranges = list(zip(rows[2][::3], rows[2][1::3], rows[2][2::3]))
            logs = []
            for begin, end in rows[3:]:
                boundaries = sorted(
                    {
                        begin,
                        end,
                        *(
                            p
                            for left, right, _ in owner_ranges
                            for p in (left, right)
                            if begin < p < end
                        ),
                    }
                )
                for left, right in zip(boundaries, boundaries[1:]):
                    owner_id = next(
                        (owner for a, b, owner in owner_ranges if a <= left and right <= b), None
                    )
                    logs.append(RecycledLog(left, right, self._owners.get(owner_id)))
            branch = RecycledBranch(tuple(rows[0]), tuple(logs), frozenset(rows[1]), serial)
            self._matches[serial] = branch
        return RecycledMatch(branch, matched_len)

    def discard(self, branch: RecycledBranch) -> None:
        if self._matches.get(branch._serial) is branch:
            self.revision += 1
            self._native["discard"](branch._serial)
            del self._matches[branch._serial]
            self._release_owners()

    def touch(self, branch: RecycledBranch) -> None:
        self._native["touch"](branch._serial)

    def __len__(self) -> int:
        return self._native["usage"]()[1]
