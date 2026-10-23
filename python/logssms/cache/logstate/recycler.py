from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from weakref import WeakValueDictionary

import torch

from .prefix_cache import LogRadixTree, LogRadixTreeNode, LogStateEntry, LogStateKind

if TYPE_CHECKING:
    from .online_analysis_scheduler import MaterializeCandidate, CheckpointCoalesce


@dataclass(frozen=True)
class RecycledLog:
    begin: int
    end  : int


@dataclass(frozen=True)
class RecycledBranch:
    tokens            : tuple[int, ...]
    logs              : tuple[RecycledLog, ...]
    protected_endpoints: frozenset[int]
    _serial           : int = field(default=0, compare=False, repr=False)
    merge_stats       : tuple[int, ...] = ()

    @property
    def bytes(self) -> int:


        return (
            256 + 8 * len(self.tokens) + 16 * len(self.logs)
            + 8 * len(self.protected_endpoints)
            + 8 * len(self.merge_stats)
        )


@dataclass(frozen=True)
class RecycledMatch:
    branch     : RecycledBranch
    matched_len: int

    def matched_tokens(self, start: int) -> int:


        return sum(
            max(0, min(log.end, self.matched_len) - max(log.begin, start))
            for log in self.branch.logs
        )


@dataclass(frozen=True)
class RecycledCheckpoint:
    begin      : int
    end        : int
    gpu_entries: tuple[LogStateEntry, ...] = ()


@dataclass(frozen=True)
class RecycledRestorePlan:

    cached_len         : int
    matched_len        : int
    checkpoints        : tuple[RecycledCheckpoint, ...]
    protected_endpoints: frozenset[int]
    gpu_materialize    : tuple[MaterializeCandidate, ...] = ()
    gpu_coalesce       : tuple[CheckpointCoalesce, ...] = ()
    intensity          : int = 2
    lock_boundaries    : frozenset[int] = frozenset()


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


    def remember(
        self,
        tree: LogRadixTree,
        node: LogRadixTreeNode,
    ) -> None:
        from logssms.kernel import logstate as planning

        logs = [
            (max(node.begin, entry.begin), min(node.depth, entry.end))
            for entry in node.entries if entry.kind is LogStateKind.LOG
        ]
        if self.capacity_bytes == 0 or not logs:
            return
        self.revision += 1
        path = tree.path(node)
        stats = node.merge_stats
        self._native["remember"](
            [part.key for part in path],
            planning._pack(logs),
            [part.depth for part in path if part.is_protected
             and any(begin <= part.depth <= end for begin, end in logs)],
            [] if stats is None else [
                stats.origin, stats.tail, stats.intensity,
                -1 if stats.next_update_tail is None else stats.next_update_tail,
                *stats.hit_positions,
            ],
        )

    def match(self, tokens: torch.Tensor, *, start: int, granularity: int) -> RecycledMatch | None:
        from logssms.kernel import logstate as planning

        assert tokens.device.type == "cpu"
        result = self._native["match"](tokens, start, granularity)
        if len(result) == 0:
            return None
        serial, matched_len = result
        branch = self._matches.get(serial)
        if branch is None:
            rows = planning._unpack(self._native["get"](serial, tokens))
            branch = RecycledBranch(
                tuple(rows[0]), tuple(RecycledLog(begin, end)
                                      for begin, end in rows[3:]), frozenset(rows[1]), serial,
                tuple(rows[2]),
            )
            self._matches[serial] = branch
        return RecycledMatch(branch, matched_len)

    def discard(self, branch: RecycledBranch) -> None:
        if self._matches.get(branch._serial) is branch:
            self.revision += 1
            self._native["discard"](branch._serial)
            del self._matches[branch._serial]

    def touch(self, branch: RecycledBranch) -> None:
        self._native["touch"](branch._serial)

    def __len__(self) -> int:
        return self._native["usage"]()[1]
