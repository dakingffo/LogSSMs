from __future__ import annotations

import bisect
import sys
from collections import OrderedDict
from dataclasses import dataclass

import torch

from .prefix_cache.logstate import LogRadixTree, LogRadixTreeNode, LogStateKind


@dataclass(frozen=True)
class RecycledEntry:
    kind      : LogStateKind
    begin     : int
    end       : int
    span_pages: int | None


@dataclass(frozen=True)
class RecycledBranch:
    tokens            : tuple[int, ...]
    entries           : tuple[RecycledEntry, ...]
    protected_endpoints: frozenset[int]

    @property
    def bytes(self) -> int:
        return (
            sys.getsizeof(self) + sys.getsizeof(self.tokens)
            + sum(sys.getsizeof(token) for token in self.tokens)
            + sys.getsizeof(self.entries)
            + sum(
                sys.getsizeof(entry)
                + sys.getsizeof(entry.begin) + sys.getsizeof(entry.end)
                + sys.getsizeof(entry.span_pages)
                for entry in self.entries
            )
            + sys.getsizeof(self.protected_endpoints)
            + sum(sys.getsizeof(position) for position in self.protected_endpoints)
            + 80  # OrderedDict node and sorted-index pointer
        )


@dataclass(frozen=True)
class RecycledMatch:
    branch     : RecycledBranch
    matched_len: int


class LogStateRecycleBin:
    """Bounded CPU-only prefix/layout history; never owns device buffers."""

    def __init__(self, capacity_bytes: int) -> None:
        assert capacity_bytes >= 0
        self.capacity_bytes = capacity_bytes
        self.used_bytes = 0
        self._branches: OrderedDict[tuple[int, ...], RecycledBranch] = OrderedDict()
        self._sorted_keys: list[tuple[int, ...]] = []

    def _remove_key(self, key: tuple[int, ...]) -> None:
        index = bisect.bisect_left(self._sorted_keys, key)
        assert self._sorted_keys[index] == key
        self._sorted_keys.pop(index)

    def remember(self, tree: LogRadixTree, node: LogRadixTreeNode) -> None:
        if self.capacity_bytes == 0:
            return
        path = tree.path(node)
        branch = RecycledBranch(
            tokens=tuple(token for part in path for token in part.key.tolist()),
            entries=tuple(
                RecycledEntry(entry.kind, entry.begin, entry.end, entry.span_pages)
                for part in path for entry in part.entries
            ),
            protected_endpoints=frozenset(
                part.depth for part in path if part.is_protected
            ),
        )
        # A previously evicted child may survive while its still-resident
        # ancestor is materialized or merged.  Once that ancestor is evicted,
        # refresh the copied prefix of every descendant record; otherwise a
        # later ghost hit could resurrect the ancestor's obsolete layout.
        prefix_len = len(branch.tokens)
        index = bisect.bisect_left(self._sorted_keys, branch.tokens)
        while index < len(self._sorted_keys):
            key = self._sorted_keys[index]
            if key[:prefix_len] != branch.tokens:
                break
            if len(key) == prefix_len:
                index += 1
                continue
            previous = self._branches[key]
            if any(
                entry.begin < prefix_len < entry.end
                for entry in previous.entries
            ):
                # A checkpoint spanning the cut cannot be reconciled with a
                # newly changed ancestor without recomputing its whole span.
                del self._branches[key]
                self.used_bytes -= previous.bytes
                self._sorted_keys.pop(index)
                continue
            updated = RecycledBranch(
                tokens=key,
                entries=branch.entries + tuple(
                    entry for entry in previous.entries
                    if entry.end > prefix_len
                ),
                protected_endpoints=(
                    branch.protected_endpoints
                    | frozenset(position for position in previous.protected_endpoints
                                if position > prefix_len)
                ),
            )
            self._branches[key] = updated
            self.used_bytes += updated.bytes - previous.bytes
            index += 1
        if branch.bytes > self.capacity_bytes:
            while self.used_bytes > self.capacity_bytes:
                _key, forgotten = self._branches.popitem(last=False)
                self._remove_key(_key)
                self.used_bytes -= forgotten.bytes
            return
        old = self._branches.pop(branch.tokens, None)
        if old is not None:
            self.used_bytes -= old.bytes
        else:
            bisect.insort(self._sorted_keys, branch.tokens)
        self._branches[branch.tokens] = branch
        self.used_bytes += branch.bytes
        while self.used_bytes > self.capacity_bytes:
            _key, forgotten = self._branches.popitem(last=False)
            self._remove_key(_key)
            self.used_bytes -= forgotten.bytes

    def match(self, tokens: torch.Tensor, *, start: int, granularity: int) -> RecycledMatch | None:
        assert tokens.device.type == "cpu"
        best: RecycledMatch | None = None
        values = tuple(tokens.tolist())
        index = bisect.bisect_left(self._sorted_keys, values)
        candidates = self._sorted_keys[max(0, index - 1):index + 1]
        for key in candidates:
            branch = self._branches[key]
            limit = min(len(key), len(values))
            if limit <= start:
                continue
            matched = 0
            for left, right in zip(key[:limit], values[:limit], strict=True):
                if left != right:
                    break
                matched += 1
            matched = matched // granularity * granularity
            if matched > start and (best is None or matched > best.matched_len):
                best = RecycledMatch(branch, matched)
        if best is not None:
            self._branches.move_to_end(best.branch.tokens)
        return best

    def discard(self, branch: RecycledBranch) -> None:
        if self._branches.get(branch.tokens) is branch:
            del self._branches[branch.tokens]
            self.used_bytes -= branch.bytes
            self._remove_key(branch.tokens)

    def __len__(self) -> int:
        return len(self._branches)
