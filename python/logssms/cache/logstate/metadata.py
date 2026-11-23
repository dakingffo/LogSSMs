from collections import deque
from dataclasses import dataclass, field
from itertools import islice

from ..metadata import CacheMetadata


@dataclass
class LogStateCacheMetadata(CacheMetadata):
    first_offsets: list[int] | None = None
    checkpoint_writes: list[tuple[int, int]] = field(default_factory=list)
    checkpoint_ranges: list[tuple[int, int, int]] = field(default_factory=list)


class ReclamationChanges:
    def __init__(self) -> None:
        self.version = 0
        self._ids: deque[int] = deque(maxlen=4096)

    def record(self, item: int) -> None:
        self._ids.append(item)
        self.version += 1

    def since(self, version: int) -> set[int] | None:
        count = self.version - version
        if count > len(self._ids):
            return None
        return set(islice(reversed(self._ids), count))
