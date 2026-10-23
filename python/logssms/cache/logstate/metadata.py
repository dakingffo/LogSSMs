from dataclasses import dataclass, field

from ..metadata import CacheMetadata


@dataclass
class LogStateCacheMetadata(CacheMetadata):
    first_offsets: list[int] | None = None
    checkpoint_writes: list[tuple[int, int]] = field(default_factory=list)
    checkpoint_ranges: list[tuple[int, int, int]] = field(default_factory=list)
