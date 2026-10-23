from dataclasses import dataclass, field

from ..metadata import CacheMetadata


@dataclass
class MarconiCacheMetadata(CacheMetadata):
    checkpoint_writes: list[list[tuple[int, int]]] | None = None
    checkpoint_copies: list[tuple[int, int]] = field(default_factory=list)
