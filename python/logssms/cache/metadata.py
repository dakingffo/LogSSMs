from dataclasses import dataclass


@dataclass
class CacheMetadata:
    replay_counts: list[int] | None = None
