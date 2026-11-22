from functools import cache


@cache
def prefill_partitions(budget: int, granularity: int) -> tuple[tuple[int, ...], ...]:
    if budget <= 0 or granularity <= 0 or budget % granularity:
        raise ValueError("mixed budget must be a positive multiple of prefill alignment")
    parts = [granularity] * (budget // granularity)
    result = [tuple(parts)]
    while len(parts) > 1:
        parts = [sum(parts[i:i + 2]) for i in range(0, len(parts), 2)]
        result.append(tuple(parts))
    return tuple(result)


@cache
def prefill_buckets(budget: int, granularity: int) -> tuple[tuple[int, ...], ...]:
    prefill_partitions(budget, granularity)
    result = []
    units = budget // granularity
    while units:
        result.extend(prefill_partitions(units * granularity, granularity))
        units //= 2
    return tuple(sorted(result, key=lambda p: (sum(p), len(p))))


def select_prefill_partition(
    buckets: tuple[tuple[int, ...], ...], lengths: tuple[int, ...],
) -> tuple[int, ...]:
    return next(p for p in buckets if len(p) >= len(lengths)
                and all(n <= capacity for n, capacity in zip(lengths, p)))
