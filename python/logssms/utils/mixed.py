from functools import cache


@cache
def prefill_partitions(budget: int, granularity: int) -> tuple[tuple[int, ...], ...]:
    if budget <= 0 or granularity <= 0 or budget % granularity:
        raise ValueError("mixed budget must be a positive multiple of prefill alignment")
    parts = [granularity] * (budget // granularity)
    result = [tuple(parts)]
    while len(parts) > 1:
        pair = next((i for i in range(len(parts) - 1)
                     if parts[i] == parts[i + 1] == parts[-1]), len(parts) - 2)
        parts[pair:pair + 2] = [sum(parts[pair:pair + 2])]
        parts.sort(reverse=True)
        result.append(tuple(parts))
    return tuple(result)
