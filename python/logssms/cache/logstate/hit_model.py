from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import count
from weakref import WeakValueDictionary


PRIOR_TAIL_ALPHA = 1. / (math.pi * math.pi / 6.)
PRIOR_INTERNAL_ALPHA = 1. - PRIOR_TAIL_ALPHA
PRIOR_EXPOSURE = 32.


@lru_cache(maxsize=512)
def prior_mass(granularity: int, bins: int, exponent: float = 2.) -> float:
    return sum(granularity * (j * granularity) ** -exponent for j in range(1, bins + 1))


@lru_cache(maxsize=512)
def bucket_avg_density(granularity: int, bucket: int, exponent: float) -> float:
    start = 1 << bucket
    return sum((j * granularity) ** -exponent for j in range(start, start << 1)) / start


@dataclass(eq=False)
class LogStateHitStats:
    origin: int
    tail: int
    uuid: int = field(default_factory=lambda: next(LogStateHitStats.counter))
    group: str | int | None = None
    hit_counts: Counter = field(default_factory=Counter)
    tail_hits: float = 0.
    exposure: Counter = field(default_factory=Counter)
    tail_exposure: float = 0.
    exposure_query: int = 0
    hit_query: int = 0
    exponent: float = 2.
    log_amplitude: float | None = None
    tail_probability: float = PRIOR_TAIL_ALPHA / PRIOR_EXPOSURE

    counter = count(1)

    def __post_init__(self):
        if self.group is None:
            self.group = self.uuid


class ReuseModel:
    def __init__(self, granularity, window, half_life, max_exponent, profile_enabled):
        self.g = granularity
        self.window = window
        self.half_life = half_life
        self.max_exponent = max_exponent
        self.query = 0
        self._owners = WeakValueDictionary()
        self._geometry_epoch = None
        self._descendants = {}
        self._rates = {}
        self._events = {}
        self._fit_epoch = -1
        self.profile = ({"aggregate": Counter(), "updates": deque(maxlen=8192)}
                        if profile_enabled else None)

    def new_chain(self, origin, group=None):
        self._fit_epoch = -1
        return LogStateHitStats(origin, origin, group=group,
                                exposure_query=self.query, hit_query=self.query)

    def publish(self, chain, tail):
        if chain.tail != tail:
            self._fit_epoch = -1
            chain.tail = tail
        if tail > chain.origin:
            self._owners[chain.uuid] = chain

    def _weight(self, age):
        return math.exp2(-age / self.half_life)

    def _exposure(self, begin, end):
        return -math.expm1(-math.log(2) * (end - begin) / self.half_life) / (
            -math.expm1(-math.log(2) / self.half_life)
        )

    def _add_widths(self, widths, low, high, share):
        low = max(self.g, low)
        high = min(self.window, high) // self.g * self.g
        while low <= high:
            bucket = (low // self.g).bit_length() - 1
            end = min(high + self.g, self.g << (bucket + 1))
            widths[bucket] += (end - low) * share
            low = end

    def _settle(self, chain, widths, tail_rate):
        elapsed = self.query - chain.exposure_query
        if not elapsed:
            return
        weight = self._weight(elapsed)
        observed = self._exposure(chain.exposure_query, self.query)
        for bucket in chain.exposure:
            chain.exposure[bucket] *= weight
        for bucket, width in widths.items():
            chain.exposure[bucket] += width * observed
        chain.tail_exposure = chain.tail_exposure * weight + tail_rate * observed
        chain.exposure_query = self.query

    def _prepare_geometry(self, tree):
        epoch = tree, tree.topology_revision
        if epoch == self._geometry_epoch:
            return
        for chain, widths in self._rates.items():
            self._settle(chain, widths, 1.)
        descendants = {}
        rates = defaultdict(Counter)
        nodes = tree._iter_nodes()
        for node in reversed(nodes):
            if node.is_root():
                continue
            if node.is_leaf():
                chain = node.analysis_owner
                if chain is None:
                    descendants[node.uuid] = {}
                    continue
                self.publish(chain, node.depth)
                self._settle(chain, {}, 0.)
                rates[chain]
                groups = {chain.group: ((chain, node.depth),)}
            else:
                branches = defaultdict(list)
                for child in node.children.values():
                    for key, targets in descendants[child.uuid].items():
                        branches[key].extend(targets)
                groups = {key: tuple(targets) for key, targets in branches.items()}
            descendants[node.uuid] = groups
            for targets in groups.values():
                share = 1. / len(targets)
                for chain, tail in targets:
                    self._add_widths(rates[chain], tail - node.depth,
                                     tail - node.begin - self.g, share)
        self._geometry_epoch = epoch
        self._descendants = descendants
        self._rates = rates

    def _record(self, targets, position):
        self.query += 1
        if not targets:
            return
        key = targets, position
        value, query = self._events.get(key, (0., self.query))
        self._events[key] = value * self._weight(self.query - query) + 1., self.query
        if len(self._events) >= 1024:
            self._flush()

    def record_query(self, tree, trace, group):
        self._prepare_geometry(tree)
        targets = ()
        if trace is not None and trace.matched_len:
            node = trace.node
            while trace.matched_len <= node.begin and not node.is_root():
                node = node.parent
            groups = self._descendants.get(node.uuid, {})
            if group in groups:
                targets = groups[group]
            elif group is None and len(groups) == 1:
                targets = next(iter(groups.values()))
        self._record(targets, trace.matched_len if trace is not None else 0)

    def record_hit(self, chain, position):
        targets = ((chain, chain.tail),) if chain is not None and 0 < position <= chain.tail else ()
        self._record(targets, position)

    def _flush(self):
        for (targets, position), (value, query) in self._events.items():
            share = value * self._weight(self.query - query) / len(targets)
            for chain, tail in targets:
                age = self.query - chain.hit_query
                if age:
                    weight = self._weight(age)
                    for key in chain.hit_counts:
                        chain.hit_counts[key] *= weight
                    chain.tail_hits *= weight
                    chain.hit_query = self.query
                distance = tail - position
                if distance == 0:
                    chain.tail_hits += share
                elif self.g <= distance <= self.window:
                    chain.hit_counts[(distance // self.g).bit_length() - 1] += share
        self._events.clear()

    def _distance(self, bucket):
        return self.g << bucket

    def _fit(self, chains):
        data = []
        for chain in chains:
            exposures = [(math.log(self._distance(bucket)), width)
                         for bucket, width in chain.exposure.items() if width > 0.]
            count = sum(chain.hit_counts.values())
            if count and len(exposures) > 1:
                observed = sum(value * math.log(self._distance(bucket))
                               for bucket, value in chain.hit_counts.items())
                data.append((exposures, count, observed))
        if not data:
            return min(2., self.max_exponent)
        low, high = 0., self.max_exponent
        for _ in range(24):
            k = (low + high) / 2.
            derivative = 2. - k
            for exposures, count, observed in data:
                weights = [width * math.exp(-k * x) for x, width in exposures]
                derivative += count * sum(x * w for (x, _), w in zip(exposures, weights)) / sum(weights)
                derivative -= observed
            if derivative > 0.:
                low = k
            else:
                high = k
        return (low + high) / 2.

    def refresh(self, extra=()):
        extra = tuple(chain for chain in extra if chain is not None)
        if self.query == self._fit_epoch and all(chain.log_amplitude is not None for chain in extra):
            return
        self._flush()
        groups = defaultdict(list)
        for chain in dict.fromkeys((*self._owners.values(), *extra)):
            widths = self._rates.get(chain)
            if widths is None and self._geometry_epoch is None:
                widths = Counter()
                self._add_widths(widths, self.g, chain.tail - chain.origin - self.g, 1.)
            self._settle(chain, widths or {}, float(widths is not None))
            age = self.query - chain.hit_query
            if age:
                weight = self._weight(age)
                for bucket in tuple(chain.hit_counts):
                    chain.hit_counts[bucket] *= weight
                    if chain.hit_counts[bucket] < 1e-8:
                        del chain.hit_counts[bucket]
                chain.tail_hits *= weight
                if chain.tail_hits < 1e-8:
                    chain.tail_hits = 0.
                chain.hit_query = self.query
            groups[chain.group].append(chain)
        for chains in groups.values():
            k = self._fit(chains)
            if self.profile is not None:
                self.profile["aggregate"]["exponent_fits"] += int(any(s.hit_counts for s in chains))
            for chain in chains:
                self._estimate(chain, k)
        self._fit_epoch = self.query

    def _estimate(self, chain, exponent):
        bins = max(1, (chain.tail - chain.origin) // self.g)
        pm = prior_mass(self.g, bins, exponent)
        exposure = sum(width * bucket_avg_density(self.g, bucket, exponent)
                       for bucket, width in chain.exposure.items())
        internal_hits = sum(chain.hit_counts.values())
        chain.exponent = exponent
        chain.log_amplitude = math.log(
            (internal_hits + PRIOR_INTERNAL_ALPHA) / (exposure + PRIOR_EXPOSURE * pm)
        )
        chain.tail_probability = (chain.tail_hits + PRIOR_TAIL_ALPHA) / (chain.tail_exposure + PRIOR_EXPOSURE)
        if self.profile is not None:
            self.profile["aggregate"]["model_updates"] += 1
            self.profile["updates"].append({
                "owner_uuid": chain.uuid, "query": self.query, "exponent": exponent,
                "log_amplitude": chain.log_amplitude, "tail_probability": chain.tail_probability,
                "weighted_internal_hits": internal_hits, "weighted_tail_hits": chain.tail_hits,
            })
