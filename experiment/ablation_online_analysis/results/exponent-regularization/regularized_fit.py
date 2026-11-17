def _fit_hit_bins(
    self, bins: dict[int, float], bucket_exposures: Sequence[float],
) -> tuple[float, float]:
    g = self.granularity
    support = [(math.log((i + 1) * g), e * g)
               for i, e in enumerate(bucket_exposures) if e > 0]
    total = sum(bins.values())
    prior_mass = sum(g * ((i + 1) * g) ** -2 for i in range(max(1, len(bucket_exposures))))
    internal_prior, _ = self._prior()
    alpha, beta = 4.0 * internal_prior, 4.0 * prior_mass
    if not support:
        return 2.0, math.log(alpha / beta)
    observed = sum(c * math.log((i + 1) * g) for i, c in bins.items())

    def terms(k):
        values = [e * math.exp(-k * x) for x, e in support]
        return sum(values), sum(v * x for v, (x, _) in zip(values, support))

    exponent = 2.0
    if total:
        lo, hi = 0.0, self.max_density_exponent
        for _ in range(40):
            mid = (lo + hi) / 2
            mass, moment = terms(mid)
            derivative = (total + alpha) * moment / (mass + beta) - observed - (mid - 2.0)
            if derivative > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-6:
                break
        exponent = (lo + hi) / 2
    mass, _ = terms(exponent)
    return exponent, math.log((total + alpha) / (mass + beta))
