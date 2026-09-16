"""
Distribution testing, divergence and interval estimation
(spec Sections 7, 26, 40).

Everything here answers one question in different ways: does the observed
digit distribution differ from a reference by more than sampling noise?

DELIBERATE NON-ASSUMPTION (Section 7): the reference distribution is a
PARAMETER, defaulting to uniform but never hard-coded as truth. The spec is
explicit that we must not assume digits are uniform, nor assume they are
not. We measure.

A NOTE ON WHAT SIGNIFICANCE BUYS YOU (Section 40): a chi-square p-value
below alpha says the distribution probably isn't exactly the reference. It
does NOT say the deviation is large enough to overcome a ~2.6% house edge.
Those are different questions, and conflating them is the single most
common way a digit bot talks itself into trading. Effect size
(total_variation_distance, and the parity_edge helper) is reported
alongside every p-value for exactly this reason, and the trade gate keys
off economics, not off p.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

UNIFORM_10 = [0.1] * 10


def _lgamma(x: float) -> float:
    return math.lgamma(x)


def chi2_sf(x: float, df: int) -> float:
    """Survival function of the chi-square distribution.

    Implemented directly (regularized upper incomplete gamma) so this module
    carries no scipy dependency; accuracy is ~1e-12, far beyond what any
    decision here needs.
    """
    if x <= 0:
        return 1.0
    if df <= 0:
        raise ValueError("df must be positive")
    return _gammaincc(df / 2.0, x / 2.0)


def _gammaincc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x)."""
    if x < 0 or a <= 0:
        raise ValueError("invalid arguments")
    if x == 0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)


def _gser(a: float, x: float, itmax: int = 500, eps: float = 1e-14) -> float:
    """Series representation of P(a, x)."""
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(itmax):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * eps:
            break
    return total * math.exp(-x + a * math.log(x) - _lgamma(a))


def _gcf(a: float, x: float, itmax: int = 500, eps: float = 1e-14) -> float:
    """Continued-fraction representation of Q(a, x) (Lentz's method)."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b if b != 0 else 1.0 / tiny
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - _lgamma(a)) * h


@dataclass(frozen=True)
class GoodnessOfFit:
    statistic: float
    df: int
    p_value: float
    n: int
    observed: list[int]
    expected: list[float]
    total_variation_distance: float
    sufficient_sample: bool   # Cochran's rule: all expected counts >= 5

    @property
    def significant_at(self) -> callable:
        return lambda alpha: self.p_value < alpha


def chi_square_gof(counts: list[int], reference: list[float] | None = None) -> GoodnessOfFit:
    """Pearson chi-square goodness-of-fit against a reference distribution."""
    reference = list(reference or UNIFORM_10)
    if len(counts) != len(reference):
        raise ValueError("counts and reference must be the same length")
    n = sum(counts)
    if n == 0:
        return GoodnessOfFit(0.0, max(len(counts) - 1, 1), 1.0, 0, list(counts),
                             [0.0] * len(counts), 0.0, False)
    total_ref = sum(reference)
    probs = [r / total_ref for r in reference]
    expected = [p * n for p in probs]
    stat = 0.0
    for o, e in zip(counts, expected):
        if e > 0:
            stat += (o - e) ** 2 / e
    df = len(counts) - 1
    tvd = 0.5 * sum(abs(o / n - p) for o, p in zip(counts, probs))
    return GoodnessOfFit(
        statistic=stat, df=df, p_value=chi2_sf(stat, df), n=n,
        observed=list(counts), expected=expected,
        total_variation_distance=tvd,
        sufficient_sample=all(e >= 5 for e in expected),
    )


def g_test(counts: list[int], reference: list[float] | None = None) -> GoodnessOfFit:
    """Likelihood-ratio (G) test. Asymptotically equivalent to chi-square but
    better behaved with small expected counts; reported alongside it so a
    disagreement between the two flags a marginal sample rather than hiding
    inside one number."""
    reference = list(reference or UNIFORM_10)
    n = sum(counts)
    if n == 0:
        return GoodnessOfFit(0.0, max(len(counts) - 1, 1), 1.0, 0, list(counts),
                             [0.0] * len(counts), 0.0, False)
    total_ref = sum(reference)
    probs = [r / total_ref for r in reference]
    expected = [p * n for p in probs]
    stat = 0.0
    for o, e in zip(counts, expected):
        if o > 0 and e > 0:
            stat += o * math.log(o / e)
    stat *= 2.0
    df = len(counts) - 1
    tvd = 0.5 * sum(abs(o / n - p) for o, p in zip(counts, probs))
    return GoodnessOfFit(
        statistic=stat, df=df, p_value=chi2_sf(stat, df), n=n,
        observed=list(counts), expected=expected,
        total_variation_distance=tvd,
        sufficient_sample=all(e >= 5 for e in expected),
    )


# ---- divergences (Section 7) ---------------------------------------------

def _normalize(p: list[float]) -> list[float]:
    t = sum(p)
    if t <= 0:
        raise ValueError("distribution sums to zero")
    return [x / t for x in p]


def kl_divergence(p: list[float], q: list[float], eps: float = 1e-12) -> float:
    """D_KL(p || q). Asymmetric; p is the 'observed' side."""
    p, q = _normalize(p), _normalize(q)
    return sum(pi * math.log((pi + eps) / (qi + eps)) for pi, qi in zip(p, q) if pi > 0)


def jensen_shannon_divergence(p: list[float], q: list[float]) -> float:
    """Symmetric, bounded in [0, ln 2]. Preferred for rolling comparisons
    because it stays finite when a digit is momentarily unobserved."""
    p, q = _normalize(p), _normalize(q)
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def total_variation_distance(p: list[float], q: list[float]) -> float:
    p, q = _normalize(p), _normalize(q)
    return 0.5 * sum(abs(pi - qi) for pi, qi in zip(p, q))


def hellinger_distance(p: list[float], q: list[float]) -> float:
    p, q = _normalize(p), _normalize(q)
    return math.sqrt(0.5 * sum((math.sqrt(pi) - math.sqrt(qi)) ** 2 for pi, qi in zip(p, q)))


# ---- intervals (Section 26) ----------------------------------------------

def inv_norm_cdf(p: float) -> float:
    """Acklam's inverse normal CDF approximation (~1e-9 accurate)."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def norm_sf(z: float) -> float:
    """Upper-tail standard normal probability."""
    return 0.5 * math.erfc(z / math.sqrt(2))


@dataclass(frozen=True)
class Interval:
    point: float
    lower: float
    upper: float
    n: int

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def half_width(self) -> float:
        return self.width / 2.0

    def excludes(self, value: float) -> bool:
        return value < self.lower or value > self.upper


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Chosen over the normal-approximation ('Wald') interval because Wald
    misbehaves badly near p=0.5 with small n in the opposite direction from
    what you want here: it is too NARROW, which would make a marginal parity
    reading look more certain than it is and wave it past the uncertainty
    gate.
    """
    if n <= 0:
        return Interval(float("nan"), 0.0, 1.0, 0)
    z = inv_norm_cdf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(point=p, lower=max(0.0, centre - margin),
                    upper=min(1.0, centre + margin), n=n)


def jeffreys_interval(successes: int, n: int, confidence: float = 0.95) -> Interval:
    """Bayesian credible interval under a Beta(0.5, 0.5) Jeffreys prior.

    Reported alongside Wilson because the two disagreeing is itself a useful
    signal that the sample is too small to be making decisions from.
    """
    if n <= 0:
        return Interval(float("nan"), 0.0, 1.0, 0)
    a = successes + 0.5
    b = n - successes + 0.5
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    z = inv_norm_cdf(1 - (1 - confidence) / 2)
    sd = math.sqrt(var)
    return Interval(point=mean, lower=max(0.0, mean - z * sd),
                    upper=min(1.0, mean + z * sd), n=n)


def parity_edge(even_count: int, n: int, confidence: float = 0.95) -> Interval:
    """Observed EVEN rate with interval -- the single most decision-relevant
    number in the bot. The interval, not the point estimate, is what the
    gate compares against break-even."""
    return wilson_interval(even_count, n, confidence)


# ---- multiple testing (Section 40) ---------------------------------------

def benjamini_hochberg(p_values: list[float], fdr: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR control. Returns a mask of which hypotheses
    survive.

    Required wherever many patterns are screened at once (Section 41):
    testing 1,000 candidate patterns at alpha=0.05 yields ~50 'significant'
    findings from pure noise, and a bot that trades them is trading its own
    multiple-comparisons error.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    keep = [False] * m
    max_k = -1
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= fdr * rank / m:
            max_k = rank
    if max_k > 0:
        for rank, idx in enumerate(order, start=1):
            if rank <= max_k:
                keep[idx] = True
    return keep


def sidak_alpha(alpha: float, n_tests: int) -> float:
    """Per-test alpha giving family-wide `alpha` across n independent tests."""
    n_tests = max(1, int(n_tests))
    return 1.0 - (1.0 - alpha) ** (1.0 / n_tests)
