"""
Information-theory engine (spec Sections 12, 13).

Entropy, conditional entropy and mutual information over the digit and
parity streams.

TWO TRAPS THIS MODULE IS BUILT AROUND, both called out by the spec:

1. "Low entropy does NOT automatically mean a trade should occur"
   (Section 13). Entropy is a descriptive statistic, not a signal. A window
   can show reduced entropy purely because it is short. Every function here
   reports the sample size used, and mutual_information() additionally
   reports a BIAS-CORRECTED value, because...

2. MUTUAL INFORMATION IS BIASED UPWARD. This is the important one. The
   plug-in MI estimator on finite samples is positive in expectation even
   when the true MI is exactly zero -- the bias is approximately
   (r-1)(s-1)/(2n) nats for an r x s table. For digit pairs that is
   81/(2n): with n=1,000 ticks you see ~0.04 nats of "dependence" in a
   perfectly independent stream, every time.

   A bot that treats raw MI as evidence of predictability will therefore
   find predictability in white noise, reliably and forever, and will be
   most confident exactly when it has least data. The Miller-Madow
   correction subtracts the leading bias term, and mi_significance() gives
   the proper test (2*n*MI is asymptotically chi-square with (r-1)(s-1) df).
   Use those, not the raw value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from app.statistics.distribution import chi2_sf


def shannon_entropy(counts: list[int] | list[float], base: float = math.e) -> float:
    """Entropy of an empirical distribution given raw counts."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h / math.log(base) if base != math.e else h


def digit_entropy(digits: list[int], base: float = math.e) -> float:
    counts = [0] * 10
    for d in digits:
        counts[d] += 1
    return shannon_entropy(counts, base)


def parity_entropy(digits: list[int], base: float = math.e) -> float:
    even = sum(1 for d in digits if d % 2 == 0)
    return shannon_entropy([even, len(digits) - even], base)


def max_digit_entropy(base: float = math.e) -> float:
    return math.log(10) / (math.log(base) if base != math.e else 1.0)


def normalized_digit_entropy(digits: list[int]) -> float:
    """Entropy as a fraction of maximum. 1.0 = perfectly uniform window.

    Interpret with care: a 20-tick window of 10 possible digits CANNOT
    reach 1.0 (at most 20 of 10 categories are populated), so short windows
    structurally look 'low entropy'. Compare like with like.
    """
    if not digits:
        return float("nan")
    return digit_entropy(digits) / math.log(10)


def conditional_entropy(pairs: list[tuple[int, int]], n_x: int = 10, n_y: int = 10) -> float:
    """H(Y | X) in nats from observed (x, y) pairs."""
    if not pairs:
        return 0.0
    joint: dict[tuple[int, int], int] = {}
    marg_x: dict[int, int] = {}
    for x, y in pairs:
        joint[(x, y)] = joint.get((x, y), 0) + 1
        marg_x[x] = marg_x.get(x, 0) + 1
    n = len(pairs)
    h = 0.0
    for (x, y), c in joint.items():
        p_xy = c / n
        p_x = marg_x[x] / n
        if p_xy > 0 and p_x > 0:
            h -= p_xy * math.log(p_xy / p_x)
    return h


@dataclass(frozen=True)
class MutualInformation:
    raw: float               # plug-in estimator -- biased UPWARD, do not use alone
    corrected: float         # Miller-Madow bias-corrected
    n: int
    df: int
    g_statistic: float       # 2*n*MI, asymptotically chi-square(df)
    p_value: float
    expected_bias: float     # (r-1)(s-1)/(2n), the noise floor for raw

    @property
    def is_significant_at(self):
        return lambda alpha: self.p_value < alpha


def mutual_information(pairs: list[tuple[int, int]], n_x: int = 10, n_y: int = 10) -> MutualInformation:
    """I(X; Y) with bias correction and a significance test.

    ALWAYS read `corrected` and `p_value`, never `raw` on its own -- see the
    module docstring. `expected_bias` is printed alongside so the size of
    the illusion is visible: if raw ~= expected_bias, you are looking at
    sampling noise, not structure.
    """
    n = len(pairs)
    if n == 0:
        return MutualInformation(0.0, 0.0, 0, 1, 0.0, 1.0, 0.0)

    joint: dict[tuple[int, int], int] = {}
    mx: dict[int, int] = {}
    my: dict[int, int] = {}
    for x, y in pairs:
        joint[(x, y)] = joint.get((x, y), 0) + 1
        mx[x] = mx.get(x, 0) + 1
        my[y] = my.get(y, 0) + 1

    mi = 0.0
    for (x, y), c in joint.items():
        p_xy = c / n
        p_x = mx[x] / n
        p_y = my[y] / n
        if p_xy > 0:
            mi += p_xy * math.log(p_xy / (p_x * p_y))

    # Miller-Madow: subtract the leading-order bias using the number of
    # OBSERVED (non-empty) categories, not the nominal alphabet size --
    # using the nominal size over-corrects on short windows where many
    # digits simply haven't appeared yet.
    r = max(1, len(mx))
    s = max(1, len(my))
    df = max(1, (r - 1) * (s - 1))
    bias = df / (2.0 * n)
    corrected = max(0.0, mi - bias)

    g = 2.0 * n * mi
    return MutualInformation(
        raw=mi, corrected=corrected, n=n, df=df,
        g_statistic=g, p_value=chi2_sf(g, df), expected_bias=bias,
    )


def digit_transition_mi(digits: list[int]) -> MutualInformation:
    """I(D_t ; D_{t-1}) -- does the previous digit tell you anything?"""
    return mutual_information(list(zip(digits, digits[1:])), 10, 10)


def parity_transition_mi(digits: list[int]) -> MutualInformation:
    """I(P_t ; P_{t-1}) -- the directly tradeable version."""
    parity = [d % 2 for d in digits]
    return mutual_information(list(zip(parity, parity[1:])), 2, 2)


def lagged_parity_mi(digits: list[int], max_lag: int = 5) -> list[tuple[int, MutualInformation]]:
    """Parity MI at several lags. Feed the p-values through
    benjamini_hochberg() before believing any of them -- testing 5 lags at
    alpha=0.05 gives a ~23% chance of at least one false positive."""
    parity = [d % 2 for d in digits]
    out = []
    for lag in range(1, max_lag + 1):
        if len(parity) <= lag + 30:
            break
        pairs = list(zip(parity[:-lag], parity[lag:]))
        out.append((lag, mutual_information(pairs, 2, 2)))
    return out
