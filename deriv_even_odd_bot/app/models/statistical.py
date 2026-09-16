"""
Statistical models A-D (spec Sections 6, 8, 10, 11).

  DirichletFrequencyModel  -- Bayesian digit frequencies with configurable
                              smoothing (Laplace / Jeffreys / custom prior)
  BetaBinomialParityModel  -- direct parity posterior, hierarchical over
                              rolling windows
  EWMAFrequencyModel       -- recency-weighted frequencies
  MarkovDigitModel         -- P(D_t | D_{t-1}, ...) with order selection
  MarkovParityModel        -- P(P_t | P_{t-1}, ...) with order selection
  RunLengthParityModel     -- P(parity | current run length), WITHOUT
                              gambler's-fallacy assumptions

SMOOTHING IS NOT COSMETIC (Section 6). With 10 digits and a 100-tick
window, an unsmoothed empirical frequency routinely assigns probability
0.00 to a digit that simply hasn't appeared yet. Feeding that into a log
loss or an ensemble gives infinite confidence in a non-event. Every model
here is smoothed, and the prior strength is a tunable that trades
responsiveness against exactly this failure.

MODEL ORDER SELECTION IS GUARDED (Section 10: "Do not blindly increase
order"). A 2nd-order digit Markov chain has 1,000 free parameters; with
5,000 ticks that is 5 observations per cell, and the model will fit the
history perfectly while predicting nothing. Order is selected by BIC, which
penalizes parameter count against sample size, and high orders are simply
unavailable below a minimum-sample threshold.

GAMBLER'S FALLACY IS EXPLICITLY NOT ASSUMED (Section 11). The run-length
model does not encode "five evens means odd is due". It MEASURES
P(parity | run length) from data and lets that be flat if it is flat --
which, on an independent stream, it will be.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

from app.models.base import (
    EVEN_DIGITS,
    DigitModel,
    DigitPrediction,
    normalize_distribution,
)

# Smoothing priors (Section 6: "select the method empirically").
LAPLACE = "laplace"      # alpha = 1.0   -- strong, safe, slow to move
JEFFREYS = "jeffreys"    # alpha = 0.5   -- standard non-informative choice
PERKS = "perks"          # alpha = 1/K   -- weakest; most responsive
SMOOTHING_ALPHAS = {LAPLACE: 1.0, JEFFREYS: 0.5, PERKS: 0.1}


class DirichletFrequencyModel(DigitModel):
    """Model A: Dirichlet-multinomial posterior over the 10 digits.

    Posterior mean for digit d is (count_d + alpha) / (n + 10*alpha), which
    is the Bayes estimate under a symmetric Dirichlet(alpha) prior. Uses a
    bounded window so it tracks the recent stream rather than averaging over
    all history forever.
    """

    def __init__(self, window: int = 1000, smoothing: str = JEFFREYS, name: str | None = None):
        super().__init__(name or f"dirichlet_{window}")
        self.window = window
        self.alpha = SMOOTHING_ALPHAS.get(smoothing, 0.5)
        self.min_samples = max(100, window // 10)
        self._buf: deque[int] = deque(maxlen=window)
        self._counts = [0] * 10

    def observe(self, digit: int) -> None:
        if len(self._buf) == self._buf.maxlen:
            self._counts[self._buf[0]] -= 1
        self._buf.append(digit)
        self._counts[digit] += 1
        self._n_seen += 1

    def predict(self) -> DigitPrediction:
        n = len(self._buf)
        if n == 0:
            return self.uninformative()
        denom = n + 10.0 * self.alpha
        probs = [(self._counts[d] + self.alpha) / denom for d in range(10)]
        return DigitPrediction(self.name, normalize_distribution(probs), n)

    def posterior_concentration(self) -> float:
        """Total pseudo-count behind the posterior -- an honest measure of
        how much evidence the estimate rests on, used by the ensemble."""
        return len(self._buf) + 10.0 * self.alpha


class BetaBinomialParityModel(DigitModel):
    """Model B: direct Beta-Binomial posterior on P(EVEN).

    Hierarchical in the sense that several window lengths are maintained and
    blended by their own posterior precision, so short windows dominate when
    they are sharply informative and long windows carry the estimate when
    they are not. Publishes `direct_p_even`, which the ensemble compares
    against the digit-derived value (Section 3).
    """

    def __init__(self, windows: tuple[int, ...] = (100, 500, 2000),
                 prior_strength: float = 2.0, name: str | None = None):
        super().__init__(name or "beta_binomial_parity")
        self.windows = windows
        self.prior = prior_strength / 2.0    # symmetric Beta(a, a)
        self.min_samples = min(windows)
        self._bufs = {w: deque(maxlen=w) for w in windows}
        self._even = {w: 0 for w in windows}

    def observe(self, digit: int) -> None:
        is_even = 1 if digit % 2 == 0 else 0
        for w, buf in self._bufs.items():
            if len(buf) == buf.maxlen and buf[0] == 1:
                self._even[w] -= 1
            elif len(buf) == buf.maxlen:
                pass
            buf.append(is_even)
            if is_even:
                self._even[w] += 1
        self._n_seen += 1

    def _posterior(self, w: int) -> tuple[float, float]:
        n = len(self._bufs[w])
        a = self._even[w] + self.prior
        b = (n - self._even[w]) + self.prior
        return a, b

    def predict(self) -> DigitPrediction:
        if self._n_seen == 0:
            return self.uninformative()
        # Precision-weighted blend across windows. Precision of a Beta is
        # (a+b+1)/(a*b/(a+b)^2 ...) -- we use a+b (evidence count) directly,
        # which is monotone in precision and numerically better behaved.
        num = 0.0
        den = 0.0
        for w in self.windows:
            a, b = self._posterior(w)
            n = a + b
            if n <= 0:
                continue
            mean = a / n
            num += n * mean
            den += n
        p_even = num / den if den > 0 else 0.5
        # Spread the parity estimate evenly across the five digits of each
        # class: this model makes no claim about WHICH even digit.
        probs = [p_even / 5 if d in EVEN_DIGITS else (1 - p_even) / 5 for d in range(10)]
        return DigitPrediction(self.name, normalize_distribution(probs),
                               self._n_seen, direct_p_even=p_even)


class EWMAFrequencyModel(DigitModel):
    """Recency-weighted digit frequencies (Section 6).

    Effective sample size converges to 1/(1-decay) from below and never
    reaches it -- see app/data/tick_store.py's note on why any downstream
    minimum-sample gate must sit well under that ceiling.
    """

    def __init__(self, decay: float = 0.995, smoothing: float = 0.5, name: str | None = None):
        super().__init__(name or f"ewma_{decay}")
        if not 0 < decay < 1:
            raise ValueError("decay must be in (0, 1)")
        self.decay = decay
        self.alpha = smoothing
        self.ceiling = 1.0 / (1.0 - decay)
        self.min_samples = int(min(500, self.ceiling / 4))
        self._w = [0.0] * 10

    def observe(self, digit: int) -> None:
        for i in range(10):
            self._w[i] *= self.decay
        self._w[digit] += 1.0
        self._n_seen += 1

    def predict(self) -> DigitPrediction:
        total = sum(self._w)
        if total <= 0:
            return self.uninformative()
        probs = [(self._w[d] + self.alpha) / (total + 10 * self.alpha) for d in range(10)]
        return DigitPrediction(self.name, normalize_distribution(probs), self._n_seen)


class MarkovDigitModel(DigitModel):
    """Models C/D: P(D_t | D_{t-1}, ..., D_{t-k}) with BIC order selection.

    Order is chosen by BIC over the observed history rather than fixed:
        BIC = -2*loglik + k_params*ln(n)
    A 1st-order digit chain has 90 free parameters, 2nd-order has 900. BIC
    will refuse the larger model unless the data genuinely supports it,
    which on an independent stream it never will -- the model correctly
    collapses to order 0 (the marginal distribution).
    """

    def __init__(self, max_order: int = 2, smoothing: float = 0.5,
                 min_per_cell: float = 5.0, name: str | None = None):
        super().__init__(name or f"markov_digit_o{max_order}")
        self.max_order = max_order
        self.alpha = smoothing
        self.min_per_cell = min_per_cell
        self.min_samples = 500
        self._history: deque[int] = deque(maxlen=50000)
        self._counts: dict[int, dict[tuple, list[int]]] = {
            o: defaultdict(lambda: [0] * 10) for o in range(self.max_order + 1)
        }

    def observe(self, digit: int) -> None:
        h = list(self._history)
        for o in range(self.max_order + 1):
            if len(h) >= o:
                ctx = tuple(h[-o:]) if o > 0 else ()
                self._counts[o][ctx][digit] += 1
        self._history.append(digit)
        self._n_seen += 1

    def _order_bic(self, order: int) -> float:
        """Lower is better. Returns +inf when the order is unsupportable."""
        table = self._counts[order]
        if not table:
            return float("inf")
        n = sum(sum(row) for row in table.values())
        if n <= 0:
            return float("inf")
        n_contexts = 10 ** order
        # Section 10: don't trust sparse transitions.
        if order > 0 and n / n_contexts < self.min_per_cell:
            return float("inf")
        loglik = 0.0
        for ctx, row in table.items():
            row_n = sum(row)
            if row_n <= 0:
                continue
            denom = row_n + 10 * self.alpha
            for d in range(10):
                if row[d] > 0:
                    loglik += row[d] * math.log((row[d] + self.alpha) / denom)
        k_params = n_contexts * 9
        return -2.0 * loglik + k_params * math.log(n)

    def best_order(self) -> int:
        scores = {o: self._order_bic(o) for o in range(self.max_order + 1)}
        finite = {o: s for o, s in scores.items() if s != float("inf")}
        if not finite:
            return 0
        return min(finite, key=finite.get)

    def predict(self) -> DigitPrediction:
        if self._n_seen < self.min_samples:
            return self.uninformative()
        order = self.best_order()
        h = list(self._history)
        if order == 0 or len(h) < order:
            row = self._counts[0].get((), [0] * 10)
        else:
            row = self._counts[order].get(tuple(h[-order:]))
            if row is None or sum(row) < self.min_per_cell:
                row = self._counts[0].get((), [0] * 10)
        total = sum(row)
        if total <= 0:
            return self.uninformative()
        probs = [(row[d] + self.alpha) / (total + 10 * self.alpha) for d in range(10)]
        return DigitPrediction(self.name, normalize_distribution(probs), self._n_seen)


class MarkovParityModel(DigitModel):
    """Model C restricted to parity: P(P_t | P_{t-1}, ..., P_{t-k}).

    Far fewer parameters than the digit chain (2^k contexts vs 10^k), so
    higher orders are statistically affordable -- this is the model most
    likely to detect alternation or streaking if any exists.
    """

    def __init__(self, max_order: int = 4, smoothing: float = 0.5,
                 min_per_cell: float = 20.0, name: str | None = None):
        super().__init__(name or f"markov_parity_o{max_order}")
        self.max_order = max_order
        self.alpha = smoothing
        self.min_per_cell = min_per_cell
        self.min_samples = 300
        self._history: deque[int] = deque(maxlen=50000)
        self._counts: dict[int, dict[tuple, list[int]]] = {
            o: defaultdict(lambda: [0, 0]) for o in range(self.max_order + 1)
        }

    def observe(self, digit: int) -> None:
        p = digit % 2
        h = list(self._history)
        for o in range(self.max_order + 1):
            if len(h) >= o:
                ctx = tuple(h[-o:]) if o > 0 else ()
                self._counts[o][ctx][p] += 1
        self._history.append(p)
        self._n_seen += 1

    def _order_bic(self, order: int) -> float:
        table = self._counts[order]
        if not table:
            return float("inf")
        n = sum(sum(r) for r in table.values())
        if n <= 0:
            return float("inf")
        n_contexts = 2 ** order
        if order > 0 and n / n_contexts < self.min_per_cell:
            return float("inf")
        loglik = 0.0
        for ctx, row in table.items():
            row_n = sum(row)
            if row_n <= 0:
                continue
            denom = row_n + 2 * self.alpha
            for i in (0, 1):
                if row[i] > 0:
                    loglik += row[i] * math.log((row[i] + self.alpha) / denom)
        return -2.0 * loglik + n_contexts * math.log(n)

    def best_order(self) -> int:
        scores = {o: self._order_bic(o) for o in range(self.max_order + 1)}
        finite = {o: s for o, s in scores.items() if s != float("inf")}
        return min(finite, key=finite.get) if finite else 0

    def predict(self) -> DigitPrediction:
        if self._n_seen < self.min_samples:
            return self.uninformative()
        order = self.best_order()
        h = list(self._history)
        row = None
        if order > 0 and len(h) >= order:
            row = self._counts[order].get(tuple(h[-order:]))
            if row is not None and sum(row) < self.min_per_cell:
                row = None
        if row is None:
            row = self._counts[0].get((), [0, 0])
        total = sum(row)
        if total <= 0:
            return self.uninformative()
        p_even = (row[0] + self.alpha) / (total + 2 * self.alpha)
        probs = [p_even / 5 if d in EVEN_DIGITS else (1 - p_even) / 5 for d in range(10)]
        return DigitPrediction(self.name, normalize_distribution(probs),
                               self._n_seen, direct_p_even=p_even)


class RunLengthParityModel(DigitModel):
    """Section 11: P(parity | current parity run length), MEASURED.

    THIS MODEL DOES NOT ASSUME MEAN REVERSION. The spec is explicit that
    E E E E E does not make O "due". What this does is estimate, from
    history, the conditional probability of the next parity given the
    current run length -- and on an independent stream that estimate comes
    out flat at 0.5 for every run length, which is the correct answer and
    exactly what the model will then report.

    Run lengths are bucketed and capped, because long runs are rare by
    construction: on a fair stream a run of 10 occurs about once per 1,024
    ticks, so per-length estimates beyond the cap are noise no matter how
    much history exists.
    """

    def __init__(self, max_run: int = 8, smoothing: float = 1.0, name: str | None = None):
        super().__init__(name or "run_length_parity")
        self.max_run = max_run
        self.alpha = smoothing
        self.min_samples = 1000
        # counts[(parity_of_run, bucketed_run_length)][next_parity]
        self._counts: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
        self._cur_parity: int | None = None
        self._cur_run = 0

    def _bucket(self, run: int) -> int:
        return min(run, self.max_run)

    def observe(self, digit: int) -> None:
        p = digit % 2
        if self._cur_parity is not None:
            key = (self._cur_parity, self._bucket(self._cur_run))
            self._counts[key][p] += 1
        if p == self._cur_parity:
            self._cur_run += 1
        else:
            self._cur_parity = p
            self._cur_run = 1
        self._n_seen += 1

    def predict(self) -> DigitPrediction:
        if self._n_seen < self.min_samples or self._cur_parity is None:
            return self.uninformative()
        row = self._counts.get((self._cur_parity, self._bucket(self._cur_run)))
        if row is None or sum(row) < 30:
            return self.uninformative()
        total = sum(row)
        p_even = (row[0] + self.alpha) / (total + 2 * self.alpha)
        probs = [p_even / 5 if d in EVEN_DIGITS else (1 - p_even) / 5 for d in range(10)]
        return DigitPrediction(self.name, normalize_distribution(probs),
                               self._n_seen, direct_p_even=p_even)

    def conditional_table(self) -> dict[tuple[int, int], float]:
        """Diagnostic: P(EVEN | run) for every observed context. Flat output
        across run lengths is the expected, correct result on a fair stream
        and is what refutes gambler's-fallacy reasoning empirically."""
        out = {}
        for key, row in sorted(self._counts.items()):
            total = sum(row)
            if total >= 30:
                out[key] = (row[0] + self.alpha) / (total + 2 * self.alpha)
        return out
