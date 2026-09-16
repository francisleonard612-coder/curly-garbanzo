"""
Probability calibration (spec Section 20 -- mandatory).

Raw model confidence is not probability. A model that says 0.58 must be
right about 58% of the time for the expected-value arithmetic in
app/economics/ to mean anything; if it is actually right 51% of the time,
every EV calculation downstream is fiction and the bot will trade
confidently into negative expectancy.

IMPLEMENTED: Platt scaling (logistic, parametric, works from ~100 samples)
and isotonic regression (non-parametric, needs more data but can fix
non-monotone miscalibration). Both are fit on a rolling window of
(predicted, realized) pairs and refreshed periodically.

THREE HARD-WON BEHAVIOURS, each from a real failure in a sibling bot:

1. A CALIBRATOR THAT HAS NOT FIT MUST SAY SO. A live log was once found
   where every trade showed raw_probability == calibrated_probability bit
   for bit: the calibrator had never collected enough samples to fit, so a
   downstream gate comparing "raw AND calibrated both exceed threshold" was
   comparing one number to itself twice and calling it two independent
   confirmations. `is_fitted` is therefore public and the gate must check
   it (see app/execution/gating.py).

2. A FITTED-BUT-BAD CALIBRATOR IS WORSE THAN NONE. It looks authoritative
   while being wrong. quality_score() (ECE-based, 0..1) exposes this, and
   the gate can require a minimum.

3. GATING ON QUALITY CREATES A PERMANENT LOCKOUT UNLESS YOU BREAK IT
   DELIBERATELY. If a low quality_score blocks trading, no new trades
   settle, so no new calibration samples arrive, so quality_score can never
   improve -- the block is self-sustaining and permanent, fixable only by
   restarting with fresh state. This is the same shape as the decay-ceiling
   lockout documented in app/data/tick_store.py. The escape hatch is
   `should_probe()`: after N consecutive blocks, one candidate is allowed
   past the QUALITY gate only (never past the economic gates), purely so a
   real settlement can feed a fresh sample in. The streak resets only when
   a probe actually produces a settled outcome, not merely when one is
   attempted.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    n: int
    mean_predicted: float
    observed_rate: float

    @property
    def gap(self) -> float:
        return abs(self.mean_predicted - self.observed_rate)


class CalibrationTracker:
    """Rolling calibration for one (symbol, side) stream."""

    def __init__(self, min_samples: int = 300, window: int = 3000,
                 method: str = "platt", refit_every: int = 100,
                 probe_interval: int = 50):
        if method not in ("platt", "isotonic"):
            raise ValueError("method must be 'platt' or 'isotonic'")
        self.min_samples = min_samples
        self.method = method
        self.refit_every = refit_every
        self.probe_interval = probe_interval
        self._raw: deque[float] = deque(maxlen=window)
        self._outcome: deque[int] = deque(maxlen=window)
        self._since_refit = 0
        self._a = 1.0       # Platt slope
        self._b = 0.0       # Platt intercept
        self._iso: list[tuple[float, float]] | None = None
        self._fitted = False
        self._blocked_streak = 0

    # ---- recording -------------------------------------------------------

    def record(self, raw_probability: float, outcome: int) -> None:
        """`outcome` is 1 if the predicted-for EVENT occurred, else 0.

        Must be the REALIZED EVENT (did EVEN happen), never "did the trade
        win" -- those differ whenever the traded side wasn't the side this
        probability referred to, and conflating them biases the fit toward
        whatever was traded.
        """
        if outcome not in (0, 1):
            raise ValueError("outcome must be 0 or 1")
        self._raw.append(min(max(float(raw_probability), 1e-6), 1 - 1e-6))
        self._outcome.append(int(outcome))
        self._since_refit += 1
        if self._since_refit >= self.refit_every and len(self._raw) >= self.min_samples:
            self.fit()

    @property
    def n_samples(self) -> int:
        return len(self._raw)

    @property
    def is_fitted(self) -> bool:
        """See docstring point 1 -- gates MUST check this rather than
        assuming calibrate() did something."""
        return self._fitted

    # ---- fitting ---------------------------------------------------------

    def fit(self) -> bool:
        if len(self._raw) < self.min_samples:
            return False
        ok = self._fit_platt() if self.method == "platt" else self._fit_isotonic()
        self._since_refit = 0
        self._fitted = self._fitted or ok
        return ok

    def _fit_platt(self) -> bool:
        """Logistic regression of outcome on log-odds of the raw probability."""
        xs = [math.log(p / (1 - p)) for p in self._raw]
        ys = list(self._outcome)
        if len(set(ys)) < 2:
            return False
        a, b = self._a, self._b
        for _ in range(200):
            ga = gb = 0.0
            for x, y in zip(xs, ys):
                z = max(min(a * x + b, 35.0), -35.0)
                p = 1.0 / (1.0 + math.exp(-z))
                ga += (p - y) * x
                gb += (p - y)
            n = len(xs)
            a -= 0.1 * ga / n
            b -= 0.1 * gb / n
        if not (math.isfinite(a) and math.isfinite(b)):
            return False
        self._a, self._b = a, b
        return True

    def _fit_isotonic(self) -> bool:
        """Pool-adjacent-violators isotonic regression."""
        pairs = sorted(zip(self._raw, self._outcome))
        if not pairs:
            return False
        blocks = [[p, float(y), 1] for p, y in pairs]   # [x, sum_y, count]
        changed = True
        while changed:
            changed = False
            out: list[list[float]] = []
            for blk in blocks:
                if out and out[-1][1] / out[-1][2] > blk[1] / blk[2]:
                    prev = out.pop()
                    merged = [prev[0], prev[1] + blk[1], prev[2] + blk[2]]
                    out.append(merged)
                    changed = True
                else:
                    out.append(blk)
            blocks = out
        self._iso = [(b[0], b[1] / b[2]) for b in blocks]
        return True

    # ---- applying --------------------------------------------------------

    def calibrate(self, raw_probability: float) -> float:
        """Maps a raw probability to a calibrated one.

        Returns the input UNCHANGED when not fitted. Callers must therefore
        consult is_fitted rather than treating the output as independent
        evidence (docstring point 1).
        """
        p = min(max(float(raw_probability), 1e-6), 1 - 1e-6)
        if not self._fitted:
            return p
        if self.method == "platt":
            x = math.log(p / (1 - p))
            z = max(min(self._a * x + self._b, 35.0), -35.0)
            return 1.0 / (1.0 + math.exp(-z))
        if not self._iso:
            return p
        lo, hi = 0, len(self._iso) - 1
        if p <= self._iso[0][0]:
            return self._iso[0][1]
        if p >= self._iso[-1][0]:
            return self._iso[-1][1]
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self._iso[mid][0] <= p:
                lo = mid
            else:
                hi = mid
        x0, y0 = self._iso[lo]
        x1, y1 = self._iso[hi]
        if x1 == x0:
            return y0
        return y0 + (y1 - y0) * (p - x0) / (x1 - x0)

    # ---- diagnostics -----------------------------------------------------

    def reliability_curve(self, n_bins: int = 10) -> list[ReliabilityBin]:
        if not self._raw:
            return []
        bins: list[ReliabilityBin] = []
        for i in range(n_bins):
            lo, hi = i / n_bins, (i + 1) / n_bins
            sel = [(p, y) for p, y in zip(self._raw, self._outcome)
                   if (lo <= p < hi or (i == n_bins - 1 and p == 1.0))]
            if not sel:
                continue
            mp = sum(p for p, _ in sel) / len(sel)
            orate = sum(y for _, y in sel) / len(sel)
            bins.append(ReliabilityBin(lo, hi, len(sel), mp, orate))
        return bins

    def expected_calibration_error(self, n_bins: int = 10) -> float:
        bins = self.reliability_curve(n_bins)
        n = sum(b.n for b in bins)
        if n == 0:
            return float("nan")
        return sum(b.n / n * b.gap for b in bins)

    def maximum_calibration_error(self, n_bins: int = 10) -> float:
        bins = self.reliability_curve(n_bins)
        return max((b.gap for b in bins), default=float("nan"))

    def brier_score(self) -> float:
        if not self._raw:
            return float("nan")
        return sum((p - y) ** 2 for p, y in zip(self._raw, self._outcome)) / len(self._raw)

    def log_loss(self) -> float:
        if not self._raw:
            return float("nan")
        return -sum(y * math.log(p) + (1 - y) * math.log(1 - p)
                    for p, y in zip(self._raw, self._outcome)) / len(self._raw)

    def quality_score(self) -> float:
        """0..1 reliability score derived from ECE. Returns a fixed low
        placeholder below 30 samples -- see the probe machinery for why
        gating on that placeholder without an escape hatch is a lockout."""
        if len(self._raw) < 30:
            return 0.0
        ece = self.expected_calibration_error()
        if ece != ece:
            return 0.0
        return max(0.0, min(1.0, 1.0 - ece / 0.25))

    # ---- lockout escape (docstring point 3) ------------------------------

    def note_blocked(self) -> None:
        self._blocked_streak += 1

    def note_settled(self) -> None:
        """Called when a real outcome was recorded for this stream: the
        lockout is broken, so the streak resets. Deliberately NOT called on
        'a probe was attempted' -- a probe that gets crowded out by a better
        candidate achieved nothing and must remain eligible."""
        self._blocked_streak = 0

    def should_probe(self) -> bool:
        if self.probe_interval <= 0:
            return False
        return self._blocked_streak >= self.probe_interval

    @property
    def blocked_streak(self) -> int:
        return self._blocked_streak
