"""
Agreement-outcome calibration.

WHY THIS EXISTS. The opportunity scorer already has a `model_agreement`
contributor (app/evidence/opportunity.py), but it is a small, additive
weight (0.07) chosen for score-blending purposes, not a claim that agreement
predicts the realized parity. Nothing in this codebase has ever measured
whether "the models agree" is actually associated with "the models were
right" on THIS stream. Before agreement can be used as a trading gate --
hard or soft -- that relationship has to be measured, not assumed.

WHAT THIS RECORDS, PER TICK, FOR EVERY CANDIDATE (not just trades):
    - the model agreement fraction at prediction time (Section 22 ordering:
      this is captured in evaluate(), BEFORE the tick is observed)
    - which side the ensemble favoured (its predicted parity)
    - whether that prediction was actually correct once the tick resolved

WHAT IT DOES NOT DO. It does not gate anything. Like ShadowThresholdAnalyzer,
this is a one-way, observation-only accumulator: record() is called from the
learn() path and nothing reads this to make a trade decision. It exists to
answer "if we required agreement >= X, would that X actually have picked out
above-chance outcomes, and with how much data behind it" -- honestly, with a
real interval, not a single point estimate.

WHY WILSON, NOT A RAW HIT RATE. A raw hit rate on a bucket with 12 samples is
noise wearing a percentage sign. The Wilson score interval gives a defensible
lower bound that accounts for sample size, so a bucket only gets reported as
"looks better than chance" once it has enough evidence to support that, not
because 4 of 6 tosses happened to land right.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def _wilson_lower_bound(hits: int, n: int, *, z: float = 1.96) -> float | None:
    """95% Wilson lower bound on a binomial proportion. None if n == 0."""
    if n == 0:
        return None
    p = hits / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (centre - margin) / denom


@dataclass
class AgreementBucket:
    lower: float
    upper: float
    n: int = 0
    hits: int = 0

    @property
    def hit_rate(self) -> float | None:
        return self.hits / self.n if self.n else None

    @property
    def wilson_lower(self) -> float | None:
        return _wilson_lower_bound(self.hits, self.n)


@dataclass
class AgreementCalibrationReport:
    buckets: list[AgreementBucket]
    total: int
    break_even_accuracy: float
    min_samples_for_signal: int
    suggested_threshold: float | None
    summary: str


class AgreementOutcomeTracker:
    """Bins candidates by model-agreement fraction and tracks realized
    accuracy per bin. min_samples_for_signal is deliberately the same order
    of magnitude as CalibrationTracker.min_samples (300): fewer than that in
    a bucket and its Wilson bound is too wide to act on.

    Buckets span the FULL [0.0, 1.0] range, not [0.5, 1.0]. FIXED: the
    original version assumed agreement_fraction (from ensemble.py) can't go
    below 0.5 and clamped anything lower into the [0.5-0.55) bucket. That
    assumption was wrong -- agreement_fraction is "fraction of individual
    members whose own p_even matches the ENSEMBLE's blended direction", not
    a majority-vote fraction, and the blend (derived+direct averaged) can
    end up favouring a side most individual members don't. Observed live:
    agreement_fraction=0.22 on a real tick. Clamping that into the same
    bucket as genuine ~50% splits silently merged two different
    populations -- "roughly even split" and "most models actively
    disagreed with the ensemble's own pick" -- into one bucket's stats.
    """

    def __init__(self, *, n_buckets: int = 10,
                 min_samples_for_signal: int = 300):
        self.n_buckets = int(n_buckets)
        self.min_samples_for_signal = int(min_samples_for_signal)
        edges = [i * (1.0 / self.n_buckets) for i in range(self.n_buckets + 1)]
        self._buckets = [AgreementBucket(edges[i], edges[i + 1])
                         for i in range(self.n_buckets)]
        self._total = 0

    def _bucket_for(self, agreement: float) -> AgreementBucket:
        a = min(max(agreement, 0.0), 1.0)
        idx = min(int(a / (1.0 / self.n_buckets)), self.n_buckets - 1)
        return self._buckets[idx]

    def record(self, *, agreement_fraction: float, predicted_even: bool,
              actual_even: bool) -> None:
        """Called once per resolved tick, from the same place calibrators
        are updated -- after the outcome is known, never before."""
        b = self._bucket_for(agreement_fraction)
        b.n += 1
        if predicted_even == actual_even:
            b.hits += 1
        self._total += 1

    def report(self, *, break_even_accuracy: float = 0.5208) -> AgreementCalibrationReport:
        """break_even_accuracy defaults to the ~1.92x-payout break-even
        (see README); pass the live proposal's break-even when known so the
        bar this compares against matches the actual quoted payout."""
        candidates = [
            b for b in self._buckets
            if b.n >= self.min_samples_for_signal
            and b.wilson_lower is not None
            and b.wilson_lower > break_even_accuracy
        ]
        suggested = min((b.lower for b in candidates), default=None)

        if not any(b.n for b in self._buckets):
            summary = "No candidates recorded yet."
        elif not any(b.n >= self.min_samples_for_signal for b in self._buckets):
            most = max(self._buckets, key=lambda b: b.n)
            summary = (
                f"{self._total} candidates recorded; no bucket has reached "
                f"{self.min_samples_for_signal} samples yet (fullest: "
                f"[{most.lower:.2f}-{most.upper:.2f}) at {most.n}). "
                "Too little data to derive a threshold -- keep collecting."
            )
        elif suggested is None:
            summary = (
                f"{self._total} candidates recorded across buckets with "
                f"enough data, but none clears break-even "
                f"({break_even_accuracy:.4f}) with a 95% lower bound. "
                "No agreement level tested so far predicts the outcome "
                "better than the payout requires -- this is what a fair "
                "stream looks like, and is itself evidence worth keeping."
            )
        else:
            summary = (
                f"Agreement >= {suggested:.2f} clears break-even "
                f"({break_even_accuracy:.4f}) with a 95% confidence lower "
                f"bound, backed by >= {self.min_samples_for_signal} samples. "
                "This is the lowest such bucket found, not a recommendation "
                "on its own -- check the full bucket table before wiring it "
                "into a gate."
            )

        return AgreementCalibrationReport(
            buckets=list(self._buckets), total=self._total,
            break_even_accuracy=break_even_accuracy,
            min_samples_for_signal=self.min_samples_for_signal,
            suggested_threshold=suggested, summary=summary,
        )
