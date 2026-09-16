"""
Model interface, prediction container and health scoring
(spec Sections 8, 9, 53).

EVERY model in this repo returns a full 10-digit distribution, never a bare
parity call. Parity is then DERIVED (Section 3). This is deliberate: it
makes the parity prediction auditable against the digit structure that
supposedly supports it. A model claiming P(EVEN)=0.58 while its digit
distribution is flat is contradicting itself, and that contradiction is
visible only if both are exposed.

HEALTH (Section 53) is scored from proper scoring rules (Brier, log loss)
against a CHANCE BASELINE, not against an absolute threshold. On a fair
stream every model scores exactly at baseline, which must read as HEALTHY-
but-uninformative rather than DEGRADED -- otherwise the bot would disable
its entire ensemble for the crime of correctly reporting 50/50, then
"recover" it on noise.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

EVEN_DIGITS = (0, 2, 4, 6, 8)
ODD_DIGITS = (1, 3, 5, 7, 9)

HEALTHY = "HEALTHY"
WARNING = "WARNING"
DEGRADED = "DEGRADED"
DISABLED = "DISABLED"


def normalize_distribution(p: list[float]) -> list[float]:
    """Projects onto the probability simplex. Spec Section 9 requires
    sum(P(d)) == 1; models that compute unnormalized scores route through
    here rather than each re-implementing (and subtly mis-implementing) it."""
    if len(p) != 10:
        raise ValueError(f"expected 10 digit probabilities, got {len(p)}")
    clipped = [max(0.0, float(x)) for x in p]
    total = sum(clipped)
    if total <= 0:
        return [0.1] * 10
    return [x / total for x in clipped]


@dataclass(frozen=True)
class DigitPrediction:
    """One model's output for one tick."""
    model: str
    digit_probabilities: list[float]
    n_samples: int
    # Models may publish a parity estimate that is NOT simply the sum of
    # their digit probabilities (e.g. a direct parity model). When None,
    # parity is derived. Section 3 wants both views available.
    direct_p_even: float | None = None

    def __post_init__(self) -> None:
        if abs(sum(self.digit_probabilities) - 1.0) > 1e-6:
            raise ValueError(f"{self.model}: digit probabilities must sum to 1")

    @property
    def derived_p_even(self) -> float:
        return sum(self.digit_probabilities[d] for d in EVEN_DIGITS)

    @property
    def derived_p_odd(self) -> float:
        return 1.0 - self.derived_p_even

    @property
    def p_even(self) -> float:
        return self.direct_p_even if self.direct_p_even is not None else self.derived_p_even

    @property
    def p_odd(self) -> float:
        return 1.0 - self.p_even

    @property
    def parity_consistency(self) -> float:
        """|direct - derived|. Large values mean the model's own two views
        disagree, which is a reason to distrust it rather than to pick the
        more favourable number."""
        if self.direct_p_even is None:
            return 0.0
        return abs(self.direct_p_even - self.derived_p_even)


@dataclass
class ModelPerformance:
    """Rolling proper-scoring-rule performance for one model.

    Brier and log loss are computed on the PARITY call, since that is what
    is traded. Both are compared against the 0.5 baseline.
    """
    window: int = 500
    _brier: deque[float] = field(default_factory=lambda: deque(maxlen=500))
    _logloss: deque[float] = field(default_factory=lambda: deque(maxlen=500))
    _correct: deque[int] = field(default_factory=lambda: deque(maxlen=500))
    total_updates: int = 0

    def __post_init__(self) -> None:
        self._brier = deque(maxlen=self.window)
        self._logloss = deque(maxlen=self.window)
        self._correct = deque(maxlen=self.window)

    def record(self, p_even: float, actual_parity: int) -> None:
        """actual_parity: 0 if the realized digit was EVEN, 1 if ODD.

        NOTE (same requirement as calibration and adaptive weighting): this
        is the REALIZED PARITY, never 'did the trade win'. Those differ
        whenever the traded side wasn't the model's call, and conflating
        them teaches every model that it agreed with whatever was traded.
        """
        y = 1.0 if actual_parity == 0 else 0.0   # y = 1 means EVEN happened
        p = min(max(p_even, 1e-9), 1 - 1e-9)
        self._brier.append((p - y) ** 2)
        self._logloss.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
        self._correct.append(1 if (p >= 0.5) == (y == 1.0) else 0)
        self.total_updates += 1

    @property
    def n(self) -> int:
        return len(self._brier)

    @property
    def brier(self) -> float:
        return sum(self._brier) / len(self._brier) if self._brier else float("nan")

    @property
    def log_loss(self) -> float:
        return sum(self._logloss) / len(self._logloss) if self._logloss else float("nan")

    @property
    def accuracy(self) -> float:
        return sum(self._correct) / len(self._correct) if self._correct else float("nan")

    # Chance baselines for a binary outcome at p=0.5.
    BASELINE_BRIER = 0.25
    BASELINE_LOGLOSS = math.log(2)

    @property
    def brier_skill(self) -> float:
        """1 - brier/baseline. >0 beats chance, <0 is worse than chance,
        0 is exactly chance (the expected value on a fair stream)."""
        if not self._brier:
            return float("nan")
        return 1.0 - self.brier / self.BASELINE_BRIER

    def health(self, min_samples: int = 100) -> str:
        """Section 53. Chance-level performance is HEALTHY-uninformative,
        not DEGRADED -- see module docstring."""
        if self.n < min_samples:
            return WARNING          # not yet judgeable
        skill = self.brier_skill
        if skill != skill:          # NaN
            return WARNING
        if skill < -0.05:
            return DISABLED         # reliably worse than a coin: actively harmful
        if skill < -0.01:
            return DEGRADED
        return HEALTHY


class DigitModel(ABC):
    """Base class for every predictive model (Sections 8-10).

    Contract:
      update(digits)  -- fold in the realized history. Must never see the
                         digit it is about to predict (Section 22).
      predict()       -- full 10-digit distribution as of everything folded
                         in so far.
    """

    name: str = "base"
    min_samples: int = 100

    def __init__(self, name: str | None = None):
        if name:
            self.name = name
        self.performance = ModelPerformance()
        self._n_seen = 0

    @property
    def n_seen(self) -> int:
        return self._n_seen

    @property
    def is_ready(self) -> bool:
        return self._n_seen >= self.min_samples

    @abstractmethod
    def observe(self, digit: int) -> None:
        """Fold one realized digit into state."""

    @abstractmethod
    def predict(self) -> DigitPrediction:
        """Distribution over the NEXT digit."""

    def uninformative(self) -> DigitPrediction:
        """The honest output before a model has earned an opinion.

        Returning a flat distribution rather than None matters: it lets the
        ensemble average over a fixed set of models instead of silently
        changing composition as models warm up, which would make its output
        jump for reasons unrelated to the market.
        """
        return DigitPrediction(self.name, [0.1] * 10, self._n_seen)

    def health(self) -> str:
        return self.performance.health()

    def record_outcome(self, p_even: float, actual_parity: int) -> None:
        self.performance.record(p_even, actual_parity)
