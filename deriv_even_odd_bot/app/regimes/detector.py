"""
Regime detection and trading state (spec Section 17).

Regimes here describe the PREDICTABILITY of the stream, not the direction
of price. That framing matters: this bot does not care whether the market
is trending, it cares whether the digit process currently departs from
randomness in a way any model could exploit.

States map directly onto Section 17's required set. The default and
overwhelmingly likely steady state on a CSPRNG instrument is
STABLE_UNPREDICTABLE, which routes to WAIT forever -- correct behaviour,
not a malfunction.

CHANGE-POINT DETECTION uses a windowed JSD against the long-run
distribution plus a CUSUM on the parity rate. Both are cheap and
incremental. Neither is trusted on its own: a change point is an
instruction to RE-EXAMINE (and to reset calibration, since a distribution
shift invalidates a fitted calibrator), never an instruction to trade.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.statistics.distribution import chi_square_gof, jensen_shannon_divergence
from app.statistics.information import normalized_digit_entropy

STABLE_UNPREDICTABLE = "STABLE_UNPREDICTABLE"   # random-looking: the expected state
HIGH_PREDICTABILITY = "HIGH_PREDICTABILITY"
LOW_PREDICTABILITY = "LOW_PREDICTABILITY"
DISTRIBUTION_SHIFT = "DISTRIBUTION_SHIFT"
TRANSITION_INSTABILITY = "TRANSITION_INSTABILITY"
MODEL_DISAGREEMENT = "MODEL_DISAGREEMENT"
DEGRADED_EDGE = "DEGRADED_EDGE"
WARMING_UP = "WARMING_UP"

# Trading states (Section 17)
TRADE = "TRADE"
WAIT = "WAIT"
COOLDOWN = "COOLDOWN"
MODEL_RECALIBRATION = "MODEL_RECALIBRATION"

TRADEABLE_REGIMES = {HIGH_PREDICTABILITY}


@dataclass
class RegimeState:
    regime: str
    trading_state: str
    reason: str
    jsd_vs_long_run: float = 0.0
    entropy: float = 1.0
    chi2_p: float = 1.0
    cusum: float = 0.0
    change_point_detected: bool = False
    details: dict = field(default_factory=dict)

    @property
    def is_tradeable(self) -> bool:
        return self.trading_state == TRADE


class RegimeDetector:
    """Incremental regime classifier.

    `cusum_threshold` is in units of standard deviations of the parity rate
    and deliberately high (default 5.0): a CUSUM that fires often would
    force constant recalibration, throwing away calibration samples faster
    than they accumulate and re-creating the cold-start lockout the
    calibrator's probe machinery exists to avoid.
    """

    def __init__(self, *, min_samples: int = 2000, jsd_threshold: float = 0.02,
                 entropy_floor: float = 0.985, cusum_threshold: float = 5.0,
                 dispersion_threshold: float = 0.10):
        self.min_samples = min_samples
        self.jsd_threshold = jsd_threshold
        self.entropy_floor = entropy_floor
        self.cusum_threshold = cusum_threshold
        self.dispersion_threshold = dispersion_threshold
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._n = 0

    def update_cusum(self, parity: int) -> float:
        """One-sided CUSUM pair on the parity stream (target rate 0.5)."""
        self._n += 1
        x = (1.0 if parity == 0 else 0.0) - 0.5
        self._cusum_pos = max(0.0, self._cusum_pos + x - 0.01)
        self._cusum_neg = max(0.0, self._cusum_neg - x - 0.01)
        return max(self._cusum_pos, self._cusum_neg)

    def reset_cusum(self) -> None:
        self._cusum_pos = self._cusum_neg = 0.0

    def classify(self, *, recent_digits: list[int], long_run_freq: list[float],
                 edge_verdict=None, ensemble_result=None,
                 calibration_quality: float | None = None) -> RegimeState:
        n = len(recent_digits)
        if n < self.min_samples:
            return RegimeState(WARMING_UP, WAIT,
                               f"warming up: {n}/{self.min_samples} ticks")

        counts = [0] * 10
        for d in recent_digits:
            counts[d] += 1
        recent_freq = [c / n for c in counts]
        jsd = jensen_shannon_divergence(recent_freq, long_run_freq) if sum(long_run_freq) > 0 else 0.0
        entropy = normalized_digit_entropy(recent_digits)
        gof = chi_square_gof(counts)
        cusum = max(self._cusum_pos, self._cusum_neg)
        sd = 0.5 / math.sqrt(max(self._n, 1))
        change = cusum > self.cusum_threshold * sd

        base = dict(jsd_vs_long_run=jsd, entropy=entropy, chi2_p=gof.p_value,
                    cusum=cusum, change_point_detected=change)

        if change or jsd > self.jsd_threshold:
            return RegimeState(
                DISTRIBUTION_SHIFT, MODEL_RECALIBRATION,
                f"distribution shift (JSD={jsd:.4f}, CUSUM={cusum:.3f}) "
                f"-- models and calibration must re-fit before trading", **base)

        if ensemble_result is not None:
            if ensemble_result.extreme_without_evidence:
                return RegimeState(MODEL_DISAGREEMENT, WAIT,
                                   "extreme probability without supporting evidence "
                                   "(Section 45)", **base)
            if ensemble_result.dispersion > self.dispersion_threshold:
                return RegimeState(MODEL_DISAGREEMENT, WAIT,
                                   f"models disagree (dispersion={ensemble_result.dispersion:.4f})",
                                   **base)

        if calibration_quality is not None and calibration_quality < 0.3:
            return RegimeState(DEGRADED_EDGE, MODEL_RECALIBRATION,
                               f"calibration quality {calibration_quality:.3f} too low", **base)

        if edge_verdict is not None and edge_verdict.tradeable:
            return RegimeState(HIGH_PREDICTABILITY, TRADE,
                               "randomness battery found an economically relevant departure",
                               **base)

        if entropy >= self.entropy_floor and gof.p_value > 0.01:
            return RegimeState(
                STABLE_UNPREDICTABLE, WAIT,
                f"stream behaves as random (entropy={entropy:.4f}, chi2 p={gof.p_value:.3f}) "
                f"-- Section 43: TRADE=FALSE", **base)

        return RegimeState(LOW_PREDICTABILITY, WAIT,
                           "some structure detected but not established as tradeable", **base)
