"""
Model ensemble, adaptive weighting and agreement scoring
(spec Sections 18, 19, 27, 45).

WEIGHTING (Section 19). Weights adapt from rolling Brier skill, with three
safeguards against the failure mode the spec explicitly warns about
("avoid overweighting a model simply because of a tiny recent sample"):

  1. SHRINKAGE. A model's weight is pulled toward the equal-weight baseline
     in proportion to how little data backs its score. A model with 30
     samples barely moves off baseline no matter how good those 30 look.
  2. FLOOR AND CEILING. No model can reach zero weight (which would
     silently change the ensemble's composition) or dominate it.
  3. RATE LIMITING. Weights move by at most `max_weight_step` per update,
     so a short streak cannot reorganize the ensemble before it is clear
     the streak means anything.

WHY NEGATIVE-SKILL MODELS ARE DOWN-WEIGHTED BUT NOT INVERTED: a sibling bot
audit found a signal layer that was reliably ANTI-predictive (45.9% when
agreeing with the traded side vs 55.2% when disagreeing) and it was very
tempting to simply flip its sign. That would have been overfitting to one
sample. A model that appears inverted is far more likely to be noise than
to be a reliable contrarian indicator, so the ensemble reduces its weight
toward the floor and lets further evidence decide -- if the inversion is
real and persists, its weight stays low and it stops doing damage; if it
was noise, it drifts back without anyone having baked in a wrong conclusion.

AGREEMENT (Section 27). Reported as both dispersion (standard deviation of
the parity estimates) and a directional agreement fraction. The spec's own
example is exactly right: four models at 0.56/0.58/0.57/0.51 is a different
epistemic situation from 0.56/0.43/0.61/0.52, even though the means are
similar.

FAILURE PROTECTION (Section 45). collapse_to_chance and extreme_without_
evidence are surfaced explicitly, because both are reasons to NOT trade
that look superficially like ordinary outputs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.models.base import (
    DISABLED,
    EVEN_DIGITS,
    DigitModel,
    DigitPrediction,
    normalize_distribution,
)


@dataclass
class EnsembleResult:
    digit_probabilities: list[float]
    p_even_digit_derived: float
    p_even_direct: float | None
    p_even_ensemble: float
    weights: dict[str, float]
    member_p_even: dict[str, float]
    n_members: int
    n_ready: int

    # Section 27
    dispersion: float
    agreement_fraction: float
    model_entropy: float

    # Section 45
    collapsed_to_chance: bool
    extreme_without_evidence: bool
    members_disabled: list[str] = field(default_factory=list)

    @property
    def p_odd_ensemble(self) -> float:
        return 1.0 - self.p_even_ensemble

    @property
    def side(self) -> str:
        return "EVEN" if self.p_even_ensemble >= 0.5 else "ODD"

    @property
    def confidence(self) -> float:
        """max(P(EVEN), P(ODD)) -- the probability of the favoured side."""
        return max(self.p_even_ensemble, self.p_odd_ensemble)

    @property
    def digit_parity_consistency(self) -> float:
        """How far the direct parity view sits from the digit-derived one
        (Section 3). Large disagreement is a reason for caution, not a
        licence to pick whichever number is more favourable."""
        if self.p_even_direct is None:
            return 0.0
        return abs(self.p_even_direct - self.p_even_digit_derived)


class AdaptiveEnsemble:
    def __init__(self, models: list[DigitModel], *,
                 min_weight: float = 0.02, max_weight: float = 0.40,
                 max_weight_step: float = 0.02, shrinkage_samples: int = 300,
                 chance_band: float = 0.005, extreme_threshold: float = 0.75):
        if not models:
            raise ValueError("ensemble needs at least one model")
        self.models = models
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.max_weight_step = max_weight_step
        self.shrinkage_samples = shrinkage_samples
        self.chance_band = chance_band
        self.extreme_threshold = extreme_threshold
        n = len(models)
        self._weights = {m.name: 1.0 / n for m in models}

    # ---- weighting -------------------------------------------------------

    def _target_weights(self) -> dict[str, float]:
        base = 1.0 / len(self.models)
        raw: dict[str, float] = {}
        for m in self.models:
            perf = m.performance
            skill = perf.brier_skill
            if skill != skill or perf.n == 0:
                raw[m.name] = base
                continue
            # Shrinkage toward the equal-weight baseline (safeguard 1).
            trust = perf.n / (perf.n + self.shrinkage_samples)
            # exp scaling keeps weights positive and is smooth in skill;
            # the factor of 4 makes a 0.05 skill difference meaningful
            # without letting a 0.20 skill difference run away.
            score = math.exp(4.0 * max(min(skill, 0.5), -0.5) * trust)
            if m.health() == DISABLED:
                score *= 0.1      # reliably worse than chance: minimise, don't invert
            raw[m.name] = base * score
        total = sum(raw.values())
        if total <= 0:
            return {m.name: base for m in self.models}
        norm = {k: v / total for k, v in raw.items()}
        clipped = {k: max(self.min_weight, min(self.max_weight, v)) for k, v in norm.items()}
        t = sum(clipped.values())
        return {k: v / t for k, v in clipped.items()}

    def update_weights(self) -> dict[str, float]:
        """Move current weights toward target, rate-limited (safeguard 3)."""
        target = self._target_weights()
        moved = {}
        for name, cur in self._weights.items():
            want = target.get(name, cur)
            delta = max(-self.max_weight_step, min(self.max_weight_step, want - cur))
            moved[name] = max(self.min_weight, cur + delta)
        total = sum(moved.values())
        self._weights = {k: v / total for k, v in moved.items()}
        return dict(self._weights)

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    # ---- prediction ------------------------------------------------------

    def predict(self) -> EnsembleResult:
        preds: list[tuple[DigitModel, DigitPrediction]] = [(m, m.predict()) for m in self.models]
        ready = [(m, p) for m, p in preds if m.is_ready]

        combined = [0.0] * 10
        total_w = 0.0
        member_p: dict[str, float] = {}
        direct_num = 0.0
        direct_den = 0.0

        for m, p in preds:
            w = self._weights.get(m.name, 0.0)
            if m.health() == DISABLED:
                w *= 0.1
            member_p[m.name] = p.p_even
            for d in range(10):
                combined[d] += w * p.digit_probabilities[d]
            total_w += w
            if p.direct_p_even is not None:
                direct_num += w * p.direct_p_even
                direct_den += w

        if total_w <= 0:
            combined = [0.1] * 10
        else:
            combined = [c / total_w for c in combined]
        combined = normalize_distribution(combined)

        derived = sum(combined[d] for d in EVEN_DIGITS)
        direct = (direct_num / direct_den) if direct_den > 0 else None

        # Section 3: blend the digit-derived and direct parity views. Equal
        # weighting when both exist -- neither has earned precedence, and
        # letting one dominate would discard the cross-check that having
        # both is for.
        p_even = derived if direct is None else 0.5 * (derived + direct)

        vals = [p.p_even for m, p in ready] or [0.5]
        mean = sum(vals) / len(vals)
        dispersion = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0
        favoured_even = p_even >= 0.5
        agreeing = sum(1 for v in vals if (v >= 0.5) == favoured_even)
        agreement = agreeing / len(vals)

        # Entropy across normalized member deviations -- high means the
        # members are spread out rather than concentrated.
        devs = [abs(v - 0.5) for v in vals]
        tot = sum(devs)
        if tot > 0:
            q = [d / tot for d in devs]
            model_entropy = -sum(x * math.log(x) for x in q if x > 0) / math.log(max(len(q), 2))
        else:
            model_entropy = 1.0

        return EnsembleResult(
            digit_probabilities=combined,
            p_even_digit_derived=derived,
            p_even_direct=direct,
            p_even_ensemble=p_even,
            weights=self.weights,
            member_p_even=member_p,
            n_members=len(self.models),
            n_ready=len(ready),
            dispersion=dispersion,
            agreement_fraction=agreement,
            model_entropy=model_entropy,
            collapsed_to_chance=abs(p_even - 0.5) < self.chance_band,
            extreme_without_evidence=(
                max(p_even, 1 - p_even) > self.extreme_threshold
                and (len(ready) < 2 or dispersion > 0.15)
            ),
            members_disabled=[m.name for m in self.models if m.health() == DISABLED],
        )

    # ---- learning --------------------------------------------------------

    def observe(self, digit: int) -> None:
        """Fold the realized digit into every member, then rescore.

        Order matters: each model is scored on the prediction it made
        BEFORE seeing this digit, so record_outcome must be called with the
        pre-observation prediction. The executor holds that snapshot.
        """
        for m in self.models:
            m.observe(digit)

    def record_outcomes(self, member_p_even: dict[str, float], digit: int) -> None:
        parity = digit % 2
        for m in self.models:
            if m.name in member_p_even:
                m.record_outcome(member_p_even[m.name], parity)
        self.update_weights()

    def health_report(self) -> dict[str, dict]:
        return {
            m.name: {
                "health": m.health(),
                "weight": self._weights.get(m.name, 0.0),
                "n": m.performance.n,
                "brier": m.performance.brier,
                "brier_skill": m.performance.brier_skill,
                "log_loss": m.performance.log_loss,
                "accuracy": m.performance.accuracy,
                "ready": m.is_ready,
            }
            for m in self.models
        }
