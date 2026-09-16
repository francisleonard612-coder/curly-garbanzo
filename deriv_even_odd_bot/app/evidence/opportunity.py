"""
Opportunity scoring, trade zones and adaptive selectivity
(spec Sections 6, 13, 14, 15).

WHAT THIS REPLACES. The previous gate was a chain of `if metric < threshold:
return NO_TRADE`, eleven deep. Section 6 identifies the failure mode exactly:
with eleven independent cutoffs each passing 70% of the time, 3% of genuine
opportunities survive, and the surviving set is determined by whichever cutoff
happens to be tightest rather than by the merit of the opportunity. This module
replaces that chain with a continuous score.

HOW THE SCORE IS BUILT. Each contributor produces a [0,1] quality and carries a
weight. The blend is a weighted arithmetic mean, NOT a product -- a product is
just a soft version of the same filter multiplication, since one near-zero term
zeroes the result. Section 13's requirement that "one weak soft metric must not
completely veto a strong opportunity" is what forces the additive form.

THE FLOOR MECHANISM. Purely additive scoring has the opposite failure: eight
mediocre contributors can outvote a catastrophic one. So each contributor also
declares a `floor_at` level; a contributor scoring below its floor caps the
FINAL score rather than vetoing it. A contributor at 0.0 with floor 0.35 caps
the total at 35, which lands the candidate in BORDERLINE -- visible, diagnosed,
and not traded, but also not silently deleted from the statistics.

WHAT THE SCORE IS NOT. It is not a probability, not an edge, and not a reason
to trade on its own. A candidate reaching STRONG has already passed the hard
gates in app/execution/hard_gates.py, including positive expected value. The
score decides how good a *qualifying* opportunity is; it never decides whether
a losing contract is worth buying.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.evidence.signals import clamp

# --- soft reason codes. Disjoint from hard_gates.HARD_CODES by design. -----
NO_TRADE_LOW_OPPORTUNITY = "NO_TRADE_LOW_OPPORTUNITY"
NO_TRADE_LOW_EDGE = "NO_TRADE_LOW_EDGE"
NO_TRADE_HIGH_UNCERTAINTY = "NO_TRADE_HIGH_UNCERTAINTY"
NO_TRADE_MODEL_DISAGREEMENT = "NO_TRADE_MODEL_DISAGREEMENT"
NO_TRADE_MODEL_DEGRADED = "NO_TRADE_MODEL_DEGRADED"
NO_TRADE_CALIBRATION_DEGRADED = "NO_TRADE_CALIBRATION_DEGRADED"
NO_TRADE_REGIME = "NO_TRADE_REGIME"
NO_TRADE_RANDOM_STREAM_EVIDENCE = "NO_TRADE_RANDOM_STREAM_EVIDENCE"

# --- trade zones (Section 14) ---------------------------------------------
STRONG = "STRONG"
VALIDATED = "VALIDATED"
BORDERLINE = "BORDERLINE"
INVALID = "INVALID"

TRADING_ZONES = frozenset({STRONG, VALIDATED})


@dataclass(frozen=True)
class Contribution:
    name: str
    quality: float        # [0, 1]
    weight: float
    floor_at: float       # below this, the contributor caps the final score
    detail: str = ""

    @property
    def weighted(self) -> float:
        return self.quality * self.weight

    @property
    def below_floor(self) -> bool:
        return self.quality < self.floor_at


@dataclass
class OpportunityAssessment:
    score: float                       # [0, 100]
    zone: str
    threshold: float                   # the score required, after adaptation
    contributions: list[Contribution] = field(default_factory=list)
    capped_by: list[str] = field(default_factory=list)
    limiting_factor: str = ""
    reason_code: str = ""
    explanation: str = ""
    selectivity_multiplier: float = 1.0

    @property
    def tradeable(self) -> bool:
        return self.zone in TRADING_ZONES

    def top_deficits(self, k: int = 3) -> list[Contribution]:
        """The contributors costing the most score, weight-adjusted. This is
        what Section 17's operator message is built from."""
        ranked = sorted(self.contributions,
                        key=lambda c: (1.0 - c.quality) * c.weight, reverse=True)
        return ranked[:k]


# ---------------------------------------------------------------------------
# Section 15: adaptive selectivity
# ---------------------------------------------------------------------------

class AdaptiveSelectivity:
    """Raises the required score when the system's own estimates get less
    trustworthy.

    THE ASYMMETRY IS DELIBERATE AND IS THE POINT OF SECTION 15. Deteriorating
    model health tightens the requirement; nothing loosens it. In particular
    a long dry spell does NOT loosen it -- Section 15's closing line and
    Section 16's "DO NOT blindly lower thresholds" both forbid that, and it is
    the single most tempting bug to write in a bot that is not trading, because
    the thing the operator wants (trades) is one constant away.

    The multiplier is bounded at [1.0, 1.6]: it can demand up to 60% more
    score, and it can never demand less than the configured baseline.
    """

    def __init__(self, *, base_threshold: float = 62.0,
                 max_multiplier: float = 1.6):
        self.base_threshold = float(base_threshold)
        self.max_multiplier = float(max_multiplier)

    def multiplier(self, evidence) -> tuple[float, list[str]]:
        m = 1.0
        notes: list[str] = []

        cal = evidence.calibration
        if cal.quality < 0.5:
            m += 0.25
            notes.append(f"calibration quality {cal.quality:.2f}")
        elif cal.quality < 0.7:
            m += 0.10
            notes.append(f"calibration quality {cal.quality:.2f}")

        mod = evidence.models
        if mod.dispersion_quality < 0.4:
            m += 0.20
            notes.append(f"model dispersion {mod.dispersion:.4f}")
        if mod.model_health < 0.6:
            m += 0.20
            notes.append(f"model health {mod.model_health:.2f}")

        reg = evidence.regime
        if reg.regime_evidence <= -0.6:
            m += 0.20
            notes.append(f"regime {reg.regime}")
        if reg.change_point:
            m += 0.15
            notes.append("change point detected")

        return (min(m, self.max_multiplier), notes)

    def threshold_for(self, evidence) -> tuple[float, float, list[str]]:
        m, notes = self.multiplier(evidence)
        return (min(100.0, self.base_threshold * m), m, notes)


# ---------------------------------------------------------------------------
# Section 13: the score itself
# ---------------------------------------------------------------------------

class OpportunityScorer:
    def __init__(self, *, base_threshold: float = 62.0,
                 strong_threshold: float = 80.0,
                 borderline_threshold: float = 40.0,
                 target_edge: float = 0.02,
                 selectivity: AdaptiveSelectivity | None = None):
        #: `target_edge` is the edge at which the edge contributor saturates,
        #: not a cutoff. An edge of half this still scores 0.5 there.
        self.target_edge = float(target_edge)
        self.strong_threshold = float(strong_threshold)
        self.borderline_threshold = float(borderline_threshold)
        self.selectivity = selectivity or AdaptiveSelectivity(
            base_threshold=base_threshold)

    def score(self, *, edge_assessment, evidence,
              persistence=None) -> OpportunityAssessment:
        """Build the full assessment for one candidate.

        `edge_assessment` may be None when no quote has been fetched yet; the
        economic contributors then score 0 and the result is a pre-quote
        estimate used only for diagnostics and shadow analysis.
        """
        c: list[Contribution] = []

        # -- economics (Sections 11, 12) -- the heaviest block, correctly ----
        if edge_assessment is not None:
            point_edge = float(edge_assessment.point_edge)
            cons_edge = float(edge_assessment.conservative_edge)
            ev_unit = float(edge_assessment.ev_per_unit_staked)
            iv = edge_assessment.probability_interval
            half_width = float(getattr(iv, "half_width", 0.5))
            payout_mult = float(edge_assessment.proposal.payout_multiple)

            # Edge, scaled so that target_edge -> 1.0 and 0 -> 0.0.
            c.append(Contribution(
                "calibrated_edge", clamp(point_edge / self.target_edge), 0.22, 0.05,
                f"point edge {point_edge:+.4f} vs break-even "
                f"{edge_assessment.break_even:.4f}"))

            # Conservative edge: Section 10 moved this out of the veto set and
            # into the score. A negative lower bound is survivable if
            # everything else is strong; it is not free.
            c.append(Contribution(
                "conservative_edge", clamp(cons_edge / self.target_edge + 0.5),
                0.14, 0.10, f"lower-bound edge {cons_edge:+.4f}"))

            c.append(Contribution(
                "expected_value", clamp(ev_unit / 0.05), 0.14, 0.05,
                f"EV {ev_unit:+.4f} per unit staked"))

            # Probability of positive edge (Section 10). Approximated from the
            # interval: where does break-even fall inside it?
            span = max(float(getattr(iv, "width", 0.0)), 1e-9)
            p_pos = clamp((float(getattr(iv, "upper", 0.5))
                           - edge_assessment.break_even) / span)
            c.append(Contribution(
                "probability_of_positive_edge", p_pos, 0.08, 0.10,
                f"P(edge > 0) ~ {p_pos:.2f}"))

            c.append(Contribution(
                "probability_precision", clamp(1.0 - half_width / 0.05),
                0.08, 0.0, f"probability +-{half_width:.4f}"))

            # Payout quality: break-even is a function of payout alone, so a
            # better quote lowers the bar without the models improving.
            c.append(Contribution(
                "payout_quality", clamp((payout_mult - 1.85) / 0.15), 0.05, 0.0,
                f"payout {payout_mult:.3f}x"))
        else:
            for name, w in (("calibrated_edge", 0.22), ("conservative_edge", 0.14),
                            ("expected_value", 0.14),
                            ("probability_of_positive_edge", 0.08),
                            ("probability_precision", 0.08),
                            ("payout_quality", 0.05)):
                c.append(Contribution(name, 0.0, w, 0.0, "no quote yet"))

        # -- model evidence (Section 22) -------------------------------------
        mod = evidence.models
        c.append(Contribution(
            "model_agreement", mod.agreement_score, 0.07, 0.15,
            f"agreement {mod.agreement_score:.2f}"))
        c.append(Contribution(
            "model_dispersion", mod.dispersion_quality, 0.04, 0.0,
            f"dispersion {mod.dispersion:.4f}"))
        c.append(Contribution(
            "model_health", mod.model_health, 0.05, 0.25,
            f"health {mod.model_health:.2f}"))

        # -- calibration health (Section 9) ----------------------------------
        cal = evidence.calibration
        c.append(Contribution(
            "calibration_health", cal.quality, 0.05, 0.20,
            f"quality {cal.quality:.2f}, ECE {cal.ece:.4f}"))

        # -- regime (Section 4): continuous, NORMAL is neutral not fatal -----
        reg = evidence.regime
        c.append(Contribution(
            "regime_quality", reg.regime_quality, 0.05, 0.0,
            f"regime {reg.regime}"))

        # -- randomness (Section 3): evidence, not veto ----------------------
        rnd = evidence.randomness
        c.append(Contribution(
            "randomness_evidence", clamp((rnd.randomness_evidence + 1.0) / 2.0),
            0.06, 0.0,
            f"randomness evidence {rnd.randomness_evidence:+.2f}"))
        c.append(Contribution(
            "pattern_strength", rnd.pattern_strength, 0.03, 0.0,
            f"pattern strength {rnd.pattern_strength:.2f}"))
        c.append(Contribution(
            "transition_strength", rnd.transition_dependence, 0.02, 0.0,
            f"transition dependence {rnd.transition_dependence:.2f}"))

        # -- information (Section 23) ----------------------------------------
        info = evidence.information
        c.append(Contribution(
            "entropy_information", info.entropy_information, 0.02, 0.0,
            f"normalized entropy {info.entropy:.4f}"))

        # -- persistence (Sections 18, 19) -----------------------------------
        pers = persistence.combined if persistence is not None else 0.0
        c.append(Contribution(
            "signal_persistence", clamp(pers), 0.03, 0.0,
            (persistence.note if persistence is not None and persistence.note
             else f"persistence {pers:.2f}")))

        # -- sample size ------------------------------------------------------
        n = int(evidence.sample_size)
        c.append(Contribution(
            "sample_size", clamp(n / 5000.0), 0.02, 0.0, f"{n} ticks"))

        # --- blend -----------------------------------------------------------
        total_weight = sum(x.weight for x in c)
        raw = 100.0 * sum(x.weighted for x in c) / total_weight if total_weight else 0.0

        capped_by = [x.name for x in c if x.below_floor]
        score = raw
        if capped_by:
            cap = 100.0 * min(x.floor_at for x in c if x.below_floor)
            score = min(raw, cap)

        threshold, mult, sel_notes = self.selectivity.threshold_for(evidence)

        # --- zone (Section 14) ------------------------------------------------
        if edge_assessment is not None and edge_assessment.expected_value <= 0:
            zone = INVALID
        elif score >= max(self.strong_threshold, threshold):
            zone = STRONG
        elif score >= threshold:
            zone = VALIDATED
        elif score >= self.borderline_threshold:
            zone = BORDERLINE
        else:
            zone = INVALID

        deficits = sorted(c, key=lambda x: (1.0 - x.quality) * x.weight, reverse=True)
        limiting = deficits[0] if deficits else None
        reason_code = _reason_for(limiting.name if limiting else "", capped_by)

        if zone in TRADING_ZONES:
            explanation = (f"opportunity {score:.1f}/100 ({zone}) vs threshold "
                           f"{threshold:.1f}")
        else:
            bits = [f"opportunity {score:.1f}/100 ({zone}) below threshold "
                    f"{threshold:.1f}"]
            if capped_by:
                bits.append("capped by " + ", ".join(capped_by))
            if limiting:
                bits.append(f"largest deficit: {limiting.name} ({limiting.detail})")
            if mult > 1.0:
                bits.append(f"selectivity x{mult:.2f} ({'; '.join(sel_notes)})")
            explanation = "; ".join(bits)

        return OpportunityAssessment(
            score=score, zone=zone, threshold=threshold, contributions=c,
            capped_by=capped_by,
            limiting_factor=limiting.name if limiting else "",
            reason_code=reason_code, explanation=explanation,
            selectivity_multiplier=mult,
        )


_REASON_MAP = {
    "calibrated_edge": NO_TRADE_LOW_EDGE,
    "conservative_edge": NO_TRADE_LOW_EDGE,
    "expected_value": NO_TRADE_LOW_EDGE,
    "probability_of_positive_edge": NO_TRADE_LOW_EDGE,
    "payout_quality": NO_TRADE_LOW_EDGE,
    "probability_precision": NO_TRADE_HIGH_UNCERTAINTY,
    "model_agreement": NO_TRADE_MODEL_DISAGREEMENT,
    "model_dispersion": NO_TRADE_MODEL_DISAGREEMENT,
    "model_health": NO_TRADE_MODEL_DEGRADED,
    "calibration_health": NO_TRADE_CALIBRATION_DEGRADED,
    "regime_quality": NO_TRADE_REGIME,
    "randomness_evidence": NO_TRADE_RANDOM_STREAM_EVIDENCE,
    "pattern_strength": NO_TRADE_RANDOM_STREAM_EVIDENCE,
    "transition_strength": NO_TRADE_RANDOM_STREAM_EVIDENCE,
    "entropy_information": NO_TRADE_RANDOM_STREAM_EVIDENCE,
}


def _reason_for(limiting: str, capped_by: list[str]) -> str:
    """Prefer a floor breach as the headline cause: a capped score is a
    specific, actionable failure, whereas the largest weighted deficit is
    often just the heaviest contributor."""
    if capped_by:
        return _REASON_MAP.get(capped_by[0], NO_TRADE_LOW_OPPORTUNITY)
    return _REASON_MAP.get(limiting, NO_TRADE_LOW_OPPORTUNITY)
