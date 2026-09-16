"""
Soft predictive evidence (spec Sections 3, 4, 5, 22, 23, 26).

THE CENTRAL ARCHITECTURAL CHANGE. Previously the randomness battery and the
regime classifier each held an independent binary veto: `EdgeVerdict.tradeable`
and `RegimeState.is_tradeable` were consulted directly by the gate, and either
one returning False ended the evaluation with NO_TRADE. Section 3 removes that
architecture. Both engines still run, still log, and still matter -- but they
now emit CONTINUOUS EVIDENCE that flows into the opportunity score rather than
short-circuiting the pipeline.

Every function here converts a diagnostic into a bounded number. Two
conventions, used consistently:

    quality-style  -> [0, 1]   where 1 is favourable
    evidence-style -> [-1, 1]  where 0 is neutral and negative is adverse

Nothing in this module can veto anything. Vetoes live in
app/execution/hard_gates.py and nowhere else.

A NOTE ON WHAT THIS DOES AND DOES NOT CHANGE. Making randomness soft means a
fair-looking stream no longer produces an instant NO_TRADE_RANDOM_STREAM. It
does NOT mean a fair stream becomes tradeable. On an independent uniform digit
stream the calibrated probability converges to 0.5, the edge against a 1.95x
break-even of 0.5128 converges to -0.0128, and the trade is refused by the
economics in app/economics/edge.py -- which is a hard gate, because negative
expected value is an economic fact, not a soft preference. The difference is
that the refusal now arrives with a number attached instead of a blanket
verdict, which is what makes the diagnostics in Sections 16 and 17 possible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def signed(x: float) -> float:
    return clamp(x, -1.0, 1.0)


def _logistic(x: float, midpoint: float, scale: float) -> float:
    """Smooth 0->1 ramp. Used instead of thresholds so that a value just
    under a cutoff degrades the score slightly rather than zeroing it --
    the whole point of Section 6."""
    if scale <= 0:
        return 1.0 if x >= midpoint else 0.0
    z = (x - midpoint) / scale
    if z > 30:
        return 1.0
    if z < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------------------
# Section 3: randomness as evidence, not veto
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RandomnessEvidence:
    """Section 3's required output surface.

    `randomness_evidence` is signed: positive means the battery found a
    statistically established, economically relevant departure from a fair
    process; negative means the stream looks like a fair process. Zero means
    the battery could not tell (usually insufficient sample).
    """
    randomness_evidence: float          # [-1, 1], + = exploitable departure found
    dependence_evidence: float          # [0, 1], strength of serial dependence
    distribution_shift: float           # [0, 1], non-stationarity strength
    transition_dependence: float        # [0, 1], lag-structure strength
    serial_dependence: float            # [0, 1], runs/ACF strength
    pattern_strength: float             # [0, 1], largest economically-scaled deviation
    confidence: float                   # [0, 1], how much sample backs all of this
    n_samples: int = 0
    notes: tuple[str, ...] = ()

    @property
    def looks_fair(self) -> bool:
        """Diagnostic convenience only. NOTHING gates on this."""
        return self.randomness_evidence < 0.0


def _p_to_strength(p: float | None, alpha: float) -> float:
    """Map a p-value to [0,1] strength. p >= alpha -> 0; p -> 0 gives -> 1.

    Log-scaled because the interesting range is p in [1e-6, alpha] and a
    linear map would make every non-significant p look almost significant.
    """
    if p is None:
        return 0.0
    p = max(float(p), 1e-12)
    if p >= alpha:
        return 0.0
    return clamp(math.log10(alpha / p) / 4.0)


def randomness_evidence_from_verdict(verdict, *, break_even: float = 0.5128
                                     ) -> RandomnessEvidence:
    """Convert an EdgeVerdict into continuous evidence.

    The verdict object is unchanged and still carries `tradeable`; we simply
    stop treating that boolean as authority. What we extract instead is the
    STRENGTH of each departure and how much sample stands behind it.
    """
    if verdict is None:
        return RandomnessEvidence(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0,
                                  ("no randomness evaluation yet",))

    alpha = float(getattr(verdict, "alpha", 0.01))
    n = int(getattr(verdict, "n_samples", 0))
    notes: list[str] = []

    # --- distributional departure -----------------------------------------
    gof = getattr(verdict, "digit_uniformity", None)
    gof_g = getattr(verdict, "digit_uniformity_g", None)
    dist_strength = max(
        _p_to_strength(getattr(gof, "p_value", None), alpha),
        _p_to_strength(getattr(gof_g, "p_value", None), alpha),
    )

    # --- serial dependence (runs + autocorrelation) ------------------------
    runs = getattr(verdict, "runs", None)
    runs_strength = _p_to_strength(getattr(runs, "p_value", None), alpha)
    acf = list(getattr(verdict, "autocorrelation", []) or [])
    acf_strength = 0.0
    lag_strength = 0.0
    for a in acf:
        s = _p_to_strength(getattr(a, "p_value", None), alpha)
        acf_strength = max(acf_strength, s)
        if int(getattr(a, "lag", 0)) <= 3:
            lag_strength = max(lag_strength, s)
    serial = max(runs_strength, acf_strength)

    # --- non-stationarity ---------------------------------------------------
    stat = getattr(verdict, "stationarity", None)
    shift = _p_to_strength(getattr(stat, "p_value", None), alpha)

    dependence = max(serial, lag_strength)

    # --- economic scaling ---------------------------------------------------
    # Statistical significance is not the question; Section 12's question is
    # whether the departure is large enough to matter against the real
    # break-even. A parity rate of 0.502 can be significant at n=10^6 and
    # still be worthless at a 0.5128 break-even.
    interval = getattr(verdict, "parity_interval", None)
    pattern_strength = 0.0
    if interval is not None:
        point = float(getattr(interval, "point", 0.5))
        favoured = max(point, 1.0 - point)
        required = break_even + float(getattr(verdict, "required_margin", 0.005))
        # 0 at break-even, 1 when the observed rate exceeds it by 2 margins.
        span = max(required - 0.5, 1e-6)
        pattern_strength = clamp((favoured - 0.5) / (2.0 * span))
        if favoured < required:
            notes.append(
                f"observed parity rate {favoured:.4f} is below the "
                f"{required:.4f} needed at this payout")

    confidence = clamp(n / max(float(getattr(verdict, "n_samples", 1) or 1), 1.0)) \
        if n else 0.0
    confidence = clamp(math.log10(max(n, 1)) / 4.0)  # n=10k -> 1.0

    statistical = max(dist_strength, serial, shift)
    if not getattr(verdict, "sufficient_sample", False):
        evidence = 0.0
        notes.append("insufficient sample for a randomness verdict")
    elif statistical <= 0.0:
        # Nothing significant anywhere. This is EVIDENCE AGAINST an edge,
        # scaled by how much data supports the null. It is not a veto.
        evidence = -confidence
        notes.append("battery found no significant departure from a fair process")
    else:
        # Something is significant. Its usefulness is the product of
        # statistical strength and economic relevance -- Section 12's rule
        # that "significant" is not the same as "mispriced".
        evidence = statistical * pattern_strength
        if pattern_strength <= 0.0:
            evidence = -0.5 * confidence
            notes.append("departure is statistically real but economically irrelevant")

    return RandomnessEvidence(
        randomness_evidence=signed(evidence),
        dependence_evidence=clamp(dependence),
        distribution_shift=clamp(shift),
        transition_dependence=clamp(lag_strength),
        serial_dependence=clamp(serial),
        pattern_strength=clamp(pattern_strength),
        confidence=confidence,
        n_samples=n,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Section 4: regimes influence quality; they do not switch trading on and off
# ---------------------------------------------------------------------------

#: Section 4's continuous mapping. NORMAL is neutral -- explicitly NOT a
#: no-trade state, which was the old TRADEABLE_REGIMES = {HIGH_PREDICTABILITY}
#: behaviour this replaces.
REGIME_EVIDENCE = {
    # Spec vocabulary
    "HIGH_PREDICTABILITY": 1.0,
    "NORMAL": 0.0,
    "LOW_PREDICTABILITY": -0.4,
    "TRANSITION": -0.6,
    "UNSTABLE": -0.9,
    "UNKNOWN": -0.2,
    # Names the existing RegimeDetector actually emits, mapped onto the same
    # continuum. STABLE_UNPREDICTABLE is the important one: it is the state a
    # fair stream sits in permanently, and the old detector paired it with
    # trading_state=WAIT, which is precisely the master switch Section 4
    # removes. Here it is NORMAL -- neutral evidence, not a refusal.
    "STABLE_UNPREDICTABLE": 0.0,
    "DISTRIBUTION_SHIFT": -0.6,
    "TRANSITION_INSTABILITY": -0.9,
    "MODEL_DISAGREEMENT": -0.7,
    "DEGRADED_EDGE": -0.9,
    "WARMING_UP": -1.0,
}

#: Canonical Section 4 name for each detector state, for display and storage.
REGIME_CANONICAL = {
    "HIGH_PREDICTABILITY": "HIGH_PREDICTABILITY",
    "STABLE_UNPREDICTABLE": "NORMAL",
    "LOW_PREDICTABILITY": "LOW_PREDICTABILITY",
    "DISTRIBUTION_SHIFT": "TRANSITION",
    "TRANSITION_INSTABILITY": "UNSTABLE",
    "MODEL_DISAGREEMENT": "UNSTABLE",
    "DEGRADED_EDGE": "UNSTABLE",
    "WARMING_UP": "UNKNOWN",
}



@dataclass(frozen=True)
class RegimeEvidence:
    regime: str
    canonical: str
    regime_evidence: float      # [-1, 1]
    regime_quality: float       # [0, 1], rescaled for the score blend
    entropy: float
    change_point: bool
    note: str = ""


def regime_evidence_from_state(state) -> RegimeEvidence:
    if state is None:
        return RegimeEvidence("UNKNOWN", "UNKNOWN", REGIME_EVIDENCE["UNKNOWN"],
                              0.4, 1.0, False, "no regime classification yet")
    name = str(getattr(state, "regime", "UNKNOWN")).upper()
    base = REGIME_EVIDENCE.get(name, REGIME_EVIDENCE["UNKNOWN"])

    # A detected change point degrades whatever the nominal regime is: the
    # distribution the models were fitted to may no longer be the live one.
    if getattr(state, "change_point_detected", False):
        base = min(base, -0.5)

    return RegimeEvidence(
        regime=name,
        canonical=REGIME_CANONICAL.get(name, "UNKNOWN"),
        regime_evidence=signed(base),
        regime_quality=clamp((base + 1.0) / 2.0),
        entropy=float(getattr(state, "entropy", 1.0)),
        change_point=bool(getattr(state, "change_point_detected", False)),
        note=str(getattr(state, "reason", "")),
    )


# ---------------------------------------------------------------------------
# Section 22/23: model agreement and information as evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelEvidence:
    agreement_score: float        # [0, 1]
    dispersion: float             # raw
    dispersion_quality: float     # [0, 1], 1 = tight consensus
    model_health: float           # [0, 1], weighted fraction of healthy members
    n_ready: int = 0
    n_members: int = 0
    parity_consistency: float = 0.0
    collapsed_to_chance: bool = False
    extreme_without_evidence: bool = False


def model_evidence_from_ensemble(result, health_report: dict | None = None
                                 ) -> ModelEvidence:
    """Section 22: agreement becomes evidence strength, never an automatic
    veto. `agreement >= X` as a hard cutoff is exactly the binary filter
    Section 6 bans; what survives is that disagreement lowers the score."""
    agreement = float(getattr(result, "agreement_fraction", 0.5))
    dispersion = float(getattr(result, "dispersion", 0.0))

    # Agreement is a fraction in [0.5, 1]: 0.5 means the members split evenly.
    agreement_score = clamp((agreement - 0.5) * 2.0)
    # Dispersion of 0.05 in P(EVEN) across members is already substantial.
    dispersion_quality = clamp(1.0 - dispersion / 0.05)

    health = 1.0
    if health_report:
        weights = {"HEALTHY": 1.0, "WARNING": 0.6, "DEGRADED": 0.2, "DISABLED": 0.0}
        vals = [weights.get(str(v.get("health", "HEALTHY")).upper(), 0.5)
                for v in health_report.values()]
        health = sum(vals) / len(vals) if vals else 1.0

    return ModelEvidence(
        agreement_score=agreement_score,
        dispersion=dispersion,
        dispersion_quality=dispersion_quality,
        model_health=clamp(health),
        n_ready=int(getattr(result, "n_ready", 0)),
        n_members=int(getattr(result, "n_members", 0)),
        parity_consistency=float(getattr(result, "digit_parity_consistency", 0.0)),
        collapsed_to_chance=bool(getattr(result, "collapsed_to_chance", False)),
        extreme_without_evidence=bool(getattr(result, "extreme_without_evidence", False)),
    )


# ---------------------------------------------------------------------------
# Section 23: information-theoretic evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InformationEvidence:
    entropy: float                # normalized digit entropy, [0, 1]
    entropy_information: float    # [0, 1], how much structure the entropy implies
    parity_entropy: float = 1.0
    mutual_information: float = 0.0


def information_evidence(*, entropy: float, parity_entropy: float = 1.0,
                         mutual_information: float = 0.0) -> InformationEvidence:
    """Section 23 is explicit that low entropy does not mean trade and high
    entropy does not mean no-trade. So entropy contributes a SMALL term,
    scaled to the range that is actually achievable: on 10 uniform digits the
    normalized entropy sits at 1.000 +- ~0.002, so a 0.98 "floor" is not a
    meaningful discriminator and treating it as one manufactures signal."""
    deficit = clamp((1.0 - float(entropy)) / 0.02)
    return InformationEvidence(
        entropy=float(entropy),
        entropy_information=deficit,
        parity_entropy=float(parity_entropy),
        mutual_information=float(mutual_information),
    )


# ---------------------------------------------------------------------------
# Calibration health as evidence (Sections 9, 15)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationEvidence:
    is_fitted: bool
    quality: float               # [0, 1]
    n_samples: int
    ece: float = 0.0
    brier: float = 0.25
    drift: float = 0.0


def calibration_evidence(calibrator) -> CalibrationEvidence:
    fitted = bool(getattr(calibrator, "is_fitted", False))
    try:
        quality = float(calibrator.quality_score())
    except Exception:
        quality = 0.0
    try:
        ece = float(calibrator.expected_calibration_error())
    except Exception:
        ece = 0.0
    try:
        brier = float(calibrator.brier_score())
    except Exception:
        brier = 0.25
    return CalibrationEvidence(
        is_fitted=fitted,
        quality=clamp(quality),
        n_samples=int(getattr(calibrator, "n_samples", 0)),
        ece=ece,
        brier=brier,
    )


# ---------------------------------------------------------------------------
# The aggregate handed to the opportunity scorer
# ---------------------------------------------------------------------------

@dataclass
class EvidenceBundle:
    """Everything soft, in one object (Section 13's input list)."""
    randomness: RandomnessEvidence
    regime: RegimeEvidence
    models: ModelEvidence
    information: InformationEvidence
    calibration: CalibrationEvidence
    sample_size: int = 0
    persistence: float = 0.0        # [0, 1], filled by app/evidence/persistence.py
    edge_persistence: float = 0.0
    extras: dict = field(default_factory=dict)
