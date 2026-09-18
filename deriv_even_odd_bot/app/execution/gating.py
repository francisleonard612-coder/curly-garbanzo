"""
Trade gating: hard vetoes, then continuous opportunity scoring
(spec Sections 2, 5, 6, 13, 14, 16).

WHAT CHANGED FROM THE PREVIOUS VERSION. This file used to be a single
eleven-deep chain of `if metric < threshold: return NO_TRADE`. Two of those
links -- the randomness verdict and the regime classifier -- were absolute
vetoes, so a stream that tested fair, or a regime that was merely NORMAL,
ended evaluation before the economics were ever consulted. Sections 3, 4 and 6
identify that as the core architectural fault, and this rewrite removes it.

THE NEW SHAPE.

    1. HARD GATES      app/execution/hard_gates.py   -- can veto, and are the
                                                        only thing that can
    2. OPPORTUNITY     app/evidence/opportunity.py   -- continuous score,
                                                        no vetoes, floors cap
    3. ZONE            STRONG / VALIDATED            -- trade
                       BORDERLINE / INVALID          -- wait

A candidate that the old gate deleted at step 4 of 11 now reaches the
economics, gets priced, gets scored, and is recorded with a number attached.
That is the difference between "NO_TRADE_RANDOM_STREAM" and "opportunity
41.2/100, capped by conservative_edge, point edge -0.0128 against a 0.5128
break-even" -- the same decision, a far better instrument.

WHAT DELIBERATELY DID NOT CHANGE. Expected value is still hard. Section 3
softens the randomness veto because "is there structure?" is a question about
evidence and belongs on a continuum; it does not soften the economics, because
"does this contract pay more than it costs?" is arithmetic on the quote Deriv
just sent. Section 14 classifies a candidate with no economically valid edge as
INVALID, and this gate honours that. The practical consequence on a fair stream
is that the bot still does not trade -- but it now says why, in numbers, on
every tick, which is what makes the difference diagnosable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.diagnostics.deadlock import DeadlockMonitor
from app.diagnostics.shadow import ShadowThresholdAnalyzer
from app.evidence.opportunity import (
    BORDERLINE,
    INVALID,
    STRONG,
    VALIDATED,
    AdaptiveSelectivity,
    OpportunityScorer,
)
from app.execution.hard_gates import HARD_CODES, GateResult, HardGates

TRADE_EVEN = "TRADE_EVEN"
TRADE_ODD = "TRADE_ODD"
NO_TRADE = "NO_TRADE"

__all__ = [
    "TRADE_EVEN", "TRADE_ODD", "NO_TRADE", "Decision", "GateResult",
    "TradeGate", "trade_quality_score", "STRONG", "VALIDATED", "BORDERLINE",
    "INVALID",
]


@dataclass
class Decision:
    """Complete prediction record (Section 23).

    Everything needed to reconstruct why this decision was made, months later,
    from the database alone. The soft-evidence fields added in this revision
    are what make Sections 16 and 17 possible -- a rejection that records only
    its reason code cannot be aggregated into a diagnosis.
    """
    timestamp: float
    symbol: str
    decision: str
    reason_code: str
    explanation: str

    current_quote: float | None = None
    current_digit: int | None = None
    previous_digits: list[int] = field(default_factory=list)

    digit_probabilities: list[float] = field(default_factory=list)
    p_even_digit_derived: float | None = None
    p_even_direct: float | None = None
    p_even_ensemble: float | None = None
    calibrated_p_even: float | None = None
    member_p_even: dict[str, float] = field(default_factory=dict)
    model_weights: dict[str, float] = field(default_factory=dict)

    probability_lower: float | None = None
    probability_upper: float | None = None
    dispersion: float | None = None
    agreement_fraction: float | None = None
    regime: str | None = None
    entropy: float | None = None
    calibration_quality: float | None = None
    randomness_tradeable: bool | None = None

    contract_type: str | None = None
    stake: float | None = None
    payout: float | None = None
    break_even_probability: float | None = None
    edge: float | None = None
    conservative_edge: float | None = None
    expected_value: float | None = None
    quality_score: float | None = None
    is_probe: bool = False

    # --- the continuous layer (Sections 13-19) -----------------------------
    opportunity_score: float | None = None
    opportunity_zone: str | None = None
    opportunity_threshold: float | None = None
    selectivity_multiplier: float | None = None
    limiting_factor: str | None = None
    capped_by: list[str] = field(default_factory=list)
    randomness_evidence: float | None = None
    regime_evidence: float | None = None
    signal_persistence: float | None = None
    contribution_detail: dict[str, float] = field(default_factory=dict)
    hard_gate_failed: bool = False

    gates: list[GateResult] = field(default_factory=list)
    outcome: str | None = None          # filled in on settlement
    pnl: float | None = None

    @property
    def will_trade(self) -> bool:
        return self.decision in (TRADE_EVEN, TRADE_ODD)

    def apply_opportunity(self, assessment) -> None:
        self.opportunity_score = assessment.score
        self.opportunity_zone = assessment.zone
        self.opportunity_threshold = assessment.threshold
        self.selectivity_multiplier = assessment.selectivity_multiplier
        self.limiting_factor = assessment.limiting_factor
        self.capped_by = list(assessment.capped_by)
        self.contribution_detail = {c.name: round(c.quality, 4)
                                    for c in assessment.contributions}
        # Alias kept so dashboards and queries written against the old schema
        # still resolve. Same number, same 0-100 scale.
        self.quality_score = assessment.score

    def apply_evidence(self, evidence) -> None:
        self.randomness_evidence = evidence.randomness.randomness_evidence
        self.regime_evidence = evidence.regime.regime_evidence
        self.signal_persistence = evidence.persistence


def trade_quality_score(**kwargs) -> float:
    """Deprecated shim.

    The hand-tuned weighted sum this used to compute has been replaced by
    OpportunityScorer, which carries the same information with floors,
    adaptive thresholds and per-contributor attribution. Retained only so
    older analysis scripts import cleanly.
    """
    edge = float(kwargs.get("conservative_edge", 0.0))
    return max(0.0, min(100.0, 100.0 * edge / 0.05))


class TradeGate:
    """Orchestrates hard gates, scoring and diagnostics for one symbol."""

    def __init__(self, settings_gating: dict, *, research_mode: bool = False,
                 risk_settings: dict | None = None):
        g = settings_gating or {}
        r = risk_settings or {}
        self.hard = HardGates(
            min_samples=g.get("min_samples", 2000),
            stale_tick_seconds=r.get("stale_tick_seconds", 30.0),
            max_proposal_age_seconds=g.get("max_proposal_age_seconds", 5.0),
            research_mode=research_mode)
        self.min_expected_value = g.get("min_expected_value", 0.0)
        # Explicit, user-chosen volume filter -- see hard_gates.py's
        # check_execution for why this is NOT a profitability claim.
        self.min_agreement_fraction = g.get("min_agreement_fraction", 0.0)
        self.scorer = OpportunityScorer(
            target_edge=g.get("target_edge", g.get("min_edge", 0.02)),
            borderline_threshold=g.get("borderline_threshold", 40.0),
            strong_threshold=g.get("strong_opportunity_score", 80.0),
            selectivity=AdaptiveSelectivity(
                base_threshold=g.get("min_opportunity_score", 62.0)))
        self.deadlock = DeadlockMonitor(
            idle_alert_seconds=g.get("idle_alert_seconds", 900.0),
            idle_alert_ticks=g.get("idle_alert_ticks", 2000))
        self.shadow = ShadowThresholdAnalyzer()
        self.research_mode = research_mode

    # -- phase 1 -------------------------------------------------------------

    def check_hard_preconditions(self, **kwargs):
        return self.hard.check_preconditions(**kwargs)

    # -- phase 2 -------------------------------------------------------------

    def evaluate(self, *, decision: Decision, edge_assessment, evidence,
                 persistence, risk_decision, now: float | None = None
                 ) -> Decision:
        """Final decision for a candidate that already has a real quote.

        Order matters and is the reverse of the old gate's: the soft score is
        computed for EVERY candidate, including ones the hard gates will
        refuse. Scoring only the survivors would mean the rejection statistics
        cover just the candidates that were already nearly good enough, and
        Section 16's diagnosis would be blind to exactly the cases it exists
        to explain.
        """
        self.deadlock.note_candidate()

        assessment = self.scorer.score(
            edge_assessment=edge_assessment, evidence=evidence,
            persistence=persistence)
        decision.apply_opportunity(assessment)
        decision.apply_evidence(evidence)
        self.shadow.record(assessment, edge_assessment)

        hard = self.hard.check_execution(
            edge_assessment=edge_assessment, risk_decision=risk_decision,
            now=now, min_expected_value=self.min_expected_value,
            # FIXED: previously reconstructed as 0.5 + evidence.models.
            # agreement_score / 2.0. That reconstruction assumed raw
            # agreement_fraction never goes below 0.5, which is false --
            # ensemble.py's agreement_fraction is "fraction of individual
            # members whose own p_even matches the ENSEMBLE's blended
            # direction", and the ensemble's blend can favour a direction
            # most individual members don't (derived+direct averaging, not
            # a member vote). Observed live: agreement_fraction=0.22 with
            # p_even~0.50, i.e. only 22% of members agreed with the
            # ensemble's own pick that tick. clamp() in signals.py maps any
            # agreement_fraction <= 0.5 to agreement_score=0.0 -- a many-
            # to-one collapse -- so reconstructing from that score always
            # produced exactly 0.5, silently misreporting every true value
            # below 0.5 as "borderline 50/50" instead of "most models
            # actively disagreed with the pick". decision.agreement_fraction
            # is set earlier in engine.py directly from the ensemble result
            # (see _pending_agreement in engine.py, which correctly feeds
            # agreement_calibration.py's tracker from the same source) --
            # use that instead of round-tripping through the lossy score.
            agreement_fraction=decision.agreement_fraction,
            min_agreement_fraction=self.min_agreement_fraction)
        decision.gates.extend(hard.trail)

        if hard.blocked:
            decision.decision = NO_TRADE
            decision.reason_code = hard.code
            decision.explanation = hard.explanation
            decision.hard_gate_failed = True
            self.deadlock.note_rejection(
                reason_code=hard.code, zone=assessment.zone,
                opportunity_score=assessment.score, hard=True,
                edge_assessment=edge_assessment,
                limiting_factor=assessment.limiting_factor, now=now)
            return decision

        if not assessment.tradeable:
            decision.decision = NO_TRADE
            decision.reason_code = assessment.reason_code
            decision.explanation = assessment.explanation
            decision.gates.append(
                GateResult(False, assessment.reason_code, assessment.explanation))
            self.deadlock.note_rejection(
                reason_code=assessment.reason_code, zone=assessment.zone,
                opportunity_score=assessment.score, hard=False,
                edge_assessment=edge_assessment,
                limiting_factor=assessment.limiting_factor, now=now)
            return decision

        proposal = edge_assessment.proposal
        side = TRADE_EVEN if proposal.contract_type.upper() == "DIGITEVEN" else TRADE_ODD
        msg = (f"{proposal.contract_type} stake {proposal.stake:.2f} "
               f"payout {proposal.payout:.2f} ({proposal.payout_multiple:.3f}x), "
               f"break-even {proposal.break_even_probability:.4f}, "
               f"edge {edge_assessment.point_edge:+.4f}, "
               f"EV {edge_assessment.expected_value:+.4f}, "
               f"opportunity {assessment.score:.1f}/100 [{assessment.zone}]")
        decision.decision = side
        decision.reason_code = side
        decision.explanation = msg
        decision.gates.append(GateResult(True, side, msg))
        self.deadlock.note_trade(now=now)
        return decision

    # -- diagnostics ---------------------------------------------------------

    def diagnostics(self) -> dict:
        """Sections 16 and 17, for the dashboard and the operator log."""
        return {"deadlock": self.deadlock.report(),
                "shadow": self.shadow.report()}

    @staticmethod
    def is_hard_code(code: str) -> bool:
        return code in HARD_CODES
