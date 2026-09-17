"""
Hard safety gates (spec Section 5).

THIS FILE CONTAINS EVERY VETO IN THE SYSTEM. If a condition can stop a trade
outright, it is here. If it is not here, it cannot stop a trade -- it can only
lower the opportunity score. That separation is the whole of Section 5, and it
is enforced structurally: app/evidence/ imports nothing from this module and
returns no booleans that the gate consults.

THE TEST FOR MEMBERSHIP. A condition belongs here if acting despite it would
be unsafe, incorrect, or impossible -- not merely unwise. "The models disagree"
is unwise and lives in the score. "We are not connected to the API" makes the
trade impossible. "The contract has negative expected value" makes it
arithmetically a losing bet, which is the economic equivalent of impossible:
there is no opportunity score high enough to make a -EV contract worth buying,
because the score measures the quality of an opportunity and a -EV contract is
not an opportunity.

THAT LAST ONE IS THE IMPORTANT DISTINCTION and it is worth being explicit,
because Section 3 removes the randomness veto and it would be easy to remove
the economic one by the same reasoning. They are not the same kind of rule.
The randomness battery answers "does structure exist?", which is a question
about evidence and belongs on a continuum. The EV check answers "does this
specific quoted contract pay more than it costs in expectation?", which is
arithmetic on two numbers Deriv just sent us. Section 14 classifies a
negative-edge candidate as INVALID, and Section 11 makes the proposal
authoritative. So EV stays hard.

Gates are ordered cheapest-first and short-circuit, so a candidate that was
never going to trade does not spend an API call.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

# --- reason codes (Section 16's diagnostic vocabulary) ---------------------
NO_TRADE_API = "NO_TRADE_API"
NO_TRADE_DATA_STALE = "NO_TRADE_DATA_STALE"
NO_TRADE_MALFORMED_TICK = "NO_TRADE_MALFORMED_TICK"
NO_TRADE_MINIMUM_DATA = "NO_TRADE_MINIMUM_DATA"
NO_TRADE_MODEL_SYSTEM_FAILURE = "NO_TRADE_MODEL_SYSTEM_FAILURE"
NO_TRADE_CALIBRATION_UNFITTED = "NO_TRADE_CALIBRATION_UNFITTED"
NO_TRADE_DATABASE = "NO_TRADE_DATABASE"
NO_TRADE_BAD_PROPOSAL = "NO_TRADE_BAD_PROPOSAL"
NO_TRADE_STALE_PROPOSAL = "NO_TRADE_STALE_PROPOSAL"
NO_TRADE_NEGATIVE_EV = "NO_TRADE_NEGATIVE_EV"
NO_TRADE_RISK = "NO_TRADE_RISK"
NO_TRADE_OPEN_CONTRACT = "NO_TRADE_OPEN_CONTRACT"
NO_TRADE_EMERGENCY_STOP = "NO_TRADE_EMERGENCY_STOP"
NO_TRADE_ACCOUNT = "NO_TRADE_ACCOUNT"
NO_TRADE_RESEARCH_MODE = "NO_TRADE_RESEARCH_MODE"

#: Everything above is a veto. Codes emitted by the *soft* layer live in
#: app/evidence/opportunity.py and are deliberately disjoint from this set, so
#: a glance at a reason code tells you which half of the architecture spoke.
HARD_CODES = frozenset({
    NO_TRADE_API, NO_TRADE_DATA_STALE, NO_TRADE_MALFORMED_TICK,
    NO_TRADE_MINIMUM_DATA, NO_TRADE_MODEL_SYSTEM_FAILURE,
    NO_TRADE_CALIBRATION_UNFITTED, NO_TRADE_DATABASE, NO_TRADE_BAD_PROPOSAL,
    NO_TRADE_STALE_PROPOSAL, NO_TRADE_NEGATIVE_EV, NO_TRADE_RISK,
    NO_TRADE_OPEN_CONTRACT, NO_TRADE_EMERGENCY_STOP, NO_TRADE_ACCOUNT,
    NO_TRADE_RESEARCH_MODE,
})


@dataclass
class GateResult:
    passed: bool
    code: str
    explanation: str


@dataclass
class HardGateOutcome:
    passed: bool
    code: str | None = None
    explanation: str = ""
    trail: list[GateResult] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return not self.passed


class HardGates:
    """Stateless evaluator. Two phases, because the first runs before we
    spend an API call on a quote and the second needs that quote."""

    def __init__(self, *, min_samples: int = 2000,
                 stale_tick_seconds: float = 30.0,
                 max_proposal_age_seconds: float = 5.0,
                 research_mode: bool = False):
        self.min_samples = int(min_samples)
        self.stale_tick_seconds = float(stale_tick_seconds)
        self.max_proposal_age_seconds = float(max_proposal_age_seconds)
        self.research_mode = bool(research_mode)

    # -- phase 1: everything checkable without a quote ----------------------

    def check_preconditions(self, *, state, ensemble_result, calibrator,
                            api_connected: bool, db_healthy: bool = True,
                            emergency_stop: bool = False,
                            open_contract: bool = False,
                            account_available: bool = True,
                            tick_valid: bool = True) -> HardGateOutcome:
        trail: list[GateResult] = []

        def fail(code: str, msg: str) -> HardGateOutcome:
            trail.append(GateResult(False, code, msg))
            return HardGateOutcome(False, code, msg, trail)

        def ok(code: str, msg: str) -> None:
            trail.append(GateResult(True, code, msg))

        if emergency_stop:
            return fail(NO_TRADE_EMERGENCY_STOP, "emergency stop engaged")
        ok(NO_TRADE_EMERGENCY_STOP, "no emergency stop")

        if not api_connected:
            return fail(NO_TRADE_API, "API not connected")
        ok(NO_TRADE_API, "API connected")

        if not account_available:
            return fail(NO_TRADE_ACCOUNT, "account unavailable or unauthenticated")
        ok(NO_TRADE_ACCOUNT, "account available")

        if not tick_valid:
            return fail(NO_TRADE_MALFORMED_TICK, "tick malformed or digit invalid")
        ok(NO_TRADE_MALFORMED_TICK, "tick well-formed")

        if state.is_stale(self.stale_tick_seconds):
            return fail(NO_TRADE_DATA_STALE,
                        f"no tick within {self.stale_tick_seconds:.0f}s")
        ok(NO_TRADE_DATA_STALE, "tick stream fresh")

        if not db_healthy:
            # Section 5: a trade we cannot record is a trade we cannot
            # reconcile, and an unreconcilable position is worse than a
            # missed one.
            return fail(NO_TRADE_DATABASE, "database unavailable; trade could not be recorded")
        ok(NO_TRADE_DATABASE, "database healthy")

        if open_contract:
            return fail(NO_TRADE_OPEN_CONTRACT, "a contract is already open for this symbol")
        ok(NO_TRADE_OPEN_CONTRACT, "no conflicting open contract")

        if state.total_count < self.min_samples:
            return fail(NO_TRADE_MINIMUM_DATA,
                        f"{state.total_count}/{self.min_samples} ticks observed")
        ok(NO_TRADE_MINIMUM_DATA, f"{state.total_count} ticks observed")

        # Model system failure -- not "models disagree" (soft) but "the model
        # layer did not produce a usable distribution at all".
        if ensemble_result is None or getattr(ensemble_result, "n_ready", 0) <= 0:
            return fail(NO_TRADE_MODEL_SYSTEM_FAILURE, "no model produced a prediction")
        probs = list(getattr(ensemble_result, "digit_probabilities", []) or [])
        if len(probs) != 10 or abs(sum(probs) - 1.0) > 1e-6 or any(
                p < 0 or p != p for p in probs):
            return fail(NO_TRADE_MODEL_SYSTEM_FAILURE,
                        "digit distribution is corrupt (Section 7 invariant violated)")
        ok(NO_TRADE_MODEL_SYSTEM_FAILURE, "model system healthy")

        # An unfitted calibrator returns its input unchanged, so every
        # downstream "calibrated" number would silently be the raw one. That
        # is a corrupted-state condition, not a preference.
        if not getattr(calibrator, "is_fitted", False):
            return fail(NO_TRADE_CALIBRATION_UNFITTED,
                        f"calibrator not fitted ({getattr(calibrator, 'n_samples', 0)} "
                        f"samples); calibrated probability would echo the raw one")
        ok(NO_TRADE_CALIBRATION_UNFITTED, "calibrator fitted")

        return HardGateOutcome(True, None, "all hard preconditions passed", trail)

    # -- phase 2: economics and risk, once a real quote exists --------------

    def check_execution(self, *, edge_assessment, risk_decision,
                        now: float | None = None,
                        min_expected_value: float = 0.0) -> HardGateOutcome:
        trail: list[GateResult] = []
        now = time.time() if now is None else now

        def fail(code: str, msg: str) -> HardGateOutcome:
            trail.append(GateResult(False, code, msg))
            return HardGateOutcome(False, code, msg, trail)

        def ok(code: str, msg: str) -> None:
            trail.append(GateResult(True, code, msg))

        if edge_assessment is None:
            return fail(NO_TRADE_BAD_PROPOSAL, "no valid proposal obtained")
        proposal = edge_assessment.proposal
        if proposal.is_stale(now, self.max_proposal_age_seconds):
            return fail(NO_TRADE_STALE_PROPOSAL,
                        "proposal too old to execute safely")
        ok(NO_TRADE_STALE_PROPOSAL, "proposal fresh")

        # --- the economic gate: DEMOTED TO INFORMATIONAL, BY REQUEST --------
        # PREVIOUSLY hard (Sections 11, 12, 14): a candidate whose point EV
        # was <= min_expected_value was vetoed outright, on the grounds that
        # a negative-EV contract is arithmetically a losing bet regardless of
        # how the opportunity score reads.
        #
        # As of this change, EV/edge no longer stops a trade here. It is
        # still computed and still recorded on every decision (see
        # decision.expected_value / decision.edge in engine.py, and the
        # explanation string below), so nothing about the economics is
        # hidden -- it simply no longer vetoes. What decides whether a
        # candidate trades now is entirely the opportunity score / trade
        # zone in app/evidence/opportunity.py (model agreement among its
        # contributors) plus risk_decision below.
        #
        # NOTE: this file is not the only place EV was enforced. See the
        # matching change in app/evidence/opportunity.py's score(), which
        # also forced zone=INVALID on non-positive EV -- both had to change
        # together, or this one is a no-op.
        ev = float(edge_assessment.expected_value)
        ok(NO_TRADE_NEGATIVE_EV,
           f"[informational, non-blocking] EV {ev:+.4f} vs floor "
           f"{min_expected_value:+.4f} at P={edge_assessment.probability:.4f} "
           f"against break-even {edge_assessment.break_even:.4f} "
           f"(payout {proposal.payout_multiple:.3f}x), "
           f"edge {edge_assessment.point_edge:+.4f}")

        if risk_decision is None or not getattr(risk_decision, "allowed", False):
            return fail(NO_TRADE_RISK,
                        getattr(risk_decision, "reason", "risk not evaluated"))
        ok(NO_TRADE_RISK, "risk checks passed")

        if self.research_mode:
            return fail(NO_TRADE_RESEARCH_MODE,
                        "research mode: would have traded, but BUY is never called")

        return HardGateOutcome(True, None, "all hard execution gates passed", trail)
