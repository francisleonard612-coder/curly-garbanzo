"""
Per-symbol pipeline and trade executor (spec Sections 33, 36, 47, 48).

STATE MACHINE (Section 36) is explicit and observable:

  INITIALIZING -> LOADING_HISTORY -> CALIBRATING -> OBSERVING
    -> SIGNAL_DETECTED -> PROPOSAL_CHECK -> EV_CHECK -> RISK_CHECK
    -> EXECUTING -> MONITORING -> RESOLVED -> LEARNING -> OBSERVING
  plus PAUSED / COOLDOWN / ERROR / EMERGENCY_STOP

THE ORDERING THAT PREVENTS LEAKAGE (Section 22), stated because it is the
easiest thing in this file to break with an innocent refactor:

    1. features  = build(state)        <- state does NOT yet contain the tick
    2. prediction = models.predict()   <- prediction is ABOUT that tick
    3. ... decide, maybe trade ...
    4. state.add(tick)                 <- only now is the tick observed
    5. models.observe(digit)           <- label revealed, models learn

Step 4 must never move above step 2. If it does, every model trains on the
answer and the bot becomes a very expensive random number generator with
excellent backtests.

DUPLICATE-BUY PROTECTION (Section 33/35): every trade carries an
idempotency key checked against the database BEFORE the buy and recorded
immediately after. On an ambiguous buy failure the executor reconciles via
portfolio rather than retrying -- see app/api/deriv_client.py rule 4.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid

from app.calibration.calibrator import CalibrationTracker
from app.data.tick_store import SymbolState
from app.digits.extraction import DigitExtractionError, extract
from app.economics.edge import (
    Proposal,
    ProposalError,
    edge_for_side,
    probability_interval_from_samples,
)
from app.ensemble.ensemble import AdaptiveEnsemble
from app.evidence.persistence import SignalTracker
from app.evidence.signals import (
    EvidenceBundle,
    calibration_evidence,
    information_evidence,
    model_evidence_from_ensemble,
    randomness_evidence_from_verdict,
    regime_evidence_from_state,
)
from app.execution.gating import Decision, TradeGate
from app.features import engine as feature_engine
from app.models.ml import GradientBoostingParityModel, OnlineLogisticParityModel
from app.models.statistical import (
    BetaBinomialParityModel,
    DirichletFrequencyModel,
    EWMAFrequencyModel,
    MarkovDigitModel,
    MarkovParityModel,
    RunLengthParityModel,
)
from app.regimes.detector import RegimeDetector
from app.statistics.randomness import RandomnessMonitor

logger = logging.getLogger(__name__)

INITIALIZING = "INITIALIZING"
LOADING_HISTORY = "LOADING_HISTORY"
CALIBRATING = "CALIBRATING"
OBSERVING = "OBSERVING"
SIGNAL_DETECTED = "SIGNAL_DETECTED"
PROPOSAL_CHECK = "PROPOSAL_CHECK"
EV_CHECK = "EV_CHECK"
RISK_CHECK = "RISK_CHECK"
EXECUTING = "EXECUTING"
MONITORING = "MONITORING"
RESOLVED = "RESOLVED"
LEARNING = "LEARNING"
PAUSED = "PAUSED"
COOLDOWN = "COOLDOWN"
ERROR = "ERROR"
EMERGENCY_STOP = "EMERGENCY_STOP"


class SymbolPipeline:
    """Owns all predictive state for one symbol."""

    def __init__(self, symbol: str, pip_size, settings, *, db=None):
        self.symbol = symbol
        self.settings = settings
        self.db = db
        self.state_name = INITIALIZING

        precision_probe = extract("1.0", pip_size)
        self.state = SymbolState(symbol=symbol, precision=precision_probe.precision)

        n_features = len(feature_engine.feature_names())
        self.models = [
            DirichletFrequencyModel(window=1000),
            DirichletFrequencyModel(window=5000),
            BetaBinomialParityModel(),
            EWMAFrequencyModel(decay=0.995),
            MarkovDigitModel(max_order=2),
            MarkovParityModel(max_order=4),
            RunLengthParityModel(),
            OnlineLogisticParityModel(n_features=n_features),
            GradientBoostingParityModel(n_features=n_features),
        ]
        self._feature_consumers = [m for m in self.models if hasattr(m, "set_features")]
        self.ensemble = AdaptiveEnsemble(self.models)

        g = settings.gating
        self.calibrators = {
            "DIGITEVEN": CalibrationTracker(),
            "DIGITODD": CalibrationTracker(),
        }
        self.randomness = RandomnessMonitor(
            alpha=g.get("randomness_alpha", 0.01),
            min_samples=g.get("min_samples", 2000))
        self.regime_detector = RegimeDetector(min_samples=g.get("min_samples", 2000))
        self.gate = TradeGate(g, research_mode=settings.is_research,
                              risk_settings=settings.risk)

        # Sections 18/19: one tracker per side. Feeding both sides into one
        # tracker would report a flapping signal as a persistent one.
        self.signal_trackers = {
            "DIGITEVEN": SignalTracker(),
            "DIGITODD": SignalTracker(),
        }
        self._last_evidence = None
        self._last_break_even = 0.5128

        self._last_verdict = None
        self._last_regime = None
        self._verdict_every = 250
        self._ticks_since_verdict = 0
        self._pending_member_p: dict[str, float] = {}
        self._pending_calibrated: dict[str, float] = {}

    # ---- cold start (Section 48) ----------------------------------------

    def seed(self, ticks) -> int:
        self.state_name = LOADING_HISTORY
        n = 0
        for t in ticks:
            try:
                ex = extract(t.quote_raw, t.pip_size)
            except DigitExtractionError:
                continue
            self.state.add(self.symbol, t.epoch, ex)
            self.ensemble.observe(ex.digit)
            n += 1
        self.state_name = CALIBRATING if n else INITIALIZING
        return n

    # ---- per-tick evaluation --------------------------------------------

    def evaluate(self, tick_quote_raw: str, pip_size, epoch: float,
                 *, risk_manager, staking, api_connected: bool,
                 db_healthy: bool = True, open_contract: bool = False
                 ) -> tuple[Decision, object | None]:
        """Returns (decision, quote_request_or_None).

        Does NOT place trades and does NOT mutate model state -- the caller
        handles I/O, then calls learn() with the realized digit. Keeping
        prediction pure is what makes the leakage ordering auditable.

        THE FLOW CHANGED IN THIS REVISION (Sections 2, 5, 6). Previously the
        soft gates ran first and a quote was only fetched if all of them
        passed, which meant a stream that tested fair never reached the
        economics at all and every rejection collapsed into one reason code.
        Now the hard preconditions run first, and any candidate that clears
        them goes to the proposal stage -- so the economics are always
        measured and always recorded, and the opportunity score decides.
        """
        tick_valid = True
        try:
            ex = extract(tick_quote_raw, pip_size)
        except DigitExtractionError as exc:
            self._mark_absent()
            return (self._decision("NO_TRADE", "NO_TRADE_MALFORMED_TICK",
                                   f"digit extraction failed: {exc}"), None)

        # --- STEP 1: features from state WITHOUT this tick -----------------
        features = feature_engine.build(self.state)
        for m in self._feature_consumers:
            m.set_features(features.values)

        # --- STEP 2: predict ------------------------------------------------
        result = self.ensemble.predict()
        self._pending_member_p = dict(result.member_p_even)

        # --- periodic randomness battery (now evidence, not veto) ----------
        self._ticks_since_verdict += 1
        if self._last_verdict is None or self._ticks_since_verdict >= self._verdict_every:
            digits = self.state.digit_sequence(20000)
            if digits:
                self._last_verdict = self.randomness.evaluate(
                    digits, break_even_probability=self._last_break_even)
                if self.db:
                    self.db.record_randomness(self.symbol, self._last_verdict)
            self._ticks_since_verdict = 0

        self.regime_detector.update_cusum(ex.parity)
        cal_even = self.calibrators["DIGITEVEN"]
        self._last_regime = self.regime_detector.classify(
            recent_digits=self.state.digit_sequence(5000),
            long_run_freq=self.state.long_run_frequencies(),
            edge_verdict=self._last_verdict,
            ensemble_result=result,
            calibration_quality=cal_even.quality_score() if cal_even.is_fitted else None,
        )

        calibrated = cal_even.calibrate(result.p_even_ensemble)
        self._pending_calibrated = {"DIGITEVEN": calibrated, "DIGITODD": 1 - calibrated}

        side = "DIGITEVEN" if calibrated >= 0.5 else "DIGITODD"
        calibrator = self.calibrators[side]
        is_probe = calibrator.should_probe()

        interval = probability_interval_from_samples(
            calibrated, max(calibrator.n_samples, 1), confidence=0.95)

        # --- soft evidence (Sections 3, 4, 22, 23) --------------------------
        self._last_evidence = self._build_evidence(result, calibrator)

        decision = self._decision(
            "NO_TRADE", "PENDING", "", ex=ex, result=result,
            calibrated=calibrated, interval=interval, is_probe=is_probe)

        # --- hard preconditions: the ONLY things that can veto here ---------
        hard = self.gate.check_hard_preconditions(
            state=self.state, ensemble_result=result, calibrator=calibrator,
            api_connected=api_connected, db_healthy=db_healthy,
            # NB: `emergency_stop` is the METHOD that engages the stop;
            # `emergency_stopped` is the flag. Reading the former gives a
            # truthy bound method and freezes the bot permanently.
            emergency_stop=bool(getattr(risk_manager, "emergency_stopped", False)),
            open_contract=open_contract, account_available=True,
            tick_valid=tick_valid)
        decision.gates = list(hard.trail)

        if hard.blocked:
            decision.decision = "NO_TRADE"
            decision.reason_code = hard.code
            decision.explanation = hard.explanation
            decision.hard_gate_failed = True
            decision.apply_evidence(self._last_evidence)
            self.gate.deadlock.note_rejection(
                reason_code=hard.code, hard=True, opportunity_score=0.0)
            self._mark_absent()
            return (decision, None)

        # Cleared every veto -- go and price it. Section 11: the proposal is
        # authoritative, so nothing downstream is decided without one.
        self.state_name = PROPOSAL_CHECK
        return (decision, (side, is_probe))

    def finalize_with_quote(self, decision: Decision, proposal: Proposal, *,
                            risk_manager, staking, side: str, is_probe: bool
                            ) -> Decision:
        """Second half of evaluation, once a real quote exists."""
        calibrator = self.calibrators[side]
        interval_even = probability_interval_from_samples(
            self._pending_calibrated["DIGITEVEN"], max(calibrator.n_samples, 1))

        # Feed the real break-even back into the randomness battery: Section
        # 12 says economic relevance is measured against the actual contract,
        # and 0.5128 is only a default until a quote tells us otherwise.
        self._last_break_even = proposal.break_even_probability

        try:
            assessment = edge_for_side(
                proposal, self._pending_calibrated["DIGITEVEN"], interval_even,
                min_edge=self.settings.gating.get("min_edge", 0.02),
                min_ev=self.settings.gating.get("min_expected_value", 0.0))
        except ProposalError as exc:
            decision.decision = "NO_TRADE"
            decision.reason_code = "NO_TRADE_BAD_PROPOSAL"
            decision.explanation = str(exc)
            decision.hard_gate_failed = True
            self.gate.deadlock.note_rejection(
                reason_code="NO_TRADE_BAD_PROPOSAL", hard=True)
            self._mark_absent()
            return decision

        # --- Sections 18/19: persistence, measured on THIS side -------------
        tracker = self.signal_trackers[side]
        tracker.observe(
            side=side,
            probability=assessment.probability,
            edge=assessment.point_edge,
            agreement=decision.agreement_fraction or 0.5)
        persistence = tracker.read()
        for other, t in self.signal_trackers.items():
            if other != side:
                t.note_absent()

        evidence = self._last_evidence or self._build_evidence(
            self.ensemble.predict(), calibrator)
        evidence.persistence = persistence.combined
        evidence.edge_persistence = persistence.edge_persistence

        stake, _why = staking.stake_for(
            balance=risk_manager.balance,
            probability_lower_bound=assessment.probability_interval.lower,
            payout_multiple=proposal.payout_multiple)
        if is_probe:
            # A candidate on probation because its own calibration is
            # questionable is exactly the wrong one to size up on.
            stake = staking.base_stake
        risk_decision = risk_manager.can_trade(stake)

        self.state_name = EV_CHECK
        decision = self.gate.evaluate(
            decision=decision, edge_assessment=assessment, evidence=evidence,
            persistence=persistence, risk_decision=risk_decision)

        decision.contract_type = proposal.contract_type
        decision.stake = stake
        decision.payout = proposal.payout
        decision.break_even_probability = proposal.break_even_probability
        decision.edge = assessment.point_edge
        decision.conservative_edge = assessment.conservative_edge
        decision.expected_value = assessment.expected_value
        return decision

    # ---- evidence assembly (Sections 3, 4, 22, 23) -----------------------

    def _build_evidence(self, result, calibrator) -> EvidenceBundle:
        regime_ev = regime_evidence_from_state(self._last_regime)
        return EvidenceBundle(
            randomness=randomness_evidence_from_verdict(
                self._last_verdict, break_even=self._last_break_even),
            regime=regime_ev,
            models=model_evidence_from_ensemble(
                result, self.ensemble.health_report()),
            information=information_evidence(entropy=regime_ev.entropy),
            calibration=calibration_evidence(calibrator),
            sample_size=self.state.total_count,
        )

    def _mark_absent(self) -> None:
        """Section 19: a tick that produced no candidate must erode the
        persistence history, not leave it frozen in place."""
        for t in self.signal_trackers.values():
            t.note_absent()

    # ---- learning (Sections 21, 47) --------------------------------------

    def learn(self, tick_quote_raw: str, pip_size, epoch: float) -> int | None:
        """STEPS 4 and 5: observe the tick, then update every model.

        Called AFTER evaluate() for the same tick. Returns the realized digit.
        """
        try:
            ex = extract(tick_quote_raw, pip_size)
        except DigitExtractionError:
            return None
        self.state.add(self.symbol, epoch, ex)
        self.ensemble.observe(ex.digit)
        if self._pending_member_p:
            self.ensemble.record_outcomes(self._pending_member_p, ex.digit)
            self._pending_member_p = {}
        # Calibration learns from the REALIZED EVENT, never from trade P/L.
        if self._pending_calibrated:
            occurred_even = 1 if ex.parity == 0 else 0
            self.calibrators["DIGITEVEN"].record(
                self._pending_calibrated["DIGITEVEN"], occurred_even)
            self.calibrators["DIGITODD"].record(
                self._pending_calibrated["DIGITODD"], 1 - occurred_even)
            self._pending_calibrated = {}
        self.state_name = OBSERVING
        return ex.digit

    def maybe_refit(self) -> None:
        """Off the tick path (Section 62)."""
        for m in self.models:
            if hasattr(m, "maybe_refit"):
                m.maybe_refit()

    # ---- helpers ---------------------------------------------------------

    def _decision(self, decision: str, code: str, expl: str, *, ex=None, result=None,
                  calibrated=None, interval=None, is_probe: bool = False) -> Decision:
        d = Decision(timestamp=time.time(), symbol=self.symbol, decision=decision,
                     reason_code=code, explanation=expl, is_probe=is_probe)
        if ex is not None:
            d.current_quote = float(ex.normalized_quote)
            d.current_digit = ex.digit
            d.previous_digits = self.state.digit_sequence(10)
        if result is not None:
            d.digit_probabilities = result.digit_probabilities
            d.p_even_digit_derived = result.p_even_digit_derived
            d.p_even_direct = result.p_even_direct
            d.p_even_ensemble = result.p_even_ensemble
            d.member_p_even = result.member_p_even
            d.model_weights = result.weights
            d.dispersion = result.dispersion
            d.agreement_fraction = result.agreement_fraction
        if calibrated is not None:
            d.calibrated_p_even = calibrated
        if interval is not None:
            d.probability_lower = interval.lower
            d.probability_upper = interval.upper
        if self._last_regime is not None:
            d.regime = self._last_regime.regime
            d.entropy = self._last_regime.entropy
        if self._last_verdict is not None:
            d.randomness_tradeable = self._last_verdict.tradeable
        d.calibration_quality = self.calibrators["DIGITEVEN"].quality_score()
        return d


def new_idempotency_key(symbol: str, contract_type: str) -> str:
    """Section 35. Unique per intended trade; checked against the database
    before the buy so a retry at ANY layer cannot open a second contract."""
    return f"{symbol}:{contract_type}:{uuid.uuid4().hex[:16]}"
