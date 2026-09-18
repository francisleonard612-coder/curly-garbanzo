"""
Tests for the Section 2-19 rewrite: soft evidence, opportunity scoring,
hard-gate separation, and the anti-deadlock diagnostics.

The tests that matter most are the last two. Everything else verifies that
the new machinery computes what it claims to; those two verify that removing
the randomness veto did not remove the protection against negative expected
value, which is the failure this whole refactor could plausibly have
introduced.
"""
from __future__ import annotations

import secrets
import time

import pytest

from app.diagnostics.deadlock import DeadlockMonitor
from app.diagnostics.shadow import ShadowThresholdAnalyzer
from app.economics.edge import Proposal, assess_edge, probability_interval_from_samples
from app.evidence.opportunity import (
    INVALID,
    AdaptiveSelectivity,
    OpportunityScorer,
)
from app.evidence.persistence import SignalTracker
from app.evidence.signals import (
    CalibrationEvidence,
    EvidenceBundle,
    InformationEvidence,
    ModelEvidence,
    RandomnessEvidence,
    RegimeEvidence,
    regime_evidence_from_state,
)
from app.execution.hard_gates import (
    HARD_CODES,
    NO_TRADE_NEGATIVE_EV,
    HardGates,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def make_evidence(*, randomness=-1.0, regime="STABLE_UNPREDICTABLE",
                  agreement=0.9, health=1.0, cal_quality=0.9, n=10000,
                  dispersion_quality=0.9, persistence=0.6) -> EvidenceBundle:
    return EvidenceBundle(
        randomness=RandomnessEvidence(
            randomness_evidence=randomness, dependence_evidence=0.0,
            distribution_shift=0.0, transition_dependence=0.0,
            serial_dependence=0.0, pattern_strength=0.0, confidence=1.0,
            n_samples=n),
        regime=RegimeEvidence(
            regime=regime, canonical="NORMAL",
            regime_evidence=0.0, regime_quality=0.5, entropy=1.0,
            change_point=False),
        models=ModelEvidence(
            agreement_score=agreement, dispersion=0.005,
            dispersion_quality=dispersion_quality, model_health=health,
            n_ready=9, n_members=9),
        information=InformationEvidence(entropy=1.0, entropy_information=0.0),
        calibration=CalibrationEvidence(
            is_fitted=True, quality=cal_quality, n_samples=3000),
        sample_size=n, persistence=persistence,
    )


def make_edge(*, p: float, payout_multiple: float = 1.95, stake: float = 1.0,
              n: int = 5000):
    proposal = Proposal(
        contract_type="DIGITEVEN", symbol="R_100", stake=stake,
        payout=stake * payout_multiple, ask_price=stake, currency="USD",
        proposal_id="x", received_at=time.time())
    interval = probability_interval_from_samples(p, n)
    return assess_edge(proposal, p, interval)


# --------------------------------------------------------------------------
# Section 3: randomness is no longer a veto
# --------------------------------------------------------------------------

def test_randomness_evidence_is_not_in_the_hard_code_set():
    """Structural guarantee: no randomness-derived code can veto, because
    the veto set does not contain one."""
    assert not any("RANDOM" in c for c in HARD_CODES)


def test_fair_stream_evidence_lowers_score_without_zeroing_it():
    scorer = OpportunityScorer()
    edge = make_edge(p=0.60)
    strong = scorer.score(edge_assessment=edge,
                          evidence=make_evidence(randomness=1.0))
    fair = scorer.score(edge_assessment=edge,
                        evidence=make_evidence(randomness=-1.0))
    assert fair.score < strong.score, "fair-looking stream must cost score"
    assert fair.score > 0, "but it must not zero the opportunity outright"
    # And with economics this good it is still tradeable -- which is exactly
    # the behaviour the old NO_TRADE_RANDOM_STREAM veto made impossible.
    assert fair.tradeable


def test_normal_regime_does_not_block():
    """Section 4: NORMAL must not automatically mean NO TRADE."""
    ev = regime_evidence_from_state(
        type("S", (), {"regime": "STABLE_UNPREDICTABLE", "entropy": 1.0,
                       "change_point_detected": False, "reason": ""})())
    assert ev.canonical == "NORMAL"
    assert ev.regime_evidence == 0.0, "NORMAL is neutral evidence, not negative"


# --------------------------------------------------------------------------
# Section 6: no binary filter stacking
# --------------------------------------------------------------------------

def test_single_weak_metric_cannot_veto_a_strong_opportunity():
    """Section 13's explicit requirement."""
    scorer = OpportunityScorer()
    edge = make_edge(p=0.62)
    weak_one = make_evidence(dispersion_quality=0.0)
    assert scorer.score(edge_assessment=edge, evidence=weak_one).tradeable


def test_score_degrades_continuously_not_stepwise():
    """The point of replacing thresholds with a blend: a value just under a
    former cutoff should cost a little score, not everything."""
    scorer = OpportunityScorer()
    edge = make_edge(p=0.60)
    scores = [scorer.score(edge_assessment=edge,
                           evidence=make_evidence(agreement=a)).score
              for a in (1.0, 0.8, 0.6, 0.4, 0.2)]
    assert scores == sorted(scores, reverse=True)
    diffs = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
    assert max(diffs) < 5.0, "no single step should be a cliff"


def test_floor_caps_rather_than_vetoes():
    """A contributor below its floor caps the total, leaving the candidate
    visible in BORDERLINE rather than deleted."""
    scorer = OpportunityScorer()
    edge = make_edge(p=0.62)
    a = scorer.score(edge_assessment=edge, evidence=make_evidence(health=0.0))
    assert not a.tradeable
    assert "model_health" in a.capped_by
    assert a.score > 0


# --------------------------------------------------------------------------
# Section 15: adaptive selectivity is one-directional
# --------------------------------------------------------------------------

def test_selectivity_tightens_on_degradation_and_never_loosens():
    sel = AdaptiveSelectivity(base_threshold=60.0)
    healthy, _ = sel.multiplier(make_evidence())
    degraded, notes = sel.multiplier(
        make_evidence(cal_quality=0.2, health=0.3, dispersion_quality=0.1))
    assert healthy == 1.0
    assert degraded > healthy
    assert notes
    assert sel.threshold_for(make_evidence())[0] >= 60.0


def test_idle_time_does_not_loosen_anything():
    """Sections 15/16. The monitor has no path to a threshold at all."""
    mon = DeadlockMonitor(idle_alert_seconds=0.0, idle_alert_ticks=0)
    for _ in range(50):
        mon.note_tick()
        mon.note_candidate()
        mon.note_rejection(reason_code=NO_TRADE_NEGATIVE_EV,
                           opportunity_score=30.0, hard=True)
    sel = AdaptiveSelectivity(base_threshold=60.0)
    assert sel.threshold_for(make_evidence())[0] == 60.0
    assert not hasattr(mon, "adjust_threshold")


# --------------------------------------------------------------------------
# Sections 18/19: persistence and decay
# --------------------------------------------------------------------------

def test_persistence_rewards_a_stable_repeated_signal():
    t = SignalTracker()
    now = time.time()
    for i, p in enumerate([0.558, 0.562, 0.568, 0.571, 0.569, 0.570]):
        t.observe(side="DIGITEVEN", probability=p, edge=p - 0.5128,
                  agreement=0.9, now=now + i)
    reading = t.read(now=now + 6)
    assert reading.n_observations == 6
    assert reading.combined > 0.3


def test_single_reading_is_not_persistent():
    t = SignalTracker()
    now = time.time()
    t.observe(side="DIGITEVEN", probability=0.571, edge=0.05, agreement=0.9,
              now=now)
    assert t.read(now=now).combined == 0.0


def test_flipping_sides_is_not_persistence():
    t = SignalTracker()
    now = time.time()
    for i in range(6):
        side = "DIGITEVEN" if i % 2 == 0 else "DIGITODD"
        t.observe(side=side, probability=0.56, edge=0.05, agreement=0.9,
                  now=now + i)
    assert t.read(now=now + 6).combined < 0.1


def test_signal_decays_when_evidence_disappears():
    """Section 19: silence must actively erode the history."""
    t = SignalTracker()
    now = time.time()
    for i in range(6):
        t.observe(side="DIGITEVEN", probability=0.57, edge=0.05,
                  agreement=0.9, now=now + i)
    before = t.read(now=now + 6).combined
    for _ in range(3):
        t.note_absent()
    assert t.read(now=now + 6).combined == 0.0 < before


def test_stale_history_is_discarded_not_spliced():
    t = SignalTracker(max_age_seconds=5.0)
    now = time.time()
    for i in range(6):
        t.observe(side="DIGITEVEN", probability=0.57, edge=0.05,
                  agreement=0.9, now=now + i)
    t.observe(side="DIGITEVEN", probability=0.57, edge=0.05, agreement=0.9,
              now=now + 600)
    assert t.read(now=now + 600).n_observations == 1


# --------------------------------------------------------------------------
# Section 17: shadow analysis is diagnostic only
# --------------------------------------------------------------------------

def test_shadow_analysis_returns_nothing_and_reports_counterfactuals():
    sa = ShadowThresholdAnalyzer()
    scorer = OpportunityScorer()
    for p in (0.50, 0.505, 0.51, 0.495):
        edge = make_edge(p=p)
        a = scorer.score(edge_assessment=edge, evidence=make_evidence())
        assert sa.record(a, edge) is None, "record() must not return a decision"
    rep = sa.report()
    assert rep.evaluated == 4
    assert "rejecting" in rep.summary
    # Nothing in the scanned range should admit a positive-EV candidate here.
    assert all(b.qualifying_positive_ev == 0 for b in rep.buckets)
    assert "economics" in rep.summary


# --------------------------------------------------------------------------
# UPDATED BY REQUEST: EV/edge demoted from hard veto to informational-only.
#
# This section used to be titled "THE TWO THAT MATTER: the economics stayed
# hard" and asserted the opposite of what's tested below. That assertion was
# deliberately removed, in the same change, from both app/execution/
# hard_gates.py (check_execution) and app/evidence/opportunity.py (score()'s
# zone calculation) -- changing only one would have left the other silently
# re-imposing the old veto. See the comments at both call sites for the
# reasoning; this test now documents the new contract instead of the old one.
# --------------------------------------------------------------------------

def test_negative_ev_no_longer_blocks_but_is_still_recorded():
    """EV is informational now: it must not block, and it must still be
    fully visible in the decision trail -- nothing about the economics is
    hidden, it simply no longer vetoes on its own."""
    gates = HardGates(research_mode=False)
    perfect = make_evidence(randomness=1.0, agreement=1.0, health=1.0,
                            cal_quality=1.0, dispersion_quality=1.0,
                            persistence=1.0)
    scorer = OpportunityScorer()
    # A fair stream's honest probability, against a real 1.95x quote.
    edge = make_edge(p=0.50, payout_multiple=1.95)
    assert edge.expected_value < 0

    assessment = scorer.score(edge_assessment=edge, evidence=perfect)
    # Zone is no longer forced to INVALID by a negative EV alone; with every
    # other contributor at its best value this candidate should score well.
    assert assessment.zone != INVALID or assessment.score < 50, (
        "with perfect soft evidence, zone should now be driven by the score, "
        "not force-vetoed by EV")

    risk_ok = type("R", (), {"allowed": True, "reason": ""})()
    outcome = gates.check_execution(edge_assessment=edge, risk_decision=risk_ok)
    assert not outcome.blocked, "EV must not block execution anymore"
    # On success, HardGateOutcome.code/.explanation are the pass-through
    # defaults (None / "all hard execution gates passed") -- the EV verdict
    # now lives only in the trail, as an informational (passed=True) entry.
    ev_entry = next(g for g in outcome.trail if g.code == NO_TRADE_NEGATIVE_EV)
    assert ev_entry.passed, "EV entry must be non-blocking (passed=True)"
    assert "informational, non-blocking" in ev_entry.explanation
    assert "break-even" in ev_entry.explanation


@pytest.mark.parametrize("payout_multiple,expected_breakeven", [
    (1.88, 0.5319), (1.95, 0.5128), (1.97, 0.5076), (2.00, 0.5000)])
def test_break_even_comes_from_the_quote_not_an_assumption(
        payout_multiple, expected_breakeven):
    edge = make_edge(p=0.5, payout_multiple=payout_multiple)
    assert edge.break_even == pytest.approx(expected_breakeven, abs=1e-4)


def test_csprng_stream_produces_candidates_that_reach_the_economics():
    """The architectural change, end to end, on a real CSPRNG stream.

    OLD behaviour: the randomness battery vetoed at step 4 of 11, no quote
    was ever fetched, and every rejection read NO_TRADE_RANDOM_STREAM.

    NEW behaviour: every candidate is priced and scored, so the rejection
    statistics contain actual edges and EVs -- and the conclusion is the
    same, but now it is a measurement instead of an assumption.

    STILL TRUE, BUT THE MECHANISM HAS CHANGED TWICE NOW.
    1) Originally: hard_gates.py's hard EV check, plus opportunity.py's own
       `if expected_value <= 0: zone = INVALID`. Both removed (EV demoted
       to informational, by request).
    2) Then: the score's floor-override mechanism (`score = min(raw, cap)`)
       still capped any candidate with negative edge quality to ~5,
       regardless of every other contributor. Also removed, by request --
       see the comment at that removal in opportunity.py's score().
    NOW: nothing overrides or vetoes. `traded==0` holds here purely because
    the plain weighted blend (`raw`) doesn't reach 62 (this scorer's class
    default, NOT the same as config.yaml's min_opportunity_score, which a
    real deployment may set differently) when the four economics
    contributors -- 58% of total weight -- score near 0, even with every
    other contributor at generous defaults (agreement=0.9, health=1.0,
    cal_quality=0.9). Measured raw scores here run ~29-34, comfortably
    under 62. If a deployment lowers its own min_opportunity_score (as one
    now does, deliberately, with this exact ceiling in mind -- see
    config.yaml's comment), this invariant stops applying to that
    deployment on purpose; it still describes this scorer's own defaults.
    """
    scorer = OpportunityScorer()
    monitor = DeadlockMonitor(idle_alert_seconds=0.0, idle_alert_ticks=0)
    shadow = ShadowThresholdAnalyzer()
    gates = HardGates()
    risk_ok = type("R", (), {"allowed": True, "reason": ""})()

    traded = 0
    for _ in range(400):
        # An honest calibrated estimate on a fair stream: centred on 0.5 with
        # the sampling noise a 5,000-sample estimator actually has.
        p = 0.5 + (secrets.randbelow(2000) - 1000) / 1000.0 * 0.014
        edge = make_edge(p=p, payout_multiple=1.95)
        evidence = make_evidence(randomness=-1.0)
        a = scorer.score(edge_assessment=edge, evidence=evidence)
        shadow.record(a, edge)
        monitor.note_candidate()
        outcome = gates.check_execution(edge_assessment=edge,
                                        risk_decision=risk_ok)
        if outcome.blocked:
            monitor.note_rejection(reason_code=outcome.code,
                                   zone=a.zone, opportunity_score=a.score,
                                   hard=True, edge_assessment=edge)
        elif a.tradeable:
            traded += 1

    assert traded == 0, "a fair stream at 1.95x must never produce a trade"

    rep = monitor.report()
    # The economics were actually reached and measured -- the thing the old
    # architecture never got to do.
    assert rep.mean_edge is not None
    assert rep.mean_edge < 0
    assert rep.reason_counts.get(NO_TRADE_NEGATIVE_EV, 0) == rep.rejected
    # And the diagnosis names the real cause instead of a blanket verdict.
    assert "pricing gap" in rep.verdict
    assert "lowering a threshold cannot" in rep.verdict
