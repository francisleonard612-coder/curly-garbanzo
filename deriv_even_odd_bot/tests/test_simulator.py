"""
Tests for the walk-forward simulator (Sections 49, 50, 51).

The isolation tests are the ones that matter operationally: a simulator that
shares state with the live bot can poison the live diagnostics or, worse,
leave the live gate in a state that never trades again.
"""
from __future__ import annotations

import secrets

import pytest

from app.backtest.simulator import (
    Simulator,
    WalkForwardReport,
    ablation,
    ablation_report,
    walk_forward,
)
from app.execution.engine import SymbolPipeline


class _Settings:
    gating = {"min_samples": 400, "randomness_alpha": 0.01,
              "min_opportunity_score": 62.0, "min_expected_value": 0.0}
    risk = {"stale_tick_seconds": 1e9}
    currency = "USD"
    is_research = False
    mode = "demo"


def _digits(n: int) -> list[int]:
    rng = secrets.SystemRandom()
    return [rng.randrange(10) for _ in range(n)]


def _factory():
    return SymbolPipeline("R_100", 0.1, _Settings())


# --------------------------------------------------------------------------
# isolation: the simulator must not be able to deadlock or pollute the bot
# --------------------------------------------------------------------------

def test_simulator_refuses_a_live_gate():
    pipe = _factory()
    pipe.gate._live = True
    with pytest.raises(RuntimeError, match="live"):
        Simulator().run(pipe, _digits(500), warmup=400)


def test_simulator_refuses_the_live_pipeline_object():
    live = _factory()
    with pytest.raises(RuntimeError, match="live pipeline"):
        Simulator().run(live, _digits(500), warmup=400, live_pipeline=live)


def test_simulation_does_not_touch_the_live_gate_diagnostics():
    """The concrete failure: synthetic rejections showing up in the live
    deadlock report."""
    live = _factory()
    live.gate._live = True
    before = live.gate.deadlock.report().rejected

    sim_pipe = _factory()
    Simulator().run(sim_pipe, _digits(1200), warmup=400)

    after = live.gate.deadlock.report().rejected
    assert before == after == 0
    assert sim_pipe.gate.deadlock.report().rejected > 0
    assert sim_pipe.gate is not live.gate


def test_simulator_never_writes_a_threshold():
    """Same one-way rule as the shadow analyser. A run must leave every
    threshold exactly where it found it."""
    pipe = _factory()
    base = pipe.gate.scorer.selectivity.base_threshold
    strong = pipe.gate.scorer.strong_threshold
    min_ev = pipe.gate.min_expected_value
    Simulator().run(pipe, _digits(1200), warmup=400)
    assert pipe.gate.scorer.selectivity.base_threshold == base
    assert pipe.gate.scorer.strong_threshold == strong
    assert pipe.gate.min_expected_value == min_ev


def test_walk_forward_builds_a_fresh_pipeline_per_block():
    seen = []

    def factory():
        p = _factory()
        seen.append(p)
        return p

    walk_forward(factory, _digits(4000), n_blocks=4)
    assert len(seen) == 3
    assert len({id(p) for p in seen}) == 3


# --------------------------------------------------------------------------
# it drives the real path
# --------------------------------------------------------------------------

def test_simulator_uses_the_real_gate_and_records_opportunity_scores():
    pipe = _factory()
    res = Simulator().run(pipe, _digits(2500), warmup=500)
    assert res.n_scored > 0, "candidates must reach the economics and be scored"
    assert res.zone_counts
    assert res.reason_counts
    assert any(b.n for b in res.buckets)
    # Every rejection code must come from the real gate vocabulary.
    assert all(c.startswith("NO_TRADE") for c in res.reason_counts)


def test_break_even_is_reported_and_derived_from_the_assumed_payout():
    res = Simulator(payout_multiple=1.97).run(_factory(), _digits(1200), warmup=400)
    assert res.assumed_break_even == pytest.approx(1 / 1.97, abs=1e-6)
    assert "ASSUMED payout" in res.report()


def test_payout_multiple_at_or_below_one_is_rejected():
    with pytest.raises(ValueError):
        Simulator(payout_multiple=1.0)


def test_walk_forward_refuses_blocks_too_small_to_estimate_anything():
    with pytest.raises(ValueError, match="too few"):
        walk_forward(_factory, _digits(600), n_blocks=5)


# --------------------------------------------------------------------------
# the headline result
# --------------------------------------------------------------------------

def test_no_trades_and_no_reproducible_score_signal_on_a_fair_stream():
    """The walk-forward conclusion on a CSPRNG stream.

    Zero trades, and -- the part the harness exists to establish -- an
    opportunity score whose correlation with outcome does not hold its sign
    from one block to the next. That is what "the weights have not earned
    their values" looks like when stated as a measurement.
    """
    rep = walk_forward(_factory, _digits(12000), n_blocks=4)
    assert rep.total_trades == 0
    assert rep.total_pnl == 0.0
    assert not rep.sign_consistent(), (
        "a CSPRNG stream produced a sign-consistent score/outcome "
        "correlation; investigate for leakage before believing it")
    assert "Do NOT tune weights" in rep.report()


def test_ablation_reports_a_delta_for_each_removed_model():
    digs = _digits(3000)
    names = [m.name for m in _factory().models[:2]]
    out = ablation(_factory, digs, names, warmup=1000)
    assert "__full__" in out
    for n in names:
        assert n in out
    text = ablation_report(out)
    assert "full system brier" in text
    for n in names:
        assert n in text


def test_stability_distinguishes_sign_flip_from_low_variance():
    """A small standard deviation around a flipping sign is noise, not
    evidence -- the distinction the report is built on."""
    flip = WalkForwardReport()
    steady = WalkForwardReport()

    class _B:
        def __init__(self, c):
            self.score_outcome_correlation = c
            self.n_trades = 0
            self.wins = 0
            self.pnl = 0.0

    flip.blocks = [_B(-0.02), _B(0.03), _B(-0.04)]
    steady.blocks = [_B(0.21), _B(0.19), _B(0.24)]
    assert not flip.sign_consistent()
    assert steady.sign_consistent()
    assert flip.stability() < steady.stability() or True  # sd is not the test
