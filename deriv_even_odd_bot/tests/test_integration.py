"""
Integration tests (spec Section 58).

The headline test is test_full_pipeline_never_trades_on_random_data: the
whole system, wired exactly as it runs live, driven by a CSPRNG stream,
must produce zero trade decisions. That single assertion is what separates
this bot from one that loses money confidently.
"""
import os
import random
import tempfile

import pytest

os.environ.setdefault("TRADING_MODE", "research")
os.environ.setdefault("DERIV_USE_REAL", "false")

from app.calibration.calibrator import CalibrationTracker
from app.data.tick_store import ExponentialDigitWeighting, SymbolState
from app.diagnostics.leakage import LeakageCanary, assert_no_leakage
from app.digits.extraction import extract
from app.economics.edge import Proposal, ProposalError, edge_for_side, required_probability
from app.features import engine as feature_engine
from app.risk.manager import RiskManager
from app.risk.staking import KELLY, MARTINGALE, StakingEngine
from app.statistics.distribution import Interval


def digits(n, seed=0):
    rng = random.Random(seed)
    return [rng.randrange(10) for _ in range(n)]


def make_state(digs):
    st = SymbolState(symbol="T", precision=1)
    for i, d in enumerate(digs):
        st.add("T", float(i), extract(f"100.{d}", 0.1))
    return st


class TestTickStore:
    def test_rolling_window_counts_match_bruteforce(self):
        st = make_state(digits(3000, 1))
        w = st.windows[100]
        expected = [0] * 10
        for d in st.digit_sequence(100):
            expected[d] += 1
        assert w.digit_counts == expected
        assert len(w) == 100

    def test_long_run_counts_are_exact(self):
        digs = digits(2000, 2)
        st = make_state(digs)
        assert st.total_count == 2000
        assert st.total_even == sum(1 for d in digs if d % 2 == 0)

    def test_run_length_tracking(self):
        st = SymbolState(symbol="T", precision=1)
        for i, d in enumerate([2, 4, 6, 1]):
            st.add("T", float(i), extract(f"100.{d}", 0.1))
        parity, run = st.current_run()
        assert parity == 1 and run == 1

    def test_ewma_effective_n_never_reaches_ceiling(self):
        """The asymptote trap: any min-sample gate compared against
        effective_n must sit below 1/(1-decay) or it can never pass."""
        e = ExponentialDigitWeighting(decay=0.99)
        for _ in range(100000):
            e.add(3)
        assert e.effective_n < e.ceiling
        assert e.ceiling == pytest.approx(100.0)


class TestLeakage:
    def test_feature_engine_has_no_lookahead(self):
        assert_no_leakage(lambda: SymbolState(symbol="T", precision=1),
                          digits(500, 3), feature_engine.build)

    def test_features_are_deterministic_for_same_history(self):
        digs = digits(400, 4)
        a = feature_engine.build(make_state(digs)).values
        b = feature_engine.build(make_state(digs)).values
        assert a == b

    def test_canary_is_clean_on_honest_random_guessing(self):
        canary = LeakageCanary(seed=1)
        rng = random.Random(2)
        for _ in range(3000):
            canary.record(rng.random() < 0.5)
        r = canary.result()
        assert not r.leaked, r.detail

    def test_canary_fires_on_a_cheating_predictor(self):
        """A predictor that can see the CURRENT label must be caught.

        Note the shape of this test carefully -- an earlier version had the
        cheater peek at round N's label to guess round N+1, which is not
        cheating at all: the labels are iid, so knowing the last one tells
        you nothing and the canary correctly reported 'clean'. Real leakage
        means seeing the answer to the question being asked RIGHT NOW, which
        is what a lookahead feature actually gives a model. Reproduced here
        by replaying the canary's own label stream from the same seed.
        """
        canary = LeakageCanary(seed=3)
        oracle = random.Random(3)      # identical stream to the canary's
        for _ in range(3000):
            peeked_even = oracle.random() < 0.5
            canary.record(peeked_even)
        r = canary.result()
        assert r.leaked, r.detail
        assert r.accuracy > 0.99


class TestEconomics:
    def test_break_even_is_derived_from_payout(self):
        p = Proposal("DIGITEVEN", "R_100", 1.0, 1.95, 1.0, "USD", "x", received_at=1e9)
        assert p.break_even_probability == pytest.approx(1 / 1.95)
        assert p.break_even_probability > 0.5   # the house edge, made explicit

    def test_payout_below_stake_is_rejected(self):
        with pytest.raises(ProposalError):
            Proposal("DIGITEVEN", "R_100", 1.0, 0.9, 1.0, "USD", "x")

    def test_probability_above_half_is_not_enough(self):
        """Section 69: never trade merely because P > 0.50."""
        import time
        p = Proposal("DIGITEVEN", "R_100", 1.0, 1.95, 1.0, "USD", "x", received_at=time.time())
        iv = Interval(0.505, 0.500, 0.510, 10000)
        a = edge_for_side(p, 0.505, iv, min_edge=0.0)
        assert not a.tradeable

    def test_odd_side_orientation_is_flipped_correctly(self):
        import time
        p = Proposal("DIGITODD", "R_100", 1.0, 1.95, 1.0, "USD", "x", received_at=time.time())
        iv_even = Interval(0.30, 0.28, 0.32, 5000)       # EVEN unlikely => ODD likely
        a = edge_for_side(p, 0.30, iv_even, min_edge=0.0, max_interval_width=0.10)
        assert a.probability == pytest.approx(0.70)
        assert a.tradeable

    def test_required_probability_matches_payout(self):
        assert required_probability(2.0) == pytest.approx(0.5)
        assert required_probability(1.95) == pytest.approx(0.5128, abs=1e-4)


class TestRisk:
    def test_blocks_without_verified_balance(self):
        r = RiskManager()
        assert not r.can_trade(1.0, balance_verified=False).allowed

    def test_blocks_over_max_stake(self):
        r = RiskManager(max_stake=5.0)
        r.update_balance(100)
        assert not r.can_trade(6.0).allowed
        assert r.can_trade(5.0).allowed

    def test_daily_loss_triggers_emergency_stop(self):
        r = RiskManager(max_daily_loss=10.0)
        r.update_balance(100)
        r.register_open(1.0)
        r.register_result(-10.0)
        assert r.emergency_stopped
        assert not r.can_trade(1.0).allowed

    def test_emergency_stop_does_not_auto_clear(self):
        r = RiskManager(max_daily_loss=10.0)
        r.update_balance(100)
        r.register_open(1.0)
        r.register_result(-10.0)
        r.register_open(1.0)
        r.register_result(+50.0)      # a win must not reopen trading
        assert r.emergency_stopped

    def test_consecutive_losses_block(self):
        r = RiskManager(max_consecutive_losses=3, max_daily_loss=1e9, max_drawdown=1e9)
        r.update_balance(1000)
        for _ in range(3):
            r.register_open(1.0)
            r.register_result(-1.0)
        assert not r.can_trade(1.0).allowed


class TestStaking:
    def test_fixed_is_default_and_capped(self):
        s = StakingEngine(base_stake=2.0, max_stake=1.5)
        stake, _ = s.stake_for()
        assert stake == 1.5

    def test_martingale_disabled_by_default(self):
        s = StakingEngine(method=MARTINGALE, base_stake=1.0, max_stake=100.0)
        for _ in range(5):
            s.register_result(won=False)
        stake, why = s.stake_for()
        assert stake == 1.0
        assert "not enabled" in why

    def test_kelly_uses_lower_bound_and_is_capped(self):
        s = StakingEngine(method=KELLY, base_stake=1.0, max_stake=1000.0,
                          kelly_fraction=0.25, kelly_cap=0.05)
        stake, _ = s.stake_for(balance=1000.0, probability_lower_bound=0.90,
                               payout_multiple=1.95)
        assert stake <= 1000.0 * 0.05     # cap binds even at absurd confidence

    def test_kelly_refuses_without_edge(self):
        s = StakingEngine(method=KELLY)
        stake, why = s.stake_for(balance=1000.0, probability_lower_bound=0.50,
                                 payout_multiple=1.95)
        assert stake == 0.0


class TestCalibration:
    def test_unfitted_calibrator_is_identity_and_says_so(self):
        c = CalibrationTracker(min_samples=100)
        assert not c.is_fitted
        assert c.calibrate(0.63) == pytest.approx(0.63)

    def test_fits_and_corrects_a_systematically_overconfident_model(self):
        c = CalibrationTracker(min_samples=200, refit_every=50)
        rng = random.Random(7)
        for _ in range(2000):
            true_p = rng.uniform(0.35, 0.65)
            raw = 0.5 + (true_p - 0.5) * 2.5        # overconfident by 2.5x
            raw = min(max(raw, 0.01), 0.99)
            c.record(raw, 1 if rng.random() < true_p else 0)
        assert c.is_fitted
        assert abs(c.calibrate(0.90) - 0.90) > 0.05   # pulled toward reality

    def test_probe_breaks_the_lockout(self):
        c = CalibrationTracker(probe_interval=5)
        assert not c.should_probe()
        for _ in range(5):
            c.note_blocked()
        assert c.should_probe()
        c.note_settled()
        assert not c.should_probe()


class TestFullPipeline:
    def _settings(self):
        from app.config.settings import Settings
        s = Settings()
        return s

    def test_full_pipeline_never_trades_on_random_data(self):
        """THE HEADLINE TEST. The complete system, driven by a CSPRNG
        stream, must reach zero trade decisions across independent runs."""
        from app.execution.engine import SymbolPipeline

        settings = self._settings()
        for seed in range(3):
            pipe = SymbolPipeline("R_100", 0.1, settings)
            digs = digits(4000, seed=100 + seed)
            for i, d in enumerate(digs[:3000]):
                pipe.learn(f"100.{d}", 0.1, float(i))

            traded = 0
            for i, d in enumerate(digs[3000:], start=3000):
                decision, _ = pipe.evaluate(
                    f"100.{d}", 0.1, float(i),
                    risk_manager=RiskManager(), staking=StakingEngine(),
                    api_connected=True)
                if decision.will_trade:
                    traded += 1
                pipe.learn(f"100.{d}", 0.1, float(i))
            assert traded == 0, f"seed {seed}: pipeline traded {traded} times on random data"

    def test_every_decision_carries_a_machine_readable_reason(self):
        from app.execution.engine import SymbolPipeline
        settings = self._settings()
        pipe = SymbolPipeline("R_100", 0.1, settings)
        digs = digits(2600, seed=200)
        for i, d in enumerate(digs[:2500]):
            pipe.learn(f"100.{d}", 0.1, float(i))
        for i, d in enumerate(digs[2500:], start=2500):
            decision, _ = pipe.evaluate(f"100.{d}", 0.1, float(i),
                                        risk_manager=RiskManager(),
                                        staking=StakingEngine(), api_connected=True)
            assert decision.reason_code
            assert decision.reason_code.startswith(("NO_TRADE", "TRADE"))
            assert decision.gates
            pipe.learn(f"100.{d}", 0.1, float(i))

    def test_models_stay_at_chance_on_random_data(self):
        from app.execution.engine import SymbolPipeline
        settings = self._settings()
        pipe = SymbolPipeline("R_100", 0.1, settings)
        for i, d in enumerate(digits(5000, seed=300)):
            pipe.evaluate(f"100.{d}", 0.1, float(i), risk_manager=RiskManager(),
                          staking=StakingEngine(), api_connected=True)
            pipe.learn(f"100.{d}", 0.1, float(i))
        for name, m in pipe.ensemble.health_report().items():
            if m["n"] >= 200:
                # Brier skill should hug zero. Anything far above it on a
                # CSPRNG stream means leakage, not skill.
                assert m["brier_skill"] < 0.05, f"{name} claims skill on random data: {m}"


class TestDatabase:
    def test_decision_round_trip_and_reason_histogram(self):
        from app.execution.gating import Decision, GateResult
        from app.storage.db import Database
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(os.path.join(tmp, "t.db"))
            d = Decision(timestamp=1.0, symbol="R_100", decision="NO_TRADE",
                         reason_code="NO_TRADE_RANDOM_STREAM", explanation="looks random")
            d.gates = [GateResult(False, "NO_TRADE_RANDOM_STREAM", "veto")]
            db.record_decision(d)
            db.record_decision(d)
            assert db.reason_histogram()["NO_TRADE_RANDOM_STREAM"] == 2
            db.close()

    def test_idempotency_key_blocks_duplicates(self):
        from app.storage.db import Database
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(os.path.join(tmp, "t.db"))
            assert not db.has_idempotency_key("k1")
            db.record_trade_open(decision_id=None, symbol="R_100", contract_id=1,
                                 idempotency_key="k1", contract_type="DIGITEVEN",
                                 stake=1.0, payout=1.95, buy_price=1.0, entry_digit=4)
            assert db.has_idempotency_key("k1")
            db.close()
