"""
Walk-forward simulator, opportunity-score calibration and ablation
(spec Sections 13, 17, 49, 50, 51).

WHY THIS FILE WAS REWRITTEN. The previous simulator had its own private
decision rule -- `confidence > break_even` -- so it validated a bot that does
not exist. With the Section 13 opportunity score now deciding trades, a
simulator that never computes that score cannot answer the only question
worth asking of it: DOES A HIGHER OPPORTUNITY SCORE ACTUALLY PREDICT A HIGHER
WIN RATE? The weights in OpportunityScorer are currently my judgement. This
harness is the mechanism by which they earn their values, or fail to.

It drives the REAL pipeline path: evaluate() -> proposal -> finalize_with_quote()
-> the real TradeGate. A leakage bug or a gating bug shows up identically here
and live. The legacy implementation is preserved in simulator_legacy.py for
comparison against historical runs; nothing imports it.

THREE THINGS IT DELIBERATELY WILL NOT DO.

1. IT CANNOT DEADLOCK THE LIVE BOT. Every run builds its OWN SymbolPipeline
   via a factory, so it touches its own TradeGate, its own DeadlockMonitor and
   its own SignalTrackers. It never accepts a live pipeline, never mutates
   settings, and never writes a threshold -- the same one-way rule that governs
   app/diagnostics/shadow.py. `assert_isolated()` checks this at runtime and
   there is a test for it.

2. IT DOES NOT TUNE ANYTHING AUTOMATICALLY. It reports which weights would
   have helped. A harness that rewrote its own weights from the same data it
   scored on would be fitting noise with extra steps; the walk-forward split
   exists precisely so a human can see out-of-sample behaviour before changing
   a constant.

3. THE PAYOUT ASSUMPTION IS EXPLICIT. Historical proposals are not available,
   so a fixed multiple is assumed and NAMED in every report. Assuming 1.97x
   when live quotes average 1.92x turns a losing system into a winning
   backtest, and that is the most common way a binary-options backtest lies.

CHRONOLOGICAL INTEGRITY IS ABSOLUTE (Section 50). Ticks replay in order.
There is no `shuffle` option, deliberately: an option that must never be used
is better not offered.
"""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from app.economics.edge import Proposal, required_probability


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass
class ScoreBucket:
    """One decile of the opportunity score, with its realized outcomes."""
    low: float
    high: float
    n: int = 0
    wins: int = 0
    sum_edge: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else float("nan")

    @property
    def mean_edge(self) -> float:
        return self.sum_edge / self.n if self.n else float("nan")


@dataclass
class SimulationResult:
    n_ticks: int
    n_evaluated: int
    n_scored: int
    n_trades: int
    wins: int
    losses: int
    pnl: float
    stake: float
    payout_multiple: float
    assumed_break_even: float

    brier: float = float("nan")
    log_loss: float = float("nan")
    accuracy: float = float("nan")
    max_losing_streak: int = 0
    max_drawdown: float = 0.0

    reason_counts: dict = field(default_factory=dict)
    zone_counts: dict = field(default_factory=dict)
    buckets: list[ScoreBucket] = field(default_factory=list)
    score_outcome_correlation: float = float("nan")
    contributor_correlation: dict = field(default_factory=dict)
    mean_score: float = float("nan")

    @property
    def win_rate(self) -> float:
        return self.wins / self.n_trades if self.n_trades else float("nan")

    @property
    def selectivity(self) -> float:
        return self.n_trades / self.n_scored if self.n_scored else 0.0

    def report(self) -> str:
        L = ["=" * 72,
             f"SIMULATION  ticks={self.n_ticks} evaluated={self.n_evaluated} "
             f"scored={self.n_scored} trades={self.n_trades} "
             f"selectivity={self.selectivity:.5f}",
             f"  ASSUMED payout {self.payout_multiple:.3f}x -> break-even "
             f"{self.assumed_break_even:.4f} (accuracy needed merely to not lose)",
             f"  win rate {self.win_rate:.4f}  pnl {self.pnl:+.2f}  "
             f"max losing streak {self.max_losing_streak}  max dd {self.max_drawdown:.2f}",
             f"  prediction quality: brier={self.brier:.4f} "
             f"log_loss={self.log_loss:.4f} accuracy={self.accuracy:.4f}"]
        if self.buckets:
            L.append(f"  OPPORTUNITY SCORE vs REALIZED OUTCOME "
                     f"(corr={self.score_outcome_correlation:+.4f}, "
                     f"mean score {self.mean_score:.1f}):")
            L.append(f"    {'score range':<16}{'n':>7}{'win rate':>11}{'mean edge':>12}")
            for b in self.buckets:
                if b.n:
                    L.append(f"    {b.low:5.1f}-{b.high:<10.1f}{b.n:>7}"
                             f"{b.win_rate:>11.4f}{b.mean_edge:>+12.4f}")
            L.append(f"    break-even win rate at this payout: "
                     f"{self.assumed_break_even:.4f}")
        if self.contributor_correlation:
            L.append("  CONTRIBUTOR vs OUTCOME correlation (does the weight earn it?):")
            ranked = sorted(self.contributor_correlation.items(),
                            key=lambda kv: -abs(kv[1]))
            for name, c in ranked[:12]:
                L.append(f"    {name:<32}{c:+.4f}")
        if self.zone_counts:
            L.append("  zones: " + ", ".join(
                f"{k}={v}" for k, v in sorted(self.zone_counts.items())))
        if self.reason_counts:
            L.append("  no-trade reasons:")
            for code, n in sorted(self.reason_counts.items(), key=lambda kv: -kv[1])[:10]:
                L.append(f"    {code:<38}{n}")
        L.append("=" * 72)
        return "\n".join(L)


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


# ---------------------------------------------------------------------------
# the simulator
# ---------------------------------------------------------------------------

class _SimRisk:
    """Minimal risk stand-in. Always allows, so that REJECTIONS IN A
    SIMULATION ARE ATTRIBUTABLE TO THE PREDICTIVE STACK rather than to a
    daily-loss limit that the live manager would have applied. Risk limits are
    tested directly in tests/test_integration.py; mixing them in here would
    make an ablation result depend on the order of simulated losses."""

    allowed = True
    reason = ""
    balance = 1000.0
    emergency_stopped = False

    def can_trade(self, stake: float):
        return self


class _SimStaking:
    def __init__(self, stake: float = 1.0):
        self.base_stake = stake

    def stake_for(self, **kwargs):
        return (self.base_stake, "simulated fixed stake")


class Simulator:
    """Replays a digit sequence through the real pipeline and the real gate."""

    def __init__(self, *, stake: float = 1.0, payout_multiple: float = 1.95,
                 pip_size: float = 0.1, n_buckets: int = 10):
        if payout_multiple <= 1.0:
            raise ValueError("payout multiple must exceed 1.0")
        self.stake = float(stake)
        self.payout_multiple = float(payout_multiple)
        self.break_even = required_probability(payout_multiple)
        self.pip_size = pip_size
        self.n_buckets = n_buckets

    # -- isolation guard (see module docstring, point 1) --------------------

    @staticmethod
    def assert_isolated(pipeline, live_pipeline=None) -> None:
        """Refuses to run against a pipeline the live bot is using.

        The failure this prevents is subtle and would be very hard to find:
        sharing a pipeline means sharing its TradeGate, so a simulated run
        would pour thousands of synthetic rejections into the live
        DeadlockMonitor and ShadowThresholdAnalyzer. The operator would then
        read a deadlock diagnosis describing a backtest.
        """
        if live_pipeline is not None and pipeline is live_pipeline:
            raise RuntimeError(
                "simulator was handed the live pipeline; build a fresh one "
                "from a factory so the live gate's diagnostics stay clean")
        gate = getattr(pipeline, "gate", None)
        if gate is not None and getattr(gate, "_live", False):
            raise RuntimeError("refusing to simulate through a gate marked live")

    # -- the run -------------------------------------------------------------

    def run(self, pipeline, digits: list[int], *, warmup: int = 2000,
            live_pipeline=None) -> SimulationResult:
        self.assert_isolated(pipeline, live_pipeline)

        risk, staking = _SimRisk(), _SimStaking(self.stake)
        n_eval = n_scored = trades = wins = losses = 0
        pnl = peak = max_dd = 0.0
        streak = max_streak = 0
        briers: list[float] = []
        lls: list[float] = []
        correct = 0
        reasons: Counter = Counter()
        zones: Counter = Counter()
        scores: list[float] = []
        outcomes: list[float] = []
        contrib: dict[str, list[float]] = defaultdict(list)
        edges: list[float] = []

        for i, digit in enumerate(digits):
            quote = f"100.{digit}"
            y = 1.0 if digit % 2 == 0 else 0.0

            if i >= warmup:
                n_eval += 1
                decision, request = pipeline.evaluate(
                    quote, self.pip_size, float(i), risk_manager=risk,
                    staking=staking, api_connected=True, db_healthy=True,
                    open_contract=False)
                pipeline.gate.deadlock.note_tick()

                if request is not None:
                    side, is_probe = request
                    proposal = Proposal(
                        contract_type=side, symbol=pipeline.symbol,
                        stake=self.stake,
                        payout=self.stake * self.payout_multiple,
                        ask_price=self.stake, currency="USD",
                        proposal_id=f"sim-{i}", received_at=time.time())
                    decision = pipeline.finalize_with_quote(
                        decision, proposal, risk_manager=risk, staking=staking,
                        side=side, is_probe=is_probe)

                # Prediction quality is measured on EVERY evaluated tick, not
                # only traded ones -- a selective bot produces too few trades
                # to estimate Brier from, and the calibrated probability is
                # emitted regardless of whether it is acted on.
                if decision.calibrated_p_even is not None:
                    p = min(max(decision.calibrated_p_even, 1e-9), 1 - 1e-9)
                    briers.append((p - y) ** 2)
                    lls.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
                    if (p >= 0.5) == (y == 1.0):
                        correct += 1

                if decision.opportunity_score is not None:
                    n_scored += 1
                    zones[decision.opportunity_zone or "?"] += 1
                    # The counterfactual outcome: WOULD this candidate have
                    # won, had it been taken? Recording it for rejected
                    # candidates too is what makes the score-vs-outcome
                    # correlation estimable at all -- restricting it to taken
                    # trades would condition on the very variable under test.
                    predicted_even = (decision.contract_type or "").upper() == "DIGITEVEN"
                    would_win = 1.0 if predicted_even == (y == 1.0) else 0.0
                    scores.append(decision.opportunity_score)
                    outcomes.append(would_win)
                    edges.append(decision.edge if decision.edge is not None else 0.0)
                    for name, q in (decision.contribution_detail or {}).items():
                        contrib[name].append(q)

                if decision.will_trade:
                    trades += 1
                    won = ((decision.contract_type or "").upper() == "DIGITEVEN") == (y == 1.0)
                    if won:
                        wins += 1
                        pnl += self.stake * (self.payout_multiple - 1)
                        streak = 0
                    else:
                        losses += 1
                        pnl -= self.stake
                        streak += 1
                        max_streak = max(max_streak, streak)
                    peak = max(peak, pnl)
                    max_dd = max(max_dd, peak - pnl)
                else:
                    reasons[decision.reason_code] += 1

            pipeline.learn(quote, self.pip_size, float(i))

        # --- score calibration (the point of this harness) -------------------
        buckets: list[ScoreBucket] = []
        width = 100.0 / self.n_buckets
        for b in range(self.n_buckets):
            buckets.append(ScoreBucket(low=b * width, high=(b + 1) * width))
        for s, o, e in zip(scores, outcomes, edges):
            idx = min(int(s / width), self.n_buckets - 1)
            bk = buckets[idx]
            bk.n += 1
            bk.wins += int(o)
            bk.sum_edge += e

        corr = {}
        for name, vals in contrib.items():
            if len(vals) < 3:
                continue
            c = _pearson(vals, outcomes[-len(vals):])
            # A contributor that never varied has no correlation to report.
            # Showing it as nan pushes a real signal off the top of the list.
            if c == c:
                corr[name] = c

        nb = len(briers)
        return SimulationResult(
            n_ticks=len(digits), n_evaluated=n_eval, n_scored=n_scored,
            n_trades=trades, wins=wins, losses=losses, pnl=pnl,
            stake=self.stake, payout_multiple=self.payout_multiple,
            assumed_break_even=self.break_even,
            brier=sum(briers) / nb if nb else float("nan"),
            log_loss=sum(lls) / nb if nb else float("nan"),
            accuracy=correct / nb if nb else float("nan"),
            max_losing_streak=max_streak, max_drawdown=max_dd,
            reason_counts=dict(reasons), zone_counts=dict(zones),
            buckets=buckets,
            score_outcome_correlation=_pearson(scores, outcomes),
            contributor_correlation=corr,
            mean_score=sum(scores) / len(scores) if scores else float("nan"),
        )


# ---------------------------------------------------------------------------
# walk-forward (Section 50)
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardReport:
    blocks: list[SimulationResult] = field(default_factory=list)
    payout_multiple: float = 1.95

    @property
    def total_trades(self) -> int:
        return sum(b.n_trades for b in self.blocks)

    @property
    def total_pnl(self) -> float:
        return sum(b.pnl for b in self.blocks)

    @property
    def pooled_win_rate(self) -> float:
        n = self.total_trades
        return sum(b.wins for b in self.blocks) / n if n else float("nan")

    def _corrs(self) -> list[float]:
        return [b.score_outcome_correlation for b in self.blocks
                if b.score_outcome_correlation == b.score_outcome_correlation]

    def stability(self) -> float:
        """Std dev of the per-block score/outcome correlation."""
        vals = self._corrs()
        if len(vals) < 2:
            return float("nan")
        m = sum(vals) / len(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

    #: Below this, a correlation is not worth acting on whatever its sign.
    #: Explaining 0.25% of the variance in outcome is not a finding.
    MIN_USEFUL_CORRELATION = 0.05

    def sign_consistent(self) -> bool:
        """Does the score/outcome relationship reproduce across blocks?

        TWO CONDITIONS, AND THE SECOND IS NOT OPTIONAL. The sign must hold
        across every block, AND every block's magnitude must clear
        MIN_USEFUL_CORRELATION.

        The magnitude floor is here because sign alone is a coin flip. With
        three blocks, a pure-noise series lands on a single sign 25% of the
        time; with four, 12.5%. Reporting that as "consistent" would tell the
        operator that re-weighting is defensible roughly one run in four on a
        stream with no signal in it at all -- the precise failure this harness
        exists to prevent. Observed on CSPRNG data: correlations of -0.001,
        -0.009, -0.021, all negative, all meaningless.

        Standard deviation is not the test either. A tiny sd around a sign
        that flips is low-variance noise, and a tiny sd around +0.002 is
        low-variance nothing.
        """
        vals = self._corrs()
        if len(vals) < 2:
            return False
        same_sign = all(v > 0 for v in vals) or all(v < 0 for v in vals)
        strong_enough = all(abs(v) >= self.MIN_USEFUL_CORRELATION for v in vals)
        return same_sign and strong_enough

    def report(self) -> str:
        L = ["#" * 72,
             f"WALK-FORWARD  {len(self.blocks)} blocks  "
             f"assumed payout {self.payout_multiple:.3f}x",
             "#" * 72]
        for i, b in enumerate(self.blocks, 1):
            L.append(f"\n--- block {i} ---")
            L.append(b.report())
        L.append("")
        L.append(f"POOLED  trades={self.total_trades}  "
                 f"win rate={self.pooled_win_rate:.4f}  pnl={self.total_pnl:+.2f}")
        corrs = [b.score_outcome_correlation for b in self.blocks]
        L.append(f"score/outcome correlation per block: "
                 + ", ".join(f"{c:+.4f}" for c in corrs))
        sd = self.stability()
        L.append(f"correlation stability (sd across blocks): {sd:.4f}")
        if not self.sign_consistent():
            vals = self._corrs()
            weak = vals and all(abs(v) < self.MIN_USEFUL_CORRELATION for v in vals)
            why = ("every block is below |{:.2f}|, which is nothing to tune "
                   "towards even where the sign happens to agree"
                   .format(self.MIN_USEFUL_CORRELATION) if weak
                   else "the sign does not survive from one block to the next")
            L.append(f"  -> NO REPRODUCIBLE SCORE/OUTCOME RELATIONSHIP: {why}. "
                     f"Do NOT tune weights on this: any constant fitted to one "
                     f"block is fitted to that block's noise.")
        elif sd == sd and sd > 0.05:
            L.append("  -> sign is consistent but the magnitude is not. Treat "
                     "as weak evidence and re-run on more blocks before acting.")
        else:
            L.append("  -> sign is consistent across blocks. This is the only "
                     "condition under which re-weighting is defensible, and it "
                     "still needs a fresh out-of-sample block to confirm.")
        L.append("#" * 72)
        return "\n".join(L)


def walk_forward(pipeline_factory, digits: list[int], *, n_blocks: int = 5,
                 stake: float = 1.0, payout_multiple: float = 1.95,
                 warmup: int = 0) -> WalkForwardReport:
    """Section 50: train on everything before a block, evaluate ON the block,
    advance. Never evaluates on data the models have already seen.

    A FRESH PIPELINE PER BLOCK, from the factory. Reusing one would leak the
    later blocks' statistics backwards through the models' own accumulators,
    which is the same class of error the leakage canary looks for live.
    """
    report = WalkForwardReport(payout_multiple=payout_multiple)
    block = len(digits) // n_blocks
    if block < 200:
        raise ValueError(
            f"{len(digits)} digits over {n_blocks} blocks gives {block} per "
            f"block; too few to estimate anything. Use more data or fewer blocks.")
    for b in range(1, n_blocks):
        train = digits[: b * block]
        test = digits[b * block: (b + 1) * block]
        if len(test) < 200:
            break
        pipe = pipeline_factory()
        for i, d in enumerate(train):
            pipe.learn(f"100.{d}", 0.1, float(i))
        sim = Simulator(stake=stake, payout_multiple=payout_multiple)
        report.blocks.append(sim.run(pipe, test, warmup=warmup))
    return report


# ---------------------------------------------------------------------------
# ablation (Section 51)
# ---------------------------------------------------------------------------

def ablation(pipeline_factory, digits: list[int], model_names: list[str],
             *, payout_multiple: float = 1.95, warmup: int = 2000) -> dict:
    """Does each model actually help?

    Re-runs with each model removed and reports the delta in Brier score. A
    component whose removal does not worsen prediction is complexity without
    benefit and should be deleted. Note the sign convention: Brier is a loss,
    so a POSITIVE delta (ablated worse than full) means the model earned its
    place.
    """
    from app.ensemble.ensemble import AdaptiveEnsemble

    sim = Simulator(payout_multiple=payout_multiple)
    full = sim.run(pipeline_factory(), digits, warmup=warmup)
    out = {"__full__": full}
    for name in model_names:
        pipe = pipeline_factory()
        remaining = [m for m in pipe.models if m.name != name]
        if len(remaining) == len(pipe.models) or not remaining:
            continue
        pipe.models = remaining
        pipe._feature_consumers = [m for m in remaining if hasattr(m, "set_features")]
        pipe.ensemble = AdaptiveEnsemble(remaining)
        out[name] = sim.run(pipe, digits, warmup=warmup)
    return out


def ablation_report(results: dict) -> str:
    full = results.get("__full__")
    if full is None:
        return "no full-system baseline"
    L = ["ABLATION (Brier is a loss: positive delta = the model earned its place)",
         f"  full system brier {full.brier:.6f}",
         f"  {'removed model':<34}{'brier':>10}{'delta':>10}"]
    for name, r in sorted(results.items()):
        if name == "__full__":
            continue
        L.append(f"  {name:<34}{r.brier:>10.6f}{r.brier - full.brier:>+10.6f}")
    return "\n".join(L)
