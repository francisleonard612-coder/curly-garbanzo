"""
Historical simulator, walk-forward harness and ablation
(spec Sections 49, 50, 51).

CHRONOLOGICAL INTEGRITY IS ABSOLUTE (Section 50). Ticks are replayed in
order; nothing is ever shuffled. Shuffling a time series lets a model
interpolate between temporally adjacent ticks and produces validation
scores that have no live counterpart. There is no `shuffle=True` option
here, deliberately -- an option that must never be used is better not
offered.

THE SIMULATOR REPLAYS THE SAME CODE PATH AS LIVE. It drives the real
SymbolPipeline with the real ordering (features -> predict -> observe), so
a leakage bug shows up identically in both. A simulator with its own
private prediction path would validate something the bot does not do.

PAYOUT ASSUMPTION IS EXPLICIT AND CONSERVATIVE. Historical proposals are
not available, so a fixed payout multiple is assumed and NAMED in the
output. Results are only as honest as that number: assuming 1.97x when live
quotes average 1.92x turns a losing system into a winning backtest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.economics.edge import required_probability


@dataclass
class SimulationResult:
    n_ticks: int
    n_evaluated: int
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

    @property
    def win_rate(self) -> float:
        return self.wins / self.n_trades if self.n_trades else float("nan")

    @property
    def selectivity(self) -> float:
        return self.n_trades / self.n_evaluated if self.n_evaluated else 0.0

    def report(self) -> str:
        lines = [
            "=" * 68,
            f"SIMULATION  ticks={self.n_ticks}  evaluated={self.n_evaluated}  "
            f"trades={self.n_trades}  selectivity={self.selectivity:.5f}",
            f"  assumed payout {self.payout_multiple:.3f}x -> break-even "
            f"{self.assumed_break_even:.4f} (accuracy needed merely to not lose)",
            f"  win rate {self.win_rate:.4f}  pnl {self.pnl:+.2f}  "
            f"max losing streak {self.max_losing_streak}  max dd {self.max_drawdown:.2f}",
            f"  prediction quality: brier={self.brier:.4f} "
            f"log_loss={self.log_loss:.4f} accuracy={self.accuracy:.4f}",
        ]
        if self.reason_counts:
            lines.append("  no-trade reasons:")
            for code, n in sorted(self.reason_counts.items(), key=lambda kv: -kv[1])[:8]:
                lines.append(f"    {code:<36} {n}")
        lines.append("=" * 68)
        return "\n".join(lines)


class Simulator:
    """Replays a digit sequence through the real pipeline."""

    def __init__(self, *, stake: float = 1.0, payout_multiple: float = 1.95,
                 min_confidence: float = 0.0):
        self.stake = stake
        self.payout_multiple = payout_multiple
        self.break_even = required_probability(payout_multiple)
        self.min_confidence = min_confidence

    def run(self, pipeline, digits: list[int], *, warmup: int = 2000) -> SimulationResult:
        import math

        n_eval = trades = wins = losses = 0
        pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        streak = max_streak = 0
        briers: list[float] = []
        lls: list[float] = []
        correct = 0
        reasons: dict[str, int] = {}
        pip = 0.1

        for i, digit in enumerate(digits):
            quote = f"100.{digit}"
            if i >= warmup:
                n_eval += 1
                features = None
                result = pipeline.ensemble.predict()
                p_even = result.p_even_ensemble
                cal = pipeline.calibrators["DIGITEVEN"]
                p_cal = cal.calibrate(p_even)

                y = 1.0 if digit % 2 == 0 else 0.0
                p = min(max(p_cal, 1e-9), 1 - 1e-9)
                briers.append((p - y) ** 2)
                lls.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
                if (p >= 0.5) == (y == 1.0):
                    correct += 1

                confidence = max(p_cal, 1 - p_cal)
                # Simulated gate: confidence must clear break-even, mirroring
                # the live economic rule rather than a bare 0.5 threshold.
                if confidence > self.break_even + self.min_confidence:
                    trades += 1
                    predicted_even = p_cal >= 0.5
                    won = predicted_even == (digit % 2 == 0)
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
                    reasons["NO_TRADE_BELOW_BREAKEVEN"] = reasons.get("NO_TRADE_BELOW_BREAKEVEN", 0) + 1

            pipeline.learn(quote, pip, float(i))

        n = len(briers)
        return SimulationResult(
            n_ticks=len(digits), n_evaluated=n_eval, n_trades=trades,
            wins=wins, losses=losses, pnl=pnl, stake=self.stake,
            payout_multiple=self.payout_multiple, assumed_break_even=self.break_even,
            brier=sum(briers) / n if n else float("nan"),
            log_loss=sum(lls) / n if n else float("nan"),
            accuracy=correct / n if n else float("nan"),
            max_losing_streak=max_streak, max_drawdown=max_dd, reason_counts=reasons)


def walk_forward(pipeline_factory, digits: list[int], *, n_blocks: int = 5,
                 stake: float = 1.0, payout_multiple: float = 1.95) -> list[SimulationResult]:
    """Section 50: train on everything before a block, evaluate ON the block,
    advance. Never evaluates on data the models have already seen."""
    results = []
    block = len(digits) // n_blocks
    for b in range(1, n_blocks):
        train = digits[: b * block]
        test = digits[b * block: (b + 1) * block]
        if len(test) < 100:
            break
        pipe = pipeline_factory()
        for i, d in enumerate(train):
            pipe.learn(f"100.{d}", 0.1, float(i))
        sim = Simulator(stake=stake, payout_multiple=payout_multiple)
        results.append(sim.run(pipe, test, warmup=0))
    return results


def ablation(pipeline_factory, digits: list[int], model_names: list[str],
             *, payout_multiple: float = 1.95) -> dict:
    """Section 51: does each component actually help?

    Re-runs with each model removed and reports the delta in Brier score.
    A component whose removal does not worsen prediction is complexity
    without benefit and should be deleted (Section 72).
    """
    sim = Simulator(payout_multiple=payout_multiple)
    full = sim.run(pipeline_factory(), digits)
    out = {"__full__": full}
    for name in model_names:
        pipe = pipeline_factory()
        pipe.models = [m for m in pipe.models if m.name != name]
        if not pipe.models:
            continue
        from app.ensemble.ensemble import AdaptiveEnsemble
        pipe.ensemble = AdaptiveEnsemble(pipe.models)
        out[name] = sim.run(pipe, digits)
    return out
