"""
Anti-deadlock monitoring (spec Section 16).

THE PROBLEM THIS SOLVES IS OBSERVABILITY, NOT INACTIVITY. A bot that has not
traded in six hours is in one of two completely different situations:

  (a) it is correctly declining, because nothing it has seen has positive
      expected value at the quoted payout; or
  (b) it is broken -- a proposal stream that stopped, a calibrator stuck
      unfitted, a risk limit latched on, a threshold nobody meant to set.

From the outside these are indistinguishable, and that ambiguity is what makes
operators reach for the threshold constant. This module removes the ambiguity
by attributing every rejection to a cause and reporting the distribution.

IT DOES NOT ADJUST ANYTHING. Section 16 is explicit: "DO NOT blindly lower
thresholds. Instead diagnose the cause." There is no code path here that writes
to a threshold, and there should never be one. `diagnose()` returns text.

READ THE OUTPUT HONESTLY. If the report says 100% of rejections are
NO_TRADE_NEGATIVE_EV with a mean point edge of -0.0128 and a mean payout of
1.95x, the bot is not deadlocked and no configuration change will help: -0.0128
is the house edge, and a threshold cannot be lowered past arithmetic. The
correct responses to that report are a better payout, a different instrument,
or not trading -- never a smaller number in config.yaml.
"""
from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field


@dataclass
class RejectionRecord:
    timestamp: float
    reason_code: str
    zone: str
    opportunity_score: float
    point_edge: float | None = None
    expected_value: float | None = None
    payout_multiple: float | None = None
    limiting_factor: str = ""
    hard: bool = False


@dataclass
class DeadlockReport:
    idle_seconds: float
    idle_ticks: int
    alerting: bool
    candidates: int
    rejected: int
    hard_rejected: int
    soft_rejected: int
    reason_counts: dict[str, int] = field(default_factory=dict)
    limiting_counts: dict[str, int] = field(default_factory=dict)
    mean_score: float = 0.0
    mean_edge: float | None = None
    mean_ev: float | None = None
    mean_payout: float | None = None
    best_score: float = 0.0
    diagnosis: str = ""
    verdict: str = ""

    def as_lines(self) -> list[str]:
        out = [self.diagnosis]
        if self.verdict:
            out.append(self.verdict)
        return out


class DeadlockMonitor:
    """Tracks decision flow and explains inactivity."""

    def __init__(self, *, idle_alert_seconds: float = 900.0,
                 idle_alert_ticks: int = 2000, history: int = 5000):
        self.idle_alert_seconds = float(idle_alert_seconds)
        self.idle_alert_ticks = int(idle_alert_ticks)
        self._records: deque[RejectionRecord] = deque(maxlen=history)
        self._last_trade_at: float = time.time()
        self._ticks_since_trade: int = 0
        self._candidates: int = 0
        self._trades: int = 0

    # -- input ---------------------------------------------------------------

    def note_tick(self) -> None:
        self._ticks_since_trade += 1

    def note_candidate(self) -> None:
        """A candidate is an evaluation that got far enough to be scored."""
        self._candidates += 1

    def note_rejection(self, *, reason_code: str, zone: str = "",
                       opportunity_score: float = 0.0, hard: bool = False,
                       edge_assessment=None, limiting_factor: str = "",
                       now: float | None = None) -> None:
        now = time.time() if now is None else now
        rec = RejectionRecord(
            timestamp=now, reason_code=reason_code, zone=zone,
            opportunity_score=float(opportunity_score),
            limiting_factor=limiting_factor, hard=bool(hard))
        if edge_assessment is not None:
            rec.point_edge = float(edge_assessment.point_edge)
            rec.expected_value = float(edge_assessment.expected_value)
            rec.payout_multiple = float(edge_assessment.proposal.payout_multiple)
        self._records.append(rec)

    def note_trade(self, now: float | None = None) -> None:
        self._last_trade_at = time.time() if now is None else now
        self._ticks_since_trade = 0
        self._trades += 1

    # -- output --------------------------------------------------------------

    @property
    def idle_seconds(self) -> float:
        return time.time() - self._last_trade_at

    @property
    def is_alerting(self) -> bool:
        return (self.idle_seconds >= self.idle_alert_seconds
                and self._ticks_since_trade >= self.idle_alert_ticks)

    def report(self) -> DeadlockReport:
        recs = list(self._records)
        n = len(recs)
        reason_counts = Counter(r.reason_code for r in recs)
        limiting_counts = Counter(r.limiting_factor for r in recs if r.limiting_factor)
        hard_n = sum(1 for r in recs if r.hard)

        def _mean(vals):
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        rep = DeadlockReport(
            idle_seconds=self.idle_seconds,
            idle_ticks=self._ticks_since_trade,
            alerting=self.is_alerting,
            candidates=self._candidates,
            rejected=n,
            hard_rejected=hard_n,
            soft_rejected=n - hard_n,
            reason_counts=dict(reason_counts),
            limiting_counts=dict(limiting_counts),
            mean_score=(sum(r.opportunity_score for r in recs) / n) if n else 0.0,
            mean_edge=_mean([r.point_edge for r in recs]),
            mean_ev=_mean([r.expected_value for r in recs]),
            mean_payout=_mean([r.payout_multiple for r in recs]),
            best_score=max((r.opportunity_score for r in recs), default=0.0),
        )
        rep.diagnosis, rep.verdict = self._diagnose(rep, reason_counts, n)
        return rep

    def _diagnose(self, rep: DeadlockReport, counts: Counter, n: int
                  ) -> tuple[str, str]:
        if self._trades > 0 and not rep.alerting:
            return (f"Active: {self._trades} trades, last "
                    f"{rep.idle_seconds / 60:.0f} min ago.", "")
        if n == 0:
            return ("No candidates have been evaluated yet.",
                    "If this persists past the warm-up window, check the tick "
                    "subscription and the minimum-sample setting.")

        top, top_n = counts.most_common(1)[0]
        pct = 100.0 * top_n / n
        head = (f"{pct:.0f}% of {n} rejections are {top}; mean opportunity "
                f"score {rep.mean_score:.1f}, best {rep.best_score:.1f}.")

        verdict = _VERDICTS.get(top, "")
        if top == "NO_TRADE_NEGATIVE_EV" and rep.mean_edge is not None:
            verdict = (
                f"Mean point edge {rep.mean_edge:+.4f} at mean payout "
                f"{rep.mean_payout:.3f}x. The break-even probability at that "
                f"payout is {1.0 / rep.mean_payout:.4f}; the models are "
                f"averaging {1.0 / rep.mean_payout + rep.mean_edge:.4f}. "
                f"This is a pricing gap, not a configuration problem -- "
                f"lowering a threshold cannot make a negative-EV contract "
                f"profitable. Change the payout, the instrument, or stop.")
        return (head, verdict)


_VERDICTS = {
    "NO_TRADE_MINIMUM_DATA":
        "Still warming up. Nothing to do but wait for ticks.",
    "NO_TRADE_CALIBRATION_UNFITTED":
        "The calibrator has not reached its minimum sample. If this persists "
        "well past warm-up, check that learn() is being called with realized "
        "digits -- an unfitted calibrator after thousands of ticks means the "
        "outcome path is broken, not that the data is unusual.",
    "NO_TRADE_BAD_PROPOSAL":
        "Proposals are not arriving or are malformed. This is an API problem, "
        "not a market one -- check the subscription and the contract "
        "parameters before touching anything predictive.",
    "NO_TRADE_STALE_PROPOSAL":
        "Quotes are arriving but ageing out before execution. Look at loop "
        "latency and the proposal refresh interval.",
    "NO_TRADE_RISK":
        "A risk limit is doing its job. Confirm which one, and whether it "
        "latched on a past loss streak that has since ended.",
    "NO_TRADE_LOW_EDGE":
        "Candidates are economically thin rather than structurally blocked. "
        "Check the mean edge below: if it is negative, the models are not "
        "finding anything and no threshold change is appropriate.",
    "NO_TRADE_HIGH_UNCERTAINTY":
        "Probability intervals are too wide to act on. This is usually a "
        "sample-size problem in the cells the prediction rests on.",
    "NO_TRADE_MODEL_DISAGREEMENT":
        "Members are pointing in different directions, which at these "
        "magnitudes usually means they are each fitting noise.",
    "NO_TRADE_MODEL_DEGRADED":
        "Model health has fallen. Check the per-model Brier skill: values at "
        "or below zero mean the members are performing at chance.",
    "NO_TRADE_CALIBRATION_DEGRADED":
        "Calibration reliability has dropped. The probabilities are not "
        "trustworthy enough to price against until it recovers.",
    "NO_TRADE_RANDOM_STREAM_EVIDENCE":
        "The randomness battery is finding no exploitable departure. Note "
        "this is now evidence rather than a veto -- it is lowering scores, "
        "not blocking trades. If it dominates rejections, the honest reading "
        "is that the stream is behaving like a fair process.",
    "NO_TRADE_LOW_OPPORTUNITY":
        "No single dominant cause; scores are broadly mediocre. Check the "
        "shadow threshold report for what a different cutoff would admit.",
    "NO_TRADE_RESEARCH_MODE":
        "Research mode. The pipeline reached a trade decision and stopped "
        "before BUY, as configured.",
}
