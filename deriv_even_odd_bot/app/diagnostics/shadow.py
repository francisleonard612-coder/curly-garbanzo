"""
Shadow threshold analysis (spec Section 17).

Counts, continuously, how many candidates WOULD have qualified under nearby
thresholds. This turns "the bot isn't trading" into "at 62 it takes 0%, at 50
it would take 0.4%, and those 0.4% have a mean edge of -0.009" -- which is an
answer, and specifically an answer that tells you not to move the threshold.

TWO RULES, BOTH STRUCTURAL.

1. IT NEVER INFLUENCES A DECISION. `record()` is called after the decision is
   final and returns None. No other module reads this one; nothing here is
   importable into a code path that trades. Section 17 says the analysis "must
   NOT influence the already-made decision retrospectively", and the only
   reliable way to guarantee that is to keep the data flowing one way.

2. IT COUNTS SCORES, NOT PROFITS. A shadow counter that reported hypothetical
   P/L would be backtesting on the live decision stream, with no settlement and
   no slippage, and it would be wrong in the optimistic direction. It reports
   how many candidates clear each threshold and what their economics looked
   like. What those trades would have RETURNED is a question for the
   walk-forward simulator, which resolves outcomes properly.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class ShadowBucket:
    threshold: float
    qualifying: int = 0
    qualifying_positive_ev: int = 0
    sum_edge: float = 0.0
    sum_ev: float = 0.0

    @property
    def mean_edge(self) -> float | None:
        return self.sum_edge / self.qualifying if self.qualifying else None

    @property
    def mean_ev(self) -> float | None:
        return self.sum_ev / self.qualifying if self.qualifying else None


@dataclass
class ShadowReport:
    evaluated: int
    buckets: list[ShadowBucket] = field(default_factory=list)
    current_threshold: float = 0.0
    dominant_deficit: str = ""
    dominant_share: float = 0.0
    summary: str = ""


class ShadowThresholdAnalyzer:
    def __init__(self, *, offsets: tuple[float, ...] = (-20.0, -10.0, -5.0,
                                                        0.0, 5.0, 10.0),
                 history: int = 5000):
        self.offsets = offsets
        self._evaluated = 0
        self._deficits: deque[str] = deque(maxlen=history)
        self._rows: deque[tuple[float, float, float, float]] = deque(maxlen=history)

    def record(self, assessment, edge_assessment=None) -> None:
        """Log one scored candidate. Returns nothing, by design."""
        self._evaluated += 1
        if assessment.limiting_factor:
            self._deficits.append(assessment.limiting_factor)
        edge = float(edge_assessment.point_edge) if edge_assessment else 0.0
        ev = float(edge_assessment.expected_value) if edge_assessment else 0.0
        self._rows.append((assessment.score, assessment.threshold, edge, ev))

    def report(self) -> ShadowReport:
        rows = list(self._rows)
        current = rows[-1][1] if rows else 0.0
        buckets: list[ShadowBucket] = []
        for off in self.offsets:
            t = max(0.0, min(100.0, current + off))
            b = ShadowBucket(threshold=t)
            for score, _thr, edge, ev in rows:
                if score >= t:
                    b.qualifying += 1
                    b.sum_edge += edge
                    b.sum_ev += ev
                    if ev > 0:
                        b.qualifying_positive_ev += 1
            buckets.append(b)

        dominant, share = "", 0.0
        if self._deficits:
            from collections import Counter
            dominant, count = Counter(self._deficits).most_common(1)[0]
            share = count / len(self._deficits)

        return ShadowReport(
            evaluated=self._evaluated, buckets=buckets,
            current_threshold=current, dominant_deficit=dominant,
            dominant_share=share, summary=self._summary(rows, buckets, dominant, share),
        )

    def _summary(self, rows, buckets, dominant, share) -> str:
        n = len(rows)
        if not n:
            return "No candidates scored yet."
        at_current = next((b for b in buckets if abs(b.threshold - rows[-1][1]) < 1e-9), None)
        taken = at_current.qualifying if at_current else 0
        rejected_pct = 100.0 * (1 - taken / n)
        line = (f"Current configuration is rejecting {rejected_pct:.0f}% of "
                f"{n} candidate opportunities")
        if dominant:
            line += f", primarily because {dominant} is the largest deficit " \
                    f"in {100 * share:.0f}% of them"
        line += "."

        # The number that actually matters: does ANY nearby threshold admit a
        # positive-EV candidate? If not, the threshold is not the constraint.
        best = max(buckets, key=lambda b: b.qualifying_positive_ev)
        if best.qualifying_positive_ev == 0:
            line += (" No threshold in the scanned range admits a single "
                     "positive-EV candidate, so the cutoff is not what is "
                     "holding the bot back -- the contract economics are.")
        else:
            line += (f" At a threshold of {best.threshold:.0f}, "
                     f"{best.qualifying_positive_ev} of {n} candidates would "
                     f"have had positive expected value "
                     f"(mean edge {best.mean_edge:+.4f}).")
        return line
