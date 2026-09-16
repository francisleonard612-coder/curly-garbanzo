"""
Automated pattern discovery (spec Sections 41, 42, 40).

Enumerates candidate contexts (last k digits, last k parities, run state,
frequency-imbalance state) and estimates P(EVEN | context) for each.

THIS MODULE IS THE MOST DANGEROUS ONE IN THE REPO, and is written
defensively for that reason. Enumerating last-1 through last-4 digit
contexts alone gives 10 + 100 + 1,000 + 10,000 = 11,110 hypotheses. Testing
all of them at alpha=0.05 yields roughly 555 "significant" patterns from
pure noise, every single time, with beautiful confidence intervals. A bot
that trades the best of them is trading its own multiple-comparisons error
and will lose money with total statistical conviction.

FOUR MANDATORY DEFENCES, all on by default:

1. BENJAMINI-HOCHBERG FDR CONTROL across the entire candidate set at once
   (Section 40). Not per-family, not per-context-length -- the whole scan,
   because that is what was actually searched.
2. MINIMUM SUPPORT. A context with fewer than `min_support` observations is
   not tested at all. Rare contexts produce the most extreme lifts and the
   least reliable ones.
3. OUT-OF-SAMPLE CONFIRMATION. Every surviving pattern is re-estimated on a
   held-out later segment it was not discovered on. Patterns are ranked by
   OUT-OF-SAMPLE lift, never in-sample. This is the defence that actually
   works when the others are fooled -- a noise pattern has, by construction,
   no reason to persist into data it did not come from.
4. ECONOMIC SCREENING. A surviving pattern must clear the contract's
   break-even probability, not merely 0.5 (Section 25).

A pattern that passes all four on a CSPRNG stream should occur at
approximately the FDR rate and fail on the next re-scan. Persistence across
independent re-scans, not the strength of any single finding, is the only
evidence worth acting on.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.statistics.distribution import (
    benjamini_hochberg,
    norm_sf,
    wilson_interval,
)


@dataclass(frozen=True)
class Pattern:
    kind: str                 # "digits" | "parity" | "run"
    context: tuple
    support: int
    p_even: float
    baseline: float
    lift: float
    p_value: float
    ci_lower: float
    ci_upper: float
    oos_support: int = 0
    oos_p_even: float = float("nan")
    oos_lift: float = float("nan")
    survives_fdr: bool = False
    confirmed_out_of_sample: bool = False
    economically_relevant: bool = False

    @property
    def actionable(self) -> bool:
        return (self.survives_fdr and self.confirmed_out_of_sample
                and self.economically_relevant)

    def describe(self) -> str:
        return (f"{self.kind}{self.context}: n={self.support} "
                f"P(EVEN)={self.p_even:.4f} (lift {self.lift:+.4f}, p={self.p_value:.2e}) "
                f"| OOS n={self.oos_support} P={self.oos_p_even:.4f} "
                f"(lift {self.oos_lift:+.4f}) | actionable={self.actionable}")


def _contexts(digits: list[int], k: int, kind: str):
    """Yields (context, next_parity) pairs."""
    if kind == "digits":
        seq = digits
    else:
        seq = [d % 2 for d in digits]
    for i in range(k, len(seq)):
        yield tuple(seq[i - k:i]), digits[i] % 2


class PatternDiscovery:
    def __init__(self, *, max_digit_context: int = 3, max_parity_context: int = 6,
                 min_support: int = 200, fdr: float = 0.05,
                 oos_fraction: float = 0.3, break_even: float = 0.5128,
                 required_margin: float = 0.005):
        self.max_digit_context = max_digit_context
        self.max_parity_context = max_parity_context
        self.min_support = min_support
        self.fdr = fdr
        self.oos_fraction = oos_fraction
        self.break_even = break_even
        self.required_margin = required_margin

    def scan(self, digits: list[int]) -> list[Pattern]:
        n = len(digits)
        if n < self.min_support * 4:
            return []
        split = int(n * (1 - self.oos_fraction))
        in_sample, oos = digits[:split], digits[split:]
        baseline = sum(1 for d in in_sample if d % 2 == 0) / len(in_sample)

        candidates: list[Pattern] = []
        for kind, max_k in (("digits", self.max_digit_context),
                            ("parity", self.max_parity_context)):
            for k in range(1, max_k + 1):
                tally: dict[tuple, list[int]] = {}
                for ctx, par in _contexts(in_sample, k, kind):
                    row = tally.setdefault(ctx, [0, 0])
                    row[0 if par == 0 else 1] += 1
                for ctx, (ev, od) in tally.items():
                    support = ev + od
                    if support < self.min_support:
                        continue
                    p = ev / support
                    se = (baseline * (1 - baseline) / support) ** 0.5
                    z = (p - baseline) / se if se > 0 else 0.0
                    ci = wilson_interval(ev, support)
                    candidates.append(Pattern(
                        kind=kind, context=ctx, support=support, p_even=p,
                        baseline=baseline, lift=p - baseline,
                        p_value=2.0 * norm_sf(abs(z)),
                        ci_lower=ci.lower, ci_upper=ci.upper))

        if not candidates:
            return []

        keep = benjamini_hochberg([c.p_value for c in candidates], self.fdr)
        survivors = [c for c, k in zip(candidates, keep) if k]
        if not survivors:
            return []

        confirmed: list[Pattern] = []
        oos_baseline = (sum(1 for d in oos if d % 2 == 0) / len(oos)) if oos else 0.5
        for pat in survivors:
            k = len(pat.context)
            ev = od = 0
            for ctx, par in _contexts(oos, k, pat.kind):
                if ctx == pat.context:
                    if par == 0:
                        ev += 1
                    else:
                        od += 1
            sup = ev + od
            oos_p = ev / sup if sup else float("nan")
            oos_lift = (oos_p - oos_baseline) if sup else float("nan")
            # Confirmation requires the OOS effect to point the SAME WAY and
            # retain at least half the magnitude. Sign agreement alone would
            # be satisfied by chance half the time.
            confirmed_flag = bool(
                sup >= max(50, self.min_support // 4)
                and oos_lift == oos_lift
                and (oos_lift > 0) == (pat.lift > 0)
                and abs(oos_lift) >= abs(pat.lift) * 0.5)
            side_p = oos_p if oos_p == oos_p else pat.p_even
            best = max(side_p, 1 - side_p)
            econ = best > self.break_even + self.required_margin
            confirmed.append(Pattern(
                **{**pat.__dict__,
                   "oos_support": sup, "oos_p_even": oos_p, "oos_lift": oos_lift,
                   "survives_fdr": True, "confirmed_out_of_sample": confirmed_flag,
                   "economically_relevant": econ}))

        confirmed.sort(key=lambda p: (p.actionable, abs(p.oos_lift) if p.oos_lift == p.oos_lift else 0),
                       reverse=True)
        return confirmed

    def actionable(self, digits: list[int]) -> list[Pattern]:
        return [p for p in self.scan(digits) if p.actionable]
