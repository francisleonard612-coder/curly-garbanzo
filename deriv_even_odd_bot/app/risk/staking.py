"""
Staking engine (spec Section 32).

SEPARATE MODULE FROM PREDICTION, BY MANDATE. Sizing never feeds back into
whether to trade.

TWO INCIDENTS FROM THIS ACCOUNT ARE ENCODED HERE AS HARD RULES:

1. BASE STAKE IS NEVER BALANCE-SCALED AT THE PREDICTION LAYER. A sibling
   bot had a balance-scaled base stake AND an independent risk allocator
   that also scaled with balance; the two disagreed by 527x and were caught
   only by a hard ceiling. Percentage and Kelly sizing here take balance as
   an explicit argument and are always clamped by max_stake, which is owned
   by the risk manager, not by this module.

2. MARTINGALE IS NOT AN EDGE (Section 31, and a documented live loss). It
   is available for completeness, OFF by default, and refuses to engage
   unless explicitly enabled. Live evidence from a sibling Rise/Fall bot:
   at negative expectancy, progression converts a slow bleed into a fast
   one. A 12-loss streak was observed there in ordinary operation -- at
   factor 2.0 that is a 4,096x terminal stake.

KELLY IS ALWAYS FRACTIONAL AND ALWAYS CAPPED (Section 32: "never allow
unrestricted Kelly"). Full Kelly assumes the probability estimate is exact.
Ours is not -- it comes with a confidence interval wide enough that full
Kelly on the point estimate would routinely bet many times the optimal
amount. We size from the interval's LOWER bound and then take a fraction
of that.
"""
from __future__ import annotations

from dataclasses import dataclass

FIXED = "fixed"
PERCENTAGE = "percentage"
KELLY = "kelly"
MARTINGALE = "martingale"


@dataclass
class StakingEngine:
    method: str = FIXED
    base_stake: float = 1.0
    max_stake: float = 5.0
    percentage: float = 0.01
    kelly_fraction: float = 0.25
    kelly_cap: float = 0.05           # never stake >5% of balance, whatever Kelly says
    martingale_enabled: bool = False
    martingale_factor: float = 2.0
    martingale_max_steps: int = 3
    min_consecutive_losses: int = 2

    _consecutive_losses: int = 0

    def stake_for(self, *, balance: float | None = None,
                  probability_lower_bound: float | None = None,
                  payout_multiple: float | None = None) -> tuple[float, str]:
        if self.method == FIXED:
            stake, why = self.base_stake, "fixed base stake"
        elif self.method == PERCENTAGE:
            if balance is None:
                return 0.0, "percentage staking requires a verified balance"
            stake = balance * self.percentage
            why = f"{self.percentage:.2%} of balance {balance:.2f}"
        elif self.method == KELLY:
            if balance is None or probability_lower_bound is None or payout_multiple is None:
                return 0.0, "Kelly staking requires balance, probability bound and payout"
            b = payout_multiple - 1.0
            if b <= 0:
                return 0.0, "non-positive net odds"
            p = probability_lower_bound
            full = (p * b - (1 - p)) / b
            if full <= 0:
                return 0.0, f"Kelly non-positive at lower bound p={p:.4f}"
            frac = min(full * self.kelly_fraction, self.kelly_cap)
            stake = balance * frac
            why = (f"{self.kelly_fraction:.2f}-Kelly on lower bound p={p:.4f} "
                   f"-> {frac:.4%} of balance")
        elif self.method == MARTINGALE:
            if not self.martingale_enabled:
                return self.base_stake, "martingale not enabled -- using base stake"
            steps = max(0, min(self._consecutive_losses - self.min_consecutive_losses + 1,
                               self.martingale_max_steps))
            stake = self.base_stake * (self.martingale_factor ** steps) if steps > 0 else self.base_stake
            why = f"martingale step {steps} after {self._consecutive_losses} losses"
        else:
            return 0.0, f"unknown staking method {self.method!r}"

        capped = min(stake, self.max_stake)
        if capped < stake:
            why += f" (capped at max_stake {self.max_stake:.2f})"
        return round(max(0.0, capped), 2), why

    def register_result(self, won: bool) -> None:
        self._consecutive_losses = 0 if won else self._consecutive_losses + 1

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses
