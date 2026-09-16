"""
Contract economics, expected value and mispricing detection
(spec Sections 24, 25, 26, 55).

THIS IS THE MODULE THAT DECIDES WHETHER MONEY IS MADE. Everything upstream
estimates P(EVEN); this converts that estimate plus a REAL quoted payout
into an expected value, and refuses the trade unless the edge survives the
uncertainty in the estimate.

BREAK-EVEN IS DERIVED FROM THE ACTUAL PROPOSAL, NEVER ASSUMED (Section 24).
For a binary contract staking S to return a total payout P on a win:

    win  -> profit = P - S
    loss -> profit = -S

    EV = p(P - S) - (1-p)S = pP - S

    EV > 0  <=>  p > S / P

So break-even probability is simply stake/payout. At a typical Even/Odd
payout of 1.95x that is 0.5128 -- i.e. the model must be right 51.28% of
the time merely to break even. This single number is why "P(EVEN) > 0.50"
is never a reason to trade, and the spec (Section 69) bans exactly that.

THE UNCERTAINTY RULE (Sections 26, 55). The spec's own worked example:
P=0.586 +- 0.045 against a break-even of 0.565 must NOT trade, because the
lower bound (0.541) is below break-even. We implement precisely that: the
edge is computed from the LOWER CONFIDENCE BOUND of the probability
estimate, not the point estimate. This is the single most important line of
defence against the multiple-comparisons problem -- across thousands of
evaluations, point estimates will wander above break-even constantly; lower
bounds will not.

PAYOUT QUALITY IS ITS OWN LEVER. Break-even is a function of the quoted
payout alone. Raising the minimum acceptable payout lowers the bar the
models must clear, and costs nothing in model complexity. A bot that
accepts 1.88x needs 53.2% accuracy; at 1.97x it needs 50.8%. On an
instrument where true accuracy is 50%, NEITHER is profitable -- but the
second is far less unprofitable, and if any real edge ever exists, payout
selection is what converts it into money.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from app.statistics.distribution import Interval


class ProposalError(ValueError):
    pass


@dataclass(frozen=True)
class Proposal:
    """A real quote from Deriv. Never synthesised, never assumed."""
    contract_type: str          # "DIGITEVEN" | "DIGITODD"
    symbol: str
    stake: float
    payout: float               # TOTAL returned on a win, including stake
    ask_price: float
    currency: str
    proposal_id: str
    spot: float | None = None
    received_at: float = 0.0

    def __post_init__(self) -> None:
        if self.stake <= 0:
            raise ProposalError(f"non-positive stake: {self.stake}")
        if self.payout <= self.stake:
            # A payout at or below stake cannot be profitable at any
            # probability; treat as a malformed quote rather than trading it.
            raise ProposalError(
                f"payout {self.payout} <= stake {self.stake}: contract cannot profit")

    @property
    def profit_if_win(self) -> float:
        return self.payout - self.stake

    @property
    def loss_if_lose(self) -> float:
        return self.stake

    @property
    def payout_multiple(self) -> float:
        return self.payout / self.stake

    @property
    def break_even_probability(self) -> float:
        """p* = stake / payout. Derived, never hard-coded (Section 24)."""
        return self.stake / self.payout

    def is_stale(self, now: float, max_age_seconds: float = 5.0) -> bool:
        """Section 33: a stale proposal must never be executed."""
        if not self.received_at:
            return True
        return (now - self.received_at) > max_age_seconds


@dataclass(frozen=True)
class EdgeAssessment:
    proposal: Proposal
    probability: float                 # calibrated point estimate for THIS side
    probability_interval: Interval
    break_even: float

    point_edge: float                  # p_hat - break_even
    conservative_edge: float           # p_lower - break_even   <- the decision number
    expected_value: float              # currency, at the point estimate
    conservative_expected_value: float # currency, at the lower bound
    ev_per_unit_staked: float
    kelly_fraction: float

    robust: bool
    reasons: tuple[str, ...]

    @property
    def tradeable(self) -> bool:
        return self.robust


def assess_edge(proposal: Proposal, probability: float, interval: Interval,
                *, min_edge: float = 0.02, min_ev: float = 0.0,
                max_interval_width: float = 0.06) -> EdgeAssessment:
    """Full economic assessment for one side of one contract.

    `probability` and `interval` must refer to the SAME side as
    `proposal.contract_type`. Passing P(EVEN) alongside a DIGITODD proposal
    is the kind of mismatch that produces a bot confidently trading the
    wrong way, so callers should use edge_for_side() below rather than
    assembling this by hand.
    """
    be = proposal.break_even_probability
    p = min(max(float(probability), 0.0), 1.0)
    p_lo = min(max(interval.lower, 0.0), 1.0)

    point_edge = p - be
    conservative_edge = p_lo - be

    ev = p * proposal.payout - proposal.stake
    ev_cons = p_lo * proposal.payout - proposal.stake

    b = proposal.profit_if_win / proposal.stake        # net odds
    kelly = (p * b - (1 - p)) / b if b > 0 else 0.0

    reasons: list[str] = []
    if conservative_edge < min_edge:
        reasons.append(
            f"conservative edge {conservative_edge:+.4f} < required {min_edge:.4f} "
            f"(point {point_edge:+.4f}, break-even {be:.4f})")
    if ev_cons <= min_ev:
        reasons.append(f"conservative EV {ev_cons:+.4f} <= required {min_ev:.4f}")
    if interval.width > max_interval_width:
        reasons.append(
            f"probability interval too wide: {interval.width:.4f} > {max_interval_width:.4f} "
            f"-- estimate not precise enough to act on")
    if kelly <= 0:
        reasons.append(f"Kelly fraction non-positive ({kelly:+.4f})")

    robust = not reasons
    if robust:
        reasons.append(
            f"edge {conservative_edge:+.4f} (lower bound {p_lo:.4f} vs break-even {be:.4f}), "
            f"EV {ev_cons:+.4f}, payout {proposal.payout_multiple:.3f}x")

    return EdgeAssessment(
        proposal=proposal, probability=p, probability_interval=interval,
        break_even=be, point_edge=point_edge, conservative_edge=conservative_edge,
        expected_value=ev, conservative_expected_value=ev_cons,
        ev_per_unit_staked=ev / proposal.stake,
        kelly_fraction=kelly, robust=robust, reasons=tuple(reasons),
    )


def edge_for_side(proposal: Proposal, p_even: float, interval_even: Interval,
                  **kwargs) -> EdgeAssessment:
    """Assess a proposal using the parity probability oriented to the
    proposal's OWN side -- the guard against the P(EVEN)/DIGITODD mismatch
    described in assess_edge()."""
    ct = proposal.contract_type.upper()
    if ct == "DIGITEVEN":
        return assess_edge(proposal, p_even, interval_even, **kwargs)
    if ct == "DIGITODD":
        flipped = Interval(point=1 - interval_even.point,
                           lower=1 - interval_even.upper,
                           upper=1 - interval_even.lower,
                           n=interval_even.n)
        return assess_edge(proposal, 1 - p_even, flipped, **kwargs)
    raise ProposalError(f"unsupported contract type for parity trading: {proposal.contract_type}")


def required_probability(payout_multiple: float, margin: float = 0.0) -> float:
    """Inverse view: what accuracy does this payout demand?

    Useful as a reality check before any modelling -- if the answer is
    higher than the model could plausibly reach, no amount of model work
    will fix it and the payout is the thing to change.
    """
    if payout_multiple <= 1.0:
        raise ProposalError("payout multiple must exceed 1.0")
    return 1.0 / payout_multiple + margin


def payout_for_probability(probability: float, margin: float = 0.0) -> float:
    """What payout would make a given accuracy break even?"""
    p = probability - margin
    if p <= 0:
        raise ProposalError("probability net of margin must be positive")
    return 1.0 / p


def probability_interval_from_samples(p_hat: float, n: int, confidence: float = 0.95) -> Interval:
    """Wilson-style interval around a model probability given the effective
    sample size behind it.

    IMPORTANT CAVEAT, stated because it is easy to over-trust this: `n` is
    the number of independent observations supporting the estimate, NOT the
    number of ticks seen. A model trained on 10,000 ticks whose prediction
    rests on a 200-sample Markov cell has n=200 here. Overstating n narrows
    the interval and defeats the conservative-edge rule that the whole gate
    depends on.
    """
    if n <= 0:
        return Interval(p_hat, 0.0, 1.0, 0)
    from app.statistics.distribution import inv_norm_cdf
    z = inv_norm_cdf(1 - (1 - confidence) / 2)
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return Interval(point=p_hat, lower=max(0.0, centre - margin),
                    upper=min(1.0, centre + margin), n=n)
