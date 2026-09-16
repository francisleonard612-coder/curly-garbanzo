"""
Feature engine (spec Sections 14, 15, 16, 22).

LEAKAGE IS THE ONLY THING THAT MATTERS HERE (Section 22). Every feature is
computed from a `SymbolState` that contains ONLY ticks already observed.
The contract is enforced structurally: build() takes the state and returns
a vector describing the state AS IT IS, and the caller must invoke it
BEFORE appending the tick being predicted. app/diagnostics/leakage.py
verifies that ordering holds in the live loop.

The subtle version of this bug is worth naming, because it is easy to write
and impossible to see in the output: computing a rolling mean over a window
that INCLUDES the target tick gives a feature that is ~1/window correlated
with the label. On 10,000 ticks that produces a model with impressive
backtest accuracy and exactly zero live edge -- it has learned to read the
answer, slightly.

FEATURE COUNT IS DELIBERATELY BOUNDED (Section 14: "do not create thousands
of meaningless correlated features"). ~40 features against a few thousand
samples is already an aggressive ratio for a binary target; a wider matrix
would fit noise faster than the regularizer can suppress it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from app.data.tick_store import SymbolState
from app.statistics.distribution import chi_square_gof, jensen_shannon_divergence
from app.statistics.information import normalized_digit_entropy, parity_entropy

FEATURE_WINDOWS = (20, 50, 100, 250, 500, 1000)


@dataclass(frozen=True)
class FeatureVector:
    names: tuple[str, ...]
    values: tuple[float, ...]
    n_history: int

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values))

    def __len__(self) -> int:
        return len(self.values)


def _safe(x: float, default: float = 0.0) -> float:
    return default if (x is None or x != x or x in (float("inf"), float("-inf"))) else float(x)


def build(state: SymbolState) -> FeatureVector:
    """Feature vector describing everything observed SO FAR.

    Must be called before the target tick is appended (Section 22).
    """
    names: list[str] = []
    values: list[float] = []

    def add(name: str, value: float) -> None:
        names.append(name)
        values.append(_safe(value))

    n = state.total_count
    long_run = state.long_run_frequencies()

    # ---- rolling parity rates, centred on 0.5 ----------------------------
    for w in FEATURE_WINDOWS:
        win = state.windows.get(w)
        if win is None or len(win) < max(10, w // 10):
            add(f"even_rate_{w}", 0.0)
            add(f"even_z_{w}", 0.0)
            continue
        rate = win.even_rate()
        add(f"even_rate_{w}", rate - 0.5)
        # z-score of the even count under a fair-coin null
        k = len(win)
        add(f"even_z_{w}", (rate - 0.5) / (0.5 / math.sqrt(k)) if k > 0 else 0.0)

    # ---- digit distribution shape ----------------------------------------
    for w in (100, 500, 1000):
        win = state.windows.get(w)
        if win is None or len(win) < max(30, w // 10):
            add(f"chi2_{w}", 0.0)
            add(f"jsd_vs_long_{w}", 0.0)
            add(f"entropy_{w}", 0.0)
            continue
        gof = chi_square_gof(win.digit_counts)
        add(f"chi2_{w}", gof.statistic)
        add(f"jsd_vs_long_{w}", jensen_shannon_divergence(win.digit_frequencies(), long_run)
            if n > 0 else 0.0)
        add(f"entropy_{w}", normalized_digit_entropy(win.digits()))

    # ---- recent-vs-long-term drift ---------------------------------------
    short = state.windows.get(100)
    if short is not None and len(short) >= 30 and n >= 500:
        add("even_rate_delta_short_long", short.even_rate() - state.long_run_even_rate())
    else:
        add("even_rate_delta_short_long", 0.0)

    # ---- exponentially weighted ------------------------------------------
    for decay, e in state.ewma.items():
        add(f"ewma_even_{decay}", e.even_rate() - 0.5)

    # ---- run / streak state ----------------------------------------------
    parity_val, run_len = state.current_run()
    add("run_length", float(run_len))
    add("run_is_even", 1.0 if parity_val == 0 else 0.0)
    add("digit_run_length", float(state.current_digit_run()))

    # ---- lagged parities (the direct Markov evidence) ---------------------
    parity_seq = state.parity_sequence(8)
    for lag in range(1, 9):
        if len(parity_seq) >= lag:
            add(f"parity_lag_{lag}", 1.0 if parity_seq[-lag] == 0 else -1.0)
        else:
            add(f"parity_lag_{lag}", 0.0)

    # ---- lagged digits, scaled -------------------------------------------
    digit_seq = state.digit_sequence(3)
    for lag in range(1, 4):
        add(f"digit_lag_{lag}", (digit_seq[-lag] / 9.0 - 0.5) if len(digit_seq) >= lag else 0.0)

    # ---- alternation / repetition scores ---------------------------------
    ps = state.parity_sequence(100)
    if len(ps) >= 20:
        alternations = sum(1 for a, b in zip(ps, ps[1:]) if a != b)
        add("alternation_score", alternations / (len(ps) - 1) - 0.5)
        add("parity_entropy_100", parity_entropy(state.digit_sequence(100), base=2))
    else:
        add("alternation_score", 0.0)
        add("parity_entropy_100", 1.0)

    ds = state.digit_sequence(100)
    if len(ds) >= 20:
        add("repetition_score", sum(1 for a, b in zip(ds, ds[1:]) if a == b) / (len(ds) - 1) - 0.1)
    else:
        add("repetition_score", 0.0)

    # ---- price-derived (Section 15) --------------------------------------
    # Retained because the digit is a deterministic function of the price;
    # if price dynamics carry any information about the final decimal, it
    # would show up here. On a CSPRNG stream it will not.
    add("realized_vol_100", state.realized_volatility(100))
    last = state.last
    add("last_delta", last.delta if last and last.delta is not None else 0.0)
    add("last_delta_sign", math.copysign(1.0, last.delta) if last and last.delta else 0.0)
    add("mean_interval_100", state.mean_interval(100))
    add("inexact_fraction", state.inexact_count / n if n > 0 else 0.0)

    return FeatureVector(tuple(names), tuple(values), n)


def feature_names() -> tuple[str, ...]:
    """Stable ordering, for model initialization before any data arrives."""
    from app.data.tick_store import SymbolState
    return build(SymbolState(symbol="_probe", precision=2)).names
