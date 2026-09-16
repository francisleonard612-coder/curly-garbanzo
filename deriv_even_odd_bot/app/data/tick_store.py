"""
Tick store and multi-scale rolling state (spec Sections 4, 16, 62, 63).

Every statistic here is INCREMENTAL. The spec (Section 62) forbids
recomputing whole rolling datasets per tick, and at 1 tick/second across
several windows the naive version would dominate the event loop.

Design: one append-only bounded deque of ticks, plus a set of
RollingDigitWindow objects that each maintain their own O(1)-updated digit
counts. Adding a tick is O(number_of_windows), not O(history).

LEAKAGE (Section 22): this module stores only what has already been
observed. `snapshot()` returns state as of the ticks appended SO FAR, and
the prediction path must call it BEFORE the target tick is appended.
app/diagnostics/leakage.py tests that ordering.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

from app.digits.extraction import ExtractedDigit


@dataclass(frozen=True)
class Tick:
    symbol: str
    epoch: float
    quote_raw: str
    quote: float
    precision: int
    digit: int
    parity: int          # 0 EVEN, 1 ODD
    exact: bool
    received_at: float
    prev_digit: int | None
    delta: float | None       # tick-to-tick price change
    interval: float | None    # seconds since previous tick

    @property
    def is_even(self) -> bool:
        return self.parity == 0


class RollingDigitWindow:
    """Fixed-length window with O(1) digit/parity counts.

    Counts are maintained on add/evict rather than recounted, so cost per
    tick is constant regardless of window size -- the 10,000-tick window
    costs the same per tick as the 20-tick one.
    """

    __slots__ = ("size", "_buf", "_digit_counts", "_even")

    def __init__(self, size: int):
        if size < 1:
            raise ValueError("window size must be >= 1")
        self.size = size
        self._buf: deque[int] = deque()
        self._digit_counts = [0] * 10
        self._even = 0

    def add(self, digit: int) -> None:
        self._buf.append(digit)
        self._digit_counts[digit] += 1
        if digit % 2 == 0:
            self._even += 1
        while len(self._buf) > self.size:
            old = self._buf.popleft()
            self._digit_counts[old] -= 1
            if old % 2 == 0:
                self._even -= 1

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def is_full(self) -> bool:
        return len(self._buf) >= self.size

    @property
    def digit_counts(self) -> list[int]:
        return list(self._digit_counts)

    @property
    def even_count(self) -> int:
        return self._even

    @property
    def odd_count(self) -> int:
        return len(self._buf) - self._even

    def digit_frequencies(self) -> list[float]:
        n = len(self._buf)
        if n == 0:
            return [0.0] * 10
        return [c / n for c in self._digit_counts]

    def even_rate(self) -> float:
        n = len(self._buf)
        return float("nan") if n == 0 else self._even / n

    def digits(self) -> list[int]:
        return list(self._buf)


class ExponentialDigitWeighting:
    """Exponentially weighted digit distribution (spec Section 6).

    Weight of an observation k ticks ago is decay**k, so the effective
    sample size converges to 1/(1-decay).

    NOTE, carried from a bug found in a sibling bot's adaptive weighting:
    that effective sample size is an ASYMPTOTE approached from below and
    never reached. Any downstream minimum-sample gate compared against
    effective_n must therefore sit well below 1/(1-decay), or it can never
    be satisfied and the gate becomes a permanent silent lockout that looks
    exactly like "no signal found". effective_n is exposed here precisely so
    such gates can be checked against it.
    """

    __slots__ = ("decay", "_w", "_n")

    def __init__(self, decay: float = 0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        self.decay = decay
        self._w = [0.0] * 10
        self._n = 0.0

    @property
    def ceiling(self) -> float:
        """Asymptotic effective sample size -- never actually attained."""
        return 1.0 / (1.0 - self.decay)

    @property
    def effective_n(self) -> float:
        return self._n

    def add(self, digit: int) -> None:
        for i in range(10):
            self._w[i] *= self.decay
        self._n = self._n * self.decay + 1.0
        self._w[digit] += 1.0

    def frequencies(self) -> list[float]:
        if self._n <= 0:
            return [0.1] * 10
        total = sum(self._w)
        if total <= 0:
            return [0.1] * 10
        return [w / total for w in self._w]

    def even_rate(self) -> float:
        f = self.frequencies()
        return sum(f[d] for d in (0, 2, 4, 6, 8))


@dataclass
class SymbolState:
    """All observed state for one symbol. Owns nothing predictive -- models
    read from here, they don't live here."""

    symbol: str
    precision: int
    window_sizes: tuple[int, ...] = (20, 50, 100, 250, 500, 1000, 2500, 5000, 10000)
    max_history: int = 20000
    ewma_decays: tuple[float, ...] = (0.99, 0.999)

    ticks: deque[Tick] = field(init=False)
    windows: dict[int, RollingDigitWindow] = field(init=False)
    ewma: dict[float, ExponentialDigitWeighting] = field(init=False)
    total_digit_counts: list[int] = field(init=False)
    total_count: int = field(init=False, default=0)
    total_even: int = field(init=False, default=0)
    inexact_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.ticks = deque(maxlen=self.max_history)
        self.windows = {s: RollingDigitWindow(s) for s in self.window_sizes}
        self.ewma = {d: ExponentialDigitWeighting(d) for d in self.ewma_decays}
        self.total_digit_counts = [0] * 10

    def add(self, symbol: str, epoch: float, extracted: ExtractedDigit) -> Tick:
        prev = self.ticks[-1] if self.ticks else None
        quote = float(extracted.normalized_quote)
        tick = Tick(
            symbol=symbol,
            epoch=float(epoch),
            quote_raw=extracted.raw_quote,
            quote=quote,
            precision=extracted.precision,
            digit=extracted.digit,
            parity=extracted.parity,
            exact=extracted.exact,
            received_at=time.time(),
            prev_digit=prev.digit if prev else None,
            delta=(quote - prev.quote) if prev else None,
            interval=(float(epoch) - prev.epoch) if prev else None,
        )
        self.ticks.append(tick)
        for w in self.windows.values():
            w.add(tick.digit)
        for e in self.ewma.values():
            e.add(tick.digit)
        self.total_digit_counts[tick.digit] += 1
        self.total_count += 1
        if tick.parity == 0:
            self.total_even += 1
        if not extracted.exact:
            self.inexact_count += 1
        return tick

    # ---- read-only views -------------------------------------------------

    @property
    def last(self) -> Tick | None:
        return self.ticks[-1] if self.ticks else None

    def digit_sequence(self, n: int | None = None) -> list[int]:
        if n is None:
            return [t.digit for t in self.ticks]
        if n <= 0:
            return []
        return [t.digit for t in list(self.ticks)[-n:]]

    def parity_sequence(self, n: int | None = None) -> list[int]:
        if n is None:
            return [t.parity for t in self.ticks]
        if n <= 0:
            return []
        return [t.parity for t in list(self.ticks)[-n:]]

    def long_run_frequencies(self) -> list[float]:
        if self.total_count == 0:
            return [0.1] * 10
        return [c / self.total_count for c in self.total_digit_counts]

    def long_run_even_rate(self) -> float:
        if self.total_count == 0:
            return float("nan")
        return self.total_even / self.total_count

    def current_run(self) -> tuple[int, int]:
        """(parity_value, run_length) of the current parity streak."""
        if not self.ticks:
            return (-1, 0)
        seq = self.parity_sequence()
        last = seq[-1]
        n = 0
        for p in reversed(seq):
            if p != last:
                break
            n += 1
        return (last, n)

    def current_digit_run(self) -> int:
        if not self.ticks:
            return 0
        seq = self.digit_sequence()
        last = seq[-1]
        n = 0
        for d in reversed(seq):
            if d != last:
                break
            n += 1
        return n

    def is_stale(self, max_age_seconds: float) -> bool:
        """Spec Section 59: stale data must halt trading."""
        if not self.ticks:
            return True
        return (time.time() - self.ticks[-1].received_at) > max_age_seconds

    def mean_interval(self, n: int = 100) -> float:
        vals = [t.interval for t in list(self.ticks)[-n:] if t.interval is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    def realized_volatility(self, n: int = 100) -> float:
        deltas = [t.delta for t in list(self.ticks)[-n:] if t.delta is not None]
        if len(deltas) < 2:
            return float("nan")
        m = sum(deltas) / len(deltas)
        return math.sqrt(sum((d - m) ** 2 for d in deltas) / (len(deltas) - 1))
