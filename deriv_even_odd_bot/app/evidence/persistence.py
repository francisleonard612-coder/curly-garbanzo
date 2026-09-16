"""
Signal persistence and decay (spec Sections 18, 19).

A single tick reading of P(EVEN) = 0.571 is one draw from a noisy estimator.
The same reading sustained across several consecutive ticks, moving in a
consistent direction, is a different and stronger piece of evidence. This
module measures that difference.

IT IS EVIDENCE, NOT A REQUIREMENT. Section 18 ends with "do not require
persistence in every case" -- a genuine short-lived dislocation should still
be tradeable. So persistence raises the opportunity score and its absence
lowers it, but no code path here can refuse a trade.

SECTION 19 IS THE HALF THAT MATTERS MORE. Persistence tracking without decay
is a machine for chasing expired signals: it would remember that the edge was
strong four ticks ago and keep acting on that memory after the edge is gone.
Every accumulator here decays on its own, and `note_absent()` is called on
every tick where no candidate exists, so silence actively erodes the history
rather than merely failing to extend it.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class PersistenceReading:
    probability_persistence: float    # [0, 1]
    edge_persistence: float           # [0, 1]
    agreement_persistence: float      # [0, 1]
    direction_stability: float        # [0, 1], fraction of history on one side
    n_observations: int
    age_seconds: float
    decayed: bool
    note: str = ""

    @property
    def combined(self) -> float:
        """Single [0,1] figure for the opportunity blend."""
        return (0.45 * self.probability_persistence
                + 0.35 * self.edge_persistence
                + 0.20 * self.agreement_persistence)


class SignalTracker:
    """Rolling per-side view of how stable the current signal is.

    One tracker per (symbol, side). Feeding EVEN and ODD readings into the
    same tracker would report a flapping signal as a persistent one, since
    P(EVEN)=0.56 and P(EVEN)=0.44 are equally "strong" in magnitude while
    pointing opposite ways.
    """

    def __init__(self, *, window: int = 12, max_age_seconds: float = 30.0,
                 decay_half_life: float = 6.0):
        self.window = window
        self.max_age_seconds = max_age_seconds
        self.decay_half_life = decay_half_life
        self._prob: deque[float] = deque(maxlen=window)
        self._edge: deque[float] = deque(maxlen=window)
        self._agree: deque[float] = deque(maxlen=window)
        self._side: deque[str] = deque(maxlen=window)
        self._last_update: float = 0.0
        self._absent_streak: int = 0

    # -- input ---------------------------------------------------------------

    def observe(self, *, side: str, probability: float, edge: float,
                agreement: float, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if self._last_update and (now - self._last_update) > self.max_age_seconds:
            # A gap this long means the intervening stream is unobserved;
            # continuing the old series would splice together two unrelated
            # windows. Section 19: never chase an expired signal.
            self.reset("history expired")
        self._prob.append(float(probability))
        self._edge.append(float(edge))
        self._agree.append(float(agreement))
        self._side.append(str(side))
        self._last_update = now
        self._absent_streak = 0

    def note_absent(self) -> None:
        """Called on every tick that produced no candidate. Section 19: the
        absence of a signal must actively degrade the history, not preserve
        it. Three consecutive silent ticks clear it entirely."""
        self._absent_streak += 1
        if self._absent_streak >= 3:
            self.reset("signal disappeared")
        elif self._prob:
            self._prob.popleft()
            self._edge.popleft()
            self._agree.popleft()
            self._side.popleft()

    def reset(self, note: str = "") -> None:
        self._prob.clear()
        self._edge.clear()
        self._agree.clear()
        self._side.clear()
        self._last_update = 0.0
        self._absent_streak = 0
        self._note = note

    # -- output --------------------------------------------------------------

    def _decay_factor(self, now: float) -> float:
        if not self._last_update:
            return 0.0
        age = max(0.0, now - self._last_update)
        return 0.5 ** (age / self.decay_half_life)

    @staticmethod
    def _stability(values: list[float], scale: float) -> float:
        """1.0 when the series is tight, falling as it scatters.

        `scale` is the standard deviation at which the series stops counting
        as stable at all, chosen per-quantity: probabilities drifting by 0.02
        between consecutive ticks are already unstable at these magnitudes.
        """
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        sd = math.sqrt(var)
        return max(0.0, 1.0 - sd / scale) if scale > 0 else 0.0

    def read(self, now: float | None = None) -> PersistenceReading:
        now = time.time() if now is None else now
        n = len(self._prob)
        if n == 0:
            return PersistenceReading(0.0, 0.0, 0.0, 0.0, 0, 0.0, True,
                                      "no persistent signal")
        decay = self._decay_factor(now)
        age = now - self._last_update if self._last_update else 0.0

        # Depth: a 2-observation history should not read as maximally
        # persistent. Saturates at 6 consecutive observations.
        depth = min(1.0, n / 6.0)

        sides = list(self._side)
        dominant = max(set(sides), key=sides.count)
        direction = sides.count(dominant) / len(sides)
        # A signal that flipped sides is not persistent in any useful sense.
        direction_weight = max(0.0, (direction - 0.5) * 2.0)

        prob_stab = self._stability(list(self._prob), 0.02)
        edge_stab = self._stability(list(self._edge), 0.02)
        agree_stab = self._stability(list(self._agree), 0.15)

        factor = depth * decay * direction_weight
        note = ""
        if decay < 0.5:
            note = f"signal decaying (age {age:.1f}s)"
        elif direction_weight < 0.5:
            note = "signal has been flipping sides"

        return PersistenceReading(
            probability_persistence=prob_stab * factor,
            edge_persistence=edge_stab * factor,
            agreement_persistence=agree_stab * factor,
            direction_stability=direction,
            n_observations=n,
            age_seconds=age,
            decayed=decay < 0.5,
            note=note,
        )
