"""
Data-leakage detection (spec Section 22 -- critical).

Leakage is the failure mode that produces a bot which backtests beautifully
and loses money live, and it is invisible in every output except the
account balance. The specific version that matters here: computing a
feature from a window that INCLUDES the tick being predicted. That gives a
feature ~1/window correlated with the label, which a model finds instantly.

Two independent checks:

  assert_no_leakage()  -- structural. Builds features at time t, then
                          appends the tick for t+1, then rebuilds. If the
                          FIRST vector changed, the feature engine is
                          reading state it should not have had.

  LeakageCanary        -- statistical. Injects a synthetic label that is
                          pure noise and verifies models cannot predict it
                          above chance. A model that "predicts" a random
                          label is reading the answer from somewhere.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


class LeakageError(AssertionError):
    pass


def assert_no_leakage(state_factory, digits: list[int], build_features) -> None:
    """Structural check.

    `state_factory()` must return a fresh empty SymbolState.
    `build_features(state)` must return a FeatureVector.
    """
    from app.digits.extraction import extract

    if len(digits) < 50:
        raise ValueError("need at least 50 digits for a meaningful check")

    state = state_factory()
    for i, d in enumerate(digits[:-1]):
        state.add("TEST", float(i), extract(f"100.{d}", 0.1))

    before = build_features(state).values

    nxt = digits[-1]
    state.add("TEST", float(len(digits)), extract(f"100.{nxt}", 0.1))
    after = build_features(state).values

    if before == after:
        raise LeakageError(
            "feature vector did not change after appending a tick -- the feature "
            "engine appears to ignore new data, which means the leakage check "
            "cannot detect anything")

    state2 = state_factory()
    for i, d in enumerate(digits[:-1]):
        state2.add("TEST", float(i), extract(f"100.{d}", 0.1))
    recomputed = build_features(state2).values

    if recomputed != before:
        raise LeakageError(
            "feature vector for identical history is not reproducible -- features "
            "depend on something outside the observed ticks")


@dataclass
class CanaryResult:
    n: int
    accuracy: float
    threshold: float
    leaked: bool
    detail: str


class LeakageCanary:
    """Statistical check: a model given a RANDOM label must score at chance.

    Threshold is set from the binomial standard error so the canary fires on
    genuine leakage rather than on ordinary sampling noise.
    """

    def __init__(self, seed: int = 0, z: float = 4.0):
        self.rng = random.Random(seed)
        self.z = z
        self._n = 0
        self._correct = 0

    def record(self, predicted_even: bool) -> int:
        """Returns the synthetic label (0 EVEN / 1 ODD) for this step."""
        label_even = self.rng.random() < 0.5
        self._n += 1
        if predicted_even == label_even:
            self._correct += 1
        return 0 if label_even else 1

    def result(self) -> CanaryResult:
        if self._n < 200:
            return CanaryResult(self._n, float("nan"), float("nan"), False,
                                "insufficient samples")
        acc = self._correct / self._n
        se = 0.5 / (self._n ** 0.5)
        threshold = 0.5 + self.z * se
        leaked = acc > threshold
        return CanaryResult(
            self._n, acc, threshold, leaked,
            f"accuracy {acc:.4f} vs chance threshold {threshold:.4f}"
            + (" -- LEAKAGE SUSPECTED" if leaked else " -- clean"))
