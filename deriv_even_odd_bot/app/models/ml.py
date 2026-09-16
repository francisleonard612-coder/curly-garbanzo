"""
Machine-learning parity models E/F/G (spec Sections 8, 21, 42).

  OnlineLogisticParityModel  -- SGD logistic regression, updated per tick
  GradientBoostingParityModel -- periodic batch refit on a rolling buffer

NEURAL MODEL (Section 8 Model G): DELIBERATELY NOT IMPLEMENTED, and this is
a compliance decision rather than an omission. The spec says plainly: "Do
not include a neural model merely because it sounds sophisticated. It must
prove incremental predictive value." The ablation harness
(app/backtest/simulator.py) is the mechanism for proving that, and on the
target instrument the linear and tree models already sit at chance. Adding
a GRU to fit the same noise with more parameters would violate Sections 42
and 51 while adding a torch dependency and a training loop to maintain. If
the ablation report ever shows a non-chance signal that a linear model
cannot capture, that is the moment to add one -- with evidence.

ANTI-OVERFITTING (Section 42) is the live constraint for both models here:

  - Both are strongly regularized by default (L2 on logistic, shallow trees
    and a low learning rate on GBM).
  - Both refuse to predict below a minimum sample count, returning a flat
    distribution rather than a confident guess from a handful of rows.
  - The GBM refits on a ROLLING, TIME-ORDERED buffer and is never shown
    shuffled data (Section 50). Shuffling a time series here would let it
    interpolate between neighbouring ticks and score brilliantly on
    validation while learning nothing transferable.
  - Predictions are clipped away from 0/1. An ML model asserting p=0.99 on
    a parity call is reporting overfit, not confidence, and unclipped it
    would dominate any ensemble average.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from app.models.base import EVEN_DIGITS, DigitModel, DigitPrediction, normalize_distribution

# ML parity estimates are clipped into this band before use. The band is
# deliberately tight: a genuine parity edge on this instrument class would
# be a fraction of a percent, so anything outside it is model pathology.
P_CLIP_LO = 0.20
P_CLIP_HI = 0.80


def _parity_prediction(name: str, p_even: float, n: int) -> DigitPrediction:
    p_even = min(max(p_even, P_CLIP_LO), P_CLIP_HI)
    probs = [p_even / 5 if d in EVEN_DIGITS else (1 - p_even) / 5 for d in range(10)]
    return DigitPrediction(name, normalize_distribution(probs), n, direct_p_even=p_even)


class OnlineLogisticParityModel(DigitModel):
    """Model E: logistic regression on the feature vector, trained by SGD.

    Learning rate decays as 1/sqrt(t) so early examples do not permanently
    dominate and late ones cannot whipsaw the weights -- Section 47's
    "bounded learning rates / prevent catastrophic adaptation to a short
    sequence".
    """

    def __init__(self, n_features: int, learning_rate: float = 0.02, l2: float = 0.01,
                 min_samples: int = 1000, name: str | None = None):
        super().__init__(name or "logistic_parity")
        self.n_features = n_features
        self.lr = learning_rate
        self.l2 = l2
        self.min_samples = min_samples
        self._w = np.zeros(n_features, dtype=float)
        self._b = 0.0
        self._t = 0
        # Running feature standardization. Unscaled features (chi2 in the
        # tens, parity flags at +-1) would make a single large-magnitude
        # feature dominate the gradient regardless of its informativeness.
        self._mean = np.zeros(n_features, dtype=float)
        self._m2 = np.zeros(n_features, dtype=float)
        self._pending: np.ndarray | None = None

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        if self._t < 2:
            return np.zeros_like(x)
        var = self._m2 / max(self._t - 1, 1)
        sd = np.sqrt(np.maximum(var, 1e-12))
        return np.clip((x - self._mean) / sd, -5.0, 5.0)

    def _update_moments(self, x: np.ndarray) -> None:
        self._t += 1
        delta = x - self._mean
        self._mean += delta / self._t
        self._m2 += delta * (x - self._mean)

    def set_features(self, x) -> None:
        """Supply the CURRENT feature vector (computed before the target
        tick exists). Held until observe() reveals the label."""
        self._pending = np.asarray(x, dtype=float)

    def observe(self, digit: int) -> None:
        """Label arrives: fold the (held features, realized parity) pair in."""
        x_raw = self._pending
        self._pending = None
        self._n_seen += 1
        if x_raw is None or len(x_raw) != self.n_features:
            return
        self._update_moments(x_raw)
        x = self._standardize(x_raw)
        y = 1.0 if digit % 2 == 0 else 0.0     # 1 = EVEN
        z = float(np.dot(self._w, x) + self._b)
        p = 1.0 / (1.0 + math.exp(-max(min(z, 35.0), -35.0)))
        err = p - y
        lr = self.lr / math.sqrt(max(self._t, 1))
        self._w -= lr * (err * x + self.l2 * self._w)
        self._b -= lr * err

    def predict(self) -> DigitPrediction:
        if self._n_seen < self.min_samples or self._pending is None:
            return self.uninformative()
        x = self._standardize(np.asarray(self._pending, dtype=float))
        z = float(np.dot(self._w, x) + self._b)
        p = 1.0 / (1.0 + math.exp(-max(min(z, 35.0), -35.0)))
        return _parity_prediction(self.name, p, self._n_seen)

    def coefficients(self) -> np.ndarray:
        """Diagnostic only (Section 52): coefficient magnitude is not
        evidence of causality, and on noise these drift around zero."""
        return self._w.copy()


class GradientBoostingParityModel(DigitModel):
    """Model F: gradient boosting, refit periodically on a rolling buffer.

    Refit cadence is a real cost/benefit tradeoff: too frequent and the
    tick loop stalls (Section 62), too rare and the model goes stale. The
    default refits every `refit_every` labelled samples, off the event loop
    path (the executor calls this between ticks, not inside tick handling).
    """

    def __init__(self, n_features: int, buffer_size: int = 5000, min_samples: int = 1500,
                 refit_every: int = 500, name: str | None = None):
        super().__init__(name or "gbm_parity")
        self.n_features = n_features
        self.min_samples = min_samples
        self.refit_every = refit_every
        self._X: deque = deque(maxlen=buffer_size)
        self._y: deque = deque(maxlen=buffer_size)
        self._pending: np.ndarray | None = None
        self._model = None
        self._since_refit = 0
        self._available = self._check_sklearn()

    @staticmethod
    def _check_sklearn() -> bool:
        try:
            import sklearn  # noqa: F401
            return True
        except ImportError:
            return False

    def set_features(self, x) -> None:
        self._pending = np.asarray(x, dtype=float)

    def observe(self, digit: int) -> None:
        x = self._pending
        self._pending = None
        self._n_seen += 1
        if x is None or len(x) != self.n_features:
            return
        self._X.append(x)
        self._y.append(1 if digit % 2 == 0 else 0)
        self._since_refit += 1

    def maybe_refit(self) -> bool:
        """Call between ticks, not inside the tick handler."""
        if not self._available:
            return False
        if len(self._X) < self.min_samples or self._since_refit < self.refit_every:
            return False
        from sklearn.ensemble import HistGradientBoostingClassifier
        X = np.array(self._X, dtype=float)
        y = np.array(self._y, dtype=int)
        if len(np.unique(y)) < 2:
            return False
        # Shallow + slow + heavily leaf-regularized: Section 42's "model
        # simplicity preference". The buffer stays in time order and is
        # never shuffled (Section 50).
        model = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.03, max_iter=150,
            min_samples_leaf=50, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.2, random_state=0,
        )
        model.fit(X, y)
        self._model = model
        self._since_refit = 0
        return True

    def predict(self) -> DigitPrediction:
        if self._model is None or self._pending is None or self._n_seen < self.min_samples:
            return self.uninformative()
        try:
            p = float(self._model.predict_proba(self._pending.reshape(1, -1))[0][1])
        except Exception:
            return self.uninformative()
        return _parity_prediction(self.name, p, self._n_seen)

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def feature_importances(self) -> np.ndarray | None:
        """Section 52: permutation importance, diagnostics only."""
        if self._model is None or len(self._X) < 500:
            return None
        try:
            from sklearn.inspection import permutation_importance
            X = np.array(list(self._X)[-1000:], dtype=float)
            y = np.array(list(self._y)[-1000:], dtype=int)
            r = permutation_importance(self._model, X, y, n_repeats=3, random_state=0)
            return r.importances_mean
        except Exception:
            return None
