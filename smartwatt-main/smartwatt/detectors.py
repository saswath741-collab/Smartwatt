"""Detectors: fixed threshold, adaptive threshold, Isolation Forest, hybrid."""
import numpy as np
from sklearn.ensemble import IsolationForest

from .features import FEATURES


class FixedThreshold:
    """Baseline: flag readings outside mean +/- k*std of the training data."""

    def __init__(self, k=3.0):
        self.k = k

    def fit(self, train):
        self.mu = train["consumption"].mean()
        self.sd = train["consumption"].std()
        return self

    def predict(self, df):
        return (np.abs(df["consumption"] - self.mu) > self.k * self.sd).astype(int).values


def adaptive_z(df, window=14):
    """Time-aware robust z-score.

    Each reading is compared with the previous `window` readings taken at the
    same hour on the same day type (weekday/weekend). The baseline is their
    median and the scale is 1.4826*MAD, so the threshold follows daily and
    weekly routine and slowly adapts as habits change. Only past data is used.
    """
    d = df[["consumption", "hour", "is_weekend"]]
    g = d.groupby(["hour", "is_weekend"])["consumption"]
    med = g.transform(lambda s: s.shift(1).rolling(window, min_periods=4).median())
    mad = g.transform(lambda s: s.shift(1).rolling(window, min_periods=4)
                      .apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True))
    scale = np.maximum(1.4826 * mad, 0.05)  # floor avoids divide-by-~0
    return ((d["consumption"] - med) / scale).values


class AdaptiveThreshold:
    """Time-aware adaptive threshold only (ablation of the hybrid)."""

    def __init__(self, k=4.0, window=14):
        self.k, self.window = k, window

    def fit(self, train):
        return self

    def predict(self, df):
        z = np.nan_to_num(adaptive_z(df, self.window))
        return (np.abs(z) > self.k).astype(int)


class IsolationForestDetector:
    """Isolation Forest on time-aware features, trained on the training split."""

    def __init__(self, contamination=0.03, seed=0):
        self.model = IsolationForest(n_estimators=200, contamination=contamination,
                                     random_state=seed)

    def fit(self, train):
        self.model.fit(train[FEATURES])
        return self

    def predict(self, df):
        return (self.model.predict(df[FEATURES]) == -1).astype(int)


class HybridAdaptive:
    """Hybrid: Isolation Forest confirmed by the adaptive threshold.

    Flag when the adaptive z-score is very high on its own (k_hi), or when
    Isolation Forest flags the point AND the reading is at least moderately
    unusual for its time slot (k_lo). The adaptive threshold suppresses IF
    false alarms on normal peaks; IF catches multivariate oddities the
    threshold alone would miss.
    """

    def __init__(self, k_hi=5.0, k_lo=2.5, contamination=0.03, window=14, seed=0):
        self.k_hi, self.k_lo, self.window = k_hi, k_lo, window
        self.iforest = IsolationForestDetector(contamination, seed)

    def fit(self, train):
        self.iforest.fit(train)
        return self

    def predict(self, df):
        z = np.abs(np.nan_to_num(adaptive_z(df, self.window)))
        iso = self.iforest.predict(df)
        return ((z > self.k_hi) | ((iso == 1) & (z > self.k_lo))).astype(int)


class DriftAdaptive:
    """Online EWMA baseline per (hour, day-type) slot.

    The baseline mean and mean absolute deviation are updated after every
    reading with rate `alpha`, so the threshold follows gradual changes in
    household usage (concept drift). Flagged readings are not used for the
    update, which stops anomalies from polluting the baseline.
    """

    def __init__(self, k=4.0, alpha=0.1, warmup=4):
        self.k, self.alpha, self.warmup = k, alpha, warmup

    def fit(self, train):
        return self

    def predict(self, df):
        mean, dev, n = {}, {}, {}
        out = np.zeros(len(df), dtype=int)
        for i, (c, h, w) in enumerate(zip(df["consumption"], df["hour"], df["is_weekend"])):
            key = (h, w)
            if key not in mean:
                mean[key], dev[key], n[key] = c, 0.05, 1
                continue
            z = abs(c - mean[key]) / max(1.25 * dev[key], 0.05)
            flag = n[key] >= self.warmup and z > self.k
            out[i] = int(flag)
            if not flag:
                dev[key] += self.alpha * (abs(c - mean[key]) - dev[key])
                mean[key] += self.alpha * (c - mean[key])
            n[key] += 1
        return out
