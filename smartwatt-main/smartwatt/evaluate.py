"""Evaluation metrics and the experiment pipeline."""
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from .data import generate_synthetic
from .features import add_features
from .detectors import FixedThreshold, AdaptiveThreshold, IsolationForestDetector, HybridAdaptive, DriftAdaptive


def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0}


def make_detectors(seed=0):
    return {"Fixed threshold": FixedThreshold(),
            "Adaptive threshold": AdaptiveThreshold(),
            "Drift-adaptive (EWMA)": DriftAdaptive(),
            "Isolation Forest": IsolationForestDetector(seed=seed),
            "Hybrid adaptive": HybridAdaptive(seed=seed)}


def run_once(df, train_frac=0.6, seed=0):
    """Chronological split: fit on the first part, evaluate on the rest.

    Features and adaptive baselines use past data only, so predicting over the
    full series gives the test split its history without leaking the future.
    Metrics are only computed when labels exist (is_anomaly != -1).
    """
    feat = add_features(df)
    cut = int(len(feat) * train_frac)
    train, test = feat.iloc[:cut], feat.iloc[cut:]
    labelled = (test["is_anomaly"] >= 0).all()
    rows, preds = {}, {}
    for name, det in make_detectors(seed).items():
        det.fit(train)
        preds[name] = det.predict(feat)
        if labelled:
            rows[name] = metrics(test["is_anomaly"].values, preds[name][cut:])
    table = pd.DataFrame(rows).T if labelled else None
    return table, feat, preds, cut


def run_repeated(n_seeds=10, **gen_kwargs):
    """Repeat over independent synthetic datasets; return mean and std."""
    runs = [run_once(generate_synthetic(seed=s, **gen_kwargs), seed=s)[0]
            for s in range(n_seeds)]
    stack = pd.concat(runs, keys=range(n_seeds))
    return stack.groupby(level=1, sort=False).mean(), stack.groupby(level=1, sort=False).std()
