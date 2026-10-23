"""Metrics for cross-crop evaluation.

Test samples whose label never appeared in training ("unseen") can't be predicted by a closed-set
model, so we report overall numbers (unseen counted as wrong) AND numbers on the seen subset.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, classes: list[str]) -> dict:
    out = {"n": int(len(y_true))}
    seen = y_true >= 0
    out["unseen_fraction"] = float(1 - seen.mean()) if len(y_true) else 0.0
    out["acc_overall"] = float((y_pred == y_true).mean()) if len(y_true) else 0.0
    if seen.any():
        yt, yp = y_true[seen], y_pred[seen]
        labels = np.unique(yt)
        out["acc_seen"] = float(accuracy_score(yt, yp))
        out["macro_f1_seen"] = float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))
    if "healthy" in classes:
        h = classes.index("healthy")
        is_healthy = (y_true == h).astype(int)
        if 0 < is_healthy.sum() < len(is_healthy):
            out["healthy_auroc"] = float(roc_auc_score(is_healthy, probs[:, h]))
    return out
