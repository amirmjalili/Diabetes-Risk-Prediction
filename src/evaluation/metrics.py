"""Comprehensive classification metrics with clinical emphasis.

Clinically prioritized metrics for diabetes risk prediction:
- Sensitivity (Recall): minimize missed true positives (undiagnosed diabetes)
- Specificity: limit unnecessary follow-up testing / anxiety
- PR-AUC: informative under class imbalance
- Brier score: overall probability quality
- ROC-AUC: discrimination
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute a full suite of classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_pred : array-like
        Binary predictions (or will be derived from y_prob if provided).
    y_prob : array-like, optional
        Predicted probabilities for the positive class.
    threshold : float
        Decision threshold applied to y_prob when y_pred is not supplied.

    Returns
    -------
    dict
        Metric name → value.
    """
    y_true = np.asarray(y_true).ravel()
    if y_prob is not None:
        y_prob = np.asarray(y_prob).ravel()
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred).ravel()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),  # sensitivity
        "sensitivity": float(tp / (tp + fn) if (tp + fn) > 0 else 0.0),
        "specificity": float(tn / (tn + fp) if (tn + fp) > 0 else 0.0),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))

    return metrics


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for ROC-AUC.

    Parameters
    ----------
    y_true, y_prob
        Labels and predicted probabilities.
    n_bootstrap : int
        Number of bootstrap resamples.
    alpha : float
        Significance level (0.05 → 95% CI).
    random_state : int
        Seed.

    Returns
    -------
    auc, ci_lower, ci_upper
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    n = len(y_true)

    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))

    aucs = np.array(aucs)
    point = float(roc_auc_score(y_true, y_prob))
    lower = float(np.percentile(aucs, 100 * alpha / 2))
    upper = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return point, lower, upper


def risk_category(prob: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    """Map calibrated probability to a clinical risk category.

    Default thresholds are illustrative and should be tuned to local
    prevalence and cost–benefit trade-offs before any real-world use.
    """
    if thresholds is None:
        thresholds = {"low": 0.20, "moderate": 0.40, "high": 0.60}
    if prob < thresholds["low"]:
        return "Low"
    if prob < thresholds["moderate"]:
        return "Moderate"
    if prob < thresholds["high"]:
        return "High"
    return "Very High"