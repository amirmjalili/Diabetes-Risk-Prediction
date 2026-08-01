"""Probability calibration using Platt scaling and Isotonic regression.

Well-calibrated probabilities are essential for clinical decision-support:
risk thresholds, net-benefit calculations, and patient communication all
depend on the numerical meaning of the predicted probability.
"""

from typing import Any, Literal, Optional, Tuple

import numpy as np
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


def calibrate_model(
    estimator: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: Literal["sigmoid", "isotonic"] = "isotonic",
    cv: str = "prefit",
) -> CalibratedClassifierCV:
    """Calibrate a pre-fitted classifier on a held-out validation set.

    Parameters
    ----------
    estimator
        Already fitted classifier that implements predict_proba.
    X_val, y_val
        Validation features and labels used exclusively for calibration.
    method : {'sigmoid', 'isotonic'}
        'sigmoid' = Platt scaling (logistic); 'isotonic' = isotonic regression.
    cv : str
        Must be 'prefit' when the base estimator is already fitted.

    Returns
    -------
    CalibratedClassifierCV
        Calibrated estimator.
    """
    calibrator = CalibratedClassifierCV(estimator, method=method, cv=cv)
    calibrator.fit(X_val, y_val)

    # Quick diagnostics
    proba_raw = estimator.predict_proba(X_val)[:, 1]
    proba_cal = calibrator.predict_proba(X_val)[:, 1]
    brier_raw = brier_score_loss(y_val, proba_raw)
    brier_cal = brier_score_loss(y_val, proba_cal)

    logger.info(
        f"Calibration ({method}): Brier raw={brier_raw:.4f} → calibrated={brier_cal:.4f}"
    )
    return calibrator


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute reliability diagram data and Brier score.

    Parameters
    ----------
    y_true : array-like
        Binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    n_bins : int
        Number of bins for the calibration curve.

    Returns
    -------
    fraction_of_positives, mean_predicted_value, brier
    """
    from sklearn.calibration import calibration_curve

    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    brier = brier_score_loss(y_true, y_prob)
    return frac_pos, mean_pred, brier