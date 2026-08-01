"""Decision Curve Analysis (DCA) for clinical utility assessment.

Reference:
    Vickers AJ, Elkin EB. Decision curve analysis: a novel method for
    evaluating prediction models. Med Decis Making. 2006;26(6):565-574.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


def net_benefit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> float:
    """Compute net benefit at a given threshold probability.

    Net Benefit = (TP / N) - (FP / N) * (pt / (1 - pt))

    where pt is the threshold probability reflecting the harm:benefit ratio.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    n = len(y_true)
    if n == 0:
        return 0.0

    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))

    if threshold >= 1.0:
        return 0.0
    nb = (tp / n) - (fp / n) * (threshold / (1.0 - threshold))
    return float(nb)


def decision_curve_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Compute net benefit across a range of threshold probabilities.

    Also returns the "treat all" and "treat none" reference strategies.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes.
    y_prob : array-like
        Predicted probabilities.
    thresholds : list of float, optional
        Threshold probabilities to evaluate.

    Returns
    -------
    pd.DataFrame
        Columns: threshold, net_benefit_model, net_benefit_all, net_benefit_none
    """
    if thresholds is None:
        thresholds = list(np.arange(0.01, 0.80, 0.01))

    y_true = np.asarray(y_true).ravel()
    prevalence = float(np.mean(y_true))
    n = len(y_true)

    rows = []
    for pt in thresholds:
        nb_model = net_benefit(y_true, y_prob, pt)
        # Treat-all: assume everyone is positive
        nb_all = prevalence - (1 - prevalence) * (pt / (1 - pt)) if pt < 1 else 0.0
        nb_none = 0.0
        rows.append(
            {
                "threshold": pt,
                "net_benefit_model": nb_model,
                "net_benefit_all": nb_all,
                "net_benefit_none": nb_none,
            }
        )

    df = pd.DataFrame(rows)
    logger.info(
        f"Decision curve computed for {len(thresholds)} thresholds "
        f"(prevalence={prevalence:.3f})"
    )
    return df