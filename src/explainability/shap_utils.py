"""SHAP-based global and local explanations for tree and linear models.

SHAP values provide a consistent, theoretically grounded attribution of
each feature's contribution to an individual prediction (Lundberg & Lee,
NeurIPS 2017). Global importance is obtained by averaging absolute SHAP
values across the sample.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


def compute_shap_values(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 200,
    random_state: int = 42,
) -> Tuple[Any, np.ndarray]:
    """Compute SHAP values using an appropriate explainer.

    For tree-based models TreeExplainer is preferred (exact, fast).
    Otherwise KernelExplainer is used on a background sample.

    Parameters
    ----------
    model
        Fitted estimator with predict or predict_proba.
    X
        Feature matrix (will be subsampled if larger than max_samples).
    feature_names : list of str, optional
        Column names for reporting.
    max_samples : int
        Maximum number of instances to explain.
    random_state : int
        Seed for subsampling.

    Returns
    -------
    explainer, shap_values
        The SHAP explainer object and the array of SHAP values
        (n_samples × n_features for binary classification, positive class).
    """
    import shap

    rng = np.random.default_rng(random_state)
    if isinstance(X, pd.DataFrame):
        feature_names = feature_names or list(X.columns)
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X_arr.shape[1])]

    n = X_arr.shape[0]
    if n > max_samples:
        idx = rng.choice(n, max_samples, replace=False)
        X_sample = X_arr[idx]
    else:
        X_sample = X_arr

    # Prefer TreeExplainer when possible
    model_type = type(model).__name__.lower()
    tree_models = ("xgb", "lgbm", "catboost", "randomforest", "extratrees", "gradientboosting")
    is_tree = any(t in model_type for t in tree_models)

    # Handle CalibratedClassifierCV wrapper
    base = model
    if hasattr(model, "calibrated_classifiers_"):
        # Take the first underlying estimator
        try:
            base = model.calibrated_classifiers_[0].estimator
            model_type = type(base).__name__.lower()
            is_tree = any(t in model_type for t in tree_models)
        except Exception:
            pass

    try:
        if is_tree:
            explainer = shap.TreeExplainer(base)
            shap_vals = explainer.shap_values(X_sample)
            # Binary classification may return list [neg, pos]
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
        else:
            # KernelExplainer needs a background set and a prediction function
            background = shap.sample(X_arr, min(50, n), random_state=random_state)
            predict_fn = (
                model.predict_proba
                if hasattr(model, "predict_proba")
                else model.predict
            )
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_vals = explainer.shap_values(X_sample)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
    except Exception as e:
        logger.warning(f"SHAP computation failed ({e}); returning zeros.")
        shap_vals = np.zeros_like(X_sample, dtype=float)
        explainer = None

    logger.info(f"SHAP values computed for {X_sample.shape[0]} samples.")
    return explainer, np.asarray(shap_vals)


def global_importance_from_shap(
    shap_values: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """Aggregate absolute SHAP values into a global importance ranking.

    Parameters
    ----------
    shap_values : ndarray of shape (n_samples, n_features)
    feature_names : list of str

    Returns
    -------
    pd.DataFrame
        Sorted by mean |SHAP| descending.
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    df = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False)
    return df.reset_index(drop=True)