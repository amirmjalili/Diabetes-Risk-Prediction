"""Hyperparameter optimization with Optuna.

Primary objective: maximize ROC-AUC on stratified cross-validation.
Secondary metrics (PR-AUC, Brier score) are logged for clinical review.
"""

from typing import Any, Dict, Optional

import numpy as np
import optuna
from loguru import logger
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from src.models.train import instantiate_model
from src.utils.reproducibility import set_seed


def _xgboost_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }


def _lightgbm_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
    }


def _catboost_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "depth": trial.suggest_int("depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 128),
    }


def _rf_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


def _logistic_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l2"]),
    }


SEARCH_SPACES = {
    "xgboost": _xgboost_search_space,
    "lightgbm": _lightgbm_search_space,
    "catboost": _catboost_search_space,
    "random_forest": _rf_search_space,
    "logistic_regression": _logistic_search_space,
}


def run_optuna_optimization(
    X: np.ndarray | Any,
    y: np.ndarray | Any,
    model_name: str = "xgboost",
    n_trials: int = 50,
    n_splits: int = 5,
    n_repeats: int = 2,
    random_state: int = 42,
    metric: str = "roc_auc",
    timeout: Optional[int] = None,
) -> optuna.Study:
    """Run Optuna hyperparameter search with repeated stratified K-fold CV.

    Parameters
    ----------
    X, y
        Feature matrix and target vector.
    model_name : str
        Model identifier (must have a search space defined).
    n_trials : int
        Number of Optuna trials.
    n_splits, n_repeats : int
        RepeatedStratifiedKFold configuration.
    random_state : int
        Seed.
    metric : str
        Sklearn scoring string (default 'roc_auc').
    timeout : int, optional
        Maximum optimization time in seconds.

    Returns
    -------
    optuna.Study
        Completed study object with best_params and best_value.
    """
    if model_name not in SEARCH_SPACES:
        raise ValueError(
            f"No search space defined for '{model_name}'. "
            f"Available: {list(SEARCH_SPACES.keys())}"
        )

    set_seed(random_state)
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    def objective(trial: optuna.Trial) -> float:
        params = SEARCH_SPACES[model_name](trial)
        model = instantiate_model(model_name, random_state=random_state, hyperparams=params)
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        study_name=f"diabetes_{model_name}",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    logger.info(f"Starting Optuna optimization for {model_name} ({n_trials} trials)...")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    logger.info(f"Best {metric}: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    return study