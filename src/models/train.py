"""Model training utilities for multiple classical and boosting algorithms.

All models are trained with fixed random seeds and, where applicable,
class_weight='balanced' to address the moderate class imbalance typical
of the Pima dataset (~35% positive).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.utils.reproducibility import set_seed


def get_model_factory(random_state: int = 42) -> Dict[str, Any]:
    """Return a dictionary of uninitialized model constructors with defaults.

    Parameters
    ----------
    random_state : int
        Seed passed to stochastic estimators.

    Returns
    -------
    dict
        Mapping from short name → (estimator class, default kwargs).
    """
    return {
        "logistic_regression": (
            LogisticRegression,
            {
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": random_state,
                "solver": "lbfgs",
            },
        ),
        "random_forest": (
            RandomForestClassifier,
            {
                "n_estimators": 200,
                "class_weight": "balanced",
                "random_state": random_state,
                "n_jobs": -1,
            },
        ),
        "extra_trees": (
            ExtraTreesClassifier,
            {
                "n_estimators": 200,
                "class_weight": "balanced",
                "random_state": random_state,
                "n_jobs": -1,
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier,
            {
                "n_estimators": 150,
                "random_state": random_state,
            },
        ),
        "xgboost": None,  # handled specially to avoid hard dependency at import
        "lightgbm": None,
        "catboost": None,
        "svm": (
            SVC,
            {
                "probability": True,
                "class_weight": "balanced",
                "random_state": random_state,
                "kernel": "rbf",
            },
        ),
    }


def _build_xgboost(random_state: int = 42, **kwargs):
    from xgboost import XGBClassifier

    defaults = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "n_jobs": -1,
    }
    defaults.update(kwargs)
    # scale_pos_weight can be set by caller for imbalance
    return XGBClassifier(**defaults)


def _build_lightgbm(random_state: int = 42, **kwargs):
    from lightgbm import LGBMClassifier

    defaults = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "class_weight": "balanced",
        "n_jobs": -1,
        "verbose": -1,
    }
    defaults.update(kwargs)
    return LGBMClassifier(**defaults)


def _build_catboost(random_state: int = 42, **kwargs):
    from catboost import CatBoostClassifier

    defaults = {
        "iterations": 200,
        "learning_rate": 0.05,
        "depth": 4,
        "random_seed": random_state,
        "auto_class_weights": "Balanced",
        "verbose": 0,
        "allow_writing_files": False,
    }
    defaults.update(kwargs)
    return CatBoostClassifier(**defaults)


def instantiate_model(
    name: str,
    random_state: int = 42,
    hyperparams: Optional[Dict[str, Any]] = None,
) -> Any:
    """Instantiate a model by short name with optional hyperparameter overrides.

    Parameters
    ----------
    name : str
        One of the keys in get_model_factory().
    random_state : int
        Random seed.
    hyperparams : dict, optional
        Overrides for default constructor arguments.

    Returns
    -------
    estimator
        Unfitted scikit-learn compatible estimator.
    """
    hyperparams = hyperparams or {}
    factory = get_model_factory(random_state)

    if name == "xgboost":
        return _build_xgboost(random_state, **hyperparams)
    if name == "lightgbm":
        return _build_lightgbm(random_state, **hyperparams)
    if name == "catboost":
        return _build_catboost(random_state, **hyperparams)

    if name not in factory or factory[name] is None:
        raise ValueError(f"Unknown model: {name}")

    cls, defaults = factory[name]
    params = {**defaults, **hyperparams}
    return cls(**params)


def train_all_models(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    model_names: Optional[List[str]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train a suite of models with default hyperparameters.

    Parameters
    ----------
    X_train, y_train
        Training features and labels.
    model_names : list of str, optional
        Subset of models to train. Defaults to all supported models.
    random_state : int
        Seed.

    Returns
    -------
    dict
        Mapping model_name → fitted estimator.
    """
    set_seed(random_state)
    if model_names is None:
        model_names = [
            "logistic_regression",
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "xgboost",
            "lightgbm",
            "catboost",
            "svm",
        ]

    fitted = {}
    for name in model_names:
        logger.info(f"Training {name} ...")
        try:
            model = instantiate_model(name, random_state=random_state)
            model.fit(X_train, y_train)
            fitted[name] = model
            logger.info(f"  {name} trained successfully.")
        except Exception as e:
            logger.error(f"  Failed to train {name}: {e}")
    return fitted


def main():
    """CLI entry point for basic training smoke test."""
    from src.data.loader import load_pima_dataset
    from src.data.preprocessing import create_clinical_features, train_val_test_split
    from src.utils.config import load_config
    from src.utils.logging import setup_logging

    setup_logging()
    config = load_config()
    set_seed(config["project"]["random_seed"])

    df = load_pima_dataset()
    df = create_clinical_features(df)
    # Simple numeric selection for smoke test
    target = "Outcome"
    exclude = {target, "BMI_category", "Age_group", "Glucose_category"}
    features = [c for c in df.columns if c not in exclude]
    X = df[features].fillna(df[features].median())
    y = df[target]

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        pd.concat([X, y], axis=1),
        target=target,
        random_state=config["project"]["random_seed"],
    )
    models = train_all_models(X_train, y_train, random_state=config["project"]["random_seed"])
    logger.info(f"Trained models: {list(models.keys())}")


if __name__ == "__main__":
    main()