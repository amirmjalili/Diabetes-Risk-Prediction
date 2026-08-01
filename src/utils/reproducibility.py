"""Reproducibility utilities: fixed random seeds across libraries."""

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and common ML libraries.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Attempt to set seeds for optional libraries
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier  # noqa: F401

        # CatBoost uses random_seed parameter at model init
    except ImportError:
        pass

    # XGBoost / LightGBM respect numpy seed + their own random_state params
    # which are set explicitly in model training code.