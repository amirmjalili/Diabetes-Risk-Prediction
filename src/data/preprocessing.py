"""Preprocessing pipeline: imputation, scaling, feature engineering, splitting.

All transformations are fitted exclusively on the training set to prevent
data leakage. The fitted transformers are persisted for consistent inference.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.reproducibility import set_seed


def create_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer clinically meaningful features from raw measurements.

    Engineered features are grounded in established diabetes risk factors
    (ADA Standards of Care, WHO BMI categories, glucose diagnostic thresholds).

    Parameters
    ----------
    df : pd.DataFrame
        Input data with at least Glucose, BMI, Age columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered columns.
    """
    out = df.copy()

    # BMI categories (WHO)
    if "BMI" in out.columns:
        out["BMI_category"] = pd.cut(
            out["BMI"],
            bins=[-np.inf, 18.5, 25, 30, np.inf],
            labels=["underweight", "normal", "overweight", "obese"],
        )
        # Ordinal encoding for models that need numeric
        cat_map = {"underweight": 0, "normal": 1, "overweight": 2, "obese": 3}
        out["BMI_category_ord"] = out["BMI_category"].map(cat_map).astype(float)

    # Age groups (clinically relevant strata)
    if "Age" in out.columns:
        out["Age_group"] = pd.cut(
            out["Age"],
            bins=[-np.inf, 35, 45, 55, np.inf],
            labels=["young", "middle", "late_middle", "senior"],
        )
        age_map = {"young": 0, "middle": 1, "late_middle": 2, "senior": 3}
        out["Age_group_ord"] = out["Age_group"].map(age_map).astype(float)

    # Glucose categories (approximate diagnostic thresholds, mg/dL)
    # Note: fasting vs random distinction not available in Pima
    if "Glucose" in out.columns:
        out["Glucose_category"] = pd.cut(
            out["Glucose"],
            bins=[-np.inf, 100, 126, np.inf],
            labels=["normal", "impaired", "diabetic_range"],
        )
        glu_map = {"normal": 0, "impaired": 1, "diabetic_range": 2}
        out["Glucose_category_ord"] = out["Glucose_category"].map(glu_map).astype(float)

    # Clinically motivated interaction terms
    if "Glucose" in out.columns and "BMI" in out.columns:
        out["Glucose_x_BMI"] = out["Glucose"] * out["BMI"]
    if "Age" in out.columns and "BMI" in out.columns:
        out["Age_x_BMI"] = out["Age"] * out["BMI"]
    if "Glucose" in out.columns and "Age" in out.columns:
        out["Glucose_x_Age"] = out["Glucose"] * out["Age"]

    # Insulin resistance proxy (simple HOMA-like when both available)
    if "Insulin" in out.columns and "Glucose" in out.columns:
        # Avoid division by zero; HOMA-IR ≈ (Glucose * Insulin) / 405
        out["HOMA_IR_proxy"] = (out["Glucose"] * out["Insulin"]) / 405.0

    logger.info(
        f"Feature engineering complete. New columns: "
        f"{[c for c in out.columns if c not in df.columns]}"
    )
    return out


def build_preprocessor(
    numeric_features: List[str],
    imputation_strategy: str = "median",
    scale: bool = True,
) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for numeric clinical features.

    Parameters
    ----------
    numeric_features : list of str
        Names of numeric columns to transform.
    imputation_strategy : str
        Strategy for SimpleImputer ('median', 'mean', 'most_frequent').
    scale : bool
        Whether to apply StandardScaler after imputation.

    Returns
    -------
    ColumnTransformer
    """
    steps = [("imputer", SimpleImputer(strategy=imputation_strategy))]
    if scale:
        steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps)

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipeline, numeric_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def train_val_test_split(
    df: pd.DataFrame,
    target: str = "Outcome",
    test_size: float = 0.20,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified train / validation / test split.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including target.
    target : str
        Target column name.
    test_size : float
        Fraction of data held out as final test set.
    val_size : float
        Fraction of the remaining data used for validation.
    random_state : int
        Random seed.
    stratify : bool
        Whether to stratify on the target.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    set_seed(random_state)
    y = df[target]
    X = df.drop(columns=[target])

    strat = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    # val_size is relative to the temporary (train+val) set
    val_relative = val_size / (1 - test_size)
    strat_temp = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_relative,
        random_state=random_state,
        stratify=strat_temp,
    )

    logger.info(
        f"Split sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )
    logger.info(
        f"Prevalence — train: {y_train.mean():.3f}, val: {y_val.mean():.3f}, "
        f"test: {y_test.mean():.3f}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_pipeline(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    fit: bool = True,
    preprocessor: Optional[ColumnTransformer] = None,
) -> Tuple[pd.DataFrame, Optional[ColumnTransformer], List[str]]:
    """Full preprocessing: feature engineering → imputation → optional scaling.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or partially cleaned data (may contain target).
    config : dict, optional
        Configuration dictionary.
    fit : bool
        If True, fit a new preprocessor. If False, transform with provided one.
    preprocessor : ColumnTransformer, optional
        Pre-fitted transformer (required when fit=False).

    Returns
    -------
    X_processed : pd.DataFrame
        Transformed feature matrix (target excluded).
    preprocessor : ColumnTransformer or None
        Fitted transformer (when fit=True).
    feature_names : list of str
        Final feature names after transformation.
    """
    from src.utils.config import load_config

    if config is None:
        config = load_config()

    target = config["data"]["pima"]["target"]
    has_target = target in df.columns

    # Feature engineering (applied before split in some workflows;
    # for production we engineer then split, or engineer inside CV)
    df_eng = create_clinical_features(df)

    # Select numeric features for modeling (exclude categorical labels & target)
    exclude = {target, "BMI_category", "Age_group", "Glucose_category"}
    numeric_features = [
        c
        for c in df_eng.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df_eng[c])
    ]

    X = df_eng[numeric_features].copy()
    if has_target:
        # Keep target separate; caller handles y
        pass

    scale = config["preprocessing"].get("scaling") == "standard"
    imp_strategy = config["preprocessing"].get("imputation_strategy", "median")

    if fit:
        preprocessor = build_preprocessor(numeric_features, imp_strategy, scale)
        X_proc = preprocessor.fit_transform(X)
    else:
        if preprocessor is None:
            raise ValueError("preprocessor must be provided when fit=False")
        X_proc = preprocessor.transform(X)

    feature_names = list(X_proc.columns) if hasattr(X_proc, "columns") else numeric_features
    logger.info(f"Preprocessed feature matrix shape: {X_proc.shape}")
    return X_proc, preprocessor, feature_names