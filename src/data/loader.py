"""Dataset loading utilities for Pima Indians Diabetes Database and related sources.

The Pima Indians Diabetes Database is a classic, publicly available clinical
dataset from the National Institute of Diabetes and Digestive and Kidney
Diseases. It contains diagnostic measurements for female patients of Pima
Indian heritage aged 21 years and older.

Reference:
    Smith, J.W., et al. (1988). Using the ADAP learning algorithm to forecast
    the onset of diabetes mellitus. Proceedings of the Symposium on Computer
    Applications and Medical Care, 261-265.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from src.utils.config import get_project_root, load_config


# Canonical column order matching the original UCI / Kaggle release
PIMA_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


def download_pima_if_needed(raw_dir: Optional[Path] = None) -> Path:
    """Ensure the Pima dataset exists locally.

    Attempts to download from a public mirror if the file is missing.
    Falls back to creating a minimal synthetic version only when download
    is impossible (offline environments), with a clear warning.

    Parameters
    ----------
    raw_dir : Path, optional
        Directory to store the raw CSV. Defaults to data/raw under project root.

    Returns
    -------
    Path
        Path to the diabetes.csv file.
    """
    if raw_dir is None:
        config = load_config()
        raw_dir = Path(config["paths"]["raw_data"])
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "diabetes.csv"

    if csv_path.exists():
        logger.info(f"Pima dataset already present at {csv_path}")
        return csv_path

    # Public raw URL (Kaggle mirror / UCI alternative via raw.githubusercontent)
    url = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    )
    try:
        logger.info(f"Downloading Pima dataset from {url}")
        df = pd.read_csv(url, header=None, names=PIMA_COLUMNS)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved Pima dataset to {csv_path} ({len(df)} rows)")
        return csv_path
    except Exception as e:
        logger.warning(
            f"Could not download Pima dataset ({e}). "
            "Creating a small synthetic placeholder for offline development only."
        )
        # Minimal synthetic data that preserves column names and rough ranges
        # This is NEVER intended for scientific conclusions.
        import numpy as np

        rng = np.random.default_rng(42)
        n = 200
        synthetic = pd.DataFrame(
            {
                "Pregnancies": rng.integers(0, 15, n),
                "Glucose": rng.normal(120, 30, n).clip(50, 200).astype(int),
                "BloodPressure": rng.normal(70, 12, n).clip(40, 120).astype(int),
                "SkinThickness": rng.normal(25, 10, n).clip(5, 60).astype(int),
                "Insulin": rng.normal(100, 80, n).clip(10, 600).astype(int),
                "BMI": rng.normal(32, 7, n).clip(18, 55).round(1),
                "DiabetesPedigreeFunction": rng.uniform(0.1, 2.0, n).round(3),
                "Age": rng.integers(21, 70, n),
                "Outcome": rng.integers(0, 2, n),
            }
        )
        synthetic.to_csv(csv_path, index=False)
        logger.warning(
            "Synthetic placeholder written. Replace with real data before any analysis."
        )
        return csv_path


def load_pima_dataset(
    path: Optional[str | Path] = None,
    treat_zero_as_missing: bool = True,
) -> pd.DataFrame:
    """Load the Pima Indians Diabetes dataset into a clean DataFrame.

    Parameters
    ----------
    path : str or Path, optional
        Path to diabetes.csv. If None, uses project data/raw location
        and downloads if necessary.
    treat_zero_as_missing : bool
        If True, replace physiologically impossible zeros with NaN for
        Glucose, BloodPressure, SkinThickness, Insulin, and BMI.

    Returns
    -------
    pd.DataFrame
        DataFrame with standard column names and optional NaN replacement.
    """
    if path is None:
        path = download_pima_if_needed()
    else:
        path = Path(path)

    df = pd.read_csv(path)
    # Ensure expected columns
    missing_cols = set(PIMA_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    df = df[PIMA_COLUMNS].copy()

    if treat_zero_as_missing:
        zero_as_missing = [
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
        ]
        for col in zero_as_missing:
            if col in df.columns:
                n_zeros = (df[col] == 0).sum()
                if n_zeros > 0:
                    logger.info(
                        f"Replacing {n_zeros} zero values in '{col}' with NaN "
                        "(physiologically implausible)."
                    )
                    df[col] = df[col].replace(0, pd.NA)

    logger.info(
        f"Loaded Pima dataset: {df.shape[0]} rows, {df.shape[1]} columns, "
        f"target prevalence = {df['Outcome'].mean():.3f}"
    )
    return df


def load_early_stage_dataset(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load Early Stage Diabetes Risk Prediction dataset (UCI).

    This dataset is used for external validation / transportability assessment.
    Feature sets differ substantially from Pima; harmonization is required
    before joint modeling.

    Parameters
    ----------
    path : str or Path, optional
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    if path is None:
        config = load_config()
        path = Path(config["paths"]["external_data"]) / "early_stage_diabetes.csv"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Early-stage dataset not found at {path}. "
            "Download from UCI ML Repository and place the file manually."
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded early-stage dataset: {df.shape}")
    return df