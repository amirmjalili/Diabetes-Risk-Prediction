"""Unit tests for data loading and quality assessment."""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_pima_dataset, PIMA_COLUMNS
from src.data.quality import assess_data_quality
from src.data.preprocessing import create_clinical_features, train_val_test_split


@pytest.fixture
def sample_df():
    """Minimal valid DataFrame mimicking Pima structure."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "Pregnancies": rng.integers(0, 10, n),
            "Glucose": rng.normal(120, 25, n).clip(60, 200),
            "BloodPressure": rng.normal(70, 10, n).clip(40, 110),
            "SkinThickness": rng.normal(25, 8, n).clip(5, 50),
            "Insulin": rng.normal(100, 50, n).clip(10, 400),
            "BMI": rng.normal(30, 6, n).clip(18, 50),
            "DiabetesPedigreeFunction": rng.uniform(0.1, 1.5, n),
            "Age": rng.integers(21, 65, n),
            "Outcome": rng.integers(0, 2, n),
        }
    )


def test_pima_columns_defined():
    assert len(PIMA_COLUMNS) == 9
    assert "Outcome" in PIMA_COLUMNS


def test_load_pima_creates_file(tmp_path, monkeypatch):
    """Loader should create or find a CSV under the given directory."""
    # Force download path to tmp
    from src.data import loader

    path = loader.download_pima_if_needed(raw_dir=tmp_path)
    assert path.exists()
    df = pd.read_csv(path)
    assert set(PIMA_COLUMNS).issubset(df.columns) or len(df.columns) == 9


def test_quality_assessment(sample_df):
    report = assess_data_quality(sample_df, target="Outcome")
    assert report["n_rows"] == 100
    assert "prevalence" in report
    assert "feature_statistics" in report
    assert report["n_duplicate_rows"] >= 0


def test_clinical_features(sample_df):
    eng = create_clinical_features(sample_df)
    assert "BMI_category_ord" in eng.columns
    assert "Glucose_x_BMI" in eng.columns
    assert "HOMA_IR_proxy" in eng.columns


def test_stratified_split(sample_df):
    eng = create_clinical_features(sample_df)
    exclude = {"Outcome", "BMI_category", "Age_group", "Glucose_category"}
    feats = [c for c in eng.columns if c not in exclude]
    data = eng[feats + ["Outcome"]].fillna(eng[feats].median())
    X_tr, X_va, X_te, y_tr, y_va, y_te = train_val_test_split(
        data, target="Outcome", random_state=42
    )
    assert len(X_tr) + len(X_va) + len(X_te) == len(data)
    # Prevalence roughly preserved
    assert abs(y_tr.mean() - data["Outcome"].mean()) < 0.15