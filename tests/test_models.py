"""Unit tests for model instantiation and basic training."""

import numpy as np
import pandas as pd
import pytest

from src.models.train import instantiate_model, train_all_models, get_model_factory
from src.models.calibrate import calibrate_model, evaluate_calibration
from src.evaluation.metrics import compute_classification_metrics, risk_category


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    n = 150
    X = pd.DataFrame(
        {
            "Glucose": rng.normal(120, 30, n),
            "BMI": rng.normal(30, 6, n),
            "Age": rng.integers(25, 70, n),
            "Insulin": rng.normal(100, 40, n),
        }
    )
    # Simple rule-based label with noise
    y = ((X["Glucose"] > 130) | (X["BMI"] > 35)).astype(int)
    y = y.where(rng.random(n) > 0.15, 1 - y)  # flip 15%
    return X, y


def test_model_factory():
    factory = get_model_factory(42)
    assert "logistic_regression" in factory
    assert "random_forest" in factory


def test_instantiate_logistic(toy_data):
    X, y = toy_data
    model = instantiate_model("logistic_regression", random_state=42)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))


def test_train_subset(toy_data):
    X, y = toy_data
    models = train_all_models(
        X, y, model_names=["logistic_regression", "random_forest"], random_state=0
    )
    assert "logistic_regression" in models
    assert "random_forest" in models


def test_calibration(toy_data):
    X, y = toy_data
    model = instantiate_model("logistic_regression", random_state=0)
    model.fit(X.iloc[:100], y.iloc[:100])
    cal = calibrate_model(model, X.iloc[100:], y.iloc[100:], method="sigmoid")
    proba = cal.predict_proba(X.iloc[100:])[:, 1]
    frac, mean_pred, brier = evaluate_calibration(y.iloc[100:].values, proba)
    assert 0 <= brier <= 1


def test_metrics(toy_data):
    X, y = toy_data
    model = instantiate_model("logistic_regression", random_state=0)
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    m = compute_classification_metrics(y.values, pred, proba)
    assert "roc_auc" in m
    assert "sensitivity" in m
    assert "specificity" in m
    assert 0 <= m["roc_auc"] <= 1


def test_risk_category():
    assert risk_category(0.10) == "Low"
    assert risk_category(0.30) == "Moderate"
    assert risk_category(0.50) == "High"
    assert risk_category(0.80) == "Very High"