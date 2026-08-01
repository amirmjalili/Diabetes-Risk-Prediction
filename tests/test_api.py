"""API integration tests using FastAPI TestClient."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "model_loaded" in body


def test_predict_validation_error():
    """Missing required fields should return 422."""
    r = client.post("/predict", json={"Glucose": 120})
    assert r.status_code == 422


def test_predict_out_of_range():
    """Values outside physiological bounds should be rejected by Pydantic."""
    payload = {
        "Pregnancies": 2,
        "Glucose": 999,  # out of range
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_valid_payload_or_503():
    """Valid payload either returns prediction or 503 if model not loaded."""
    payload = {
        "Pregnancies": 2,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        body = r.json()
        assert "calibrated_probability" in body
        assert "risk_category" in body
        assert "disclaimer" in body
        assert 0 <= body["calibrated_probability"] <= 1
