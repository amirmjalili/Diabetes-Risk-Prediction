"""Production-oriented FastAPI application for diabetes risk prediction.

Endpoints
---------
GET  /health          - Liveness / readiness
POST /predict         - Single-patient risk prediction with explanation
GET  /docs            - OpenAPI / Swagger UI (automatic)
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.schemas import HealthResponse, PatientFeatures, PredictionResponse
from src.evaluation.metrics import risk_category
from src.utils.config import get_project_root, load_config
from src.utils.logging import setup_logging

# ---------------------------------------------------------------------------
# Global state (loaded at startup)
# ---------------------------------------------------------------------------
MODEL: Any = None
PREPROCESSOR: Any = None
FEATURE_NAMES: List[str] = []
CONFIG: Dict[str, Any] = {}
VERSION = "1.0.0"


def _load_artifacts() -> None:
    """Load model, preprocessor, and feature list from disk."""
    global MODEL, PREPROCESSOR, FEATURE_NAMES, CONFIG
    CONFIG = load_config()
    models_dir = Path(CONFIG["paths"]["models"])
    root = get_project_root()

    model_path = models_dir / "best_model.joblib"
    prep_path = models_dir / "preprocessor.joblib"
    feat_path = models_dir / "feature_names.joblib"

    if not model_path.exists():
        logger.warning(
            f"Model artifact not found at {model_path}. "
            "API will start but /predict will return 503 until a model is trained."
        )
        return

    MODEL = joblib.load(model_path)
    if prep_path.exists():
        PREPROCESSOR = joblib.load(prep_path)
    if feat_path.exists():
        FEATURE_NAMES = joblib.load(feat_path)
    logger.info(f"Loaded model from {model_path}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Starting Diabetes Risk Prediction API...")
    _load_artifacts()
    yield
    logger.info("Shutting down API.")


app = FastAPI(
    title="Diabetes Risk Prediction API",
    description=(
        "Clinical decision-support API for Type 2 Diabetes risk stratification. "
        "**Not a medical device.** Intended for research and educational use only."
    ),
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Health / readiness probe."""
    return HealthResponse(
        status="ok",
        version=VERSION,
        model_loaded=MODEL is not None,
    )


def _prepare_features(patient: PatientFeatures) -> pd.DataFrame:
    """Convert validated patient input into a model-ready feature row.

    Applies the same clinical feature engineering used during training.
    """
    from src.data.preprocessing import create_clinical_features

    raw = pd.DataFrame([patient.model_dump()])
    engineered = create_clinical_features(raw)

    # Select columns expected by the preprocessor / model
    if FEATURE_NAMES:
        # Ensure all expected columns exist (fill missing engineered ones)
        for col in FEATURE_NAMES:
            if col not in engineered.columns:
                engineered[col] = np.nan
        X = engineered[FEATURE_NAMES]
    else:
        # Fallback: original 8 clinical variables
        base_cols = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ]
        X = engineered[base_cols]

    if PREPROCESSOR is not None:
        X = PREPROCESSOR.transform(X)

    return X


def _local_shap_explanation(X: pd.DataFrame | np.ndarray, top_k: int = 5) -> List[Dict]:
    """Compute a lightweight local explanation if SHAP is available."""
    try:
        import shap

        # Prefer TreeExplainer on the underlying model
        base = MODEL
        if hasattr(MODEL, "calibrated_classifiers_"):
            base = MODEL.calibrated_classifiers_[0].estimator

        explainer = shap.TreeExplainer(base)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_vals = np.asarray(shap_vals).ravel()

        names = FEATURE_NAMES or [f"f{i}" for i in range(len(shap_vals))]
        pairs = sorted(
            zip(names, shap_vals),
            key=lambda t: abs(t[1]),
            reverse=True,
        )[:top_k]
        return [{"feature": n, "shap_value": round(float(v), 4)} for n, v in pairs]
    except Exception as e:
        logger.debug(f"Local SHAP unavailable: {e}")
        return []


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(patient: PatientFeatures) -> PredictionResponse:
    """Predict Type 2 Diabetes risk for a single patient.

    Returns calibrated probability, risk category, top contributing features,
    and a short clinical interpretation. **This is decision support only.**
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train and save artifacts before requesting predictions.",
        )

    try:
        X = _prepare_features(patient)
        proba_raw = float(MODEL.predict_proba(X)[0, 1])

        # If the model is already a CalibratedClassifierCV, proba is calibrated
        calibrated = proba_raw
        conf_note = (
            "Probability is produced by a calibrated model (isotonic / Platt). "
            "Calibration quality depends on the validation distribution."
        )

        category = risk_category(
            calibrated,
            thresholds=CONFIG.get("evaluation", {}).get("clinical_thresholds"),
        )

        top_feats = _local_shap_explanation(X)

        # Narrative interpretation
        interpretation = (
            f"Estimated risk category: **{category}** "
            f"(calibrated probability ≈ {calibrated:.1%}). "
        )
        if category in ("High", "Very High"):
            interpretation += (
                "Elevated predicted risk. Consider confirmatory laboratory testing "
                "(fasting glucose, HbA1c) and lifestyle assessment per local guidelines. "
            )
        elif category == "Moderate":
            interpretation += (
                "Intermediate risk. Shared decision-making regarding screening intensity "
                "and lifestyle counseling may be appropriate. "
            )
        else:
            interpretation += (
                "Lower predicted risk relative to the training population. "
                "Routine preventive counseling remains indicated. "
            )
        interpretation += (
            "Individual clinical context (symptoms, family history, comorbidities) "
            "must always guide management."
        )

        return PredictionResponse(
            probability=round(proba_raw, 4),
            calibrated_probability=round(calibrated, 4),
            risk_category=category,
            confidence_note=conf_note,
            top_contributing_features=top_feats,
            clinical_interpretation=interpretation,
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}") from e


# Allow running with: uvicorn src.api.app:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)