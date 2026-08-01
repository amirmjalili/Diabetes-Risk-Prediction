"""Pydantic schemas for request/response validation."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class PatientFeatures(BaseModel):
    """Input features for a single patient risk prediction.

    Units follow the original Pima Indians Diabetes Database conventions:
    - Glucose: plasma glucose concentration (mg/dL)
    - BloodPressure: diastolic blood pressure (mm Hg)
    - SkinThickness: triceps skin fold thickness (mm)
    - Insulin: 2-hour serum insulin (mu U/ml)
    - BMI: body mass index (kg/m²)
    - DiabetesPedigreeFunction: diabetes pedigree function
    - Age: age in years
    - Pregnancies: number of times pregnant
    """

    Pregnancies: float = Field(..., ge=0, le=20, description="Number of pregnancies")
    Glucose: float = Field(..., ge=40, le=400, description="Plasma glucose (mg/dL)")
    BloodPressure: float = Field(
        ..., ge=30, le=180, description="Diastolic blood pressure (mm Hg)"
    )
    SkinThickness: float = Field(
        ..., ge=0, le=100, description="Triceps skin fold thickness (mm)"
    )
    Insulin: float = Field(
        ..., ge=0, le=900, description="2-hour serum insulin (mu U/ml)"
    )
    BMI: float = Field(..., ge=10, le=70, description="Body mass index (kg/m²)")
    DiabetesPedigreeFunction: float = Field(
        ..., ge=0.05, le=3.0, description="Diabetes pedigree function"
    )
    Age: float = Field(..., ge=18, le=100, description="Age in years")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Pregnancies": 2,
                    "Glucose": 148,
                    "BloodPressure": 72,
                    "SkinThickness": 35,
                    "Insulin": 0,
                    "BMI": 33.6,
                    "DiabetesPedigreeFunction": 0.627,
                    "Age": 50,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Structured clinical decision-support output."""

    probability: float = Field(..., description="Raw model probability of diabetes")
    calibrated_probability: float = Field(
        ..., description="Calibrated probability (preferred for clinical use)"
    )
    risk_category: str = Field(
        ..., description="Low / Moderate / High / Very High"
    )
    confidence_note: str = Field(
        ...,
        description="Qualitative statement about prediction confidence / calibration",
    )
    top_contributing_features: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Local feature contributions (SHAP) when available",
    )
    clinical_interpretation: str = Field(
        ...,
        description="Brief narrative interpretation for the clinician",
    )
    disclaimer: str = Field(
        default=(
            "This output is a research decision-support tool only. "
            "It does not constitute a diagnosis or medical advice and must "
            "not replace clinical judgment. Always consult a qualified "
            "healthcare professional."
        )
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool