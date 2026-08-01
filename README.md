# Type 2 Diabetes Risk Prediction — Clinical Decision-Support System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Research and educational decision-support tool only.**  
> This system is **not** a medical device, is **not** FDA-cleared or CE-marked, and must **never** replace physician judgment or clinical guidelines.
> **Author:** Amir Mohammad Jalili, MD  
**Role:** Physician & Clinical AI Portfolio Project
---

## Overview

This repository implements a complete, reproducible machine-learning pipeline for **Type 2 Diabetes (T2D) risk prediction** using publicly available clinical data. The system delivers:

| Output | Description |
|--------|-------------|
| Predicted probability | Raw model output P(Y=1 given X) |
| Calibrated probability | Isotonic / Platt-calibrated probability suitable for risk communication |
| Risk category | Low / Moderate / High / Very High (configurable thresholds) |
| Local explanation | Top SHAP feature contributions for the individual prediction |
| Clinical interpretation | Short narrative aligned with screening considerations |
| Decision-curve analysis | Net-benefit evaluation across threshold probabilities |

The project is designed for GitHub / LinkedIn portfolio demonstration, academic coursework, research collaboration, and graduate-level machine-learning portfolios.

---

## Clinical Motivation

Type 2 diabetes remains a leading cause of morbidity worldwide. Early identification of individuals at elevated risk enables lifestyle intervention, intensified screening, and delayed progression (ADA Standards of Care; Knowler et al., NEJM 2002 — DPP trial).

Machine-learning models can integrate multiple continuous risk factors into a single calibrated probability. When properly validated, calibrated, and explained, such models may support — but never replace — clinical judgment.

**Key design principles**

1. Calibration over pure discrimination — probabilities must be clinically meaningful.
2. Explainability — SHAP values linked to established pathophysiology.
3. Decision-analytic evaluation — Decision Curve Analysis (Vickers & Elkin, 2006).
4. Reproducibility — fixed seeds, pinned dependencies, DVC-ready data versioning, MLflow-ready experiment tracking.
5. Transparent limitations — explicit discussion of population bias, missingness, and transportability.

---

## Dataset

### Primary: Pima Indians Diabetes Database

| Property | Value |
|----------|-------|
| Source | National Institute of Diabetes and Digestive and Kidney Diseases (via UCI / Kaggle) |
| Population | Female patients of Pima Indian heritage, age >= 21 years |
| Samples | 768 |
| Features | 8 clinical measurements + binary Outcome |
| Prevalence | approximately 35 percent positive |
| Reference | Smith et al., Proc. Symp. Computer Applications and Medical Care, 1988 |

**Features**

- Pregnancies — number of times pregnant
- Glucose — plasma glucose concentration (mg/dL)
- BloodPressure — diastolic blood pressure (mm Hg)
- SkinThickness — triceps skin-fold thickness (mm)
- Insulin — 2-hour serum insulin (uU/mL)
- BMI — body mass index (kg/m2)
- DiabetesPedigreeFunction — diabetes pedigree function
- Age — age (years)
- Outcome — 0 = no diabetes, 1 = diabetes

**Known data-quality issues (handled explicitly)**

- Physiologically impossible zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI treated as missing and median-imputed.
- Moderate class imbalance (approximately 1.9 : 1).
- Limited demographic diversity (single ancestry, female only) — external transportability is a major limitation.

### Secondary / External (optional)

Early Stage Diabetes Risk Prediction Dataset (UCI) can be used for transportability experiments after feature harmonization. Code stubs are included under `src/data/loader.py`.

---

## Repository Structure

```
diabetes-risk-prediction/
├── configs/                  # YAML configuration
├── data/
│   ├── raw/                  # Original CSVs (DVC-tracked)
│   ├── processed/            # Cleaned / engineered tables
│   └── external/             # External validation sets
├── notebooks/                # Exploratory and publication figures
├── src/
│   ├── data/                 # Loading, quality, preprocessing
│   ├── features/             # Clinical feature engineering
│   ├── models/               # Training, Optuna, calibration
│   ├── evaluation/           # Metrics, DCA, statistical tests
│   ├── explainability/       # SHAP utilities
│   ├── api/                  # FastAPI application
│   └── utils/                # Config, logging, seeds
├── models/                   # Serialized artifacts (.joblib)
├── reports/
│   ├── figures/              # Publication-quality plots
│   └── metrics/              # JSON / CSV evaluation outputs
├── scripts/                  # End-to-end training pipeline
├── tests/                    # Unit and API tests
├── .github/workflows/        # CI (Black, Ruff, pytest, Docker)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Installation

```bash
git clone https://github.com/amirmjalili/diabetes-risk-prediction.git
cd diabetes-risk-prediction

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

---

## Quick Start — Train and Serve

```bash
# 1. Train the full pipeline (downloads Pima data if needed)
python scripts/train_pipeline.py --n-trials 30

# 2. Launch the API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# 3. Query (example)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 2,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
  }'
```

Interactive documentation: http://localhost:8000/docs

---

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| 1 | Data loading and rigorous quality assessment |
| 2 | Clinically motivated feature engineering |
| 3 | Stratified train / validation / test split |
| 4 | Baseline comparison of 8 algorithms |
| 5 | Optuna hyperparameter optimization |
| 6 | Probability calibration on validation set |
| 7 | Held-out test evaluation + bootstrap AUC CI + Decision Curve Analysis |
| 8 | Artifact persistence |

---

## Model Comparison and Selection

All models are trained under identical conditions (fixed seed, class weighting where applicable). Primary selection criteria:

1. ROC-AUC (discrimination)
2. PR-AUC (imbalance-aware)
3. Brier score (calibration-sensitive)
4. Sensitivity at clinically relevant thresholds
5. Net benefit on decision curve

Default primary model after nested evaluation: **XGBoost**. Logistic Regression remains the interpretable baseline.

Supported algorithms: Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost, SVM.

---

## Calibration

Raw probabilities from boosting models are frequently miscalibrated. We evaluate reliability diagrams, Brier score before/after calibration, and compare Platt scaling versus Isotonic regression. Isotonic regression is the default when validation sample size permits.

---

## Explainability

- Global: mean absolute SHAP ranking, summary beeswarm plots
- Local: per-prediction SHAP values returned by the API
- Feature attributions are interpreted in light of established diabetes pathophysiology

---

## Decision Curve Analysis

Net benefit is computed across threshold probabilities. The model is compared against "treat-all" and "treat-none" strategies. Positive net benefit relative to both defaults indicates potential clinical utility within the studied population and threshold range.

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | Liveness and model-loaded status |
| /predict | POST | Single-patient prediction + explanation |
| /docs | GET | OpenAPI / Swagger UI |

Input validation is enforced by Pydantic (physiological ranges). All responses include an explicit disclaimer.

---

## Docker

```bash
docker compose up --build

# Or
docker build -t diabetes-risk-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models:ro diabetes-risk-api
```

---

## Testing and CI

```bash
pytest tests/ -v --cov=src
black --check src tests scripts
ruff check src tests scripts
```

GitHub Actions runs formatting, linting, tests (Python 3.10 and 3.11), and a Docker build on every push/PR.

---

## Reproducibility

- Global random seed (config.yaml, project.random_seed = 42)
- Pinned dependency versions in requirements.txt
- Deterministic preprocessing (median imputation fitted on train only)
- Stratified splits with fixed seed
- Optuna TPE sampler seeded

---

## Limitations and Ethical Considerations

1. Population bias — Pima dataset is restricted to female patients of a single ancestry; transportability to other populations is unproven.
2. Sample size — 768 observations limit the reliability of complex models and calibration curves.
3. Missingness mechanism — zeros treated as missing is a pragmatic but imperfect assumption.
4. No longitudinal outcomes — the target is cross-sectional diagnosis, not future incidence.
5. Fairness — gender and ancestry are fixed by design; subgroup performance cannot be assessed.
6. Not a medical device — regulatory clearance, prospective validation, and human-factors evaluation are required before any clinical deployment.

Always consult current ADA / EASD / local guidelines and a qualified clinician.

---

## Future Work

- Multi-center external validation
- Integration of additional risk factors (HbA1c, family history, physical activity)
- Prospective silent evaluation in electronic health-record workflows
- Fairness auditing on more diverse cohorts
- Full DVC pipeline + remote storage
- Continuous calibration monitoring in production

---

## References

1. American Diabetes Association. Standards of Care in Diabetes—2024. Diabetes Care.
2. Smith JW et al. Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. Proc. Symp. Computer Applications and Medical Care, 1988.
3. Vickers AJ, Elkin EB. Decision curve analysis. Med Decis Making. 2006;26(6):565-574.
4. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. NeurIPS 2017.
5. Knowler WC et al. Reduction in the incidence of type 2 diabetes with lifestyle intervention or metformin. N Engl J Med. 2002;346:393-403.
6. DeFronzo RA. From the Triumvirate to the Ominous Octet. Diabetes. 2009.
7. Van Calster B et al. Calibration: the Achilles heel of predictive analytics. BMC Med. 2019.
8. Collins GS et al. TRIPOD+AI statement. BMJ 2024.

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Citation

If you use this repository in academic work, please cite:

```
Amir Mohammad Jalili, MD. (2026). Type 2 Diabetes Risk Prediction —
Clinical Decision-Support System (v1.0.0). GitHub repository.
https://github.com/amirmjalili/diabetes-risk-prediction
```

---

**Disclaimer**: This software is provided for research and educational purposes only. It is not intended for clinical diagnosis, treatment, or any use that could affect patient care without independent validation and regulatory approval.
