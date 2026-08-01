# Software Architecture

## Design Goals

- Clean separation of concerns (data → features → models → evaluation → serving)
- Reproducibility (seeds, config-driven, version-pinned dependencies)
- Clinical interpretability (calibration, SHAP, decision curves, narrative output)
- Production readiness (FastAPI, Docker, CI, typed schemas, health checks)

## Layer Overview

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI Layer                        │
│  /health  /predict  OpenAPI  Pydantic validation         │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│              Explainability + Evaluation                 │
│  SHAP  Decision Curve  Metrics  Statistical Tests        │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                    Model Layer                           │
│  Train  Optuna  Calibration  Persist (joblib / MLflow)   │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│              Feature Engineering                         │
│  Clinical categories  Interactions  HOMA-IR proxy        │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│              Data Layer                                  │
│  Loader  Quality Assessment  Preprocessing  Split        │
└─────────────────────────────────────────────────────────┘
```

## Configuration

All tunable parameters live in `configs/config.yaml`. Paths, seeds, model candidates, Optuna settings, clinical thresholds, and API metadata are centralized so experiments remain reproducible and auditable.

## Artifact Contract

After `scripts/train_pipeline.py` succeeds, the following artifacts exist under `models/`:

| File | Content |
|------|---------|
| best_model.joblib | Calibrated classifier (CalibratedClassifierCV) |
| feature_names.joblib | Ordered list of feature names expected at inference |
| preprocessor.joblib | Median values + feature list (lightweight) |

The API loads these at startup. If missing, `/predict` returns HTTP 503.

## Extensibility

- Add a new algorithm: extend `get_model_factory` / `instantiate_model` and optionally a search space in `optimize.py`.
- Add external validation: implement harmonization in `src/data/loader.py` and a separate evaluation script.
- Swap calibration method via config (`platt` → sigmoid, `isotonic`).
- Enable MLflow by wrapping the training loop with `mlflow.start_run()`.