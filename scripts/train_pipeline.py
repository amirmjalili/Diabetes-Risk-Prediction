#!/usr/bin/env python
"""End-to-end training pipeline for diabetes risk prediction.

Steps
-----
1. Load & quality-assess data
2. Feature engineering
3. Train / validation / test split (stratified)
4. Train candidate models
5. Hyperparameter optimization (Optuna) for primary model
6. Calibration on validation set
7. Evaluation on held-out test set
8. Persist artifacts (model, preprocessor, metrics, figures)
9. Optional MLflow logging

Usage
-----
    python scripts/train_pipeline.py
    python scripts/train_pipeline.py --n-trials 30 --primary-model xgboost
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_pima_dataset
from src.data.preprocessing import create_clinical_features, train_val_test_split
from src.data.quality import assess_data_quality, print_quality_summary
from src.evaluation.decision_curve import decision_curve_analysis
from src.evaluation.metrics import bootstrap_auc_ci, compute_classification_metrics
from src.models.calibrate import calibrate_model
from src.models.optimize import run_optuna_optimization
from src.models.train import instantiate_model, train_all_models
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train diabetes risk models")
    p.add_argument("--n-trials", type=int, default=None, help="Optuna trials")
    p.add_argument("--primary-model", type=str, default=None)
    p.add_argument("--skip-optuna", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config()
    seed = args.seed or config["project"]["random_seed"]
    set_seed(seed)
    setup_logging(level="INFO", log_dir=config["paths"].get("logs"))

    models_dir = Path(config["paths"]["models"])
    figures_dir = Path(config["paths"]["figures"])
    metrics_dir = Path(config["paths"]["metrics"])
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 1: Data loading & quality assessment")
    logger.info("=" * 60)
    df = load_pima_dataset()
    quality = assess_data_quality(df, target="Outcome")
    print_quality_summary(quality)
    with open(metrics_dir / "data_quality.json", "w") as f:
        json.dump(quality, f, indent=2)

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 2: Feature engineering")
    logger.info("=" * 60)
    df_eng = create_clinical_features(df)
    target = "Outcome"
    exclude = {target, "BMI_category", "Age_group", "Glucose_category"}
    feature_cols = [
        c
        for c in df_eng.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df_eng[c])
    ]
    logger.info(f"Modeling features ({len(feature_cols)}): {feature_cols}")

    # Simple median imputation for training script (full pipeline uses ColumnTransformer)
    X_full = df_eng[feature_cols].copy()
    medians = X_full.median()
    X_full = X_full.fillna(medians)
    y_full = df_eng[target]

    # ------------------------------------------------------------------
    # 3. Split
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 3: Stratified train/val/test split")
    logger.info("=" * 60)
    combined = X_full.copy()
    combined[target] = y_full
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        combined,
        target=target,
        test_size=config["preprocessing"]["test_size"],
        val_size=config["preprocessing"]["val_size"],
        random_state=seed,
        stratify=True,
    )

    # ------------------------------------------------------------------
    # 4. Baseline models
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 4: Train baseline models")
    logger.info("=" * 60)
    baselines = train_all_models(X_train, y_train, random_state=seed)

    baseline_metrics = {}
    for name, model in baselines.items():
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = compute_classification_metrics(y_test, pred, proba)
        baseline_metrics[name] = m
        logger.info(
            f"  {name:25s}  ROC-AUC={m.get('roc_auc', 0):.4f}  "
            f"PR-AUC={m.get('pr_auc', 0):.4f}  F1={m.get('f1', 0):.4f}"
        )

    with open(metrics_dir / "baseline_metrics.json", "w") as f:
        json.dump(baseline_metrics, f, indent=2)

    # ------------------------------------------------------------------
    # 5. Optuna optimization for primary model
    # ------------------------------------------------------------------
    primary = args.primary_model or config["models"]["primary_model"]
    n_trials = args.n_trials or config["optuna"]["n_trials"]

    if not args.skip_optuna and primary in ("xgboost", "lightgbm", "catboost", "random_forest", "logistic_regression"):
        logger.info("=" * 60)
        logger.info(f"STAGE 5: Optuna optimization ({primary})")
        logger.info("=" * 60)
        # Use train+val for CV-based optimization
        X_cv = pd.concat([X_train, X_val])
        y_cv = pd.concat([y_train, y_val])
        study = run_optuna_optimization(
            X_cv,
            y_cv,
            model_name=primary,
            n_trials=n_trials,
            n_splits=config["cross_validation"]["n_splits"],
            n_repeats=config["cross_validation"]["n_repeats"],
            random_state=seed,
            metric=config["optuna"]["metric"],
            timeout=config["optuna"].get("timeout"),
        )
        best_params = study.best_params
        with open(metrics_dir / f"optuna_{primary}_best.json", "w") as f:
            json.dump(
                {"best_value": study.best_value, "best_params": best_params},
                f,
                indent=2,
            )
    else:
        logger.info("Skipping Optuna; using default hyperparameters.")
        best_params = {}

    # ------------------------------------------------------------------
    # 6. Final model + calibration
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 6: Final model training & calibration")
    logger.info("=" * 60)
    final_model = instantiate_model(primary, random_state=seed, hyperparams=best_params)
    final_model.fit(X_train, y_train)

    cal_method = config["models"].get("calibration_method", "isotonic")
    if cal_method == "platt":
        cal_method = "sigmoid"
    calibrated = calibrate_model(final_model, X_val, y_val, method=cal_method)

    # ------------------------------------------------------------------
    # 7. Test-set evaluation
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 7: Held-out test evaluation")
    logger.info("=" * 60)
    proba_test = calibrated.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    test_metrics = compute_classification_metrics(y_test, pred_test, proba_test)
    auc, lo, hi = bootstrap_auc_ci(y_test.values, proba_test, random_state=seed)
    test_metrics["roc_auc_ci_lower"] = lo
    test_metrics["roc_auc_ci_upper"] = hi
    logger.info(f"Test ROC-AUC: {auc:.4f} (95% CI {lo:.4f}–{hi:.4f})")
    logger.info(f"Test metrics: {json.dumps(test_metrics, indent=2)}")

    with open(metrics_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Decision curve
    dca = decision_curve_analysis(y_test.values, proba_test)
    dca.to_csv(metrics_dir / "decision_curve.csv", index=False)

    # ------------------------------------------------------------------
    # 8. Persist artifacts
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 8: Persist artifacts")
    logger.info("=" * 60)
    joblib.dump(calibrated, models_dir / "best_model.joblib")
    joblib.dump(feature_cols, models_dir / "feature_names.joblib")
    # Store medians as a simple "preprocessor" for the API fallback path
    joblib.dump({"medians": medians.to_dict(), "features": feature_cols}, models_dir / "preprocessor.joblib")
    logger.info(f"Artifacts saved to {models_dir}")

    logger.info("Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())