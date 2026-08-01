#!/usr/bin/env python
"""Generate publication-quality evaluation figures after training.

Produces:
- ROC curves (multiple models)
- Precision-Recall curves
- Calibration (reliability) diagram
- Confusion matrix
- Decision curve
- SHAP summary (if model available)

Run after scripts/train_pipeline.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    calibration_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_pima_dataset
from src.data.preprocessing import create_clinical_features, train_val_test_split
from src.evaluation.decision_curve import decision_curve_analysis
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed


def main():
    setup_logging()
    config = load_config()
    set_seed(config["project"]["random_seed"])
    figures_dir = Path(config["paths"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(config["paths"]["models"])

    model_path = models_dir / "best_model.joblib"
    feat_path = models_dir / "feature_names.joblib"
    if not model_path.exists():
        logger.error("No trained model found. Run scripts/train_pipeline.py first.")
        return 1

    model = joblib.load(model_path)
    feature_cols = joblib.load(feat_path)

    df = load_pima_dataset()
    df_eng = create_clinical_features(df)
    X_full = df_eng[feature_cols].fillna(df_eng[feature_cols].median())
    y_full = df_eng["Outcome"]
    combined = X_full.copy()
    combined["Outcome"] = y_full
    _, _, X_test, _, _, y_test = train_val_test_split(
        combined, target="Outcome", random_state=config["project"]["random_seed"]
    )

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300

    # ROC
    fig, ax = plt.subplots(figsize=(5.5, 5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name="Calibrated model")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_title("ROC Curve â Held-out Test Set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curve.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved roc_curve.png")

    # PR
    fig, ax = plt.subplots(figsize=(5.5, 5))
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax, name="Calibrated model")
    ax.set_title("Precision-Recall Curve â Held-out Test Set")
    fig.tight_layout()
    fig.savefig(figures_dir / "pr_curve.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved pr_curve.png")

    # Calibration
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=8, strategy="uniform")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(mean_pred, frac_pos, "s-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(figures_dir / "calibration_curve.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved calibration_curve.png")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["No DM", "DM"], ax=ax, cmap="Blues"
    )
    ax.set_title("Confusion Matrix (threshold = 0.5)")
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved confusion_matrix.png")

    # Decision curve
    dca = decision_curve_analysis(y_test.values, proba)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(dca["threshold"], dca["net_benefit_model"], label="Model")
    ax.plot(dca["threshold"], dca["net_benefit_all"], label="Treat all", linestyle="--")
    ax.plot(dca["threshold"], dca["net_benefit_none"], label="Treat none", linestyle=":")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve Analysis")
    ax.legend()
    ax.set_xlim(0, 0.8)
    fig.tight_layout()
    fig.savefig(figures_dir / "decision_curve.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved decision_curve.png")

    # SHAP summary (optional)
    try:
        from src.explainability.shap_utils import compute_shap_values, global_importance_from_shap

        explainer, shap_vals = compute_shap_values(
            model, X_test, feature_names=feature_cols, max_samples=150
        )
        if shap_vals is not None and shap_vals.size > 0:
            imp = global_importance_from_shap(shap_vals, feature_cols)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(imp["feature"][::-1], imp["mean_abs_shap"][::-1], color="#4C72B0")
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title("Global Feature Importance (SHAP)")
            fig.tight_layout()
            fig.savefig(figures_dir / "shap_importance.png", bbox_inches="tight")
            plt.close()
            logger.info("Saved shap_importance.png")
    except Exception as e:
        logger.warning(f"SHAP figure skipped: {e}")

    logger.info(f"All figures written to {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
