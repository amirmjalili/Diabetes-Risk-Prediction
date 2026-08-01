"""Rigorous data quality assessment for clinical datasets.

Produces a structured report covering missingness, duplicates, impossible
values, outliers, class imbalance, multicollinearity, and basic leakage checks.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


def assess_data_quality(
    df: pd.DataFrame,
    target: str = "Outcome",
    continuous_features: List[str] | None = None,
) -> Dict[str, Any]:
    """Perform comprehensive data quality assessment.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target : str
        Name of the binary target column.
    continuous_features : list of str, optional
        Columns treated as continuous. If None, inferred from dtypes.

    Returns
    -------
    dict
        Structured quality report suitable for logging and JSON serialization.
    """
    report: Dict[str, Any] = {}
    n = len(df)

    # --- Basic shape ---
    report["n_rows"] = n
    report["n_columns"] = df.shape[1]
    report["memory_mb"] = round(df.memory_usage(deep=True).sum() / 1e6, 3)

    # --- Missing values ---
    missing = df.isna().sum()
    missing_pct = (missing / n * 100).round(2)
    report["missing"] = {
        col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
        for col in df.columns
        if missing[col] > 0
    }
    report["complete_cases"] = int(df.dropna().shape[0])
    report["complete_case_pct"] = round(report["complete_cases"] / n * 100, 2)

    # --- Duplicates ---
    report["n_duplicate_rows"] = int(df.duplicated().sum())
    report["duplicate_pct"] = round(report["n_duplicate_rows"] / n * 100, 2)

    # --- Target distribution ---
    if target in df.columns:
        value_counts = df[target].value_counts(dropna=False).to_dict()
        report["target_distribution"] = {
            str(k): int(v) for k, v in value_counts.items()
        }
        pos = df[target].sum() if df[target].dtype != object else (df[target] == 1).sum()
        report["prevalence"] = round(float(pos) / n, 4)
        report["class_imbalance_ratio"] = round(
            (n - pos) / max(pos, 1), 3
        )  # majority / minority

    # --- Continuous feature summaries & outliers ---
    if continuous_features is None:
        continuous_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in continuous_features:
            continuous_features.remove(target)

    feature_stats = {}
    for col in continuous_features:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = int(((s < lower) | (s > upper)).sum())
        feature_stats[col] = {
            "mean": round(float(s.mean()), 3),
            "std": round(float(s.std()), 3),
            "median": round(float(s.median()), 3),
            "min": round(float(s.min()), 3),
            "max": round(float(s.max()), 3),
            "skewness": round(float(stats.skew(s)), 3),
            "kurtosis": round(float(stats.kurtosis(s)), 3),
            "n_outliers_iqr": n_outliers,
            "outlier_pct": round(n_outliers / len(s) * 100, 2),
        }
    report["feature_statistics"] = feature_stats

    # --- Multicollinearity (Pearson correlation) ---
    if len(continuous_features) >= 2:
        corr = df[continuous_features].corr(method="pearson")
        high_corr_pairs = []
        for i, c1 in enumerate(corr.columns):
            for c2 in corr.columns[i + 1 :]:
                r = corr.loc[c1, c2]
                if abs(r) >= 0.7:
                    high_corr_pairs.append(
                        {"feature_1": c1, "feature_2": c2, "correlation": round(float(r), 3)}
                    )
        report["high_correlation_pairs"] = high_corr_pairs

    # --- Impossible / clinically implausible values (domain knowledge) ---
    implausible = {}
    checks = {
        "Glucose": (40, 600),
        "BloodPressure": (30, 200),
        "BMI": (10, 70),
        "Age": (18, 100),
        "Insulin": (0, 1000),
        "SkinThickness": (0, 100),
        "Pregnancies": (0, 20),
    }
    for col, (lo, hi) in checks.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        n_impl = int(((s < lo) | (s > hi)).sum())
        if n_impl > 0:
            implausible[col] = {
                "n_implausible": n_impl,
                "range_checked": [lo, hi],
            }
    report["implausible_values"] = implausible

    # --- Leakage heuristics ---
    # Target should not appear in feature names; no perfect predictors expected
    report["potential_leakage_notes"] = []
    if target in continuous_features:
        report["potential_leakage_notes"].append(
            "Target column is listed among continuous features — check pipeline."
        )

    logger.info(
        f"Data quality assessment complete: {n} rows, "
        f"{len(report['missing'])} columns with missing values, "
        f"prevalence={report.get('prevalence', 'N/A')}"
    )
    return report


def print_quality_summary(report: Dict[str, Any]) -> None:
    """Pretty-print key quality metrics to the logger."""
    logger.info("=" * 60)
    logger.info("DATA QUALITY SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Rows: {report['n_rows']} | Columns: {report['n_columns']}")
    logger.info(f"Complete cases: {report['complete_cases']} ({report['complete_case_pct']}%)")
    logger.info(f"Duplicate rows: {report['n_duplicate_rows']}")
    if "prevalence" in report:
        logger.info(
            f"Target prevalence: {report['prevalence']:.3f} | "
            f"Imbalance ratio: {report['class_imbalance_ratio']:.2f}"
        )
    if report["missing"]:
        logger.info("Missing values:")
        for col, info in report["missing"].items():
            logger.info(f"  {col}: {info['count']} ({info['pct']}%)")
    if report["high_correlation_pairs"]:
        logger.info("High correlations (|r| >= 0.7):")
        for pair in report["high_correlation_pairs"]:
            logger.info(
                f"  {pair['feature_1']} ~ {pair['feature_2']}: {pair['correlation']}"
            )
    logger.info("=" * 60)