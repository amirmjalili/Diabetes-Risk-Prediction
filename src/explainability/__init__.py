"""Explainable AI utilities: SHAP, feature importance, partial dependence."""

from .shap_utils import compute_shap_values, global_importance_from_shap

__all__ = ["compute_shap_values", "global_importance_from_shap"]
