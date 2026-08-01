"""Clinically motivated feature engineering.

Re-exports the core function from the data module for cleaner import paths
and provides additional documentation linking features to medical evidence.
"""

from src.data.preprocessing import create_clinical_features

__all__ = ["create_clinical_features"]

# Clinical rationale (for documentation & reviewers):
#
# BMI categories
#   - WHO classification of overweight/obesity is a major modifiable risk factor
#     for type 2 diabetes (ADA Standards of Care 2024; Lancet 2017).
#
# Age groups
#   - Incidence of type 2 diabetes rises sharply after age 45; screening
#     recommendations are age-stratified (ADA Standards of Care).
#
# Glucose categories
#   - Fasting glucose thresholds of 100 mg/dL (impaired) and 126 mg/dL (diabetes)
#     are diagnostic cut-points (ADA). Pima Glucose is a 2-hour post-load value
#     in the original protocol; categories remain useful for risk stratification.
#
# Interaction terms (Glucose × BMI, Age × BMI, Glucose × Age)
#   - Synergistic effects of adiposity and glycemia on insulin resistance are
#     well documented (DeFronzo, Diabetes Care 2009; Kahn et al., Nature 2006).
#
# HOMA-IR proxy
#   - Homeostatic Model Assessment of Insulin Resistance is a standard surrogate
#     when fasting insulin and glucose are available (Matthews et al., 1985).
#     Applied here with the caveat that Pima insulin/glucose timing may differ.