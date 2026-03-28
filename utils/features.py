"""
User-facing base features + engineered signals (computable at inference).

All engineered columns are derived only from the six primary inputs so
training and production stay aligned.
"""

from __future__ import annotations

import pandas as pd

BASE_FEATURE_COLUMNS = [
    "Age",
    "Income",
    "CreditScore",
    "LoanAmount",
    "EmploymentLength",
    "DebtToIncome",
]

ENGINEERED_COLUMNS = [
    "LoanToIncomeRatio",
    "AffordabilityScore",
    "CreditEmploymentIndex",
]

ALL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ENGINEERED_COLUMNS


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add at least three meaningful derived features (in-place copy)."""
    out = df.copy()
    income = out["Income"].astype(float).clip(lower=0)
    loan = out["LoanAmount"].astype(float).clip(lower=0)
    dti = out["DebtToIncome"].astype(float).clip(0, 100)
    cs = out["CreditScore"].astype(float)
    emp = out["EmploymentLength"].astype(float).clip(lower=0)

    # Loan burden relative to stated income (higher → more risk)
    out["LoanToIncomeRatio"] = loan / (income + 1.0) * 100.0

    # Higher score → more capacity to service debt relative to loan size
    out["AffordabilityScore"] = (income * (100.0 - dti) / 100.0) / (loan + 1.0)

    # Strong credit + tenure often reduces default risk
    out["CreditEmploymentIndex"] = (cs * (emp + 1.0) ** 0.5) / 1000.0

    return out


def row_dict_to_frame(base: dict, base_columns: list[str] | None = None) -> pd.DataFrame:
    """Build a single-row DataFrame with base columns only."""
    cols = base_columns or BASE_FEATURE_COLUMNS
    row = {k: base[k] for k in cols}
    return pd.DataFrame([row])


def prepare_model_matrix(
    base: dict,
    base_columns: list[str] | None = None,
    output_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Base user input → full feature matrix for the booster."""
    df = row_dict_to_frame(base, base_columns)
    out = add_engineered_features(df)
    oc = output_columns or ALL_FEATURE_COLUMNS
    return out[oc]
