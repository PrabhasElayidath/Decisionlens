"""Tests for feature engineering (train/serve alignment)."""

from __future__ import annotations

import pandas as pd

from utils.features import (
    ALL_FEATURE_COLUMNS,
    BASE_FEATURE_COLUMNS,
    add_engineered_features,
    prepare_model_matrix,
)


def test_engineered_columns_present():
    base = pd.DataFrame(
        [
            {
                "Age": 35,
                "Income": 100_000.0,
                "CreditScore": 650.0,
                "LoanAmount": 20_000.0,
                "EmploymentLength": 5.0,
                "DebtToIncome": 30.0,
            }
        ]
    )
    out = add_engineered_features(base)
    for c in ALL_FEATURE_COLUMNS:
        assert c in out.columns
    assert out["LoanToIncomeRatio"].iloc[0] > 0
    assert out["AffordabilityScore"].iloc[0] > 0


def test_prepare_model_matrix_order():
    base = {
        "Age": 40,
        "Income": 80_000.0,
        "CreditScore": 700.0,
        "LoanAmount": 10_000.0,
        "EmploymentLength": 8,
        "DebtToIncome": 25.0,
    }
    X = prepare_model_matrix(base)
    assert list(X.columns) == ALL_FEATURE_COLUMNS
    assert X.shape == (1, len(ALL_FEATURE_COLUMNS))


def test_base_feature_columns_count():
    assert len(BASE_FEATURE_COLUMNS) == 6
    assert len(ALL_FEATURE_COLUMNS) == 9
