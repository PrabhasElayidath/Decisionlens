"""Model service tests (requires trained artifacts)."""

from __future__ import annotations

import pytest

from services.model import ModelService


@pytest.mark.usefixtures("ensure_models")
def test_predict_shape_and_threshold():
    m = ModelService()
    base = {
        "Age": 35,
        "Income": 120_000.0,
        "CreditScore": 640.0,
        "LoanAmount": 25_000.0,
        "EmploymentLength": 6,
        "DebtToIncome": 32.0,
    }
    out = m.predict({**base, "threshold": 0.5})
    assert "prediction" in out
    assert "probability" in out
    assert "drift_alert" in out
    assert out["threshold_used"] == 0.5

    out_high = m.predict({**base, "threshold": 0.9})
    assert out_high["threshold_used"] == 0.9


@pytest.mark.usefixtures("ensure_models")
def test_predict_proba_in_unit_interval():
    m = ModelService()
    p = m.predict_proba_positive(
        {
            "Age": 35,
            "Income": 120_000.0,
            "CreditScore": 640.0,
            "LoanAmount": 25_000.0,
            "EmploymentLength": 6,
            "DebtToIncome": 32.0,
        }
    )
    assert 0.0 <= p <= 1.0
