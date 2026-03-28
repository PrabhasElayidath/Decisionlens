"""FastAPI route tests (requires trained artifacts)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


SAMPLE = {
    "Age": 35,
    "Income": 120_000.0,
    "CreditScore": 640.0,
    "LoanAmount": 25_000.0,
    "EmploymentLength": 6,
    "DebtToIncome": 32.0,
    "threshold": 0.5,
}


@pytest.fixture
def client(ensure_models):
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_metrics(client: TestClient):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "metrics" in body
    assert "roc_auc" in body["metrics"] or "accuracy" in body["metrics"]


def test_predict_threshold(client: TestClient):
    r = client.post("/predict", json={**SAMPLE, "threshold": 0.2})
    assert r.status_code == 200
    j = r.json()
    assert "prediction" in j
    assert j["threshold_used"] == 0.2


def test_predict_and_explain(client: TestClient):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    e = client.post("/explain", json=SAMPLE)
    assert e.status_code == 200
    assert "local_explanation" in e.json()


def test_what_if(client: TestClient):
    body = {
        "baseline": SAMPLE,
        "scenario": {**SAMPLE, "Income": 200_000.0},
    }
    r = client.post("/what-if", json=body)
    assert r.status_code == 200
    j = r.json()
    assert "delta_probability" in j
