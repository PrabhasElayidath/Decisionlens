"""
DecisionLens FastAPI application: predict, explain (SHAP), what-if, health, metrics.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from utils.logging_config import setup_api_logging

logger = setup_api_logging("decisionlens")

model_service = None
explain_service = None


def _ensure_artifacts() -> None:
    """Train if allowed and artifacts are missing (dev convenience)."""
    from utils.paths import META_PATH, MODEL_PATH

    if MODEL_PATH.exists() and META_PATH.exists():
        return
    auto = os.getenv("DECISIONLENS_AUTO_TRAIN", "true").lower() in ("1", "true", "yes")
    if not auto:
        raise FileNotFoundError("Model artifacts missing and DECISIONLENS_AUTO_TRAIN is disabled.")
    logger.warning("Model artifacts missing; running training pipeline...")
    from data.train import main as train_main

    train_main()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_service, explain_service
    _ensure_artifacts()
    from services.explain import ExplainService
    from services.model import ModelService

    model_service = ModelService()
    explain_service = ExplainService(model_service)
    logger.info("Model and explainer loaded.")
    yield


app = FastAPI(
    title="DecisionLens: Explainable ML Engine",
    description="Production-style API for credit-default risk scoring with SHAP and what-if analysis.",
    version="1.1.0",
    lifespan=lifespan,
)


class PredictionInput(BaseModel):
    Age: int = Field(..., ge=18, le=100)
    Income: float = Field(..., ge=0, le=10_000_000)
    CreditScore: float = Field(..., ge=300, le=850)
    LoanAmount: float = Field(..., ge=0, le=10_000_000)
    EmploymentLength: int = Field(..., ge=0, le=60)
    DebtToIncome: float = Field(..., ge=0, le=100)
    threshold: float = Field(
        default=0.5,
        ge=0.1,
        le=0.9,
        description="Classification threshold on P(class=1); higher → fewer positives (higher precision, lower recall).",
    )


class PredictionOutput(BaseModel):
    prediction: int = Field(..., description="1 = elevated default / credit risk")
    risk_category: str
    probability: float
    confidence: float
    threshold_used: float = 0.5
    drift_alert: bool = False
    drift_notes: list[str] = Field(default_factory=list)


class ExplainResponse(BaseModel):
    local_explanation: dict[str, Any]
    global_importance: dict[str, Any]


class WhatIfRequest(BaseModel):
    """Compare two full feature vectors (baseline vs counterfactual scenario)."""

    baseline: PredictionInput
    scenario: PredictionInput


class WhatIfResponse(BaseModel):
    baseline: PredictionOutput
    scenario: PredictionOutput
    delta_probability: float
    interpretation: str


def _payload_for_explain(p: PredictionInput) -> dict[str, Any]:
    """SHAP only needs base features, not threshold."""
    return p.model_dump(exclude={"threshold"})


@app.get("/health")
async def health_check() -> dict[str, Any]:
    if model_service is None:
        return {"status": "starting", "model_loaded": False}
    meta = getattr(model_service, "meta", {}) or {}
    return {
        "status": "ok",
        "model_loaded": True,
        "data_source": meta.get("data_source"),
        "train_version": meta.get("train_version"),
        "holdout_metrics": meta.get("metrics", {}),
    }


@app.get("/metrics")
async def metrics_endpoint() -> dict[str, Any]:
    """Hold-out metrics, confusion matrix, training metadata (for monitoring / dashboards)."""
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    meta = model_service.meta
    return {
        "metrics": meta.get("metrics", {}),
        "confusion_matrix": meta.get("confusion_matrix", {}),
        "train_version": meta.get("train_version"),
        "scale_pos_weight": meta.get("scale_pos_weight"),
        "default_threshold": meta.get("default_threshold", 0.5),
        "data_source": meta.get("data_source"),
        "features": meta.get("features", []),
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict_endpoint(input_data: PredictionInput) -> PredictionOutput:
    # FIX: guard against requests arriving before startup completes
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model not ready yet. Please retry in a moment.")
    try:
        out = model_service.predict(input_data.model_dump())
        return PredictionOutput(**out)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception:
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from None


@app.post("/explain", response_model=ExplainResponse)
async def explain_endpoint(input_data: PredictionInput) -> ExplainResponse:
    # FIX: guard against requests arriving before startup completes
    if model_service is None or explain_service is None:
        raise HTTPException(status_code=503, detail="Model not ready yet. Please retry in a moment.")
    try:
        payload = _payload_for_explain(input_data)
        local_exp = explain_service.generate_local_explanation(payload)
        global_imp = explain_service.generate_global_importance()
        return ExplainResponse(local_explanation=local_exp, global_importance=global_imp)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception:
        logger.exception("explain failed")
        raise HTTPException(status_code=500, detail="Explanation failed.")


@app.post("/what-if", response_model=WhatIfResponse)
async def what_if_endpoint(body: WhatIfRequest) -> WhatIfResponse:
    # FIX: guard against requests arriving before startup completes
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model not ready yet. Please retry in a moment.")
    try:
        # Use baseline threshold for both arms so comparison is apples-to-apples
        t = body.baseline.threshold
        b = body.baseline.model_dump()
        s = body.scenario.model_dump()
        b["threshold"] = t
        s["threshold"] = t
        base_out = model_service.predict(b)
        scen_out = model_service.predict(s)
        delta = float(scen_out["probability"] - base_out["probability"])
        if delta > 0.02:
            interp = f"Scenario increases estimated risk by {delta * 100:.1f} percentage points versus baseline."
        elif delta < -0.02:
            interp = f"Scenario decreases estimated risk by {abs(delta) * 100:.1f} percentage points versus baseline."
        else:
            interp = "Scenario is close to baseline; estimated risk barely moves."
        return WhatIfResponse(
            baseline=PredictionOutput(**base_out),
            scenario=PredictionOutput(**scen_out),
            delta_probability=delta,
            interpretation=interp,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception:
        logger.exception("what-if failed")
        raise HTTPException(status_code=500, detail="What-if analysis failed.")