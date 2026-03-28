"""Load XGBoost artifact and run inference on user-facing base features."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import xgboost as xgb

from utils.drift import check_feature_drift
from utils.features import ALL_FEATURE_COLUMNS, BASE_FEATURE_COLUMNS, prepare_model_matrix
from utils.paths import META_PATH, MODEL_PATH


class ModelService:
    def __init__(self) -> None:
        self.model = xgb.XGBClassifier()
        self.features: list[str] = []
        self.base_features: list[str] = []
        self.metrics: dict[str, float] = {}
        self.meta: dict[str, Any] = {}
        self.feature_stats: dict[str, dict[str, float]] = {}
        self._load()

    def _load(self) -> None:
        if not MODEL_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(
                f"Model artifacts missing. Run: python data/train.py (expected {MODEL_PATH} and {META_PATH})."
            )
        self.model.load_model(str(MODEL_PATH))
        with open(META_PATH, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.features = list(self.meta.get("features", ALL_FEATURE_COLUMNS))
        self.base_features = list(self.meta.get("base_features", BASE_FEATURE_COLUMNS))
        self.metrics = dict(self.meta.get("metrics", {}))
        self.feature_stats = dict(self.meta.get("feature_stats", {}))

    def prepare_matrix(self, base_input: dict[str, Any]) -> pd.DataFrame:
        missing = [k for k in self.base_features if k not in base_input]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        return prepare_model_matrix(
            {k: base_input[k] for k in self.base_features},
            self.base_features,
            self.features,
        )

    def predict_proba_positive(self, base_input: dict[str, Any]) -> float:
        X = self.prepare_matrix(base_input)
        if list(X.columns) != self.features:
            X = X[self.features]
        prob = float(self.model.predict_proba(X)[0, 1])
        return prob

    def predict(self, base_input: dict[str, Any]) -> dict[str, Any]:
        """
        Returns binary prediction where label 1 = elevated default / credit risk.
        Optional keys: ``threshold`` (0.1–0.9), popped before feature matrix build.
        """
        data = dict(base_input)
        threshold = float(data.pop("threshold", self.meta.get("default_threshold", 0.5)))
        threshold = max(0.1, min(0.9, threshold))

        drift_alert, drift_notes = check_feature_drift(data, self.feature_stats or None)

        prob = self.predict_proba_positive(data)
        pred = int(prob >= threshold)
        risk_category = "HIGH RISK" if pred == 1 else "LOW RISK"
        confidence = float(prob if pred == 1 else 1.0 - prob)
        return {
            "prediction": pred,
            "risk_category": risk_category,
            "probability": prob,
            "confidence": confidence,
            "threshold_used": threshold,
            "drift_alert": drift_alert,
            "drift_notes": drift_notes,
        }
