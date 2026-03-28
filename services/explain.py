"""SHAP-backed explanations: global importance, local attribution, plain language."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import shap

from utils.paths import GLOBAL_SHAP_PATH

# Readable names for non-technical summaries
FEATURE_LABELS = {
    "Age": "age",
    "Income": "income (credit limit proxy)",
    "CreditScore": "credit behavior score",
    "LoanAmount": "outstanding balance / exposure",
    "EmploymentLength": "employment tenure proxy",
    "DebtToIncome": "debt-to-income pressure",
    "LoanToIncomeRatio": "loan size relative to income",
    "AffordabilityScore": "ability to absorb payments",
    "CreditEmploymentIndex": "credit strength with tenure",
}


class ExplainService:
    def __init__(self, model_service):
        self.model_service = model_service
        self.explainer = shap.TreeExplainer(model_service.model)

    def _expected_value_positive_class(self) -> float:
        ev = np.asarray(self.explainer.expected_value, dtype=float).ravel()
        if ev.size >= 2:
            return float(ev[1])
        return float(ev[0])

    def _shap_row(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        """SHAP values for positive class (index 1) and expected value (margin) base."""
        sv = self.explainer.shap_values(X)
        if isinstance(sv, list):
            sv = np.asarray(sv[1])
        else:
            sv = np.asarray(sv)
        if sv.ndim == 1:
            sv = sv.reshape(1, -1)
        base = self._expected_value_positive_class()
        return sv[0], base

    def load_global_shap(self) -> dict[str, float]:
        if GLOBAL_SHAP_PATH.exists():
            with open(GLOBAL_SHAP_PATH, encoding="utf-8") as f:
                return {k: float(v) for k, v in json.load(f).items()}
        return self.compute_global_shap_from_background()

    def compute_global_shap_from_background(self, max_rows: int = 600) -> dict[str, float]:
        """Mean |SHAP| over a background sample (fallback if JSON missing)."""
        from utils.paths import BACKGROUND_PATH

        if not BACKGROUND_PATH.exists():
            return {}
        X_bg = np.load(BACKGROUND_PATH)
        n = min(max_rows, X_bg.shape[0])
        X = pd.DataFrame(X_bg[:n], columns=self.model_service.features)
        sv = self.explainer.shap_values(X)
        if isinstance(sv, list):
            sv = np.asarray(sv[1])
        mean_abs = np.mean(np.abs(sv), axis=0)
        return {f: float(v) for f, v in zip(self.model_service.features, mean_abs)}

    def generate_local_explanation(self, base_input: dict[str, Any]) -> dict[str, Any]:
        X = self.model_service.prepare_matrix(base_input)
        if list(X.columns) != self.model_service.features:
            X = X[self.model_service.features]

        sv_row, base_value = self._shap_row(X)
        feats = self.model_service.features

        pairs = sorted(
            [(feats[i], float(sv_row[i])) for i in range(len(feats))],
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        top = pairs[:5]

        pos = [{"feature": f, "impact": v, "direction": "increases_risk"} for f, v in top if v > 0]
        neg = [{"feature": f, "impact": v, "direction": "decreases_risk"} for f, v in top if v < 0]

        prob = self.model_service.predict_proba_positive(base_input)
        text = self._plain_language_summary(pairs, base_value, prob)

        return {
            "base_value": base_value,
            "shap_values": {f: float(sv_row[i]) for i, f in enumerate(feats)},
            "top_features": [{"feature": f, "shap_value": v, "human_label": FEATURE_LABELS.get(f, f)} for f, v in top],
            "positive_impact": pos,
            "negative_impact": neg,
            "plain_english": text,
            "text_explanation": text,
        }

    def _plain_language_summary(
        self,
        sorted_pairs: list[tuple[str, float]],
        base_value: float,
        probability_positive: float,
    ) -> str:
        top3 = sorted_pairs[:3]
        if not top3:
            return "Not enough information to summarize this prediction."

        def phrase(feat: str, val: float) -> str:
            label = FEATURE_LABELS.get(feat, feat)
            mag = abs(val)
            if val > 0:
                return f"{label} pushes the outcome toward higher risk (strength {mag:.3f})"
            return f"{label} pulls the outcome toward lower risk (strength {mag:.3f})"

        lead = (
            f"The model estimates a {probability_positive * 100:.1f}% chance of the adverse outcome. "
            "In SHAP terms (margin space), it starts from a baseline near "
            f"{base_value:.3f} and the largest adjustments come from: "
        )
        detail = "; ".join(phrase(f, v) for f, v in top3) + "."
        closing = (
            " Positive SHAP contributions increase estimated default risk; "
            "negative contributions decrease it."
        )
        return lead + detail + closing

    def generate_global_importance(self) -> dict[str, Any]:
        raw = self.load_global_shap()
        if not raw:
            raw = {f: 0.0 for f in self.model_service.features}
        ordered = sorted(raw.items(), key=lambda x: x[1], reverse=True)
        return {
            "mean_abs_shap": raw,
            "ranked": [{"feature": f, "mean_abs_shap": v, "human_label": FEATURE_LABELS.get(f, f)} for f, v in ordered],
        }
