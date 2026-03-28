"""
Train DecisionLens XGBoost model on the UCI *default of credit card clients* dataset
(Taiwan credit card default), with cleaning, domain mapping, and engineered features.

Falls back to calibrated synthetic data if the source file cannot be fetched.
"""

from __future__ import annotations

import json
from typing import Any
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# Repo root on path for `utils`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.features import (  # noqa: E402
    ALL_FEATURE_COLUMNS,
    BASE_FEATURE_COLUMNS,
    add_engineered_features,
)
from utils.paths import (  # noqa: E402
    BACKGROUND_PATH,
    DATA_RAW_DIR,
    GLOBAL_SHAP_PATH,
    META_PATH,
    MODEL_PATH,
    MODELS_DIR,
)
from utils.metrics_np import (  # noqa: E402
    accuracy,
    precision_recall_f1,
    roc_auc_binary,
)

try:
    from sklearn.metrics import (  # type: ignore
        accuracy_score,
        confusion_matrix as sklearn_confusion_matrix,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split  # type: ignore

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_SKLEARN = False
    sklearn_confusion_matrix = None  # type: ignore

UCI_XLS_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls"
)


def _ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)


def _load_uci_credit_default() -> pd.DataFrame:
    """Load real default-of-credit-card dataset (cached under data/raw/)."""
    cache_path = DATA_RAW_DIR / "default_of_credit_card_clients.xls"
    if not cache_path.exists():
        warnings.warn(f"Downloading dataset to {cache_path} ...", stacklevel=1)
        import urllib.request

        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(UCI_XLS_URL, cache_path)

    # Second row is the real header in the UCI file
    df = pd.read_excel(cache_path, header=1, engine="xlrd")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _build_training_frame_from_uci(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    pay_cols = [c for c in df.columns if str(c).upper().startswith("PAY_")]
    if not pay_cols:
        raise ValueError("No PAY_* columns in UCI frame")

    target_candidates = [c for c in df.columns if "default" in str(c).lower()]
    if not target_candidates:
        raise ValueError("No default target column found")
    y_col = target_candidates[-1]

    lim = pd.to_numeric(df["LIMIT_BAL"], errors="coerce").fillna(0).clip(lower=1)
    age = pd.to_numeric(df["AGE"], errors="coerce").fillna(35).clip(18, 100)
    bill1 = pd.to_numeric(df["BILL_AMT1"], errors="coerce").fillna(0)

    delay = df[pay_cols].apply(pd.to_numeric, errors="coerce").fillna(0).clip(-2, 8)
    avg_delay = delay.mean(axis=1)
    credit_score = (720.0 - 35.0 * avg_delay).clip(300, 850)

    edu = pd.to_numeric(df.get("EDUCATION", 2), errors="coerce").fillna(2).clip(0, 6)
    employment_length = (edu * 3.0).clip(0, 40)

    loan_amount = bill1.abs()
    income = lim.astype(float)
    dti = (bill1.clip(lower=0) / (lim + 1.0) * 100.0).clip(0, 100)

    y = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int).clip(0, 1)

    base = pd.DataFrame(
        {
            "Age": age,
            "Income": income,
            "CreditScore": credit_score,
            "LoanAmount": loan_amount,
            "EmploymentLength": employment_length,
            "DebtToIncome": dti,
        }
    )
    X = add_engineered_features(base)[ALL_FEATURE_COLUMNS]
    return X, y


def _synthetic_fallback(n_samples: int = 12000, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 71, size=n_samples)
    income = rng.normal(165_000, 80_000, size=n_samples).clip(10_000, 800_000)
    credit = rng.normal(660, 85, size=n_samples).clip(300, 850)
    loan = rng.lognormal(10.0, 0.85, size=n_samples).clip(500, 500_000)
    emp = rng.integers(0, 41, size=n_samples)
    dti = rng.normal(32, 18, size=n_samples).clip(0, 100)

    affordability = (income * (100.0 - dti) / 100.0) / (loan + 1.0)
    logit = (
        -4.0
        + 0.008 * (600 - credit)
        + 0.06 * dti
        - 0.000006 * income
        + 0.000004 * loan
        - 0.12 * emp
        - 1.1 * affordability
        + rng.normal(0, 0.45, size=n_samples)
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n_samples) < p).astype(int)

    base = pd.DataFrame(
        {
            "Age": age,
            "Income": income.astype(float),
            "CreditScore": credit,
            "LoanAmount": loan,
            "EmploymentLength": emp,
            "DebtToIncome": dti,
        }
    )
    X = add_engineered_features(base)[ALL_FEATURE_COLUMNS]
    return X, y


def _split(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42
):
    if _HAS_SKLEARN:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    idx = np.arange(len(X))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_test = max(1, int(len(X) * test_size))
    te = idx[:n_test]
    tr = idx[n_test:]
    return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]


def _confusion_dict(y_true, y_pred) -> dict[str, Any]:
    """Binary labels 0/1: rows=true, cols=pred."""
    if _HAS_SKLEARN and sklearn_confusion_matrix is not None:
        cm = sklearn_confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    else:
        yt = np.asarray(y_true).astype(int)
        yp = np.asarray(y_pred).astype(int)
        tn = int(((yt == 0) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tp = int(((yt == 1) & (yp == 1)).sum())
    return {
        "rows": ["true_0", "true_1"],
        "columns": ["pred_0", "pred_1"],
        "matrix": [[tn, fp], [fn, tp]],
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def _feature_stats(X: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for c in X.columns:
        s = X[c].astype(float)
        out[c] = {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)) or 1e-9,
            "min": float(s.min()),
            "max": float(s.max()),
        }
    return out


def _classification_metrics(y_true, y_pred, y_prob) -> dict:
    if _HAS_SKLEARN:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
        }
    prec, rec, _ = precision_recall_f1(np.asarray(y_true), np.asarray(y_pred))
    return {
        "accuracy": accuracy(np.asarray(y_true), np.asarray(y_pred)),
        "precision": prec,
        "recall": rec,
        "roc_auc": roc_auc_binary(np.asarray(y_true), np.asarray(y_prob)),
    }


def _compute_global_shap(model: xgb.XGBClassifier, X_bg: pd.DataFrame) -> dict[str, float]:
    import shap

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_bg)
    if isinstance(sv, list):
        sv = sv[1]
    mean_abs = np.mean(np.abs(sv), axis=0)
    return {f: float(v) for f, v in zip(ALL_FEATURE_COLUMNS, mean_abs)}


def main() -> None:
    _ensure_dirs()
    data_source = "uci_credit_default"

    try:
        raw = _load_uci_credit_default()
        X, y = _build_training_frame_from_uci(raw)
        print(f"Loaded UCI credit default data: {X.shape[0]} rows, {X.shape[1]} features.")
    except Exception as exc:  # pragma: no cover - network / parse
        warnings.warn(f"UCI load failed ({exc}); using synthetic fallback.", stacklevel=1)
        X, y = _synthetic_fallback()
        data_source = "synthetic_fallback"

    X_train, X_test, y_train, y_test = _split(X, y)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = float(n_neg / max(n_pos, 1))

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        objective="binary:logistic",
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = _classification_metrics(y_test, y_pred, y_prob)
    cm_dict = _confusion_dict(y_test, y_pred)
    feat_stats = _feature_stats(X_train)

    print("Evaluation (hold-out):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  scale_pos_weight (train): {scale_pos_weight:.4f}")

    model.save_model(str(MODEL_PATH))
    print(f"Model saved to {MODEL_PATH}")

    train_version = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {
        "features": ALL_FEATURE_COLUMNS,
        "base_features": BASE_FEATURE_COLUMNS,
        "metrics": metrics,
        "confusion_matrix": cm_dict,
        "feature_stats": feat_stats,
        "train_version": train_version,
        "scale_pos_weight": scale_pos_weight,
        "default_threshold": 0.5,
        "data_source": data_source,
        "positive_class_label": "default_next_period",
        "model_type": "xgboost",
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {META_PATH}")

    # Background sample for SHAP / UI (no labels needed)
    bg_n = min(800, len(X_train))
    X_bg = X_train.sample(n=bg_n, random_state=42)
    np.save(BACKGROUND_PATH, X_bg.to_numpy(dtype=np.float64))

    try:
        gshap = _compute_global_shap(model, X_bg)
        with open(GLOBAL_SHAP_PATH, "w", encoding="utf-8") as f:
            json.dump(gshap, f, indent=2)
        print(f"Global mean |SHAP| saved to {GLOBAL_SHAP_PATH}")
    except Exception as exc:
        warnings.warn(f"Could not compute global SHAP during train: {exc}", stacklevel=1)


if __name__ == "__main__":
    main()
