"""
DecisionLens dashboard: prediction, SHAP explanations, global importance, what-if.
Requires the FastAPI backend (see README) or set API_URL.
"""

from __future__ import annotations

import html
import os
from pathlib import Path

import httpx
import plotly.graph_objects as go
import streamlit as st

STYLE_PATH = Path(__file__).resolve().parent / "style.css"

st.set_page_config(
    page_title="DecisionLens | Explainable ML",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="◈",
)

if STYLE_PATH.exists():
    st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

# —— Sidebar (minimal, premium) ——
with st.sidebar:
    st.markdown("### ◈ Control")
    st.caption("Backend connection")
    st.code(API_URL, language="text")
    if st.button("Ping API", use_container_width=True):
        try:
            r = httpx.get(f"{API_URL}/health", timeout=10.0)
            if r.status_code == 200:
                h = r.json()
                st.success("Online")
                st.caption(f"Data: `{h.get('data_source', '—')}`")
            else:
                st.error(r.text[:200])
        except Exception as e:
            st.error(str(e))
    st.divider()
    st.caption("DecisionLens · XGBoost + SHAP · Production-style API")


def _payload(age, income, credit, loan, emp, dti) -> dict:
    return {
        "Age": int(age),
        "Income": float(income),
        "CreditScore": float(credit),
        "LoanAmount": float(loan),
        "EmploymentLength": int(emp),
        "DebtToIncome": float(dti),
    }


_PLOT_FONT = dict(family="DM Sans, sans-serif", color="#e2e8f0", size=12)
_PLOT_TITLE = dict(font=dict(family="DM Sans, sans-serif", color="#f1f5f9", size=14))


def _waterfall_figure(base: float, shap_values: dict[str, float], top_k: int = 8) -> go.Figure:
    items = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    names = [i[0] for i in items]
    vals = [i[1] for i in items]
    x = ["Baseline"] + names + ["Output"]
    measure = ["absolute"] + ["relative"] * len(names) + ["total"]
    total = base + sum(vals)
    y = [base] + vals + [total]
    fig = go.Figure(
        go.Waterfall(
            name="SHAP",
            orientation="v",
            measure=measure,
            x=x,
            textposition="outside",
            text=[f"{v:.3f}" for v in y],
            y=y,
            connector={"line": {"color": "rgba(148, 163, 184, 0.35)", "width": 1}},
            increasing={"marker": {"color": "rgba(248, 113, 113, 0.85)"}},
            decreasing={"marker": {"color": "rgba(96, 165, 250, 0.9)"}},
            totals={"marker": {"color": "rgba(167, 139, 250, 0.95)"}},
        )
    )
    fig.update_layout(
        title={**_PLOT_TITLE, "text": "Local attribution · SHAP waterfall (margin)"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,18,28,0.5)",
        font=_PLOT_FONT,
        height=440,
        margin=dict(l=48, r=24, t=56, b=88),
        xaxis=dict(tickangle=-28, gridcolor="rgba(148,163,184,0.08)", zeroline=False),
        yaxis=dict(gridcolor="rgba(148,163,184,0.08)", zeroline=True, zerolinecolor="rgba(148,163,184,0.2)"),
    )
    return fig


def _global_bar_figure(ranked: list[dict]) -> go.Figure:
    feats = [r["human_label"] for r in ranked][::-1]
    vals = [r["mean_abs_shap"] for r in ranked][::-1]
    colors = [f"rgba(99, 102, 241, {0.35 + 0.55 * (v / max(vals) if vals else 1)})" for v in vals]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=feats,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
        )
    )
    fig.update_layout(
        title={**_PLOT_TITLE, "text": "Global importance · mean |SHAP| (background sample)"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,18,28,0.5)",
        font=_PLOT_FONT,
        height=420,
        margin=dict(l=8, r=24, t=56, b=40),
        xaxis=dict(title="Mean |SHAP|", gridcolor="rgba(148,163,184,0.08)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


# —— Hero ——
st.markdown(
    """
<div class="hero-wrap">
  <div class="hero-badge">Explainable AI · Credit risk</div>
  <h1 class="hero-title">DecisionLens</h1>
  <p class="hero-sub">
    Live predictions with SHAP-backed drivers and side-by-side what-if analysis — built for clarity, not black-box demos.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

col_baseline, col_scenario = st.columns(2, gap="large")

with col_baseline:
    st.markdown(
        """
<div class="glass-panel">
  <p class="panel-title">Profile A</p>
  <p class="panel-head">Baseline</p>
  <p style="color:#94a3b8;font-size:0.9rem;margin:0 0 0.5rem;">Applicant / account to score.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    b_age = st.slider("Age", 18, 100, 35, key="b_age")
    b_income = st.slider("Income (limit proxy)", 10_000, 500_000, 120_000, step=5_000, key="b_inc")
    b_credit = st.slider("Credit behavior score", 300, 850, 640, key="b_cs")
    b_loan = st.slider("Exposure / balance", 0, 200_000, 25_000, step=1_000, key="b_loan")
    b_emp = st.slider("Employment tenure (years)", 0, 40, 6, key="b_emp")
    b_dti = st.slider("Debt-to-income (%)", 0, 100, 32, key="b_dti")

with col_scenario:
    st.markdown(
        """
<div class="glass-panel">
  <p class="panel-title">Profile B</p>
  <p class="panel-head">What-if scenario</p>
  <p style="color:#94a3b8;font-size:0.9rem;margin:0 0 0.5rem;">Adjust to compare against baseline.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    s_age = st.slider("Age", 18, 100, b_age, key="s_age")
    s_income = st.slider("Income (limit proxy)", 10_000, 500_000, b_income, step=5_000, key="s_inc")
    s_credit = st.slider("Credit behavior score", 300, 850, b_credit, key="s_cs")
    s_loan = st.slider("Exposure / balance", 0, 200_000, b_loan, step=1_000, key="s_loan")
    s_emp = st.slider("Employment tenure (years)", 0, 40, b_emp, key="s_emp")
    s_dti = st.slider("Debt-to-income (%)", 0, 100, b_dti, key="s_dti")

baseline = _payload(b_age, b_income, b_credit, b_loan, b_emp, b_dti)
scenario = _payload(s_age, s_income, s_credit, s_loan, s_emp, s_dti)

st.markdown('<p class="section-label">Live intelligence</p>', unsafe_allow_html=True)

try:
    with httpx.Client(timeout=30.0) as client:
        pred = client.post(f"{API_URL}/predict", json=baseline)
        expl = client.post(f"{API_URL}/explain", json=baseline)
        wif = client.post(f"{API_URL}/what-if", json={"baseline": baseline, "scenario": scenario})

    if pred.status_code != 200:
        st.error(f"/predict failed ({pred.status_code}): {pred.text}")
        st.stop()
    if expl.status_code != 200:
        st.error(f"/explain failed ({expl.status_code}): {expl.text}")
        st.stop()
    if wif.status_code != 200:
        st.error(f"/what-if failed ({wif.status_code}): {wif.text}")
        st.stop()

    p = pred.json()
    e = expl.json()
    w = wif.json()
    loc = e["local_explanation"]

    try:
        health = httpx.get(f"{API_URL}/health", timeout=5.0).json()
    except Exception:
        health = {}

    metrics = health.get("holdout_metrics") or {}
    roc = metrics.get("roc_auc", "—")

    st.markdown(
        f"""
<div class="kpi-row">
  <span class="kpi-pill">Hold-out ROC-AUC <strong>{roc if isinstance(roc, str) else f"{roc:.3f}"}</strong></span>
  <span class="kpi-pill">Model <strong>XGBoost</strong></span>
  <span class="kpi-pill">Explain <strong>SHAP TreeExplainer</strong></span>
</div>
""",
        unsafe_allow_html=True,
    )

    left, mid, right = st.columns([1.05, 1.15, 1.0], gap="medium")

    with left:
        card_class = "risk-high" if p["risk_category"] == "HIGH RISK" else "risk-low"
        st.markdown(
            f"""
            <div class="glass-panel risk-card {card_class}">
                <div class="metric-label">Baseline assessment</div>
                <h1 style="font-size: 2.4rem; margin: 0;">{p["risk_category"]}</h1>
                <div class="prob-text">
                  Adverse outcome probability<br/>
                  <strong style="color:#f1f5f9;font-size:1.15rem">{p["probability"]*100:.2f}%</strong>
                  &nbsp;·&nbsp; confidence <strong>{p["confidence"]*100:.1f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
<div class="glass-panel">
  <p class="panel-title">Counterfactual delta</p>
  <p class="panel-head">Scenario vs baseline</p>
  <p style="color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.12em;margin:0;">Scenario P(risk)</p>
  <p class="metric-big">{w["scenario"]["probability"] * 100:.2f}%</p>
  <p class="metric-delta">{w["delta_probability"] * 100:+.2f} pts vs baseline</p>
  <p class="metric-caption">{html.escape(w["interpretation"])}</p>
</div>
""",
            unsafe_allow_html=True,
        )

    with mid:
        st.plotly_chart(
            _waterfall_figure(float(loc["base_value"]), loc["shap_values"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with right:
        glob = e["global_importance"]["ranked"]
        st.plotly_chart(
            _global_bar_figure(glob),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown('<p class="section-label">Narrative & drivers</p>', unsafe_allow_html=True)

    chips = []
    for tf in loc["top_features"][:6]:
        sign = "+" if tf["shap_value"] >= 0 else ""
        label = html.escape(tf["human_label"])
        chips.append(
            f'<span class="driver-chip">{label} {sign}{tf["shap_value"]:.3f}</span>'
        )
    chips_html = " ".join(chips) if chips else "<span class='metric-caption'>No chip data.</span>"

    pos_lines = "".join(
        f"<div>· <code>{html.escape(item['feature'])}</code> → {item['impact']:+.3f}</div>"
        for item in loc.get("positive_impact", [])[:6]
    )
    neg_lines = "".join(
        f"<div>· <code>{html.escape(item['feature'])}</code> → {item['impact']:+.3f}</div>"
        for item in loc.get("negative_impact", [])[:6]
    )
    if not pos_lines:
        pos_lines = "<div class='metric-caption'>—</div>"
    if not neg_lines:
        neg_lines = "<div class='metric-caption'>—</div>"

    st.markdown(
        f"""
<div class="glass-panel">
  <p class="panel-title">Interpretation</p>
  <p class="panel-head">Why this prediction</p>
  <p class="narrative-body">{html.escape(loc["plain_english"])}</p>
  <p class="subhead">Signed contributions</p>
  <div class="chip-row">{chips_html}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="glass-panel">
  <p class="panel-title">Directional SHAP</p>
  <p class="panel-head">Risk ↑ vs ↓</p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-top:0.5rem;">
    <div class="directional-col">
      <strong style="color:#f87171;">Pushes risk up</strong>
      {pos_lines}
    </div>
    <div class="directional-col">
      <strong style="color:#60a5fa;">Pulls risk down</strong>
      {neg_lines}
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

except httpx.ConnectError:
    st.error(
        "Cannot reach the API. In another terminal run:  \n`$env:PYTHONPATH='E:\\ml'; .\\.venv\\Scripts\\Activate.ps1; python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`"
    )
except Exception as ex:
    st.exception(ex)
