"""
Project Aesclepius — 2026 Simulator Dashboard
==============================================
Streamlit front-end for the Safety Net budget-allocation engine.

Design note: there is NO button gate.  The solver runs reactively
every time a sidebar slider or input changes, which is the standard
Streamlit pattern for dashboards.
"""

import os
import pickle

import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
import xgboost as xgb
import matplotlib.pyplot as plt

from src.engine import BudgetAllocator
from src.costs import apply_meps_costs, MEPS_BASE_COSTS

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Project Aesclepius: 2026 Simulator",
    layout="wide",
)

# ── Data loading & cleaning ──────────────────────────────────────────
@st.cache_data
def load_data(path: str = "2026_forecast.csv") -> pd.DataFrame:
    """Load forecast CSV and align columns with what the engine expects."""
    df = pd.read_csv(path)

    # Rename the patient-volume column to what the engine uses
    df = df.rename(columns={"Predicted_Patients_2026": "Projected_Volume"})

    # Fix department names — CSV uses underscores, MEPS dict uses spaces
    df["Department"] = df["Department"].str.replace("_", " ")

    # Drop departments missing from the cost dictionary
    known = set(MEPS_BASE_COSTS.keys())
    unknown_mask = ~df["Department"].isin(known)
    if unknown_mask.any():
        bad = df.loc[unknown_mask, "Department"].unique().tolist()
        st.warning(f"Dropping rows with unknown departments: {bad}")
        df = df[~unknown_mask].reset_index(drop=True)

    # Attach MEPS-projected cost columns (Cost_Per_Person, Total_Group_Cost)
    df = apply_meps_costs(df)
    return df


# ── Absolute base directory (where app.py lives) ────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── ML model loading ─────────────────────────────────────────────────
def _resolve_models_dir() -> str:
    """Return the absolute path to the ``models/`` folder.

    Checks two locations so it works regardless of cwd:
      1. <BASE_DIR>/models   (app.py at project root — normal case)
      2. <BASE_DIR>/../models (if app.py were ever moved into src/)
    """
    candidate = os.path.join(BASE_DIR, "models")
    if os.path.isdir(candidate):
        return candidate
    fallback = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
    if os.path.isdir(fallback):
        return fallback
    return candidate          # return default so errors stay meaningful


@st.cache_resource
def load_models():
    """Load trained ML artefacts from the models/ directory."""
    from lifelines import CoxPHFitter          # ensure lifelines is available

    models_dir = _resolve_models_dir()
    st.sidebar.caption(f"📁 Models dir: `{models_dir}`")
    xgb_booster, cox_model, config = None, None, {}

    # XGBoost — load the raw Booster for reliable probability inference
    xgb_path = os.path.join(models_dir, "xgb_mortality.json")
    try:
        xgb_booster = xgb.Booster()
        xgb_booster.load_model(xgb_path)
    except Exception as e:
        st.sidebar.error(f"DEBUG XGBoost error: {e}")
        xgb_booster = None

    # Cox PH model
    cox_path = os.path.join(models_dir, "cox_survival.pkl")
    try:
        with open(cox_path, "rb") as f:
            cox_model = pickle.load(f)
    except Exception as e:
        st.sidebar.error(f"DEBUG Cox error: {e}")
        cox_model = None

    # Shared config / mappings
    cfg_path = os.path.join(models_dir, "config.pkl")
    try:
        with open(cfg_path, "rb") as f:
            config = pickle.load(f)
    except Exception as e:
        st.sidebar.error(f"DEBUG Config error: {e}")

    return xgb_booster, cox_model, config


def synthesize_features(
    risk_score: float, department_name: str, config: dict
) -> pd.DataFrame:
    """
    Bridge function: translate a forecast row (Risk_Score + Department)
    into the feature vector the ML models were trained on.
    """
    # ── Age & Emergency heuristics based on risk severity ──
    if risk_score > 8:
        age_numeric, emergency_flag = 4, 1      # 70+ / critical
    elif risk_score > 6:
        age_numeric, emergency_flag = 3, 1      # 50-69 / emergency
    elif risk_score > 4:
        age_numeric, emergency_flag = 2, 0      # 30-49 / stable
    elif risk_score >= 3:
        age_numeric, emergency_flag = 1, 0      # 18-29 / healthy
    else:
        age_numeric, emergency_flag = 1, 0      # young / low-risk

    # ── Department → integer code via the saved LabelEncoder ──
    dept_le = config.get("dept_label_encoder")
    if dept_le is not None and department_name in dept_le.classes_:
        dept_code = int(dept_le.transform([department_name])[0])
    else:
        dept_code = 0

    return pd.DataFrame([{
        "Age_Numeric":    age_numeric,
        "Emergency_Flag": emergency_flag,
        "Dept_Code":      dept_code,
    }])


def _xgb_predict_proba(booster, features_df):
    """Return class probabilities [n_samples, 4] from a raw Booster."""
    dm = xgb.DMatrix(features_df)
    raw = booster.predict(dm, output_margin=True)       # raw logits
    raw = np.atleast_2d(raw)
    exp_s = np.exp(raw - raw.max(axis=1, keepdims=True))
    return exp_s / exp_s.sum(axis=1, keepdims=True)


# ── Sidebar controls ─────────────────────────────────────────────────
st.sidebar.title("Project Aesclepius: 2026 Simulator")

budget = st.sidebar.number_input(
    "Total Budget ($)",
    min_value=1_000_000,
    max_value=500_000_000_000,
    value=30_000_000,
    step=1_000_000,
    format="%d",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Priority Weights")

w_efficiency = st.sidebar.slider(
    "⚙️ Efficiency Weight (cost-efficiency)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
)

w_humanity = st.sidebar.slider(
    "❤️ Humanity Weight (clinical risk)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
)

st.sidebar.caption(
    "**Score = (Efficiency Wt × Normalized_Efficiency) "
    "+ (Humanity Wt × Normalized_Risk)**  \n"
    "Risk is scaled 0–1 (Risk_Score ÷ 10).  "
    "Efficiency is inverted cost: cheapest dept → 1.0, priciest → 0.0."
)

# ── Cost reference in sidebar ────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("MEPS Cost Reference (2026)")
for dept, base in MEPS_BASE_COSTS.items():
    projected = round(base * (1.05 ** 3), 2)
    st.sidebar.text(f"{dept}: ${projected:,.2f}/person")


# ── Main area ────────────────────────────────────────────────────────
st.title("⚕️ Project Aesclepius — 2026 Safety Net Simulator")

df = load_data()

# ── Run the engine (reactive — two independent weights) ──────────────
allocator = BudgetAllocator()
result, remaining = allocator.run_allocation(
    df, budget, w_efficiency=w_efficiency, w_humanity=w_humanity,
)

spent = budget - remaining

total_lives_saved = int(result["People_Covered"].sum())
total_patients = int(result["Projected_Volume"].sum())
total_unmet = total_patients - total_lives_saved

# ── Load ML models & compute AI mortality probability ────────────────
xgb_booster, cox_model, model_config = load_models()

if xgb_booster is not None:
    ai_probs = []
    for _, row in result.iterrows():
        syn = synthesize_features(row["Risk_Score"], row["Department"], model_config)
        proba = _xgb_predict_proba(xgb_booster, syn)[0]
        # P(Major) + P(Extreme) ≈ high-mortality probability
        ai_probs.append(round(float(proba[2] + proba[3]) * 100, 1))
    result["AI Mortality Prob (%)"] = ai_probs

# ── KPI metrics ──────────────────────────────────────────────────────
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.metric("💚 Lives Covered", f"{total_lives_saved:,}")
col2.metric("💰 Budget Spent", f"${spent:,.0f}")
col3.metric("💵 Budget Remaining", f"${remaining:,.0f}")
col4.metric("🚨 Unmet Need (patients)", f"{total_unmet:,}")

# ── Triage-line callout ──────────────────────────────────────────────
partial = result[(result["Funded_Pct"] > 0) & (result["Funded_Pct"] < 1.0)]
if not partial.empty:
    for _, row in partial.iterrows():
        pct = row["Funded_Pct"]
        dept = row["Department"]
        risk = row["Risk_Score"]
        covered = int(row["People_Covered"])
        total = int(row["Projected_Volume"])
        st.warning(
            f"⚠️ **Triage Line** fell on **{dept} Risk {risk:.0f}** — "
            f"only **{pct:.1%}** funded "
            f"({covered:,} of {total:,} patients covered)"
        )

st.markdown("---")

# ── Tabs: Triage & AI Insights ───────────────────────────────────────
tab_triage, tab_survival = st.tabs(
    ["🩺 Triage & Allocation", "📈 AI Survival Analysis"]
)

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — Triage & Budget Allocation
# ═══════════════════════════════════════════════════════════════════════
with tab_triage:
    st.subheader("Coverage by Department")

    dept_summary = (
        result
        .groupby("Department", as_index=False)
        .agg(
            Predicted_Patients=("Projected_Volume", "sum"),
            Covered_Patients=("People_Covered", "sum"),
        )
    )
    dept_summary["Uncovered_Patients"] = (
        dept_summary["Predicted_Patients"] - dept_summary["Covered_Patients"]
    )
    dept_summary["Coverage_Pct"] = (
        dept_summary["Covered_Patients"] / dept_summary["Predicted_Patients"]
    ).round(4)

    chart_data = dept_summary.melt(
        id_vars=["Department", "Coverage_Pct"],
        value_vars=["Covered_Patients", "Uncovered_Patients"],
        var_name="Status",
        value_name="Patients",
    )

    bar_chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("Department:N", title="Department",
                     axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Patients:Q", title="Patient Count", stack="zero"),
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(
                    domain=["Covered_Patients", "Uncovered_Patients"],
                    range=["#2ecc71", "#e74c3c"],
                ),
                legend=alt.Legend(title="Status"),
            ),
            tooltip=[
                "Department",
                "Status",
                alt.Tooltip("Patients:Q", format=","),
                alt.Tooltip("Coverage_Pct:Q", title="Dept Coverage",
                            format=".1%"),
            ],
        )
        .properties(height=420)
    )

    st.altair_chart(bar_chart, use_container_width=True)

    # ── Department coverage summary ──────────────────────────────────
    st.subheader("Department Coverage Summary")
    dept_display = dept_summary[
        ["Department", "Predicted_Patients", "Covered_Patients",
         "Uncovered_Patients", "Coverage_Pct"]
    ].copy()
    dept_display["Coverage_Pct"] = dept_display["Coverage_Pct"].apply(
        lambda x: f"{x:.1%}"
    )
    st.dataframe(dept_display, use_container_width=True, hide_index=True)

    # ── Unfunded risk groups ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔴 Unfunded Risk Groups (0 % Coverage)")

    unfunded = result[result["Funded_Pct"] == 0.0]

    if unfunded.empty:
        st.success("All patient groups receive at least partial funding! 🎉")
    else:
        display_cols = [
            "Department", "Risk_Score", "Projected_Volume",
            "Cost_Per_Person", "Total_Group_Cost", "Priority_Score",
        ]
        display_cols = [c for c in display_cols if c in unfunded.columns]
        st.dataframe(
            unfunded[display_cols]
            .sort_values("Priority_Score", ascending=True)
            .reset_index(drop=True),
            use_container_width=True,
        )

    # ── Full allocation table (now includes AI column) ───────────────
    with st.expander("📋 Full Allocation Table (with AI Insights)"):
        st.dataframe(result, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — AI Survival Analysis
# ═══════════════════════════════════════════════════════════════════════
with tab_survival:
    st.subheader("Projected Survival Curves (Wait-Time Impact)")
    st.markdown(
        "These curves show how quickly patients in a given department "
        "deteriorate **if funding is delayed**.  A steeper drop means "
        "shorter time-to-adverse-event — funding those groups first "
        "saves more lives."
    )

    if cox_model is None:
        st.error("Cox Proportional-Hazards model could not be loaded.")
    else:
        departments = sorted(result["Department"].unique())
        sel_dept = st.selectbox("Select Department", departments)

        # Build synthetic patients at low / medium / high risk tiers
        risk_tiers = [2, 5, 8]
        tier_labels = ["Low Risk (2)", "Medium Risk (5)", "High Risk (8)"]

        fig, ax = plt.subplots(figsize=(10, 5))
        for rs, label in zip(risk_tiers, tier_labels):
            syn = synthesize_features(rs, sel_dept, model_config)
            sf = cox_model.predict_survival_function(syn)
            ax.plot(sf.index, sf.iloc[:, 0], label=label, linewidth=2)

        ax.set_xlabel("Length of Stay (days)")
        ax.set_ylabel("Survival Probability")
        ax.set_title(f"Survival Curves — {sel_dept}")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        st.info(
            "💡 **Interpretation:** Each curve represents a synthetic patient "
            "profile at a different risk level.  The *High Risk* curve drops "
            "fastest — these patients have the greatest urgency for "
            "resource allocation."
        )

        # ── Per-department AI mortality breakdown ────────────────────
        st.markdown("---")
        st.subheader("🧠 AI Mortality Risk Breakdown")

        if xgb_booster is not None and "AI Mortality Prob (%)" in result.columns:
            dept_rows = result[result["Department"] == sel_dept].copy()
            ai_cols = [
                "Department", "Risk_Score", "Projected_Volume",
                "AI Mortality Prob (%)", "Funded_Pct",
            ]
            ai_cols = [c for c in ai_cols if c in dept_rows.columns]
            st.dataframe(
                dept_rows[ai_cols]
                .sort_values("Risk_Score")
                .reset_index(drop=True),
                use_container_width=True,
            )
        else:
            st.warning(
                "XGBoost model not available — mortality breakdown skipped."
            )
