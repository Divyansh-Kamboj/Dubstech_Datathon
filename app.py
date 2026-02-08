"""
Project Aesclepius — 2026 Simulator Dashboard
==============================================
Streamlit front-end for the Safety Net budget-allocation engine.

Design note: there is NO button gate.  The solver runs reactively
every time a sidebar slider or input changes, which is the standard
Streamlit pattern for dashboards.
"""

import streamlit as st
import pandas as pd
import altair as alt

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
            alt.Tooltip("Coverage_Pct:Q", title="Dept Coverage", format=".1%"),
        ],
    )
    .properties(height=420)
)

st.altair_chart(bar_chart, use_container_width=True)
dept_display = dept_summary[
    ["Department", "Predicted_Patients", "Covered_Patients",
     "Uncovered_Patients", "Coverage_Pct"]
].copy()
dept_display["Coverage_Pct"] = dept_display["Coverage_Pct"].apply(
    lambda x: f"{x:.1%}"
)
st.dataframe(dept_display, use_container_width=True, hide_index=True)

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

# ── Full results ─────────────────────────────────────────────────────
with st.expander("📋 Full Allocation Table"):
    st.dataframe(result, use_container_width=True)
