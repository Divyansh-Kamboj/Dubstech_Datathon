import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ── Load cleaned data ──
df = pd.read_csv("cleaned_data_intermediate.csv")

# ── Step 3: Recursive Forecasting (2024 → 2026) ──
#
# The user specifies training on 2019-2023 and recursively predicting
# 2024, 2025, 2026. We average the ESTIMATE across TOPICs within each
# (Department, SUBGROUP, TIME_PERIOD) group before fitting.

TRAIN_YEARS = [2019, 2020, 2021, 2022, 2023]
FORECAST_YEARS = [2024, 2025, 2026]


def recursive_forecast(history_years, history_values, forecast_years):
    """
    Recursively predict one year at a time with Linear Regression.
    Each new prediction is appended to history before predicting the next year.
    Values are clamped to [0, 100].
    """
    years = list(history_years)
    values = list(history_values)

    for target_year in forecast_years:
        X = np.array(years).reshape(-1, 1)
        y = np.array(values)
        model = LinearRegression().fit(X, y)
        pred = model.predict(np.array([[target_year]]))[0]
        pred = float(np.clip(pred, 0.0, 100.0))
        years.append(target_year)
        values.append(pred)

    return values[-1]  # return the final year's prediction (2026)


# Aggregate: mean ESTIMATE per (Department, SUBGROUP, TIME_PERIOD)
agg = (
    df.groupby(["Department", "SUBGROUP", "Risk_Score", "TIME_PERIOD"])["ESTIMATE"]
    .mean()
    .reset_index()
)

# Filter to training window
agg_train = agg[agg["TIME_PERIOD"].isin(TRAIN_YEARS)]

# Forecast 2026 rate for each (Department, SUBGROUP) pair
forecast_rows = []
for (dept, subgrp, risk), grp in agg_train.groupby(
    ["Department", "SUBGROUP", "Risk_Score"]
):
    grp_sorted = grp.sort_values("TIME_PERIOD")
    hist_years = grp_sorted["TIME_PERIOD"].tolist()
    hist_vals = grp_sorted["ESTIMATE"].tolist()

    pred_2026 = recursive_forecast(hist_years, hist_vals, FORECAST_YEARS)

    forecast_rows.append(
        {
            "Department": dept,
            "SUBGROUP": subgrp,
            "Risk_Score": risk,
            "Predicted_Rate_2026": round(pred_2026, 4),
        }
    )

forecast_df = pd.DataFrame(forecast_rows)

print("── Predicted 2026 Rates ──")
print(forecast_df.to_string(index=False))

# ── Step 4: Convert to Patient Volume (CBO 2026 Population) ──

# Source: Congressional Budget Office (2025), "The Demographic Outlook: 2025 to 2055"
# Adjusted for 2026 projections.
# Note: dataset uses "75 years and older"; CBO label is "75 years and over".
POPULATION_2026 = {
    "18-44 years": 118_000_000,
    "45-64 years": 85_000_000,
    "65-74 years": 37_000_000,
    "75 years and older": 26_000_000,  # mapped to match dataset label
}

forecast_df["Predicted_Patients_2026"] = (
    (forecast_df["Predicted_Rate_2026"] / 100)
    * forecast_df["SUBGROUP"].map(POPULATION_2026)
).astype(int)

# ── Final Output: aggregate by Department + Risk_Score ──

final = (
    forecast_df.groupby(["Department", "Risk_Score"], as_index=False)[
        "Predicted_Patients_2026"
    ]
    .sum()
    .sort_values(["Department", "Risk_Score"])
)

print("\n── 2026 Forecast (Department × Risk_Score) ──")
print(final.to_string(index=False))

total = final["Predicted_Patients_2026"].sum()
print(f"\n✅ Total predicted sick patients in 2026: {total:,}")

# Save
final.to_csv("2026_forecast.csv", index=False)
print("Saved → 2026_forecast.csv")
