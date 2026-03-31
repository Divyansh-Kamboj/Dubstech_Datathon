"""
Safety Net Simulator 2026 — MEPS Cost Projections
===================================================
Applies real MEPS 2022/2023 base costs per department and projects
them forward to 2026 using a 5 % annual medical-inflation multiplier.
"""

# Base per-person costs from MEPS 2022/2023 data
MEPS_BASE_COSTS = {
    "Cardiology":    4_655,
    "Oncology":      10_823,
    "Endocrinology": 6_054,
    "Mental Health":  3_292,
}

INFLATION_RATE = 0.05   # 5 % annual increase
PROJECTION_YEARS = 3    # 2023 → 2026


def apply_meps_costs(df):
    """
    Enrich a demand DataFrame with projected 2026 costs.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ``Department`` and ``Projected_Volume``.

    Returns
    -------
    pandas.DataFrame
        The same DataFrame with two new columns:
        - ``Cost_Per_Person``  – MEPS base cost inflated to 2026.
        - ``Total_Group_Cost`` – Cost_Per_Person × Projected_Volume.
    """
    # Normalize headers in a working copy so env-specific spacing/casing
    # differences do not break column access.
    normalized_df = df.copy()
    normalized_df.columns = [str(col).strip().lower() for col in normalized_df.columns]

    if "projected_volume" not in normalized_df.columns:
        print(
            "DEBUG: apply_meps_costs missing 'projected_volume'. "
            f"Available columns: {df.columns.tolist()}"
        )
        return df

    if "department" not in normalized_df.columns:
        print(
            "DEBUG: apply_meps_costs missing 'department'. "
            f"Available columns: {df.columns.tolist()}"
        )
        return df

    multiplier = (1 + INFLATION_RATE) ** PROJECTION_YEARS
    normalized_costs = {k.lower(): v for k, v in MEPS_BASE_COSTS.items()}

    department_series = (
        normalized_df["department"].astype(str).str.strip().str.replace("_", " ", regex=False)
    )

    df["Cost_Per_Person"] = (
        department_series.str.lower().map(normalized_costs).mul(multiplier).round(2)
    )

    df["Total_Group_Cost"] = (df["Cost_Per_Person"] * normalized_df["projected_volume"]).round(2)

    return df
