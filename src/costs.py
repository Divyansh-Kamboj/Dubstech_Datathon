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
    multiplier = (1 + INFLATION_RATE) ** PROJECTION_YEARS

    df["Cost_Per_Person"] = (
        df["Department"]
        .map(MEPS_BASE_COSTS)
        .mul(multiplier)
        .round(2)
    )

    df["Total_Group_Cost"] = (
        df["Cost_Per_Person"] * df["Projected_Volume"]
    ).round(2)

    return df
