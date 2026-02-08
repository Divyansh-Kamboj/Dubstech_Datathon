"""
Safety Net Simulator 2026 — Main Pipeline
==========================================
Loads demand data, applies MEPS-projected costs, runs the budget
allocation engine, and prints a human-readable summary.
"""

import pandas as pd

from src.costs import apply_meps_costs
from src.engine import BudgetAllocator

# ── Configuration ────────────────────────────────────────────────────
DATA_PATH = "data/mock_demand_2026.csv"
TOTAL_BUDGET = 30_000_000        # $30 M (realistic for 200k catchment)
W_EFFICIENCY = 0.5               # Balanced: efficiency vs. humanity
W_HUMANITY = 0.5


def main():
    # 1. Load data --------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} patient groups from {DATA_PATH}\n")

    # 2. Apply MEPS 2026 projected costs ----------------------------------
    df = apply_meps_costs(df)

    # 3. Run allocation engine --------------------------------------------
    allocator = BudgetAllocator()
    result, remaining = allocator.run_allocation(
        df, TOTAL_BUDGET, w_efficiency=W_EFFICIENCY, w_humanity=W_HUMANITY
    )

    # 4. Summary ----------------------------------------------------------
    funded = result[result["Funded_Status"].isin(["Yes", "Partial"])]
    unfunded = result[result["Funded_Status"] == "No"]

    spent = TOTAL_BUDGET - remaining
    utilization = (spent / TOTAL_BUDGET) * 100
    total_patients_saved = int(result["People_Covered"].sum())

    print("=" * 55)
    print("  SAFETY NET SIMULATOR 2026 — ALLOCATION SUMMARY")
    print("=" * 55)
    print(f"  Budget:              ${TOTAL_BUDGET:>14,.2f}")
    print(f"  Efficiency Weight:   {W_EFFICIENCY}")
    print(f"  Humanity Weight:     {W_HUMANITY}")
    print("-" * 55)
    print(f"  Groups Funded:       {len(funded):>6}  / {len(result)}")
    print(f"  Groups Unfunded:     {len(unfunded):>6}  / {len(result)}")
    print(f"  Total Patients Saved:{total_patients_saved:>7,}")
    print("-" * 55)
    print(f"  Budget Spent:        ${spent:>14,.2f}")
    print(f"  Budget Remaining:    ${remaining:>14,.2f}")
    print(f"  Budget Utilization:  {utilization:>13.2f} %")
    print("=" * 55)


if __name__ == "__main__":
    main()
