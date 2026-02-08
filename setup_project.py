"""
Safety Net Simulator 2026 — Project Setup
==========================================
Creates the project directory structure and generates a synthetic
demand-forecast dataset at data/mock_demand_2026.csv (100 rows).
"""

import os
import random
import csv

# ---------------------------------------------------------------------------
# 1. Ensure project directories exist
# ---------------------------------------------------------------------------
DIRS = ["data", "src", "notebooks"]

for d in DIRS:
    os.makedirs(d, exist_ok=True)
    print(f"✔  Directory ready: {d}/")

# ---------------------------------------------------------------------------
# 2. Configuration for synthetic data
# ---------------------------------------------------------------------------
NUM_ROWS = 100
DEPARTMENTS = ["Cardiology", "Oncology", "Endocrinology", "Mental Health"]

# Risk-score ranges biased by department
RISK_RANGES = {
    "Cardiology":     (7, 10),
    "Oncology":       (7, 10),
    "Endocrinology":  (1, 6),
    "Mental Health":  (1, 6),
}

OUTPUT_PATH = os.path.join("data", "mock_demand_2026.csv")

# ---------------------------------------------------------------------------
# 3. Generate and write the dataset
# ---------------------------------------------------------------------------
random.seed(42)  # reproducibility

rows = []
for i in range(1, NUM_ROWS + 1):
    dept = random.choice(DEPARTMENTS)
    risk_lo, risk_hi = RISK_RANGES[dept]
    rows.append({
        "Group_ID":         f"P{i:03d}",
        "Department":       dept,
        "Projected_Volume": random.randint(100, 2000),
        "Risk_Score":       random.randint(risk_lo, risk_hi),
    })

with open(OUTPUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Group_ID", "Department",
                                            "Projected_Volume", "Risk_Score"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\n✔  Generated {NUM_ROWS} rows → {OUTPUT_PATH}")
print(f"   Columns: Group_ID, Department, Projected_Volume, Risk_Score")
print(f"\n🎉 Safety Net Simulator 2026 — setup complete!")
