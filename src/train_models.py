"""
Project Aesclepius — ML Model Training
=======================================
Trains two models on the NY SPARCS 2022 inpatient dataset:

  A.  **XGBoost Classifier** — predicts APR Risk-of-Mortality level (0-3).
  B.  **Cox Proportional-Hazards** — predicts survival curves using
      Length of Stay as the duration and patient expiry as the event.

Artefacts are saved under ``models/``.
"""

import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# ──────────────────────────────────────────────────────────────────────
# 1. Paths & constants
# ──────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "sparcs_2022.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# APR MDC Codes → our 4 departments
DEPT_MAP = {
    5:  "Cardiology",
    17: "Oncology",
    10: "Endocrinology",
    19: "Mental Health",
}

# Mortality-string → integer encoding
MORTALITY_MAP = {
    "Minor":    0,
    "Moderate":  1,
    "Major":     2,
    "Extreme":   3,
}

# Age-group → ordinal encoding
AGE_MAP = {
    "0 to 17":     0,
    "18 to 29":    1,
    "30 to 49":    2,
    "50 to 69":    3,
    "70 or Older":  4,
}


def main() -> None:
    # ──────────────────────────────────────────────────────────────────
    # 2. Safety: create models/ directory
    # ──────────────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────
    # 3. Load data (cap at 500 000 rows for memory safety)
    # ──────────────────────────────────────────────────────────────────
    print("📂  Loading SPARCS 2022 data …")
    df = pd.read_csv(DATA_PATH, low_memory=False, nrows=500_000)
    print(f"   Raw shape: {df.shape}")

    # ──────────────────────────────────────────────────────────────────
    # 4. Filter for our 4 departments
    # ──────────────────────────────────────────────────────────────────
    df["APR MDC Code"] = pd.to_numeric(df["APR MDC Code"], errors="coerce")
    df = df[df["APR MDC Code"].isin(DEPT_MAP.keys())].copy()
    df["Department"] = df["APR MDC Code"].map(DEPT_MAP)
    print(f"   After department filter: {df.shape}")
    print(f"   Department distribution:\n{df['Department'].value_counts().to_string()}\n")

    # ──────────────────────────────────────────────────────────────────
    # 5. Drop rows with missing targets
    # ──────────────────────────────────────────────────────────────────
    df = df.dropna(subset=["APR Risk of Mortality", "Length of Stay"])
    print(f"   After dropping NaNs: {df.shape}")

    # ──────────────────────────────────────────────────────────────────
    # 6. Preprocessing
    # ──────────────────────────────────────────────────────────────────

    # Target A — Mortality level (ordinal 0-3)
    df["Mortality_Level"] = df["APR Risk of Mortality"].map(MORTALITY_MAP)
    df = df.dropna(subset=["Mortality_Level"])          # drop unmapped values
    df["Mortality_Level"] = df["Mortality_Level"].astype(int)

    # Target B — Event column for Cox model
    df["Event"] = (df["Patient Disposition"] == "Expired").astype(int)

    # Length of Stay → numeric (some SPARCS files encode 120+ as a string)
    df["Length of Stay"] = pd.to_numeric(
        df["Length of Stay"].astype(str).str.replace("+", "", regex=False),
        errors="coerce",
    )
    df = df.dropna(subset=["Length of Stay"])
    df["Length of Stay"] = df["Length of Stay"].astype(float)
    # Cox requires durations > 0
    df["Length of Stay"] = df["Length of Stay"].clip(lower=0.5)

    # Feature: Age_Numeric
    df["Age_Numeric"] = df["Age Group"].map(AGE_MAP).fillna(-1).astype(int)

    # Feature: Emergency_Flag
    df["Emergency_Flag"] = (df["Emergency Department Indicator"] == "Y").astype(int)

    # Feature: Dept_Code (label-encoded 0-3)
    dept_le = LabelEncoder()
    df["Dept_Code"] = dept_le.fit_transform(df["Department"])

    print(f"   Final training set: {df.shape}")
    print(f"   Mortality_Level distribution:\n{df['Mortality_Level'].value_counts().sort_index().to_string()}")
    print(f"   Event (Expired) distribution:\n{df['Event'].value_counts().to_string()}\n")

    # ──────────────────────────────────────────────────────────────────
    # 7. Define feature matrix
    # ──────────────────────────────────────────────────────────────────
    FEATURES = ["Age_Numeric", "Emergency_Flag", "Dept_Code"]

    # ──────────────────────────────────────────────────────────────────
    # 8. Model A — XGBoost Classifier
    # ──────────────────────────────────────────────────────────────────
    print("🌲  Training XGBoost classifier …")
    X = df[FEATURES].values
    y = df["Mortality_Level"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=4,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    y_pred = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n   XGBoost accuracy: {acc:.4f}")
    print(classification_report(
        y_test, y_pred,
        target_names=["Minor", "Moderate", "Major", "Extreme"],
    ))

    xgb_path = os.path.join(MODELS_DIR, "xgb_mortality.json")
    xgb_model.save_model(xgb_path)
    print(f"   ✅ Saved XGBoost model → {xgb_path}")

    # ──────────────────────────────────────────────────────────────────
    # 9. Model B — Cox Proportional-Hazards
    # ──────────────────────────────────────────────────────────────────
    print("\n📈  Training Cox Proportional-Hazards model …")
    cox_cols = FEATURES + ["Length of Stay", "Event"]
    cox_df = df[cox_cols].copy()

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="Length of Stay", event_col="Event")
    cph.print_summary()

    cox_path = os.path.join(MODELS_DIR, "cox_survival.pkl")
    with open(cox_path, "wb") as f:
        pickle.dump(cph, f)
    print(f"   ✅ Saved Cox model → {cox_path}")

    # ──────────────────────────────────────────────────────────────────
    # 10. Save configuration / mappings for the app
    # ──────────────────────────────────────────────────────────────────
    config = {
        "mortality_map":     MORTALITY_MAP,
        "inv_mortality_map": {v: k for k, v in MORTALITY_MAP.items()},
        "age_map":           AGE_MAP,
        "dept_map":          DEPT_MAP,
        "dept_label_encoder": dept_le,
        "features":          FEATURES,
    }
    config_path = os.path.join(MODELS_DIR, "config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(config, f)
    print(f"   ✅ Saved config mappings → {config_path}")

    print("\n🎉  All models trained and saved successfully!")


if __name__ == "__main__":
    main()
