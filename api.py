"""
Standalone FastAPI backend for Project Aesclepius.

Run with:
    python api.py
"""

from __future__ import annotations

import os
import pickle
from typing import Any

import pandas as pd
import uvicorn
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.costs import apply_meps_costs
from src.engine import BudgetAllocator


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FORECAST_PATH = os.path.join(BASE_DIR, "2026_forecast.csv")
HARDCODED_BUDGET = 150_000_000


def load_data(path: str = FORECAST_PATH) -> pd.DataFrame:
    """
    Load 2026 forecast CSV and align columns for the allocation engine.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"Predicted_Patients_2026": "Projected_Volume"})
    df["Department"] = df["Department"].str.replace("_", " ", regex=False)
    df = apply_meps_costs(df)
    return df


def load_models() -> tuple[xgb.Booster, dict[str, Any]]:
    """
    Load XGBoost and Cox models from the existing /models directory.
    """
    xgb_path = os.path.join(MODELS_DIR, "xgb_mortality.json")
    cox_path = os.path.join(MODELS_DIR, "cox_models.pkl")

    booster = xgb.Booster()
    booster.load_model(xgb_path)

    with open(cox_path, "rb") as handle:
        cox_models = pickle.load(handle)

    return booster, cox_models


class SimulationRequest(BaseModel):
    w_efficiency: float = Field(ge=0.0, le=1.0)
    w_humanity: float = Field(ge=0.0, le=1.0)


app = FastAPI(title="Project Aesclepius API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://aesclepius.tech",
        "https://www.aesclepius.tech",
        "https://project-aesclepius.vercel.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

XGB_BOOSTER, COX_MODELS = load_models()


@app.post("/simulate")
def simulate(payload: SimulationRequest) -> dict[str, Any]:
    """
    Run allocator using fixed budget and return summary + chart data.
    """
    df = load_data()
    allocator = BudgetAllocator()
    result, remaining_budget = allocator.run_allocation(
        df,
        HARDCODED_BUDGET,
        w_efficiency=payload.w_efficiency,
        w_humanity=payload.w_humanity,
    )

    lives_covered = int(result["People_Covered"].sum())
    unmet_need = int(result["Projected_Volume"].sum()) - lives_covered

    chart_data = [
        {
            "Department": str(row["Department"]),
            "Funded_Pct": float(row["Funded_Pct"]),
        }
        for _, row in result.iterrows()
    ]

    return {
        "lives_covered": lives_covered,
        "unmet_need": unmet_need,
        "budget_remaining": float(remaining_budget),
        "chart_data": chart_data,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
