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
from fastapi import FastAPI, Request, Response
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
    print(f"DEBUG: CSV Columns are {df.columns.tolist()}")

    # Normalize header whitespace and support alternative department names.
    df.columns = [str(col).strip() for col in df.columns]
    lower_to_actual = {str(col).lower(): str(col) for col in df.columns}

    if "department" not in lower_to_actual:
        for candidate in ("dept", "sector"):
            if candidate in lower_to_actual:
                df = df.rename(columns={lower_to_actual[candidate]: "Department"})
                break
        else:
            # Fallback: use first column as Department.
            df = df.rename(columns={df.columns[0]: "Department"})
    elif lower_to_actual["department"] != "Department":
        df = df.rename(columns={lower_to_actual["department"]: "Department"})

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


@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    # Check if the request is coming from your domain
    origin = request.headers.get("origin")
    allowed_origins = [
        "https://aesclepius.tech",
        "https://www.aesclepius.tech",
        "https://project-aesclepius.vercel.app",
    ]

    if request.method == "OPTIONS":
        response = Response()
        if origin in allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
        else:
            # Fallback for testing
            response.headers["Access-Control-Allow-Origin"] = "https://aesclepius.tech"

        response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

    response = await call_next(request)
    if origin in allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
    return response

XGB_BOOSTER, COX_MODELS = load_models()


@app.get("/")
async def root():
    return {"status": "online", "message": "Project Aesclepius API"}


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
