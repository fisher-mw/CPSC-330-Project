"""
main.py  --  Phase 2: FastAPI Backend

Serves the trained Random Forest pipeline via three endpoints:

    POST /predict     Runs a prediction + per-instance SHAP breakdown
    GET  /metadata    Returns model metrics, SHAP importance, and class distribution
    GET  /eda         Returns pre-computed histogram/bar data for the EDA page

Prerequisites:
    Run train_and_export.py first to generate model.pkl,
    model_metadata.json, and eda_data.json in this directory.

Start:
    uvicorn main:app --reload
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))

# ── Global state loaded at startup ───────────────────────────────────────────
pipeline  = None
explainer = None
metadata  = None
eda_data  = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all artifacts once when the server starts."""
    global pipeline, explainer, metadata, eda_data

    model_path    = os.path.join(HERE, "model.pkl")
    metadata_path = os.path.join(HERE, "model_metadata.json")
    eda_path      = os.path.join(HERE, "eda_data.json")

    for path in (model_path, metadata_path, eda_path):
        if not os.path.exists(path):
            raise RuntimeError(
                f"Missing artifact: {path}\n"
                "Run 'python train_and_export.py' first."
            )

    print("Loading model pipeline...")
    pipeline = joblib.load(model_path)

    print("Initializing SHAP TreeExplainer...")
    rf_model  = pipeline.named_steps["randomforestclassifier"]
    explainer = shap.TreeExplainer(rf_model)

    with open(metadata_path) as f:
        metadata = json.load(f)

    with open(eda_path) as f:
        eda_data = json.load(f)

    print("API ready.")
    yield  # server runs here


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Card Default Classifier API",
    description=(
        "Explainable AI backend for the CPSC 330 credit default classifier. "
        "Built on a tuned Random Forest with SHAP-based interpretability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # CRA + Vite
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    All monetary values are in New Taiwan Dollars (TWD).

    EDUCATION:
        0 = Unknown / Other
        1 = High School
        2 = University
        3 = Graduate School

    SEX:
        1 = Male
        2 = Female

    MARRIAGE:
        1 = Married
        2 = Single
        3 = Other

    PAY_0 / PAY_2 … PAY_6  (repayment status, most recent first):
        -1 = Paid in full / paid on time
         0 = Revolving credit use (minimum payment made)
         1 = 1 month late
         2 = 2 months late
         ...
         9 = 9+ months late
    """
    # Demographics
    limit_bal: float = Field(..., ge=10_000, le=1_000_000,
                             description="Credit limit in TWD (10,000–1,000,000)")
    sex:       int   = Field(..., ge=1, le=2,
                             description="1=Male, 2=Female")
    education: int   = Field(..., ge=0, le=3,
                             description="0=Other, 1=High School, 2=University, 3=Graduate")
    marriage:  int   = Field(..., ge=1, le=3,
                             description="1=Married, 2=Single, 3=Other")
    age:       int   = Field(..., ge=18, le=100,
                             description="Age in years")

    # Repayment status — Sept → Apr 2005
    pay_0: int = Field(..., ge=-2, le=9, description="Repayment status Sept 2005")
    pay_2: int = Field(..., ge=-2, le=9, description="Repayment status Aug 2005")
    pay_3: int = Field(..., ge=-2, le=9, description="Repayment status Jul 2005")
    pay_4: int = Field(..., ge=-2, le=9, description="Repayment status Jun 2005")
    pay_5: int = Field(..., ge=-2, le=9, description="Repayment status May 2005")
    pay_6: int = Field(..., ge=-2, le=9, description="Repayment status Apr 2005")

    # Bill statement amounts — Sept → Apr 2005
    bill_amt1: float = Field(..., description="Bill amount Sept 2005 (TWD)")
    bill_amt2: float = Field(..., description="Bill amount Aug 2005 (TWD)")
    bill_amt3: float = Field(..., description="Bill amount Jul 2005 (TWD)")
    bill_amt4: float = Field(..., description="Bill amount Jun 2005 (TWD)")
    bill_amt5: float = Field(..., description="Bill amount May 2005 (TWD)")
    bill_amt6: float = Field(..., description="Bill amount Apr 2005 (TWD)")

    # Previous payment amounts — Sept → Apr 2005
    pay_amt1: float = Field(..., ge=0, description="Payment made Sept 2005 (TWD)")
    pay_amt2: float = Field(..., ge=0, description="Payment made Aug 2005 (TWD)")
    pay_amt3: float = Field(..., ge=0, description="Payment made Jul 2005 (TWD)")
    pay_amt4: float = Field(..., ge=0, description="Payment made Jun 2005 (TWD)")
    pay_amt5: float = Field(..., ge=0, description="Payment made May 2005 (TWD)")
    pay_amt6: float = Field(..., ge=0, description="Payment made Apr 2005 (TWD)")

    model_config = {"json_schema_extra": {
        "example": {
            "limit_bal": 200000, "sex": 2, "education": 2, "marriage": 1, "age": 35,
            "pay_0": 0, "pay_2": 0, "pay_3": 0, "pay_4": 0, "pay_5": -1, "pay_6": -1,
            "bill_amt1": 50000, "bill_amt2": 48000, "bill_amt3": 45000,
            "bill_amt4": 42000, "bill_amt5": 40000, "bill_amt6": 38000,
            "pay_amt1": 5000, "pay_amt2": 5000, "pay_amt3": 4000,
            "pay_amt4": 4000, "pay_amt5": 3000, "pay_amt6": 3000,
        }
    }}


class SHAPFeature(BaseModel):
    feature:      str
    display_name: str
    shap_value:   float


class PredictResponse(BaseModel):
    prediction:            int          # 0 or 1
    label:                 str          # "No Default" or "Default"
    probability_default:   float        # P(default)
    probability_no_default: float       # P(no default)
    shap_breakdown:        list[SHAPFeature]  # sorted by |shap_value| desc


# ── Helper: build a single-row DataFrame matching the training schema ─────────

def _build_dataframe(req: PredictRequest) -> pd.DataFrame:
    """
    Construct a one-row DataFrame using the exact column names the
    ColumnTransformer was fitted on. EDUCATION is already in mapped
    ordinal form (0–3) from the request.
    """
    return pd.DataFrame([{
        "ID":        0,            # column is present but will be dropped
        "LIMIT_BAL": req.limit_bal,
        "SEX":       req.sex,
        "EDUCATION": req.education,  # 0=unknown, 1=hs, 2=uni, 3=grad
        "MARRIAGE":  req.marriage,
        "AGE":       req.age,
        "PAY_0":     req.pay_0,
        "PAY_2":     req.pay_2,
        "PAY_3":     req.pay_3,
        "PAY_4":     req.pay_4,
        "PAY_5":     req.pay_5,
        "PAY_6":     req.pay_6,
        "BILL_AMT1": req.bill_amt1,
        "BILL_AMT2": req.bill_amt2,
        "BILL_AMT3": req.bill_amt3,
        "BILL_AMT4": req.bill_amt4,
        "BILL_AMT5": req.bill_amt5,
        "BILL_AMT6": req.bill_amt6,
        "PAY_AMT1":  req.pay_amt1,
        "PAY_AMT2":  req.pay_amt2,
        "PAY_AMT3":  req.pay_amt3,
        "PAY_AMT4":  req.pay_amt4,
        "PAY_AMT5":  req.pay_amt5,
        "PAY_AMT6":  req.pay_amt6,
    }])


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """
    Classify a single applicant and return:
      - Binary prediction (0/1) and human-readable label
      - Default probability
      - Per-feature SHAP values for this specific prediction
    """
    try:
        df = _build_dataframe(req)

        # Prediction
        proba      = pipeline.predict_proba(df)[0]
        prediction = int(pipeline.predict(df)[0])

        # SHAP for this instance
        preprocessor  = pipeline.named_steps["columntransformer"]
        X_transformed = preprocessor.transform(df)
        feature_names = preprocessor.get_feature_names_out().tolist()
        display_names = metadata["display_names"]

        shap_vals = explainer.shap_values(X_transformed)

        if isinstance(shap_vals, list):
            instance_shap = np.array(shap_vals[1])[0]
        else:
            instance_shap = np.array(shap_vals)[0, :, 1]

        # Sort by absolute value descending so the UI shows the biggest drivers first
        shap_items = [
            SHAPFeature(
                feature=fn,
                display_name=display_names.get(fn, fn),
                shap_value=round(float(sv), 6),
            )
            for fn, sv in zip(feature_names, instance_shap)
        ]
        shap_items.sort(key=lambda x: abs(x.shap_value), reverse=True)

        return PredictResponse(
            prediction=prediction,
            label="Default" if prediction == 1 else "No Default",
            probability_default=round(float(proba[1]), 4),
            probability_no_default=round(float(proba[0]), 4),
            shap_breakdown=shap_items,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metadata")
def get_metadata() -> dict[str, Any]:
    """
    Returns static model information for the Explainability page:
      - Test-set performance metrics vs. dummy baseline
      - Class distribution
      - SHAP-based global feature importance (mean |SHAP| over training sample)
      - Model hyperparameter info
    """
    return metadata


@app.get("/eda")
def get_eda() -> dict[str, Any]:
    """
    Returns pre-computed histogram / bar-chart data for key features,
    split by default status, in Recharts-compatible format.
    """
    return eda_data


@app.get("/health")
def health() -> dict[str, str]:
    """Simple liveness check."""
    return {"status": "ok"}
