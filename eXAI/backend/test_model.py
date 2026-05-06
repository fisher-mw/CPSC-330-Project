"""
test_model.py  --  Validate the serialized pipeline before building the frontend.

Two test modes:
  1. Direct model test  (no server needed) -- loads model.pkl and runs predictions
  2. Live API test      (server must be running) -- hits the FastAPI endpoints

Usage:
    # Mode 1 — just the model
    python test_model.py

    # Mode 2 — model + running API
    python test_model.py --api
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Two contrasting test cases ────────────────────────────────────────────────
# EDUCATION is the already-mapped ordinal: 0=unknown, 1=hs, 2=uni, 3=grad
# PAY_X: -1=paid on time, 0=revolving credit, 1=1-month late, ... 9=9+ months

CASES = [
    {
        "label": "LOW-RISK — consistent payer, high limit",
        "input": {
            "ID": 0,
            "LIMIT_BAL": 500_000,
            "SEX": 2,
            "EDUCATION": 3,    # graduate school
            "MARRIAGE": 2,     # single
            "AGE": 32,
            "PAY_0": -1, "PAY_2": -1, "PAY_3": -1,
            "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
            "BILL_AMT1": 30_000, "BILL_AMT2": 28_000, "BILL_AMT3": 25_000,
            "BILL_AMT4": 22_000, "BILL_AMT5": 20_000, "BILL_AMT6": 18_000,
            "PAY_AMT1": 30_000, "PAY_AMT2": 28_000, "PAY_AMT3": 25_000,
            "PAY_AMT4": 22_000, "PAY_AMT5": 20_000, "PAY_AMT6": 18_000,
        },
    },
    {
        "label": "HIGH-RISK — repeated late payments, low limit",
        "input": {
            "ID": 0,
            "LIMIT_BAL": 20_000,
            "SEX": 1,
            "EDUCATION": 1,    # high school
            "MARRIAGE": 1,     # married
            "AGE": 45,
            "PAY_0": 3, "PAY_2": 2, "PAY_3": 2,
            "PAY_4": 1, "PAY_5": 1, "PAY_6": 0,
            "BILL_AMT1": 19_500, "BILL_AMT2": 19_000, "BILL_AMT3": 18_500,
            "BILL_AMT4": 18_000, "BILL_AMT5": 17_500, "BILL_AMT6": 17_000,
            "PAY_AMT1": 500,  "PAY_AMT2": 500,  "PAY_AMT3": 500,
            "PAY_AMT4": 500,  "PAY_AMT5": 500,  "PAY_AMT6": 500,
        },
    },
    {
        "label": "EDGE CASE — minimum values",
        "input": {
            "ID": 0,
            "LIMIT_BAL": 10_000,
            "SEX": 1,
            "EDUCATION": 0,    # unknown
            "MARRIAGE": 3,     # other
            "AGE": 18,
            "PAY_0": 0, "PAY_2": 0, "PAY_3": 0,
            "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
            "BILL_AMT1": 0, "BILL_AMT2": 0, "BILL_AMT3": 0,
            "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
            "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0,
            "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0,
        },
    },
]


# ── Mode 1: Direct model test ─────────────────────────────────────────────────

def test_model_direct():
    model_path = os.path.join(HERE, "model.pkl")
    if not os.path.exists(model_path):
        print("ERROR: model.pkl not found. Run train_and_export.py first.")
        sys.exit(1)

    import shap
    pipeline = joblib.load(model_path)
    rf       = pipeline.named_steps["randomforestclassifier"]
    prep     = pipeline.named_steps["columntransformer"]
    explainer = shap.TreeExplainer(rf)

    feature_names = prep.get_feature_names_out().tolist()

    # Load display names from metadata if available
    meta_path = os.path.join(HERE, "model_metadata.json")
    display_names = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            display_names = json.load(f).get("display_names", {})

    print("=" * 60)
    print("DIRECT MODEL TEST  (model.pkl)")
    print("=" * 60)

    all_passed = True

    for case in CASES:
        print(f"\n>>> {case['label']}")
        print("-" * 60)

        df = pd.DataFrame([case["input"]])

        proba      = pipeline.predict_proba(df)[0]
        prediction = int(pipeline.predict(df)[0])
        label      = "DEFAULT" if prediction == 1 else "NO DEFAULT"

        print(f"  Prediction        : {label}")
        print(f"  P(default)        : {proba[1]:.4f}  ({proba[1]*100:.1f}%)")
        print(f"  P(no default)     : {proba[0]:.4f}  ({proba[0]*100:.1f}%)")

        # SHAP for this instance
        X_proc    = prep.transform(df)
        shap_vals = explainer.shap_values(X_proc)

        if isinstance(shap_vals, list):
            sv = np.array(shap_vals[1])[0]
        else:
            sv = np.array(shap_vals)[0, :, 1]

        # Top 5 drivers
        ranked = sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)
        print("  Top 5 SHAP drivers:")
        for fn, val in ranked[:5]:
            name = display_names.get(fn, fn)
            bar  = "+" * int(abs(val) * 100) if val > 0 else "-" * int(abs(val) * 100)
            direction = "-> default" if val > 0 else "-> no default"
            print(f"    {name:<30s}  {val:+.4f}  {direction}")

        # Sanity checks
        assert 0.0 <= proba[0] <= 1.0, "probability out of range"
        assert 0.0 <= proba[1] <= 1.0, "probability out of range"
        assert abs(proba[0] + proba[1] - 1.0) < 1e-6, "probabilities don't sum to 1"
        assert prediction in (0, 1), "prediction must be 0 or 1"
        assert len(sv) == len(feature_names), "SHAP length mismatch"
        print("  Assertions        : PASSED")

    print("\n" + "=" * 60)
    print("All direct model tests passed." if all_passed else "Some tests failed.")
    print("=" * 60)


# ── Mode 2: Live API test ─────────────────────────────────────────────────────

def test_api_live(base_url: str = "http://localhost:8000"):
    try:
        import requests
    except ImportError:
        print("Install requests:  pip install requests")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"LIVE API TEST  ({base_url})")
    print("=" * 60)

    # /health
    r = requests.get(f"{base_url}/health", timeout=5)
    r.raise_for_status()
    print(f"\n/health   -> {r.json()}")

    # /metadata
    r = requests.get(f"{base_url}/metadata", timeout=5)
    r.raise_for_status()
    meta = r.json()
    print(f"\n/metadata -> metrics: {meta['metrics']}")
    print(f"           top SHAP feature: {meta['shap_importance'][0]['display_name']}"
          f"  (importance={meta['shap_importance'][0]['importance']:.4f})")

    # /eda
    r = requests.get(f"{base_url}/eda", timeout=5)
    r.raise_for_status()
    eda = r.json()
    print(f"\n/eda      -> features available: {list(eda.keys())}")

    # /predict for each test case
    for case in CASES:
        inp = case["input"].copy()
        inp.pop("ID")  # API schema doesn't include ID

        # Map raw column names to API snake_case field names
        api_payload = {
            "limit_bal": inp["LIMIT_BAL"],
            "sex":       inp["SEX"],
            "education": inp["EDUCATION"],
            "marriage":  inp["MARRIAGE"],
            "age":       inp["AGE"],
            "pay_0":     inp["PAY_0"],
            "pay_2":     inp["PAY_2"],
            "pay_3":     inp["PAY_3"],
            "pay_4":     inp["PAY_4"],
            "pay_5":     inp["PAY_5"],
            "pay_6":     inp["PAY_6"],
            "bill_amt1": inp["BILL_AMT1"], "bill_amt2": inp["BILL_AMT2"],
            "bill_amt3": inp["BILL_AMT3"], "bill_amt4": inp["BILL_AMT4"],
            "bill_amt5": inp["BILL_AMT5"], "bill_amt6": inp["BILL_AMT6"],
            "pay_amt1":  inp["PAY_AMT1"],  "pay_amt2":  inp["PAY_AMT2"],
            "pay_amt3":  inp["PAY_AMT3"],  "pay_amt4":  inp["PAY_AMT4"],
            "pay_amt5":  inp["PAY_AMT5"],  "pay_amt6":  inp["PAY_AMT6"],
        }

        r = requests.post(f"{base_url}/predict", json=api_payload, timeout=10)
        r.raise_for_status()
        result = r.json()

        print(f"\n>>> {case['label']}")
        print(f"  label             : {result['label']}")
        print(f"  P(default)        : {result['probability_default']}")
        top = result["shap_breakdown"][0]
        print(f"  #1 SHAP driver    : {top['display_name']}  ({top['shap_value']:+.4f})")

    print("\n" + "=" * 60)
    print("All API tests passed.")
    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true",
                        help="Also run live API tests (requires uvicorn running)")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Base URL for the API (default: http://localhost:8000)")
    args = parser.parse_args()

    test_model_direct()

    if args.api:
        test_api_live(args.url)
    else:
        print("\nTip: run with --api to also test the live FastAPI endpoints.")
        print("     (start the server first with: uvicorn main:app --reload)")
