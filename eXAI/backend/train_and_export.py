"""
train_and_export.py  --  Phase 1: Model Serialization

Run this script ONCE from any working directory to produce three artifacts
inside eXAI/backend/:

    model.pkl           Full sklearn pipeline  (preprocessor + Random Forest)
    model_metadata.json Performance metrics, feature display names, SHAP summary
    eda_data.json       Pre-computed histogram/bar data for the EDA page

Expects the raw dataset at:
    <repo-root>/data/UCI_Credit_Card.csv

Usage:
    cd eXAI/backend
    python train_and_export.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "..", "data", "UCI_Credit_Card.csv")
RANDOM_STATE = 123

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data from:", DATA_PATH)
raw_data = pd.read_csv(DATA_PATH)

# ── 2. EDUCATION ordinal map (identical to notebook) ─────────────────────────
#   Original encoding:  1=grad school, 2=university, 3=high school, 0/4/5/6=unknown
#   Mapped encoding:    0=unknown, 1=high school, 2=university, 3=grad school
edu_map = {0: 0, 5: 0, 6: 0, 4: 0, 3: 1, 2: 2, 1: 3}
raw_data["EDUCATION"] = raw_data["EDUCATION"].map(edu_map)

# ── 3. Train / test split (identical to notebook: test_size=0.7) ──────────────
train_df, test_df = train_test_split(raw_data, test_size=0.7, random_state=RANDOM_STATE)

TARGET = "default.payment.next.month"
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test  = test_df.drop(columns=[TARGET])
y_test  = test_df[TARGET]

print(f"Training samples: {len(X_train)}  |  Test samples: {len(X_test)}")

# ── 4. Feature lists  ─────────────────────────────────
numeric_features     = [
    "EDUCATION", "LIMIT_BAL", "AGE",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
]
binary_features      = ["SEX"]
categorical_features = ["MARRIAGE"]
drop_features        = ["ID"]

# ── 5. Preprocessor ───────────────────────────────────
preprocessor = make_column_transformer(
    (StandardScaler(),                                          numeric_features),
    (OneHotEncoder(drop="if_binary", dtype=int),               binary_features),
    (OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ("drop", drop_features),
)

# ── 6. Pipeline — best hyperparams from notebook GridSearch ──────────────────
pipeline = make_pipeline(
    preprocessor,
    RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        max_features=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
)

# ── 7. Train ──────────────────────────────────────────────────────────────────
print("Training Random Forest pipeline...")
pipeline.fit(X_train, y_train)
print("Training complete.")

# ── 8. Evaluate on test set ───────────────────────────────────────────────────
print("Evaluating on test set...")
y_pred = pipeline.predict(X_test)
metrics = {
    "accuracy":          round(float(accuracy_score(y_test, y_pred)),  4),
    "precision":         round(float(precision_score(y_test, y_pred)), 4),
    "recall":            round(float(recall_score(y_test, y_pred)),    4),
    "f1":                round(float(f1_score(y_test, y_pred)),        4),
    # Dummy baseline values from the notebook for comparison on the EDA page
    "baseline_accuracy": 0.649,
    "baseline_f1":       0.225,
}
print("Metrics:", metrics)

# ── 9. SHAP feature importance (sampled for speed) ───────────────────────────
print("Computing SHAP values on 500-sample subset (this may take a minute)...")
rf_model     = pipeline.named_steps["randomforestclassifier"]
X_train_proc = pipeline.named_steps["columntransformer"].transform(X_train)
feature_names = pipeline.named_steps["columntransformer"].get_feature_names_out().tolist()

rng   = np.random.default_rng(RANDOM_STATE)
idx   = rng.choice(len(X_train_proc), size=min(500, len(X_train_proc)), replace=False)
sample = X_train_proc[idx]

explainer  = shap.TreeExplainer(rf_model)
shap_vals  = explainer.shap_values(sample)

# Handle both SHAP API versions
if isinstance(shap_vals, list):
    # Older SHAP: list[n_classes] of (n_samples, n_features)
    shap_class1 = np.array(shap_vals[1])
else:
    # Newer SHAP: (n_samples, n_features, n_classes)
    shap_class1 = np.array(shap_vals)[:, :, 1]

mean_abs_shap = np.abs(shap_class1).mean(axis=0).tolist()

# ── 10. Human-readable display names for the 25 pipeline features ─────────────
DISPLAY_NAMES = {
    "standardscaler__EDUCATION":      "Education Level",
    "standardscaler__LIMIT_BAL":      "Credit Limit (TWD)",
    "standardscaler__AGE":            "Age",
    "standardscaler__PAY_AMT1":       "Payment Amount Sept",
    "standardscaler__PAY_AMT2":       "Payment Amount Aug",
    "standardscaler__PAY_AMT3":       "Payment Amount Jul",
    "standardscaler__PAY_AMT4":       "Payment Amount Jun",
    "standardscaler__PAY_AMT5":       "Payment Amount May",
    "standardscaler__PAY_AMT6":       "Payment Amount Apr",
    "standardscaler__BILL_AMT1":      "Bill Amount Sept",
    "standardscaler__BILL_AMT2":      "Bill Amount Aug",
    "standardscaler__BILL_AMT3":      "Bill Amount Jul",
    "standardscaler__BILL_AMT4":      "Bill Amount Jun",
    "standardscaler__BILL_AMT5":      "Bill Amount May",
    "standardscaler__BILL_AMT6":      "Bill Amount Apr",
    "standardscaler__PAY_0":          "Repayment Status Sept",
    "standardscaler__PAY_2":          "Repayment Status Aug",
    "standardscaler__PAY_3":          "Repayment Status Jul",
    "standardscaler__PAY_4":          "Repayment Status Jun",
    "standardscaler__PAY_5":          "Repayment Status May",
    "standardscaler__PAY_6":          "Repayment Status Apr",
    "onehotencoder-1__SEX_2":         "Sex (Female=1)",
    "onehotencoder-2__MARRIAGE_1":    "Marital Status: Married",
    "onehotencoder-2__MARRIAGE_2":    "Marital Status: Single",
    "onehotencoder-2__MARRIAGE_3":    "Marital Status: Other",
}

shap_importance = [
    {
        "feature":      fn,
        "display_name": DISPLAY_NAMES.get(fn, fn),
        "importance":   round(v, 6),
    }
    for fn, v in sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)
]

# ── 11. Pre-compute EDA histogram / bar-chart data ────────────────────────────
print("Pre-computing EDA chart data...")

# Class distribution from the FULL dataset
class_counts = raw_data[TARGET].value_counts(normalize=True)
class_distribution = {
    "no_default": round(float(class_counts.get(0, 0)), 4),
    "default":    round(float(class_counts.get(1, 0)), 4),
}

EDA_FEATURES = {
    "PAY_0":      {"label": "Repayment Status (Sept 2005)",  "type": "ordinal"},
    "LIMIT_BAL":  {"label": "Credit Limit (TWD)",            "type": "numeric"},
    "AGE":        {"label": "Age",                           "type": "numeric"},
    "BILL_AMT1":  {"label": "Bill Amount Sept (TWD)",        "type": "numeric"},
    "PAY_AMT1":   {"label": "Payment Amount Sept (TWD)",     "type": "numeric"},
    "EDUCATION":  {"label": "Education Level (Ordinal)",     "type": "ordinal"},
}

# Recharts-friendly format: list of {name, default, no_default}
eda_data = {}

for col, meta in EDA_FEATURES.items():
    df_def = train_df[train_df[TARGET] == 1][col]
    df_ok  = train_df[train_df[TARGET] == 0][col]

    if meta["type"] == "ordinal":
        all_vals = sorted(set(df_def.dropna()) | set(df_ok.dropna()))
        eda_data[col] = {
            "label": meta["label"],
            "type":  "bar",
            "data": [
                {
                    "name":       str(v),
                    "default":    int(df_def.value_counts().get(v, 0)),
                    "no_default": int(df_ok.value_counts().get(v, 0)),
                }
                for v in all_vals
            ],
        }
    else:
        # Clip at 1st–99th percentile to suppress outlier dominance
        combined = pd.concat([df_def, df_ok]).dropna()
        lo, hi   = float(np.percentile(combined, 1)), float(np.percentile(combined, 99))
        bins     = np.linspace(lo, hi, 21)
        centers  = ((bins[:-1] + bins[1:]) / 2)

        counts_def, _ = np.histogram(df_def.clip(lo, hi), bins=bins, density=True)
        counts_ok,  _ = np.histogram(df_ok.clip(lo, hi),  bins=bins, density=True)

        eda_data[col] = {
            "label": meta["label"],
            "type":  "histogram",
            "data": [
                {
                    "name":       round(float(c), 2),
                    "default":    round(float(d), 6),
                    "no_default": round(float(o), 6),
                }
                for c, d, o in zip(centers, counts_def, counts_ok)
            ],
        }

# ── 12. Assemble model_metadata.json ─────────────────────────────────────────
metadata = {
    "metrics":            metrics,
    "class_distribution": class_distribution,
    "feature_names":      feature_names,
    "display_names":      DISPLAY_NAMES,
    "shap_importance":    shap_importance,
    "model_info": {
        "name":          "Random Forest Classifier",
        "n_estimators":  100,
        "max_depth":     10,
        "max_features":  5,
        "class_weight":  "balanced",
        "random_state":  RANDOM_STATE,
        "description":   (
            "Tuned via GridSearchCV over n_estimators, max_depth, and max_features. "
            "class_weight='balanced' handles the 78/22 class imbalance. "
            "Primary metric: F1 score."
        ),
    },
}

# ── 13. Save artifacts ────────────────────────────────────────────────────────
print("Saving artifacts to:", HERE)

joblib.dump(pipeline, os.path.join(HERE, "model.pkl"))
print("  Saved model.pkl")

with open(os.path.join(HERE, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print("  Saved model_metadata.json")

with open(os.path.join(HERE, "eda_data.json"), "w") as f:
    json.dump(eda_data, f, indent=2)
print("  Saved eda_data.json")

print("\nAll done! Start the API with:")
print("  uvicorn main:app --reload")
