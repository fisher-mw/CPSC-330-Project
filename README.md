# Credit Card Default Classifier

A full-stack machine learning web application that predicts whether a credit card client is likely to default on their next payment. Built for CPSC-330 (Applied Machine Learning) with an emphasis on explainable AI, as decisions that affect financial access need to be transparent.

**Live demo:** [Frontend](https://cpsc-330-project.vercel.app) | [API Docs](https://cpsc-330-project.onrender.com/docs)

---

## Features

- **Classifier** — Input applicant profile, repayment history, and billing data to receive a default probability and a SHAP-based breakdown of the top driving factors
- **Explainability dashboard** — Model performance metrics, class distribution, global feature importance (mean |SHAP|), and per-feature distributions split by default status

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML pipeline | scikit-learn (Random Forest + StandardScaler + OneHotEncoder) |
| Explainability | SHAP (TreeExplainer) |
| Backend API | FastAPI + joblib |
| Frontend | React 18 + Vite + Recharts |
| Backend hosting | Render |
| Frontend hosting | Vercel |

---

## Dataset

[UCI Credit Card Default Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) — 30,000 Taiwanese credit card clients, October 2005. Target variable: default payment next month (binary).

---

## Project Structure

```
CPSC-330-Project/
├── hw5.ipynb              # Original EDA, model training, and evaluation notebook
├── data/                  # UCI_Credit_Card.csv (not tracked in git)
└── eXAI/
    ├── backend/
    │   ├── main.py                # FastAPI app (/predict, /metadata, /eda, /health)
    │   ├── train_and_export.py    # Trains pipeline, exports model.pkl + JSON artifacts
    │   ├── test_model.py          # Sanity checks for the serialized model
    │   ├── model.pkl              # Serialized sklearn pipeline
    │   ├── model_metadata.json    # Metrics, SHAP importance, class distribution
    │   ├── eda_data.json          # Pre-computed feature distribution data
    │   └── requirements.txt
    └── frontend/
        ├── src/
        │   ├── pages/
        │   │   ├── PredictorPage.jsx
        │   │   └── ExplainabilityPage.jsx
        │   ├── components/NavBar.jsx
        │   ├── api.js
        │   └── index.css
        ├── vercel.json
        └── package.json
```

---

## Local Development

**Backend**
```bash
cd eXAI/backend
pip install -r requirements.txt

# Regenerate model artifacts from scratch (requires UCI_Credit_Card.csv in data/)
python train_and_export.py

# Start the API server
uvicorn main:app --reload
# Runs at http://localhost:8000 — interactive docs at /docs
```

**Frontend**
```bash
cd eXAI/frontend
npm install
npm run dev
# Runs at http://localhost:5173
```

The frontend reads `VITE_API_URL` from `.env`. Defaults to `http://localhost:8000`.

---

## Deployment

- **Backend (Render)** — Web Service, root dir `eXAI/backend`, start command `uvicorn main:app --host 0.0.0.0 --port $PORT`, Python 3.11, `scikit-learn==1.6.1` pinned to match the serialized model
- **Frontend (Vercel)** — Import repo, set root directory to `eXAI/frontend`, add env variable `VITE_API_URL=<your Render URL>`, redeploy after setting

---

## Why Explainability Matters

Credit default classification directly determines whether someone can access financial products. A model that is accurate but opaque fails the people it affects — they cannot understand or challenge its output. This project uses SHAP to surface per-prediction factor breakdowns and global importance rankings, making every decision auditable. The analysis shows that recent repayment behavior dominates the model's decisions while demographic features contribute minimally, a positive signal for fairness.

---

## Course

CPSC-330 Applied Machine Learning — University of British Columbia
