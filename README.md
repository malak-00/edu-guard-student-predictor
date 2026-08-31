# Student Outcome Predictor — Deployment

Deploys a Random Forest model trained on the EduGuard AI — SmartCity AI
Challenge 2026 dataset, predicting **Dropout / Enrolled / Graduate**, behind
a Streamlit web form.

Competition: https://www.kaggle.com/competitions/edu-guard-ai-smart-city-ai-challenge-2026

## What's in here

- `kaggle_notebook.ipynb` — the original competition notebook.
- `train.py` — preprocessing (missing-value fill, outlier capping, encoding,
  scaling), an 80/20 train/validation split, and saves `model.joblib`,
  `scaler.joblib`, and `schema.json`.
- `app.py` — the Streamlit app.
- `model.joblib`, `scaler.joblib`, `schema.json` — trained artifacts,
  included so the app runs immediately without retraining.
- `requirements.txt` — dependencies.

## Model performance (held-out validation)

- Accuracy: **77.3%**
- Macro-F1: **0.68**
- Weakest class: "Enrolled" (recall 0.27)

## Run locally

The trained model is already included, so you can run the app directly:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (defaults to http://localhost:8501).

## Deploy for others to use

1. Push this repo to GitHub as-is (model artifacts are already included).
2. Go to https://share.streamlit.io, connect the repo, and point it at
   `app.py`.

## Retraining

Download `train.csv` from the competition page above, place it in this
folder, and run:

```bash
python train.py
```

This regenerates `model.joblib`, `scaler.joblib`, and `schema.json`.
