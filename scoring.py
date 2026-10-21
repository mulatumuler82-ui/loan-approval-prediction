"""Prediction helper used by the Streamlit app (importable without Streamlit)."""
import joblib
import pandas as pd
from train_features import add_features

MODEL_PATH = "models/best_model.joblib"


def load_model(path: str = MODEL_PATH):
    return joblib.load(path)


def score(model, applicant: dict) -> dict:
    df = add_features(pd.DataFrame([applicant]))
    p = float(model.predict_proba(df)[0, 1])
    emi = df["EMI"].iloc[0]
    return {
        "probability": p,
        "approved": p >= 0.5,
        "emi": float(emi),
        "income_to_emi": float(df["IncomeToEMI"].iloc[0]),
    }
