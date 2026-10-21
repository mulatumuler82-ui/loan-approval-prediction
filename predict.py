"""Score a single applicant with the trained model.

Usage:  python predict.py
"""
import joblib
import pandas as pd
from train_features import add_features  # noqa: E402

model = joblib.load("models/best_model.joblib")

applicant = pd.DataFrame([{
    "Gender": "Male", "Married": "Yes", "Dependents": "1",
    "Education": "Graduate", "Self_Employed": "No",
    "ApplicantIncome": 5800, "CoapplicantIncome": 1500,
    "LoanAmount": 140, "Loan_Amount_Term": 360,
    "Credit_History": 1.0, "Property_Area": "Semiurban",
}])

p = model.predict_proba(add_features(applicant))[0, 1]
print(f"Approval probability: {p:.1%} -> {'APPROVED' if p >= 0.5 else 'REJECTED'}")
