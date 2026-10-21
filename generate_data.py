"""Generate a synthetic loan dataset using the classic Loan Prediction schema.

Replace data/loan_data.csv with the real Kaggle "Loan Prediction" CSV
(same column names) and train.py will work unchanged.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 2000

gender = rng.choice(["Male", "Female"], N, p=[0.8, 0.2])
married = rng.choice(["Yes", "No"], N, p=[0.65, 0.35])
dependents = rng.choice(["0", "1", "2", "3+"], N, p=[0.57, 0.17, 0.16, 0.10])
education = rng.choice(["Graduate", "Not Graduate"], N, p=[0.78, 0.22])
self_employed = rng.choice(["Yes", "No"], N, p=[0.14, 0.86])
applicant_income = np.round(np.exp(rng.normal(8.4, 0.55, N)))
coapplicant_income = np.where(rng.random(N) < 0.45, 0,
                              np.round(np.exp(rng.normal(7.5, 0.6, N))))
loan_amount = np.round(np.clip(
    (applicant_income + coapplicant_income) / 38 * rng.normal(1, 0.3, N), 20, 700))
loan_term = rng.choice([360, 180, 480, 300, 240, 120, 84], N,
                       p=[0.82, 0.07, 0.03, 0.03, 0.02, 0.02, 0.01])
credit_history = rng.choice([1.0, 0.0], N, p=[0.84, 0.16])
property_area = rng.choice(["Urban", "Semiurban", "Rural"], N, p=[0.33, 0.38, 0.29])

total_income = applicant_income + coapplicant_income
emi = loan_amount * 1000 / loan_term
score = (
    3.2 * (credit_history - 0.5)
    + 0.25 * (property_area == "Semiurban")
    + 0.2 * (married == "Yes")
    + 0.15 * (education == "Graduate")
    - 0.9 * (emi / total_income * 10 - 1.2).clip(0)
    + rng.normal(0, 0.6, N)
    + 0.6
)
approved = np.where(score > 0.5, "Y", "N")

df = pd.DataFrame({
    "Loan_ID": [f"LP{100000 + i}" for i in range(N)],
    "Gender": gender, "Married": married, "Dependents": dependents,
    "Education": education, "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income, "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount, "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history, "Property_Area": property_area,
    "Loan_Status": approved,
})

# Inject realistic missing values
for col, frac in [("Gender", .02), ("Married", .01), ("Dependents", .03),
                  ("Self_Employed", .05), ("LoanAmount", .04),
                  ("Loan_Amount_Term", .02), ("Credit_History", .08)]:
    df.loc[rng.random(N) < frac, col] = np.nan

df.to_csv("data/loan_data.csv", index=False)
print(f"Saved data/loan_data.csv  shape={df.shape}  approval rate="
      f"{(df.Loan_Status == 'Y').mean():.1%}")
