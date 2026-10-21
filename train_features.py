"""Shared feature engineering (kept separate so predict.py doesn't re-run training)."""
import numpy as np
import pandas as pd


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["TotalIncome"] = d["ApplicantIncome"] + d["CoapplicantIncome"]
    d["LogTotalIncome"] = np.log1p(d["TotalIncome"])
    d["EMI"] = d["LoanAmount"] * 1000 / d["Loan_Amount_Term"]
    d["IncomeToEMI"] = d["TotalIncome"] / d["EMI"].replace(0, np.nan)
    return d
