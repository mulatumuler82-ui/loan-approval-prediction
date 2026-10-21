"""Streamlit demo: Loan Approval Prediction Model.
Run:  streamlit run app.py
"""
import json
import os
import subprocess
import sys

import pandas as pd
import streamlit as st

from scoring import MODEL_PATH, load_model, score

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")


@st.cache_resource
def get_model():
    # First run on a fresh deploy: train the model if it isn't in the repo.
    if not os.path.exists(MODEL_PATH):
        if not os.path.exists("data/loan_data.csv"):
            subprocess.run([sys.executable, "generate_data.py"], check=True)
        subprocess.run([sys.executable, "train.py"], check=True)
    return load_model()


model = get_model()

st.title("🏦 Loan Approval Prediction")
st.caption("Scikit-Learn classification pipeline · Logistic Regression vs "
           "Decision Tree vs Random Forest")

tab_predict, tab_perf = st.tabs(["Predict", "Model performance"])

with tab_predict:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Applicant details")
        c1, c2 = st.columns(2)
        gender = c1.selectbox("Gender", ["Male", "Female"])
        married = c2.selectbox("Married", ["Yes", "No"])
        dependents = c1.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = c2.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = c1.selectbox("Self employed", ["No", "Yes"])
        property_area = c2.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
        app_income = c1.number_input("Applicant income (monthly)", 0, 200000, 5800, 100)
        co_income = c2.number_input("Coapplicant income (monthly)", 0, 200000, 1500, 100)
        loan_amount = c1.number_input("Loan amount (in thousands)", 10, 1000, 140, 5)
        term = c2.selectbox("Loan term (months)", [360, 300, 240, 180, 120, 84, 480])
        credit = st.radio("Credit history meets guidelines?", ["Yes", "No"], horizontal=True)
        go = st.button("Predict", type="primary", use_container_width=True)

    with right:
        st.subheader("Result")
        if go:
            r = score(model, {
                "Gender": gender, "Married": married, "Dependents": dependents,
                "Education": education, "Self_Employed": self_employed,
                "ApplicantIncome": app_income, "CoapplicantIncome": co_income,
                "LoanAmount": loan_amount, "Loan_Amount_Term": term,
                "Credit_History": 1.0 if credit == "Yes" else 0.0,
                "Property_Area": property_area,
            })
            if r["approved"]:
                st.success("✅ Likely APPROVED")
            else:
                st.error("❌ Likely REJECTED")
            st.metric("Approval probability", f"{r['probability']:.1%}")
            st.progress(min(max(r["probability"], 0.0), 1.0))
            m1, m2 = st.columns(2)
            m1.metric("Monthly EMI", f"{r['emi']:,.0f}")
            m2.metric("Income / EMI", f"{r['income_to_emi']:.1f}x")
        else:
            st.info("Fill in the details and press **Predict**.")
        st.caption("Demo only. Trained on synthetic data. Not financial advice.")

with tab_perf:
    st.subheader("Model comparison (hold-out test set)")
    if os.path.exists("reports/model_comparison.csv"):
        # Plain markdown table: avoids st.dataframe, which needs pyarrow
        # (pyarrow is blocked by Application Control on some Windows PCs).
        cmp = pd.read_csv("reports/model_comparison.csv", index_col=0)
        head = "| Model | " + " | ".join(cmp.columns) + " |"
        sep = "|---|" + "---|" * len(cmp.columns)
        rows = [f"| {name} | " + " | ".join(f"{v:.3f}" for v in r) + " |"
                for name, r in cmp.iterrows()]
        st.markdown("\n".join([head, sep] + rows))
    if os.path.exists("reports/metrics.json"):
        st.write("**Best model:**", json.load(open("reports/metrics.json"))["best_model"])
    a, b, c = st.columns(3)
    for col, img in zip((a, b, c), ("roc_curves", "confusion_matrix", "feature_importance")):
        p = f"reports/{img}.png"
        if os.path.exists(p):
            col.image(p, use_container_width=True)