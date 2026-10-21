"""Train and compare Logistic Regression, Decision Tree and Random Forest
for loan approval prediction. Saves the best model + evaluation reports."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import os
for _d in ("data", "models", "reports"):
    os.makedirs(_d, exist_ok=True)

SEED = 42
df = pd.read_csv("data/loan_data.csv")


from train_features import add_features

df = add_features(df)
y = (df["Loan_Status"] == "Y").astype(int)
X = df.drop(columns=["Loan_ID", "Loan_Status"])

num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

candidates = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        {"clf__C": [0.1, 1, 10]}),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=SEED, class_weight="balanced"),
        {"clf__max_depth": [3, 5, 7], "clf__min_samples_leaf": [5, 20]}),
    "Random Forest": (
        RandomForestClassifier(random_state=SEED, class_weight="balanced", n_jobs=1),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [6, 10, None],
         "clf__min_samples_leaf": [2, 5]}),
}

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
results, fitted = {}, {}
for name, (est, grid) in candidates.items():
    pipe = Pipeline([("pre", pre), ("clf", est)])
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=1).fit(X_tr, y_tr)
    pred = gs.predict(X_te)
    proba = gs.predict_proba(X_te)[:, 1]
    results[name] = {
        "best_params": gs.best_params_,
        "cv_f1": round(gs.best_score_, 4),
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "precision": round(precision_score(y_te, pred), 4),
        "recall": round(recall_score(y_te, pred), 4),
        "f1": round(f1_score(y_te, pred), 4),
        "roc_auc": round(roc_auc_score(y_te, proba), 4),
    }
    fitted[name] = gs.best_estimator_
    print(f"\n=== {name} ===")
    print(classification_report(y_te, pred, target_names=["Rejected", "Approved"]))

summary = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "roc_auc", "cv_f1"]]
print("\nModel comparison (hold-out test set):\n", summary)

best_name = summary["roc_auc"].idxmax()
best = fitted[best_name]
joblib.dump(best, "models/best_model.joblib")
json.dump({"best_model": best_name, "results": results},
          open("reports/metrics.json", "w"), indent=2)
summary.to_csv("reports/model_comparison.csv")
print(f"\nBest model by ROC-AUC: {best_name}  -> models/best_model.joblib")

# --- Plots ---
fig, ax = plt.subplots(figsize=(6, 5))
for name, m in fitted.items():
    RocCurveDisplay.from_estimator(m, X_te, y_te, name=name, ax=ax)
ax.plot([0, 1], [0, 1], "k--", alpha=.4)
ax.set_title("ROC curves")
fig.tight_layout(); fig.savefig("reports/roc_curves.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay.from_estimator(
    best, X_te, y_te, display_labels=["Rejected", "Approved"], cmap="Blues", ax=ax)
ax.set_title(f"Confusion matrix - {best_name}")
fig.tight_layout(); fig.savefig("reports/confusion_matrix.png", dpi=150); plt.close(fig)

rf = fitted["Random Forest"]
names = rf.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).nlargest(12)
fig, ax = plt.subplots(figsize=(7, 5))
imp[::-1].plot.barh(ax=ax, color="#7c6cf0")
ax.set_title("Random Forest - top 12 feature importances")
fig.tight_layout(); fig.savefig("reports/feature_importance.png", dpi=150); plt.close(fig)
print("Saved plots to reports/")
