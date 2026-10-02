"""
Predictive Risk Analytics in FinTech
------------------------------------
Base project: predict the probability that a borrower defaults on a loan,
convert it into a risk score / risk band, and evaluate the model.

Pipeline:
  1. Generate synthetic loan data (replace with a real CSV later)
  2. Preprocess + feature engineering
  3. Train Logistic Regression (baseline) and Gradient Boosting
  4. Evaluate (ROC-AUC, precision/recall, confusion matrix)
  5. Save the best model and plots
  6. Score a new applicant
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)

SEED = 42
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------- 1. DATA
def generate_data(n=5000, seed=SEED) -> pd.DataFrame:
    """Create a synthetic loan dataset with a realistic default relationship."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(21, 66, n),
        "annual_income": rng.lognormal(10.9, 0.5, n).round(0),
        "loan_amount": rng.lognormal(9.5, 0.7, n).round(0),
        "loan_term_months": rng.choice([12, 24, 36, 48, 60], n),
        "credit_score": np.clip(rng.normal(680, 70, n), 300, 850).round(0),
        "employment_years": np.clip(rng.exponential(5, n), 0, 40).round(1),
        "num_open_accounts": rng.integers(1, 15, n),
        "num_late_payments": rng.poisson(1.0, n),
        "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], n, p=[.45, .40, .15]),
        "loan_purpose": rng.choice(
            ["debt_consolidation", "home", "education", "business", "personal"], n),
    })

    # Hidden "true" default probability driven by risk factors
    dti = df["loan_amount"] / df["annual_income"]
    logit = (-3.6
             + 2.5 * dti
             - 0.012 * (df["credit_score"] - 680)
             + 0.35 * df["num_late_payments"]
             - 0.05 * df["employment_years"]
             + 0.004 * df["loan_term_months"]
             + np.where(df["home_ownership"] == "RENT", 0.3, 0)
             + np.where(df["loan_purpose"] == "business", 0.4, 0)
             + rng.normal(0, 0.5, n))
    prob = 1 / (1 + np.exp(-logit))
    df["default"] = (rng.random(n) < prob).astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain feature engineering."""
    df = df.copy()
    df["loan_to_income"] = df["loan_amount"] / df["annual_income"]
    df["monthly_payment_est"] = df["loan_amount"] / df["loan_term_months"]
    df["payment_to_income"] = df["monthly_payment_est"] / (df["annual_income"] / 12)
    return df


# ------------------------------------------------------------- 2. MODEL
def build_pipeline(model, num_cols, cat_cols) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return Pipeline([("prep", pre), ("model", model)])


def risk_band(p: float) -> str:
    if p < 0.10: return "LOW"
    if p < 0.25: return "MEDIUM"
    if p < 0.50: return "HIGH"
    return "VERY HIGH"


def credit_score_from_pd(p: float) -> int:
    """Map probability of default to a 300-850 style score (higher = safer)."""
    return int(round(850 - 550 * np.clip(p, 0, 1)))


# ----------------------------------------------------------- 3. EVALUATE
def evaluate(name, pipe, X_test, y_test):
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    print(f"\n===== {name} =====")
    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, pred, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    return auc, proba


def plot_roc(results, y_test):
    plt.figure(figsize=(6, 5))
    for name, proba in results.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Default Prediction"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=120); plt.close()


def plot_importance(pipe, num_cols, cat_cols):
    model = pipe.named_steps["model"]
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(model.feature_importances_, index=names).sort_values().tail(12)
    imp.plot(kind="barh", figsize=(7, 5), title="Top Risk Drivers")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=120); plt.close()


# ---------------------------------------------------------------- MAIN
def main():
    df = add_features(generate_data())
    df.to_csv(f"{OUT_DIR}/loan_data.csv", index=False)
    print(f"Dataset: {df.shape}, default rate = {df['default'].mean():.2%}")

    target = "default"
    cat_cols = ["home_ownership", "loan_purpose"]
    num_cols = [c for c in df.columns if c not in cat_cols + [target]]

    X, y = df[num_cols + cat_cols], df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }

    fitted, aucs, probas = {}, {}, {}
    for name, m in models.items():
        pipe = build_pipeline(m, num_cols, cat_cols).fit(X_train, y_train)
        aucs[name], probas[name] = evaluate(name, pipe, X_test, y_test)
        fitted[name] = pipe

    plot_roc(probas, y_test)
    plot_importance(fitted["Gradient Boosting"], num_cols, cat_cols)

    best = max(aucs, key=aucs.get)
    joblib.dump(fitted[best], f"{OUT_DIR}/best_model.joblib")
    print(f"\nBest model: {best} (AUC {aucs[best]:.4f}) -> saved to {OUT_DIR}/best_model.joblib")

    # ---- Score a new applicant
    applicant = add_features(pd.DataFrame([{
        "age": 34, "annual_income": 600000, "loan_amount": 450000,
        "loan_term_months": 48, "credit_score": 610, "employment_years": 3.0,
        "num_open_accounts": 6, "num_late_payments": 3,
        "home_ownership": "RENT", "loan_purpose": "business",
    }]))
    p = fitted[best].predict_proba(applicant[num_cols + cat_cols])[0, 1]
    print("\n--- New Applicant ---")
    print(f"Probability of default : {p:.2%}")
    print(f"Risk band              : {risk_band(p)}")
    print(f"Internal risk score    : {credit_score_from_pd(p)}")
    print("Decision               :", "REVIEW / DECLINE" if p >= 0.25 else "APPROVE")


if __name__ == "__main__":
    main()
