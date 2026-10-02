# Predictive Risk Analytics in FinTech

A simple base project that predicts loan default risk.

## Run
```bash
pip install -r requirements.txt
python risk_analytics.py
```

## What it does
1. Generates a synthetic loan dataset (swap in your own CSV with a `default` column)
2. Engineers features (loan-to-income, payment-to-income)
3. Trains Logistic Regression and Gradient Boosting
4. Reports ROC-AUC, precision/recall, confusion matrix
5. Saves best model, ROC curve and feature-importance plots to `outputs/`
6. Scores a new applicant with PD, risk band, score and decision

## Ideas to extend
- Use real data (Lending Club, German Credit, Give Me Some Credit on Kaggle)
- Add SHAP explainability, XGBoost/LightGBM, SMOTE for imbalance
- Wrap `best_model.joblib` in a FastAPI endpoint or Streamlit dashboard
- Add fraud detection or market-risk (VaR) modules
