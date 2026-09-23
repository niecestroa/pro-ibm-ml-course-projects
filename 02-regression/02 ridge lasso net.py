# -*- coding: utf-8 -*-
"""
Author:     Aaron Niecestro
Project:    DS Salaries – OLS, Ridge, Lasso, Elastic Net

# Created:    September 17 ,2026
# Last Edit:  September 23, 2026
Progress:     Completed

Purpose:
    This script evaluates multiple linear regression models 
    using regularization techniques (Ridge, Lasso, Elastic Net) 
    and compares them against a baseline OLS model.

What it does:
    • Loads and preprocesses the USA-only salary dataset
    • Builds predictor matrix (X) and response vector (y)
    • Applies scaling + one-hot encoding to all predictors
    • Fits OLS, RidgeCV, LassoCV, and ElasticNetCV models
    • Computes RMSE, MAE, and R² for each model
    • Compares coefficient shrinkage and sparsity
    • Generates diagnostic plots:
         – RMSE comparison bar chart
         – Coefficient shrinkage curves
         – Regularization paths (Ridge & Lasso)
         – Predicted vs Actual comparison
         – Residual plots for all models

Why it exists:
    To identify whether regularization improves predictive 
    performance or model interpretability compared to OLS, 
    especially in high-dimensional encoded feature spaces.

Usage:
    Run as a standalone analysis file. 
    No dependencies on earlier scripts—data loading and 
    preprocessing are included here.
"""

# =========================================================
# Regularization Model Comparison + Visualization Suite
# =========================================================

import numpy as np                                  # numerical operations
import pandas as pd                                 # data handling
import matplotlib.pyplot as plt                     # plotting
import seaborn as sns                               # visualization
import statsmodels.api as sm                        # OLS and statistical tools

from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNetCV  # regularization models
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score   # evaluation metrics
from sklearn.model_selection import train_test_split                            # train/test split
from sklearn.preprocessing import StandardScaler, OneHotEncoder                 # scaling + encoding
from sklearn.compose import ColumnTransformer                                   # preprocessing transformer
from sklearn.pipeline import Pipeline                                           # pipeline

sns.set(style="whitegrid", context="talk")          # set seaborn style

# =========================================================
# 1. Load Data (USA-only)
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
dss2025 = pd.read_csv(file_path)                    # load full dataset

dss2025 = dss2025.query("company_location == 'United States'").copy()  # filter to USA only

# =========================================================
# 2. PIPELINE A — FULL PREDICTOR SET (LOG RESPONSE)
# =========================================================

print("\n=========================================================")
print("PIPELINE A — FULL PREDICTOR SET (LOG RESPONSE)")
print("=========================================================\n")

# LOG RESPONSE (updated from raw salary)
y_full = np.log(dss2025["salary_in_usd"])           # log-transformed salary

# Full predictor set (same as original file)
X_full = dss2025[
    ['experience_level', 'employment_type', 'job_title',
     'employee_residence', 'remote_ratio',
     'company_location', 'company_size', 'data_age',
     'work_year_cat', 'remote_work_cat', 'job_title_group']
]

# Identify numeric and categorical predictors
numeric_features_full = X_full.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features_full = X_full.select_dtypes(include=["object", "category"]).columns.tolist()

# Preprocessing (scaling + encoding)
preprocessor_full = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features_full),
        ("categorical", OneHotEncoder(handle_unknown="ignore", drop="first"),
         categorical_features_full)
    ]
)

# Transform full predictor matrix
X_full_transformed = preprocessor_full.fit_transform(X_full)

# Convert sparse to dense if needed
if hasattr(X_full_transformed, "toarray"):
    X_full_transformed = X_full_transformed.toarray()

# Add intercept for OLS
X_full_ols = sm.add_constant(X_full_transformed)

# =========================================================
# A.1 Utility: Metrics (LOG SCALE)
# =========================================================

def compute_metrics_log(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))   # RMSE in log scale
    mae = mean_absolute_error(y_true, y_pred)            # MAE in log scale
    r2 = r2_score(y_true, y_pred)                        # R² in log scale
    return rmse, mae, r2

# =========================================================
# A.2 Fit OLS (LOG RESPONSE)
# =========================================================

ols_full = sm.OLS(y_full, X_full_ols).fit()
ols_full_pred = ols_full.predict(X_full_ols)
ols_full_rmse, ols_full_mae, ols_full_r2 = compute_metrics_log(y_full, ols_full_pred)

# =========================================================
# A.3 Ridge Regression (LOG RESPONSE)
# =========================================================

ridge_full = RidgeCV(alphas=np.logspace(-4, 4, 200), cv=10).fit(X_full_transformed, y_full)
ridge_full_pred = ridge_full.predict(X_full_transformed)
ridge_full_rmse, ridge_full_mae, ridge_full_r2 = compute_metrics_log(y_full, ridge_full_pred)

# =========================================================
# A.4 Lasso Regression (LOG RESPONSE)
# =========================================================

lasso_full = LassoCV(alphas=np.logspace(-4, 4, 200), cv=10, max_iter=5000).fit(X_full_transformed, y_full)
lasso_full_pred = lasso_full.predict(X_full_transformed)
lasso_full_rmse, lasso_full_mae, lasso_full_r2 = compute_metrics_log(y_full, lasso_full_pred)
lasso_full_nonzero = np.sum(lasso_full.coef_ != 0)

# =========================================================
# A.5 Elastic Net (LOG RESPONSE)
# =========================================================

elastic_full = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
    alphas=np.logspace(-4, 4, 200),
    cv=10,
    max_iter=5000
).fit(X_full_transformed, y_full)

elastic_full_pred = elastic_full.predict(X_full_transformed)
elastic_full_rmse, elastic_full_mae, elastic_full_r2 = compute_metrics_log(y_full, elastic_full_pred)
elastic_full_nonzero = np.sum(elastic_full.coef_ != 0)

# =========================================================
# A.6 Comparison Table (FULL PREDICTORS)
# =========================================================

comparison_full = pd.DataFrame({
    "Model": ["OLS", "Ridge", "Lasso", "Elastic Net"],
    "RMSE (log)": [ols_full_rmse, ridge_full_rmse, lasso_full_rmse, elastic_full_rmse],
    "MAE (log)": [ols_full_mae, ridge_full_mae, lasso_full_mae, elastic_full_mae],
    "R² (log)": [ols_full_r2, ridge_full_r2, lasso_full_r2, elastic_full_r2],
    "Non-zero Coefficients": [
        np.sum(ols_full.params != 0),
        np.sum(ridge_full.coef_ != 0),
        lasso_full_nonzero,
        elastic_full_nonzero
    ]
})

print("\n===== FULL PREDICTOR MODEL COMPARISON (LOG SCALE) =====")
print(comparison_full)

# =========================================================
# 3. PIPELINE B — AIC-SELECTED PREDICTOR SET (LOG RESPONSE)
# =========================================================

print("\n=========================================================")
print("PIPELINE B — AIC-SELECTED PREDICTOR SET (LOG RESPONSE)")
print("=========================================================\n")

# LOG RESPONSE (same as Pipeline A)
y_aic = np.log(dss2025["salary_in_usd"])

# AIC-selected predictors
X_aic = dss2025[
    ['experience_level', 'employment_type', 'job_title',
     'remote_ratio', 'work_year_cat']
]

# Identify numeric and categorical predictors
numeric_features_aic = X_aic.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features_aic = X_aic.select_dtypes(include=["object", "category"]).columns.tolist()

# Preprocessing (scaling + encoding)
preprocessor_aic = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features_aic),
        ("categorical", OneHotEncoder(handle_unknown="ignore", drop="first"),
         categorical_features_aic)
    ]
)

# Transform AIC predictor matrix
X_aic_transformed = preprocessor_aic.fit_transform(X_aic)

# Convert sparse to dense if needed
if hasattr(X_aic_transformed, "toarray"):
    X_aic_transformed = X_aic_transformed.toarray()

# Add intercept for OLS
X_aic_ols = sm.add_constant(X_aic_transformed)

# =========================================================
# B.1 Fit OLS (LOG RESPONSE)
# =========================================================

ols_aic = sm.OLS(y_aic, X_aic_ols).fit()
ols_aic_pred = ols_aic.predict(X_aic_ols)
ols_aic_rmse, ols_aic_mae, ols_aic_r2 = compute_metrics_log(y_aic, ols_aic_pred)

# =========================================================
# B.2 Ridge Regression (LOG RESPONSE)
# =========================================================

ridge_aic = RidgeCV(alphas=np.logspace(-4, 4, 200), cv=10).fit(X_aic_transformed, y_aic)
ridge_aic_pred = ridge_aic.predict(X_aic_transformed)
ridge_aic_rmse, ridge_aic_mae, ridge_aic_r2 = compute_metrics_log(y_aic, ridge_aic_pred)

# =========================================================
# B.3 Lasso Regression (LOG RESPONSE)
# =========================================================

lasso_aic = LassoCV(alphas=np.logspace(-4, 4, 200), cv=10, max_iter=5000).fit(X_aic_transformed, y_aic)
lasso_aic_pred = lasso_aic.predict(X_aic_transformed)
lasso_aic_rmse, lasso_aic_mae, lasso_aic_r2 = compute_metrics_log(y_aic, lasso_aic_pred)
lasso_aic_nonzero = np.sum(lasso_aic.coef_ != 0)

# =========================================================
# B.4 Elastic Net (LOG RESPONSE)
# =========================================================

elastic_aic = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
    alphas=np.logspace(-4, 4, 200),
    cv=10,
    max_iter=5000
).fit(X_aic_transformed, y_aic)

elastic_aic_pred = elastic_aic.predict(X_aic_transformed)
elastic_aic_rmse, elastic_aic_mae, elastic_aic_r2 = compute_metrics_log(y_aic, elastic_aic_pred)
elastic_aic_nonzero = np.sum(elastic_aic.coef_ != 0)

# =========================================================
# B.5 Comparison Table (AIC PREDICTORS)
# =========================================================

comparison_aic = pd.DataFrame({
    "Model": ["OLS", "Ridge", "Lasso", "Elastic Net"],
    "RMSE (log)": [ols_aic_rmse, ridge_aic_rmse, lasso_aic_rmse, elastic_aic_rmse],
    "MAE (log)": [ols_aic_mae, ridge_aic_mae, lasso_aic_mae, elastic_aic_mae],
    "R² (log)": [ols_aic_r2, ridge_aic_r2, lasso_aic_r2, elastic_aic_r2],
    "Non-zero Coefficients": [
        np.sum(ols_aic.params != 0),
        np.sum(ridge_aic.coef_ != 0),
        lasso_aic_nonzero,
        elastic_aic_nonzero
    ]
})

print("\n===== AIC PREDICTOR MODEL COMPARISON (LOG SCALE) =====")
print(comparison_aic)

# =========================================================
# 4. Visualization Suite (FULL + AIC)
# =========================================================

# ---------------------------------------------------------
# A. RMSE Comparison (Full vs AIC)
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))
combined_rmse = pd.concat([
    comparison_full.assign(Predictor_Set="Full"),
    comparison_aic.assign(Predictor_Set="AIC")
])
sns.barplot(data=combined_rmse, x="Model", y="RMSE (log)", hue="Predictor_Set")
plt.title("RMSE Comparison: Full vs AIC Predictor Sets (Log Scale)")
plt.show()

# ---------------------------------------------------------
# B. Coefficient Shrinkage (Full vs AIC)
# ---------------------------------------------------------

plt.figure(figsize=(14, 6))
plt.plot(ridge_full.coef_, label="Ridge (Full)", linewidth=2)
plt.plot(ridge_aic.coef_, label="Ridge (AIC)", linewidth=2)
plt.title("Ridge Coefficient Shrinkage: Full vs AIC Predictors")
plt.legend()
plt.show()

# ---------------------------------------------------------
# C. Predicted vs Actual (Full vs AIC)
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.scatter(y_full, ridge_full_pred, alpha=0.5, label="Ridge (Full)")
plt.scatter(y_aic, ridge_aic_pred, alpha=0.5, label="Ridge (AIC)")
plt.plot([y_full.min(), y_full.max()],
         [y_full.min(), y_full.max()],
         "k--", linewidth=2)
plt.title("Predicted vs Actual (Log Salary): Full vs AIC")
plt.xlabel("Actual log(salary)")
plt.ylabel("Predicted log(salary)")
plt.legend()
plt.show()
