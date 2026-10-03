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
# Imports
# =========================================================

import numpy as np                      # numerical operations
import pandas as pd                     # data handling
import matplotlib.pyplot as plt         # plotting
import seaborn as sns                   # visualization
import statsmodels.api as sm            # OLS model

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV   # regularization models
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

sns.set(style="whitegrid", context="talk")  # seaborn style

# =========================================================
# Load USA-only dataset
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
dss2025 = pd.read_csv(file_path)
dss2025 = dss2025.query("company_location == 'United States'").copy()

# =========================================================
# Predictor sets
# =========================================================

y_full = np.log(dss2025["salary_in_usd"])  # log salary response

X_full = dss2025[
    ['experience_level', 'employment_type', 'job_title',
     'employee_residence', 'remote_ratio',
     'company_location', 'company_size', 'data_age',
     'work_year_cat', 'remote_work_cat', 'job_title_group']
]

X_aic = dss2025[
    ['experience_level', 'employment_type', 'job_title',
     'remote_ratio', 'work_year_cat']
]

y_aic = np.log(dss2025["salary_in_usd"])  # same log response

# =========================================================
# Identify numeric + categorical predictors
# =========================================================

numeric_full = X_full.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_full = X_full.select_dtypes(include=["object", "category"]).columns.tolist()

numeric_aic = X_aic.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_aic = X_aic.select_dtypes(include=["object", "category"]).columns.tolist()

# =========================================================
# Preprocessing (scaling + one-hot encoding)
# =========================================================

pre_full = ColumnTransformer([
    ("num", StandardScaler(), numeric_full),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_full)
])

pre_aic = ColumnTransformer([
    ("num", StandardScaler(), numeric_aic),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_aic)
])

# =========================================================
# Transform predictor matrices
# =========================================================

X_full_t = pre_full.fit_transform(X_full)
X_aic_t = pre_aic.fit_transform(X_aic)

X_full_t = X_full_t.toarray() if hasattr(X_full_t, "toarray") else X_full_t
X_aic_t = X_aic_t.toarray() if hasattr(X_aic_t, "toarray") else X_aic_t

X_full_ols = sm.add_constant(X_full_t)  # add intercept
X_aic_ols = sm.add_constant(X_aic_t)

# =========================================================
# Metric function (RMSE, MAE, R², Adjusted R²)
# =========================================================

def compute_metrics_log(y_true, y_pred, p):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))      # RMSE
    mae = mean_absolute_error(y_true, y_pred)               # MAE
    r2 = r2_score(y_true, y_pred)                           # R²
    n = len(y_true)                                         # sample size
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))          # Adjusted R²
    return rmse, mae, r2, adj_r2

# =========================================================
# FULL predictor set — Fit models
# =========================================================

p_full = X_full_t.shape[1]  # number of predictors

ols_full = sm.OLS(y_full, X_full_ols).fit()
ols_full_pred = ols_full.predict(X_full_ols)
ols_full_rmse, ols_full_mae, ols_full_r2, ols_full_adj_r2 = compute_metrics_log(y_full, ols_full_pred, p_full)

ridge_full = RidgeCV(alphas=np.logspace(-4, 4, 200), cv=10).fit(X_full_t, y_full)
ridge_full_pred = ridge_full.predict(X_full_t)
ridge_full_rmse, ridge_full_mae, ridge_full_r2, ridge_full_adj_r2 = compute_metrics_log(y_full, ridge_full_pred, p_full)

lasso_full = LassoCV(alphas=np.logspace(-4, 4, 200), cv=10, max_iter=5000).fit(X_full_t, y_full)
lasso_full_pred = lasso_full.predict(X_full_t)
lasso_full_rmse, lasso_full_mae, lasso_full_r2, lasso_full_adj_r2 = compute_metrics_log(y_full, lasso_full_pred, p_full)

elastic_full = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
    alphas=np.logspace(-4, 4, 200),
    cv=10,
    max_iter=5000
).fit(X_full_t, y_full)
elastic_full_pred = elastic_full.predict(X_full_t)
elastic_full_rmse, elastic_full_mae, elastic_full_r2, elastic_full_adj_r2 = compute_metrics_log(y_full, elastic_full_pred, p_full)

# =========================================================
# AIC predictor set — Fit models
# =========================================================

p_aic = X_aic_t.shape[1]  # number of predictors

ols_aic = sm.OLS(y_aic, X_aic_ols).fit()
ols_aic_pred = ols_aic.predict(X_aic_ols)
ols_aic_rmse, ols_aic_mae, ols_aic_r2, ols_aic_adj_r2 = compute_metrics_log(y_aic, ols_aic_pred, p_aic)

ridge_aic = RidgeCV(alphas=np.logspace(-4, 4, 200), cv=10).fit(X_aic_t, y_aic)
ridge_aic_pred = ridge_aic.predict(X_aic_t)
ridge_aic_rmse, ridge_aic_mae, ridge_aic_r2, ridge_aic_adj_r2 = compute_metrics_log(y_aic, ridge_aic_pred, p_aic)

lasso_aic = LassoCV(alphas=np.logspace(-4, 4, 200), cv=10, max_iter=5000).fit(X_aic_t, y_aic)
lasso_aic_pred = lasso_aic.predict(X_aic_t)
lasso_aic_rmse, lasso_aic_mae, lasso_aic_r2, lasso_aic_adj_r2 = compute_metrics_log(y_aic, lasso_aic_pred, p_aic)

elastic_aic = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
    alphas=np.logspace(-4, 4, 200),
    cv=10,
    max_iter=5000
).fit(X_aic_t, y_aic)
elastic_aic_pred = elastic_aic.predict(X_aic_t)
elastic_aic_rmse, elastic_aic_mae, elastic_aic_r2, elastic_aic_adj_r2 = compute_metrics_log(y_aic, elastic_aic_pred, p_aic)

# =========================================================
# Export best metrics for leaderboard
# =========================================================

best_ols_rmse       = ols_full_rmse
best_ols_mae        = ols_full_mae
best_ols_r2         = ols_full_r2
best_ols_adj_r2     = ols_full_adj_r2

best_ridge_rmse     = ridge_full_rmse
best_ridge_mae      = ridge_full_mae
best_ridge_r2       = ridge_full_r2
best_ridge_adj_r2   = ridge_full_adj_r2

best_lasso_rmse     = lasso_full_rmse
best_lasso_mae      = lasso_full_mae
best_lasso_r2       = lasso_full_r2
best_lasso_adj_r2   = lasso_full_adj_r2

best_elastic_rmse   = elastic_full_rmse
best_elastic_mae    = elastic_full_mae
best_elastic_r2     = elastic_full_r2
best_elastic_adj_r2 = elastic_full_adj_r2

__all__ = [
    "best_ols_rmse", "best_ols_mae", "best_ols_r2", "best_ols_adj_r2",
    "best_ridge_rmse", "best_ridge_mae", "best_ridge_r2", "best_ridge_adj_r2",
    "best_lasso_rmse", "best_lasso_mae", "best_lasso_r2", "best_lasso_adj_r2",
    "best_elastic_rmse", "best_elastic_mae", "best_elastic_r2", "best_elastic_adj_r2"
]
