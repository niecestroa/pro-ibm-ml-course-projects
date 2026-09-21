# -*- coding: utf-8 -*-
"""
Author:     Aaron Niecestro
Project:    DS Salaries – OLS, Ridge, Lasso, Elastic Net

# Created:    September 17 ,2026
# Last Edit:  September 21, 2026
Progress:     Ongoing
# Author:     Aaron Niecestro

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
# 1. Load Data and Define X, y (same structure as main script)
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
dss2025 = pd.read_csv(file_path)                    # load full dataset

dss2025 = dss2025.query("company_location == 'United States'").copy()  # filter to USA only

# Define response and predictors (same as main script)
y_resp = dss2025["salary_in_usd"]                   # response variable: salary in USD

x_pred = dss2025[                                  # predictor variables
    ['experience_level', 'employment_type', 'job_title',
     'employee_residence', 'remote_ratio',
     'company_location', 'company_size', 'data_age',
     'work_year_cat', 'remote_work_cat', 'job_title_group']
]

X = x_pred.copy()                                   # predictor matrix
y = y_resp.copy()                                   # response vector

# Identify numeric and categorical predictors
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()                                  # numeric predictors

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()                                  # categorical predictors

# =========================================================
# 2. Preprocessing (Scaling + One-Hot Encoding)
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),    # scale numeric predictors
        ("categorical",
         OneHotEncoder(handle_unknown="ignore", drop="first"),
         categorical_features)                              # encode categorical predictors
    ]
)

# Fit preprocessor on full X and transform
X_transformed = preprocessor.fit_transform(X)       # scaled + encoded predictors

# Convert sparse matrix to dense if needed
if hasattr(X_transformed, "toarray"):               # check if sparse
    X_transformed = X_transformed.toarray()         # convert to dense

y_array = np.array(y)                               # convert response to numpy array

# Add intercept for OLS
X_ols = sm.add_constant(X_transformed)              # add intercept column

# =========================================================
# 3. Utility: Metrics
# =========================================================

def compute_metrics(y_true, y_pred):                # computes RMSE, MAE, R²
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# =========================================================
# 4. Fit OLS
# =========================================================

ols_model = sm.OLS(y_array, X_ols).fit()            # fits OLS model
ols_pred = ols_model.predict(X_ols)                 # OLS predictions
ols_rmse, ols_mae, ols_r2 = compute_metrics(y_array, ols_pred)  # OLS metrics

# =========================================================
# 5. Ridge Regression (CV)
# =========================================================

ridge_model = RidgeCV(
    alphas=np.logspace(-4, 4, 200),                 # grid of alphas
    cv=10                                           # 10-fold CV
).fit(X_transformed, y_array)                       # fits Ridge with CV

ridge_pred = ridge_model.predict(X_transformed)     # Ridge predictions
ridge_rmse, ridge_mae, ridge_r2 = compute_metrics(y_array, ridge_pred)  # Ridge metrics

# =========================================================
# 6. Lasso Regression (CV)
# =========================================================

lasso_model = LassoCV(
    alphas=np.logspace(-4, 4, 200),                 # grid of alphas
    cv=10,                                          # 10-fold CV
    max_iter=5000                                   # max iterations
).fit(X_transformed, y_array)                       # fits Lasso with CV

lasso_pred = lasso_model.predict(X_transformed)     # Lasso predictions
lasso_rmse, lasso_mae, lasso_r2 = compute_metrics(y_array, lasso_pred)  # Lasso metrics
lasso_nonzero = np.sum(lasso_model.coef_ != 0)      # number of non-zero coefficients

# =========================================================
# 7. Elastic Net (CV)
# =========================================================

elastic_model = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],             # mix of L1/L2 ratios
    alphas=np.logspace(-4, 4, 200),                 # grid of alphas
    cv=10,                                          # 10-fold CV
    max_iter=5000                                   # max iterations
).fit(X_transformed, y_array)                       # fits Elastic Net with CV

elastic_pred = elastic_model.predict(X_transformed) # Elastic Net predictions
elastic_rmse, elastic_mae, elastic_r2 = compute_metrics(y_array, elastic_pred)  # Elastic metrics
elastic_nonzero = np.sum(elastic_model.coef_ != 0)  # number of non-zero coefficients

# =========================================================
# 8. Comparison Table
# =========================================================

comparison_df = pd.DataFrame({                      # builds comparison table
    "Model": ["OLS", "Ridge", "Lasso", "Elastic Net"],
    "RMSE": [ols_rmse, ridge_rmse, lasso_rmse, elastic_rmse],
    "MAE": [ols_mae, ridge_mae, lasso_mae, elastic_mae],
    "R²": [ols_r2, ridge_r2, lasso_r2, elastic_r2],
    "Non-zero Coefficients": [
        np.sum(ols_model.params != 0),              # OLS non-zero params
        np.sum(ridge_model.coef_ != 0),             # Ridge non-zero coefficients
        lasso_nonzero,                              # Lasso non-zero coefficients
        elastic_nonzero                             # Elastic Net non-zero coefficients
    ]
})

print("\n===== MODEL COMPARISON TABLE =====")
print(comparison_df)                                # prints comparison table

# =========================================================
# 9. Visualization Suite
# =========================================================

# ---------------------------------------------------------
# A. Performance Bar Chart (RMSE)
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))                         # figure size
sns.barplot(
    data=comparison_df,
    x="Model",
    y="RMSE",
    palette="viridis"
)                                                   # barplot of RMSE by model
plt.title("RMSE Comparison Across Models")          # title
plt.ylabel("RMSE")                                  # y-axis label
plt.xlabel("Model")                                 # x-axis label
plt.show()                                          # show plot

# ---------------------------------------------------------
# B. Coefficient Shrinkage Plot
# ---------------------------------------------------------

plt.figure(figsize=(14, 6))                         # figure size
plt.plot(ridge_model.coef_, label="Ridge", linewidth=2)        # Ridge coefficients
plt.plot(lasso_model.coef_, label="Lasso", linewidth=2)        # Lasso coefficients
plt.plot(elastic_model.coef_, label="Elastic Net", linewidth=2)# Elastic Net coefficients
plt.title("Coefficient Shrinkage: Ridge vs Lasso vs Elastic Net")  # title
plt.xlabel("Coefficient Index")                     # x-axis label
plt.ylabel("Coefficient Value")                     # y-axis label
plt.legend()                                        # legend
plt.show()                                          # show plot

# ---------------------------------------------------------
# C. Regularization Paths (Ridge & Lasso)
# ---------------------------------------------------------

alphas = np.logspace(-4, 4, 50)                     # smaller grid for paths

# Ridge path
ridge_coefs = []                                    # list to store Ridge coefficients
for a in alphas:
    ridge = Ridge(alpha=a).fit(X_transformed, y_array)  # fit Ridge for given alpha
    ridge_coefs.append(ridge.coef_)                 # store coefficients

ridge_coefs = np.array(ridge_coefs)                 # convert to array

plt.figure(figsize=(14, 6))                         # figure size
plt.plot(alphas, ridge_coefs)                       # plot coefficients vs alpha
plt.xscale("log")                                   # log scale for alpha
plt.title("Ridge Regularization Path")              # title
plt.xlabel("Alpha")                                 # x-axis label
plt.ylabel("Coefficient Value")                     # y-axis label
plt.show()                                          # show plot

# Lasso path
lasso_coefs = []                                    # list to store Lasso coefficients
for a in alphas:
    lasso = Lasso(alpha=a, max_iter=5000).fit(X_transformed, y_array)  # fit Lasso
    lasso_coefs.append(lasso.coef_)                 # store coefficients

lasso_coefs = np.array(lasso_coefs)                 # convert to array

plt.figure(figsize=(14, 6))                         # figure size
plt.plot(alphas, lasso_coefs)                       # plot coefficients vs alpha
plt.xscale("log")                                   # log scale for alpha
plt.title("Lasso Regularization Path")              # title
plt.xlabel("Alpha")                                 # x-axis label
plt.ylabel("Coefficient Value")                     # y-axis label
plt.show()                                          # show plot

# ---------------------------------------------------------
# D. Predicted vs Actual (All Models)
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))                         # figure size
plt.scatter(y_array, ridge_pred, alpha=0.5, label="Ridge")       # Ridge predicted vs actual
plt.scatter(y_array, lasso_pred, alpha=0.5, label="Lasso")       # Lasso predicted vs actual
plt.scatter(y_array, elastic_pred, alpha=0.5, label="Elastic Net")  # Elastic Net predicted vs actual
plt.scatter(y_array, ols_pred, alpha=0.5, label="OLS")           # OLS predicted vs actual
plt.plot([y_array.min(), y_array.max()],
         [y_array.min(), y_array.max()],
         "k--", linewidth=2)                       # 45-degree perfect line
plt.title("Predicted vs Actual (All Models)")      # title
plt.xlabel("Actual")                               # x-axis label
plt.ylabel("Predicted")                            # y-axis label
plt.legend()                                       # legend
plt.show()                                          # show plot

# ---------------------------------------------------------
# E. Residual Plots (All Models)
# ---------------------------------------------------------

models = {
    "OLS": ols_pred,
    "Ridge": ridge_pred,
    "Lasso": lasso_pred,
    "Elastic Net": elastic_pred
}                                                  # dictionary of model predictions

plt.figure(figsize=(14, 8))                        # figure size
for i, (name, pred) in enumerate(models.items(), 1):
    residuals = y_array - pred                     # residuals for model
    plt.subplot(2, 2, i)                           # subplot grid
    sns.scatterplot(x=pred, y=residuals, alpha=0.5)  # residuals vs predicted
    plt.axhline(0, color="red", linestyle="--")    # zero line
    plt.title(f"{name} Residual Plot")             # subplot title
    plt.xlabel("Predicted")                        # x-axis label
    plt.ylabel("Residuals")                        # y-axis label

plt.tight_layout()                                 # adjust layout
plt.show()                                         # show all residual plots
