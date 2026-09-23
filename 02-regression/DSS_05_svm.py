# -*- coding: utf-8 -*-
"""
Author:     Aaron Niecestro
Project:    DS Salaries – Support Vector Machines

# Created:    September 21 ,2026
# Last Edit:  September 23, 2026
Progress:     Completed

Description - PART 4 Support Vector Regression (SVR with RBF Kernel):

Purpose:
    Model nonlinear salary patterns using Support Vector Regression.
    Includes:
        • RBF kernel SVR
        • One-hot encoding for categorical predictors
        • Hyperparameter tuning (C, epsilon, gamma)
        • Performance metrics (RMSE, MAE, R²)
        • Actual vs Predicted plot
        • Residual plot
        • Final SVR comparison table
"""

# =========================================================
# Imports
# =========================================================

import numpy as np                                      # numerical operations
import pandas as pd                                     # data handling
import matplotlib.pyplot as plt                         # plotting
import seaborn as sns                                   # visualization

from sklearn.model_selection import train_test_split     # train/test split
from sklearn.preprocessing import OneHotEncoder          # categorical encoding
from sklearn.compose import ColumnTransformer            # preprocessing transformer
from sklearn.pipeline import Pipeline                    # pipeline

from sklearn.svm import SVR                              # support vector regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # metrics

sns.set(style="whitegrid", context="talk")               # seaborn style

# =========================================================
# Load Data (USA-only)
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
df = pd.read_csv(file_path)                              # load dataset
df = df.query("company_location == 'United States'").copy()  # filter USA rows

# =========================================================
# Define Response and Predictors
# =========================================================

y_resp = df["salary_in_usd"]                             # response variable

x_pred = df[[
    'experience_level', 'employment_type', 'job_title',
    'employee_residence', 'remote_ratio',
    'company_location', 'company_size', 'data_age',
    'work_year_cat', 'remote_work_cat', 'job_title_group'
]]                                                       # predictor matrix

# =========================================================
# Identify Numeric and Categorical Predictors
# =========================================================

numeric_features = x_pred.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()                                       # numeric predictors

categorical_features = x_pred.select_dtypes(
    include=["object", "category"]
).columns.tolist()                                       # categorical predictors

# =========================================================
# Train/Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    x_pred, y_resp, test_size=0.30, random_state=72018
)                                                        # 70/30 split

# =========================================================
# Preprocessor (One-Hot Encoding)
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_features),     # numeric passthrough
        ("categorical",
         OneHotEncoder(handle_unknown="ignore", drop="first"),
         categorical_features)                            # one-hot encode categorical
    ]
)

# =========================================================
# Metrics Utility
# =========================================================

def compute_metrics(y_true, y_pred):                      # compute RMSE, MAE, R²
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# =========================================================
# Define SVR Hyperparameter Grid
# =========================================================

svr_grid = [
    {"C": 1.0, "epsilon": 0.1, "gamma": "scale"},
    {"C": 10.0, "epsilon": 0.1, "gamma": "scale"},
    {"C": 50.0, "epsilon": 0.1, "gamma": "scale"},
    {"C": 100.0, "epsilon": 0.1, "gamma": "scale"},
    {"C": 10.0, "epsilon": 0.2, "gamma": "scale"},
    {"C": 50.0, "epsilon": 0.2, "gamma": "scale"},
    {"C": 100.0, "epsilon": 0.2, "gamma": "scale"},
    {"C": 50.0, "epsilon": 0.1, "gamma": 0.01},
    {"C": 50.0, "epsilon": 0.1, "gamma": 0.005}
]

# =========================================================
# Fit SVR Models + Collect Metrics
# =========================================================

results = []                                              # store metrics
svr_models = {}                                           # store fitted models

for params in svr_grid:                                   # loop through hyperparameters
    svr = SVR(kernel="rbf", C=params["C"],
              epsilon=params["epsilon"], gamma=params["gamma"])

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),              # preprocessing
            ("model", svr)                               # SVR model
        ]
    )

    pipe.fit(X_train, y_train)                           # fit model
    y_pred = pipe.predict(X_test)                        # predict

    rmse, mae, r2 = compute_metrics(y_test, y_pred)      # compute metrics

    results.append({
        "C": params["C"],
        "epsilon": params["epsilon"],
        "gamma": params["gamma"],
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2
    })

    svr_models[(params["C"], params["epsilon"], params["gamma"])] = {
        "pipeline": pipe,
        "y_pred": y_pred
    }

# =========================================================
# SVR Comparison Table
# =========================================================

svr_df = pd.DataFrame(results).sort_values("RMSE")        # sort by RMSE
print("\n===== SVR MODEL COMPARISON (RBF Kernel) =====")
print(svr_df)

# =========================================================
# Best SVR Model
# =========================================================

best_row = svr_df.iloc[0]                                 # best row
best_params = (best_row["C"], best_row["epsilon"], best_row["gamma"])
best_pipe = svr_models[best_params]["pipeline"]
best_pred = svr_models[best_params]["y_pred"]

print(f"\nBest SVR Model: C={best_params[0]}, epsilon={best_params[1]}, gamma={best_params[2]}")

# =========================================================
# Actual vs Predicted Plot
# =========================================================

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=best_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", linewidth=2)
plt.title("SVR (RBF Kernel): Actual vs Predicted")
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.tight_layout()
plt.show()

# =========================================================
# Residual Plot
# =========================================================

residuals = y_test - best_pred                           # residuals

plt.figure(figsize=(10, 6))
sns.scatterplot(x=best_pred, y=residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.title("SVR (RBF Kernel): Residuals vs Predicted")
plt.xlabel("Predicted Salary")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()

# =========================================================
# Export Best Metrics for SVR (with Adjusted R²)
# =========================================================

# compute adjusted R² for best SVR model
X_test_transformed = best_pipe.named_steps["preprocessor"].transform(X_test)
X_test_transformed = X_test_transformed.toarray() if hasattr(X_test_transformed, "toarray") else X_test_transformed
p = X_test_transformed.shape[1]                          # number of encoded predictors

# recompute metrics including adjusted R²
best_svr_rmse = np.sqrt(mean_squared_error(y_test, best_pred))
best_svr_mae  = mean_absolute_error(y_test, best_pred)
best_svr_r2   = r2_score(y_test, best_pred)
n = len(y_test)
best_svr_adj_r2 = 1 - ((1 - best_svr_r2) * (n - 1) / (n - p - 1))

# export variables for import
__all__ = [
    "best_svr_rmse",
    "best_svr_mae",
    "best_svr_r2",
    "best_svr_adj_r2"
]

