# -*- coding: utf-8 -*-
"""
Author:     Aaron Niecestro
Project:    DS Salaries – Polynomial Regression Analysis

# Created:    September 21 ,2026
# Last Edit:  September 22, 2026
Progress:     Ongoing

Description:
    
PART 1 — Polynomial Regression (Degree 2, Numeric‑Only)

Purpose:
    Fit a polynomial regression model (degree 2) using only numeric predictors.
    Includes preprocessing, model fitting, evaluation, and basic diagnostics.

PART 2 — Interaction‑Only Model (Numeric × Numeric)

Purpose:
    Fit a regression model using ONLY interaction terms between numeric predictors.
    No polynomial (squared) terms included. Captures pairwise numeric interactions.

PART 3 — Combined Polynomial + Interaction Model

Purpose:
    Fit a regression model using BOTH:
        - Polynomial terms (degree 2) for numeric predictors
        - Interaction terms between numeric predictors
    Includes preprocessing, model fitting, evaluation, and basic diagnostics.

PART 4 — Model Selection using AIC/BIC (Forward + Backward Only)

Purpose:
    Perform model selection for the three candidate models:
        - Polynomial-only (Part 1)
        - Interaction-only (Part 2)
        - Polynomial + Interaction (Part 3)

    Uses:
        - Forward AIC
        - Backward AIC
        - Forward BIC
        - Backward BIC

    Output:
        - AIC/BIC comparison table
        - Best model selected for Part 5

PART 5 — Final Model Fit (Based on AIC/BIC Selection)

Purpose:
    Identify the best model among:
        - Polynomial-only (Part 1)
        - Interaction-only (Part 2)
        - Polynomial + Interaction (Part 3)

    Then:
        - Refit the best model cleanly
        - Generate predictions and residuals
        - Store everything needed for Part 6 diagnostics

PART 6 — Final Model Diagnostics, Plots, and Tables

Purpose:
    Perform full diagnostics on the final selected model from Part 5.
    Includes:
        - Actual vs Predicted plot
        - Residual vs Fitted plot
        - QQ plot
        - Breusch–Pagan test
        - Homoscedasticity check
        - Coefficient table
        - Performance metrics table
"""

# =========================================================
# Unified Import Section for Polynomial + Interaction Suite
# =========================================================

import numpy as np                                      # numerical operations
import pandas as pd                                     # data handling
import matplotlib.pyplot as plt                         # plotting
import seaborn as sns                                   # visualization

import statsmodels.api as sm                            # OLS + QQ plot
from statsmodels.stats.diagnostic import het_breuschpagan  # BP test

from sklearn.preprocessing import PolynomialFeatures     # polynomial + interaction generator
from sklearn.preprocessing import OneHotEncoder          # categorical encoding
from sklearn.compose import ColumnTransformer            # preprocessing transformer
from sklearn.pipeline import Pipeline                    # pipeline (Parts 1–3)
from sklearn.linear_model import LinearRegression        # regression model (Parts 1–3)
from sklearn.model_selection import train_test_split     # train/test split

from sklearn.metrics import (
    r2_score,                                            # R² metric
    mean_squared_error,                                  # MSE → RMSE
    mean_absolute_error                                  # MAE
)

"""
PART 1 — Polynomial Regression (Degree 2, Numeric‑Only)

Purpose:
    Fit a polynomial regression model (degree 2) using only numeric predictors.
    Includes preprocessing, model fitting, evaluation, and basic diagnostics.
"""

# =========================================================
# 1. Load Data (USA-only)
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
df = pd.read_csv(file_path)                              # load dataset
df = df.query("company_location == 'United States'").copy()  # filter to USA rows

# =========================================================
# 2. Define Response and Predictors
# =========================================================

y = df["salary_in_usd"]                                  # response variable

X = df[[
    'experience_level', 'employment_type', 'job_title',
    'employee_residence', 'remote_ratio',
    'company_location', 'company_size', 'data_age',
    'work_year_cat', 'remote_work_cat', 'job_title_group'
]]                                                       # predictor matrix

# =========================================================
# 3. Identify Numeric and Categorical Predictors
# =========================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()                                       # numeric predictors only

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()                                       # categorical predictors

# =========================================================
# 4. Train/Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=72018
)                                                        # 70/30 split

# =========================================================
# 5. Polynomial Preprocessing (Degree 2)
# =========================================================

poly = PolynomialFeatures(
    degree=2,                                            # second-degree polynomial
    include_bias=False                                   # no extra bias column
)                                                        # generates numeric polynomial features

preprocessor_poly = ColumnTransformer(
    transformers=[
        ("poly_numeric", poly, numeric_features),         # polynomial transform on numeric predictors
        ("categorical",
         OneHotEncoder(handle_unknown="ignore", drop="first"),
         categorical_features)                            # one-hot encode categorical predictors
    ]
)

# =========================================================
# 6. Polynomial Regression Pipeline
# =========================================================

poly_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor_poly),             # apply polynomial + encoding
        ("regressor", LinearRegression())                # linear regression model
    ]
)

# =========================================================
# 7. Fit Model
# =========================================================

poly_pipeline.fit(X_train, y_train)                      # fit polynomial regression model

# =========================================================
# 8. Predict
# =========================================================

y_pred = poly_pipeline.predict(X_test)                   # predictions on test set

# =========================================================
# 9. Metrics
# =========================================================

rmse = np.sqrt(mean_squared_error(y_test, y_pred))       # RMSE
mae = mean_absolute_error(y_test, y_pred)                # MAE
r2 = r2_score(y_test, y_pred)                            # R²

print("\n===== PART 1: Polynomial Regression (Degree 2) =====")
print(f"RMSE: {rmse:,.2f}")                              # print RMSE
print(f"MAE:  {mae:,.2f}")                               # print MAE
print(f"R²:   {r2:.4f}")                                 # print R²

# =========================================================
# 10. Actual vs Predicted Plot
# =========================================================

plt.figure(figsize=(10, 6))                              # figure size
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)           # scatter plot
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--", linewidth=2)                             # perfect prediction line
plt.title("Polynomial Regression (Degree 2): Actual vs Predicted")  # title
plt.xlabel("Actual Salary")                              # x-axis label
plt.ylabel("Predicted Salary")                           # y-axis label
plt.show()                                               # display plot

# =========================================================
# 11. Residual Plot
# =========================================================

residuals = y_test - y_pred                              # compute residuals

plt.figure(figsize=(10, 6))                              # figure size
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)        # residuals vs predicted
plt.axhline(0, color="red", linestyle="--")              # zero line
plt.title("Residuals vs Predicted (Polynomial Regression)")  # title
plt.xlabel("Predicted Salary")                           # x-axis label
plt.ylabel("Residuals")                                  # y-axis label
plt.show()                                               # display plot

"""
PART 2 — Interaction‑Only Model (Numeric × Numeric)

Purpose:
    Fit a regression model using ONLY interaction terms between numeric predictors.
    No polynomial (squared) terms included. Captures pairwise numeric interactions.
"""

# =========================================================
# 1. Load Data (USA-only)
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
df = pd.read_csv(file_path)                              # load dataset
df = df.query("company_location == 'United States'").copy()  # filter to USA rows

# =========================================================
# 2. Define Response and Predictors
# =========================================================

y = df["salary_in_usd"]                                  # response variable

X = df[[
    'experience_level', 'employment_type', 'job_title',
    'employee_residence', 'remote_ratio',
    'company_location', 'company_size', 'data_age',
    'work_year_cat', 'remote_work_cat', 'job_title_group'
]]                                                       # predictor matrix

# =========================================================
# 3. Identify Numeric and Categorical Predictors
# =========================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()                                       # numeric predictors only

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()                                       # categorical predictors

# =========================================================
# 4. Train/Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=72018
)                                                        # 70/30 split

# =========================================================
# 5. Interaction‑Only Preprocessing (Numeric × Numeric)
# =========================================================

interaction = PolynomialFeatures(
    degree=2,                                            # degree 2 but…
    include_bias=False,                                  # no bias term
    interaction_only=True                                # ONLY interaction terms (no squares)
)

preprocessor_interact = ColumnTransformer(
    transformers=[
        ("interactions", interaction, numeric_features),  # numeric × numeric interactions
        ("categorical",
         OneHotEncoder(handle_unknown="ignore", drop="first"),
         categorical_features)                            # one-hot encode categorical predictors
    ]
)

# =========================================================
# 6. Interaction‑Only Regression Pipeline
# =========================================================

interact_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor_interact),         # apply interaction + encoding
        ("regressor", LinearRegression())                # linear regression model
    ]
)

# =========================================================
# 7. Fit Model
# =========================================================

interact_pipeline.fit(X_train, y_train)                  # fit interaction-only model

# =========================================================
# 8. Predict
# =========================================================

y_pred = interact_pipeline.predict(X_test)               # predictions on test set

# =========================================================
# 9. Metrics
# =========================================================

rmse = np.sqrt(mean_squared_error(y_test, y_pred))       # RMSE
mae = mean_absolute_error(y_test, y_pred)                # MAE
r2 = r2_score(y_test, y_pred)                            # R²

print("\n===== PART 2: Interaction‑Only Regression =====")
print(f"RMSE: {rmse:,.2f}")                              # print RMSE
print(f"MAE:  {mae:,.2f}")                               # print MAE
print(f"R²:   {r2:.4f}")                                 # print R²

# =========================================================
# 10. Actual vs Predicted Plot
# =========================================================

plt.figure(figsize=(10, 6))                              # figure size
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)           # scatter plot
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--", linewidth=2)                             # perfect prediction line
plt.title("Interaction‑Only Regression: Actual vs Predicted")  # title
plt.xlabel("Actual Salary")                              # x-axis label
plt.ylabel("Predicted Salary")                           # y-axis label
plt.show()                                               # display plot

# =========================================================
# 11. Residual Plot
# =========================================================

residuals = y_test - y_pred                              # compute residuals

plt.figure(figsize=(10, 6))                              # figure size
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)        # residuals vs predicted
plt.axhline(0, color="red", linestyle="--")              # zero line
plt.title("Residuals vs Predicted (Interaction‑Only)")   # title
plt.xlabel("Predicted Salary")                           # x-axis label
plt.ylabel("Residuals")                                  # y-axis label
plt.show()                                               # display plot

"""
PART 3 — Combined Polynomial + Interaction Model

Purpose:
    Fit a regression model using BOTH:
        • Polynomial terms (degree 2) for numeric predictors
        • Interaction terms between numeric predictors
    Includes preprocessing, model fitting, evaluation, and basic diagnostics.
"""

# =========================================================
# 1. Load Data (USA-only)
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
df = pd.read_csv(file_path)                              # load dataset
df = df.query("company_location == 'United States'").copy()  # filter to USA rows

# =========================================================
# 2. Define Response and Predictors
# =========================================================

y = df["salary_in_usd"]                                  # response variable

X = df[[
    'experience_level', 'employment_type', 'job_title',
    'employee_residence', 'remote_ratio',
    'company_location', 'company_size', 'data_age',
    'work_year_cat', 'remote_work_cat', 'job_title_group'
]]                                                       # predictor matrix

# =========================================================
# 3. Identify Numeric and Categorical Predictors
# =========================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()                                       # numeric predictors only

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()                                       # categorical predictors

# =========================================================
# 4. Train/Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=72018
)                                                        # 70/30 split

# =========================================================
# 5. Polynomial + Interaction Preprocessing
# =========================================================

poly_interact = PolynomialFeatures(
    degree=2,                                            # degree 2 polynomial
    include_bias=False,                                  # no bias term
    interaction_only=False                               # include BOTH squares + interactions
)

preprocessor_poly_interact = ColumnTransformer(
    transformers=[
        ("poly_interact_numeric", poly_interact, numeric_features),  # polynomial + interaction terms
        ("categorical",
         OneHotEncoder(handle_unknown="ignore", drop="first"),
         categorical_features)                            # one-hot encode categorical predictors
    ]
)

# =========================================================
# 6. Combined Polynomial + Interaction Pipeline
# =========================================================

poly_interact_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor_poly_interact),     # apply polynomial + interactions + encoding
        ("regressor", LinearRegression())                 # linear regression model
    ]
)

# =========================================================
# 7. Fit Model
# =========================================================

poly_interact_pipeline.fit(X_train, y_train)              # fit combined model

# =========================================================
# 8. Predict
# =========================================================

y_pred = poly_interact_pipeline.predict(X_test)           # predictions on test set

# =========================================================
# 9. Metrics
# =========================================================

rmse = np.sqrt(mean_squared_error(y_test, y_pred))        # RMSE
mae = mean_absolute_error(y_test, y_pred)                 # MAE
r2 = r2_score(y_test, y_pred)                             # R²

print("\n===== PART 3: Combined Polynomial + Interaction Model =====")
print(f"RMSE: {rmse:,.2f}")                               # print RMSE
print(f"MAE:  {mae:,.2f}")                                # print MAE
print(f"R²:   {r2:.4f}")                                  # print R²

# =========================================================
# 10. Actual vs Predicted Plot
# =========================================================

plt.figure(figsize=(10, 6))                               # figure size
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)            # scatter plot
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--", linewidth=2)                              # perfect prediction line
plt.title("Combined Polynomial + Interaction: Actual vs Predicted")  # title
plt.xlabel("Actual Salary")                               # x-axis label
plt.ylabel("Predicted Salary")                            # y-axis label
plt.show()                                                # display plot

# =========================================================
# 11. Residual Plot
# =========================================================

residuals = y_test - y_pred                               # compute residuals

plt.figure(figsize=(10, 6))                               # figure size
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)         # residuals vs predicted
plt.axhline(0, color="red", linestyle="--")               # zero line
plt.title("Residuals vs Predicted (Polynomial + Interaction)")  # title
plt.xlabel("Predicted Salary")                            # x-axis label
plt.ylabel("Residuals")                                   # y-axis label
plt.show()                                                # display plot

"""
PART 4 — Model Selection using AIC/BIC (Forward + Backward Only)

Purpose:
    Perform model selection for the three candidate models:
        • Polynomial-only (Part 1)
        • Interaction-only (Part 2)
        • Polynomial + Interaction (Part 3)

    Uses:
        • Forward AIC
        • Backward AIC
        • Forward BIC
        • Backward BIC

    Output:
        • AIC/BIC comparison table
        • Best model selected for Part 5
"""

# =========================================================
# 1. Load Data (USA-only)
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
df = pd.read_csv(file_path)                              # load dataset
df = df.query("company_location == 'United States'").copy()  # filter to USA rows

# =========================================================
# 2. Define Response and Predictors
# =========================================================

y = df["salary_in_usd"]                                  # response variable

X = df[[
    'experience_level', 'employment_type', 'job_title',
    'employee_residence', 'remote_ratio',
    'company_location', 'company_size', 'data_age',
    'work_year_cat', 'remote_work_cat', 'job_title_group'
]]                                                       # predictor matrix

# =========================================================
# 3. Identify Numeric and Categorical Predictors
# =========================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()                                       # numeric predictors only

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()                                       # categorical predictors

# =========================================================
# 4. Train/Test Split (same as Parts 1–3)
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=72018
)                                                        # 70/30 split

# =========================================================
# 5. Build Design Matrices for All Three Models
# =========================================================

# ---------- Polynomial-only (Part 1) ----------
poly = PolynomialFeatures(degree=2, include_bias=False)  # polynomial degree 2
pre_poly = ColumnTransformer(
    [("poly", poly, numeric_features),
     ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)]
)
X_poly = pre_poly.fit_transform(X_train)                 # transformed polynomial design matrix
X_poly = sm.add_constant(X_poly)                         # add intercept

# ---------- Interaction-only (Part 2) ----------
interact = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
pre_interact = ColumnTransformer(
    [("interact", interact, numeric_features),
     ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)]
)
X_interact = pre_interact.fit_transform(X_train)         # transformed interaction-only matrix
X_interact = sm.add_constant(X_interact)                 # add intercept

# ---------- Polynomial + Interaction (Part 3) ----------
poly_interact = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
pre_poly_interact = ColumnTransformer(
    [("poly_interact", poly_interact, numeric_features),
     ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)]
)
X_poly_interact = pre_poly_interact.fit_transform(X_train)  # transformed combined matrix
X_poly_interact = sm.add_constant(X_poly_interact)          # add intercept

# =========================================================
# 6. Utility: Fit OLS
# =========================================================

def fit_ols(Xmat, yvec):                                  # fits OLS model
    return sm.OLS(yvec, Xmat).fit()                       # returns fitted model

# =========================================================
# 7. Forward Selection (AIC/BIC)
# =========================================================

def forward_selection(Xmat, yvec, criterion="AIC"):        # forward selection function
    remaining = list(range(1, Xmat.shape[1]))              # candidate predictors (skip intercept)
    selected = []                                          # selected predictors
    current_score = np.inf                                 # initialize best score

    while remaining:                                       # loop until no improvement
        scores = []                                        # store scores
        for cand in remaining:                             # test each candidate
            cols = [0] + selected + [cand]                 # intercept + selected + candidate
            model = fit_ols(Xmat[:, cols], yvec)           # fit model
            score = model.aic if criterion == "AIC" else model.bic
            scores.append((score, cand))                   # store score + candidate

        scores.sort()                                      # sort by best score
        best_score, best_cand = scores[0]                  # best candidate

        if best_score < current_score:                     # improvement check
            selected.append(best_cand)                     # add predictor
            remaining.remove(best_cand)                    # remove from remaining
            current_score = best_score                     # update score
        else:
            break                                          # stop if no improvement

    return selected, current_score                         # return selected predictors + score

# =========================================================
# 8. Backward Elimination (AIC/BIC)
# =========================================================

def backward_elimination(Xmat, yvec, criterion="AIC"):     # backward selection function
    selected = list(range(1, Xmat.shape[1]))               # start with all predictors
    model_full = fit_ols(Xmat[:, [0] + selected], yvec)    # full model
    current_score = model_full.aic if criterion == "AIC" else model_full.bic

    while len(selected) > 1:                               # loop until only one predictor left
        scores = []                                        # store scores
        for cand in selected:                              # test removing each predictor
            cols = [0] + [c for c in selected if c != cand]
            model = fit_ols(Xmat[:, cols], yvec)
            score = model.aic if criterion == "AIC" else model.bic
            scores.append((score, cand))

        scores.sort()                                      # sort by best score
        best_score, worst_cand = scores[0]                 # best removal

        if best_score < current_score:                     # improvement check
            selected.remove(worst_cand)                    # remove predictor
            current_score = best_score                     # update score
        else:
            break                                          # stop if no improvement

    return selected, current_score                         # return selected predictors + score

# =========================================================
# 9. Run Model Selection for All Three Models
# =========================================================

models = {
    "Polynomial Only": X_poly,
    "Interaction Only": X_interact,
    "Polynomial + Interaction": X_poly_interact
}

results = []                                               # store results

for name, Xmat in models.items():                          # loop through models
    fwd_aic_vars, fwd_aic_score = forward_selection(Xmat, y_train, "AIC")
    bwd_aic_vars, bwd_aic_score = backward_elimination(Xmat, y_train, "AIC")
    fwd_bic_vars, fwd_bic_score = forward_selection(Xmat, y_train, "BIC")
    bwd_bic_vars, bwd_bic_score = backward_elimination(Xmat, y_train, "BIC")

    results.append({
        "Model": name,
        "Forward AIC": fwd_aic_score,
        "Backward AIC": bwd_aic_score,
        "Forward BIC": fwd_bic_score,
        "Backward BIC": bwd_bic_score
    })

# =========================================================
# 10. Results Table
# =========================================================

results_df = pd.DataFrame(results)                        # convert to DataFrame

print("\n===== PART 4: AIC/BIC Model Selection =====")
print(results_df)                                         # print selection table

# =========================================================
# 11. Identify Best Model
# =========================================================

best_model_row = results_df.loc[
    results_df[["Forward AIC", "Backward AIC", "Forward BIC", "Backward BIC"]].idxmin().min()
]                                                         # find best row

print("\n===== BEST MODEL BASED ON AIC/BIC =====")
print(best_model_row)                                     # print best model info

"""
PART 5 — Final Model Fit (Based on AIC/BIC Selection)

Purpose:
    Identify the best model among:
        • Polynomial-only (Part 1)
        • Interaction-only (Part 2)
        • Polynomial + Interaction (Part 3)

    Then:
        • Refit the best model cleanly
        • Generate predictions and residuals
        • Store everything needed for Part 6 diagnostics
"""

# =========================================================
# 1. Load Data (USA-only)
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
df = pd.read_csv(file_path)                              # load dataset
df = df.query("company_location == 'United States'").copy()  # filter to USA rows

# =========================================================
# 2. Define Response and Predictors
# =========================================================

y = df["salary_in_usd"]                                  # response variable

X = df[[
    'experience_level', 'employment_type', 'job_title',
    'employee_residence', 'remote_ratio',
    'company_location', 'company_size', 'data_age',
    'work_year_cat', 'remote_work_cat', 'job_title_group'
]]                                                       # predictor matrix

# =========================================================
# 3. Identify Numeric and Categorical Predictors
# =========================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()                                       # numeric predictors only

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()                                       # categorical predictors

# =========================================================
# 4. Train/Test Split (same as Parts 1–4)
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=72018
)                                                        # 70/30 split

# =========================================================
# 5. Build Design Matrices for All Three Models
# =========================================================

# ---------- Polynomial-only ----------
poly = PolynomialFeatures(degree=2, include_bias=False)
pre_poly = ColumnTransformer(
    [("poly", poly, numeric_features),
     ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)]
)
X_poly = pre_poly.fit_transform(X_train)                 # polynomial design matrix
X_poly = sm.add_constant(X_poly)                         # add intercept

# ---------- Interaction-only ----------
interact = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
pre_interact = ColumnTransformer(
    [("interact", interact, numeric_features),
     ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)]
)
X_interact = pre_interact.fit_transform(X_train)         # interaction-only matrix
X_interact = sm.add_constant(X_interact)                 # add intercept

# ---------- Polynomial + Interaction ----------
poly_interact = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
pre_poly_interact = ColumnTransformer(
    [("poly_interact", poly_interact, numeric_features),
     ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)]
)
X_poly_interact = pre_poly_interact.fit_transform(X_train)  # combined matrix
X_poly_interact = sm.add_constant(X_poly_interact)          # add intercept

# =========================================================
# 6. Utility: Fit OLS
# =========================================================

def fit_ols(Xmat, yvec):                                  # fits OLS model
    return sm.OLS(yvec, Xmat).fit()                       # returns fitted model

# =========================================================
# 7. Compute AIC/BIC for All Three Models
# =========================================================

models = {
    "Polynomial Only": X_poly,
    "Interaction Only": X_interact,
    "Polynomial + Interaction": X_poly_interact
}

scores = []                                               # store AIC/BIC scores

for name, Xmat in models.items():                         # loop through models
    model = fit_ols(Xmat, y_train)                        # fit OLS
    scores.append({
        "Model": name,
        "AIC": model.aic,
        "BIC": model.bic
    })

scores_df = pd.DataFrame(scores)                          # convert to DataFrame

print("\n===== PART 5: AIC/BIC Scores for All Models =====")
print(scores_df)                                          # print AIC/BIC table

# =========================================================
# 8. Identify Best Model (Lowest AIC)
# =========================================================

best_row = scores_df.loc[scores_df["AIC"].idxmin()]       # best model by AIC
best_model_name = best_row["Model"]                       # extract model name

print("\n===== BEST MODEL SELECTED (Based on AIC) =====")
print(best_row)                                           # print best model info

# =========================================================
# 9. Refit Best Model Cleanly
# =========================================================

if best_model_name == "Polynomial Only":
    final_preprocessor = pre_poly                         # polynomial-only preprocessor
    X_train_final = X_poly                                # design matrix
elif best_model_name == "Interaction Only":
    final_preprocessor = pre_interact                     # interaction-only preprocessor
    X_train_final = X_interact                            # design matrix
else:
    final_preprocessor = pre_poly_interact                # combined preprocessor
    X_train_final = X_poly_interact                       # design matrix

final_model = fit_ols(X_train_final, y_train)             # refit final model

# =========================================================
# 10. Prepare Test Matrix for Final Model
# =========================================================

X_test_final = final_preprocessor.transform(X_test)       # transform test set
X_test_final = sm.add_constant(X_test_final)              # add intercept

# =========================================================
# 11. Predictions + Residuals
# =========================================================

y_pred = final_model.predict(X_test_final)                # predictions
residuals = y_test - y_pred                               # residuals

print("\n===== FINAL MODEL PERFORMANCE =====")
print(f"RMSE: {np.sqrt(np.mean(residuals**2)):,.2f}")     # RMSE
print(f"MAE:  {np.mean(np.abs(residuals)):,.2f}")          # MAE
print(f"R²:   {final_model.rsquared:.4f}")                 # R²

# =========================================================
# 12. Save Outputs for Part 6 Diagnostics
# =========================================================

# These variables are intentionally left in memory:
#     final_model
#     final_preprocessor
#     X_train_final
#     X_test_final
#     y_train
#     y_test
#     y_pred
#     residuals

print("\nFinal model, predictions, and residuals are ready for Part 6 diagnostics.")

"""
PART 6 — Final Model Diagnostics, Plots, and Tables

Purpose:
    Perform full diagnostics on the final selected model from Part 5.
    Includes:
        • Actual vs Predicted plot
        • Residual vs Fitted plot
        • QQ plot
        • Breusch–Pagan test
        • Homoscedasticity check
        • Coefficient table
        • Performance metrics table
"""

# =========================================================
# 1. Load Data (USA-only)
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
df = pd.read_csv(file_path)                              # load dataset
df = df.query("company_location == 'United States'").copy()  # filter to USA rows

# =========================================================
# 2. Define Response and Predictors
# =========================================================

y = df["salary_in_usd"]                                  # response variable

X = df[[
    'experience_level', 'employment_type', 'job_title',
    'employee_residence', 'remote_ratio',
    'company_location', 'company_size', 'data_age',
    'work_year_cat', 'remote_work_cat', 'job_title_group'
]]                                                       # predictor matrix

# =========================================================
# 3. Identify Numeric and Categorical Predictors
# =========================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()                                       # numeric predictors only

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()                                       # categorical predictors

# =========================================================
# 4. Train/Test Split (same as Parts 1–5)
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=72018
)                                                        # 70/30 split

# =========================================================
# 5. Rebuild All Three Design Matrices (same as Part 5)
# =========================================================

# ---------- Polynomial-only ----------
poly = PolynomialFeatures(degree=2, include_bias=False)
pre_poly = ColumnTransformer(
    [("poly", poly, numeric_features),
     ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)]
)
X_poly = pre_poly.fit_transform(X_train)
X_poly = sm.add_constant(X_poly)

# ---------- Interaction-only ----------
interact = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
pre_interact = ColumnTransformer(
    [("interact", interact, numeric_features),
     ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)]
)
X_interact = pre_interact.fit_transform(X_train)
X_interact = sm.add_constant(X_interact)

# ---------- Polynomial + Interaction ----------
poly_interact = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
pre_poly_interact = ColumnTransformer(
    [("poly_interact", poly_interact, numeric_features),
     ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)]
)
X_poly_interact = pre_poly_interact.fit_transform(X_train)
X_poly_interact = sm.add_constant(X_poly_interact)

# =========================================================
# 6. Utility: Fit OLS
# =========================================================

def fit_ols(Xmat, yvec):                                  # fits OLS model
    return sm.OLS(yvec, Xmat).fit()                       # returns fitted model

# =========================================================
# 7. Compute AIC/BIC for All Models (same logic as Part 5)
# =========================================================

models = {
    "Polynomial Only": X_poly,
    "Interaction Only": X_interact,
    "Polynomial + Interaction": X_poly_interact
}

scores = []                                               # store AIC/BIC scores

for name, Xmat in models.items():                         # loop through models
    model = fit_ols(Xmat, y_train)                        # fit OLS
    scores.append({
        "Model": name,
        "AIC": model.aic,
        "BIC": model.bic
    })

scores_df = pd.DataFrame(scores)                          # convert to DataFrame

best_row = scores_df.loc[scores_df["AIC"].idxmin()]       # best model by AIC
best_model_name = best_row["Model"]                       # extract model name

print("\n===== PART 6: Best Model Identified =====")
print(best_row)                                           # print best model info

# =========================================================
# 8. Refit Best Model Cleanly
# =========================================================

if best_model_name == "Polynomial Only":
    final_preprocessor = pre_poly
    X_train_final = X_poly
elif best_model_name == "Interaction Only":
    final_preprocessor = pre_interact
    X_train_final = X_interact
else:
    final_preprocessor = pre_poly_interact
    X_train_final = X_poly_interact

final_model = fit_ols(X_train_final, y_train)             # refit final model

# =========================================================
# 9. Prepare Test Matrix
# =========================================================

X_test_final = final_preprocessor.transform(X_test)       # transform test set
X_test_final = sm.add_constant(X_test_final)              # add intercept

# =========================================================
# 10. Predictions + Residuals
# =========================================================

y_pred = final_model.predict(X_test_final)                # predictions
residuals = y_test - y_pred                               # residuals

# =========================================================
# 11. Performance Table
# =========================================================

rmse = np.sqrt(np.mean(residuals**2))                     # RMSE
mae = np.mean(np.abs(residuals))                          # MAE
r2 = final_model.rsquared                                 # R²

perf_df = pd.DataFrame({
    "Metric": ["RMSE", "MAE", "R²"],
    "Value": [rmse, mae, r2]
})

print("\n===== PERFORMANCE METRICS =====")
print(perf_df)

# =========================================================
# 12. Actual vs Predicted Plot
# =========================================================

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--", linewidth=2)
plt.title(f"{best_model_name}: Actual vs Predicted")
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.show()

# =========================================================
# 13. Residual vs Fitted Plot
# =========================================================

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.title(f"{best_model_name}: Residuals vs Predicted")
plt.xlabel("Predicted Salary")
plt.ylabel("Residuals")
plt.show()

# =========================================================
# 14. QQ Plot (Normality Check)
# =========================================================

plt.figure(figsize=(8, 6))
sm.qqplot(residuals, line="45", fit=True)
plt.title("QQ Plot of Residuals")
plt.show()

# =========================================================
# 15. Breusch–Pagan Test (Homoscedasticity)
# =========================================================

bp_test = het_breuschpagan(residuals, X_test_final)

bp_labels = ["LM Statistic", "LM p-value", "F Statistic", "F p-value"]
bp_results = pd.Series(bp_test, index=bp_labels)

print("\n===== BREUSCH–PAGAN TEST =====")
print(bp_results)

# =========================================================
# 16. Coefficient Table
# =========================================================

coef_df = pd.DataFrame({
    "Coefficient": final_model.params
})

print("\n===== COEFFICIENT TABLE =====")
print(coef_df)

# =========================================================
# 17. Full Model Summary
# =========================================================

print("\n===== FINAL MODEL SUMMARY =====")
print(final_model.summary())

