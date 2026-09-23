# -*- coding: utf-8 -*-
"""
Author:     Aaron Niecestro
Project:    DS Salaries – Polynomial Regression Analysis

# Created:    September 21 ,2026
# Last Edit:  September 23, 2026
Progress:     Ongoing - Part 4 is taking long time to run

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
df = pd.read_csv(file_path)                                   # load dataset
df = df.query("company_location == 'United States'").copy()   # filter to USA rows
df["log_salary"] = np.log(df["salary_in_usd"])

# =========================================================
# 2. Define Log-Response and Predictors
# =========================================================

y = df["log_salary"]                             # LOG-transformed response

X = df[[
    'experience_level', 'employment_type', 'job_title',
    'employee_residence', 'remote_ratio',
    'company_location', 'company_size', 'data_age',
    'work_year_cat', 'remote_work_cat', 'job_title_group'
]]                                                             # predictor matrix

# =========================================================
# 3. Identify Numeric and Categorical Predictors
# =========================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()                                             # numeric predictors only

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()                                             # categorical predictors

# =========================================================
# 4. Train/Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=72018
)                                                              # 70/30 split

# =========================================================
# 5. Polynomial Preprocessing (Degree 2)
# =========================================================

poly = PolynomialFeatures(
    degree=2,                                                  # second-degree polynomial
    include_bias=False                                         # no extra bias column
)                                                              # generates numeric polynomial features

preprocessor_poly = ColumnTransformer(
    transformers=[
        ("poly_numeric", poly, numeric_features),              # polynomial transform on numeric predictors
        ("categorical",
         OneHotEncoder(handle_unknown="ignore", drop="first"),
         categorical_features)                                 # one-hot encode categorical predictors
    ]
)

# did some polynomial graphs but did not much information from them

# =========================================================
# 6. Polynomial Regression Pipeline
# =========================================================

poly_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor_poly),                   # apply polynomial + encoding
        ("regressor", LinearRegression())                      # linear regression model
    ]
)

# =========================================================
# 7. Fit Model
# =========================================================

poly_pipeline.fit(X_train, y_train)                            # fit polynomial regression model (log-response)

# =========================================================
# 8. Predict (log-scale)
# =========================================================

y_pred_log = poly_pipeline.predict(X_test)                     # predictions on log scale

# =========================================================
# 9. Metrics (log-scale)
# =========================================================

rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))     # RMSE on log scale
mae_log = mean_absolute_error(y_test, y_pred_log)              # MAE on log scale
r2_log = r2_score(y_test, y_pred_log)                          # R² on log scale

# =========================================================
# 10. AIC / BIC (log-likelihood based)
# =========================================================

n = len(y_test)                                                # number of observations
k = poly_pipeline.named_steps["regressor"].coef_.shape[0] + 1  # number of parameters (coefficients + intercept)

residuals = y_test - y_pred_log                                # residuals on log scale
sigma2 = np.var(residuals, ddof=k)                             # variance estimate

log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)   # Gaussian log-likelihood

aic = -2 * log_likelihood + 2 * k                              # AIC formula
bic = -2 * log_likelihood + k * np.log(n)                      # BIC formula

print("\n===== PART 1: Polynomial Regression (Degree 2, LOG Response) =====")
print(f"RMSE (log): {rmse_log:,.4f}")                          # print RMSE
print(f"MAE  (log): {mae_log:,.4f}")                           # print MAE
print(f"R²   (log): {r2_log:.4f}")                             # print R²
print(f"AIC (log): {aic:,.2f}")                                # print AIC
print(f"BIC (log): {bic:,.2f}")                                # print BIC

# =========================================================
# 11. Actual vs Predicted Plot (log-scale)
# =========================================================

plt.figure(figsize=(10, 6))                                    # figure size
sns.scatterplot(x=y_test, y=y_pred_log, alpha=0.6)             # scatter plot (log actual vs log predicted)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--", linewidth=2)                                   # perfect prediction line
plt.title("Polynomial Regression (Degree 2): Actual vs Predicted (LOG Salary)")  
plt.xlabel("Actual log(salary)")                               # x-axis label
plt.ylabel("Predicted log(salary)")                            # y-axis label
plt.show()                                                     # display plot

# =========================================================
# 12. Residual Plot (log-scale)
# =========================================================

plt.figure(figsize=(10, 6))                                    # figure size
sns.scatterplot(x=y_pred_log, y=residuals, alpha=0.6)          # residuals vs predicted (log)
plt.axhline(0, color="red", linestyle="--")                    # zero line
plt.title("Residuals vs Predicted (Polynomial Regression, LOG Salary)")  
plt.xlabel("Predicted log(salary)")                            # x-axis label
plt.ylabel("Residuals (log-scale)")                            # y-axis label
plt.show()                                                     # display plot

"""
PART 2 — Interaction‑Only Model (Numeric × Numeric)

Purpose:
    Fit a regression model using ONLY interaction terms between numeric predictors.
    No polynomial (squared) terms included. Captures pairwise numeric interactions.
    
Assumes df, X, y, numeric_features, categorical_features,
X_train, X_test, y_train, y_test were already defined ONCE
at the top of the script.
"""

# =========================================================
# 1. Interaction‑Only Preprocessing (Numeric × Numeric)
# =========================================================

interaction = PolynomialFeatures(
    degree=2,                                                  # degree 2
    include_bias=False,                                        # no bias term
    interaction_only=True                                      # ONLY interaction terms (no squares)
)

preprocessor_interact = ColumnTransformer(
    transformers=[
        ("interactions", interaction, numeric_features),        # numeric × numeric interactions
        ("categorical",
         OneHotEncoder(handle_unknown="ignore", drop="first"),
         categorical_features)                                  # one-hot encode categorical predictors
    ]
)

# =========================================================
# 2. Interaction‑Only Regression Pipeline
# =========================================================

interact_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor_interact),               # apply interaction + encoding
        ("regressor", LinearRegression())                      # linear regression model
    ]
)

# =========================================================
# 3. Fit Model
# =========================================================

interact_pipeline.fit(X_train, y_train)                        # fit interaction-only model (log-response)

# =========================================================
# 4. Predict (log-scale)
# =========================================================

y_pred_log = interact_pipeline.predict(X_test)                 # predictions on log scale

# =========================================================
# 5. Metrics (log-scale)
# =========================================================

rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))     # RMSE on log scale
mae_log = mean_absolute_error(y_test, y_pred_log)              # MAE on log scale
r2_log = r2_score(y_test, y_pred_log)                          # R² on log scale

print("\n===== PART 2: Interaction‑Only Regression (LOG Response) =====")
print(f"RMSE (log): {rmse_log:,.4f}")                          # print RMSE
print(f"MAE  (log): {mae_log:,.4f}")                           # print MAE
print(f"R²   (log): {r2_log:.4f}")                             # print R²

# =========================================================
# 6. AIC / BIC (log-likelihood)
# =========================================================

n = len(y_test)                                                # number of observations
k = interact_pipeline.named_steps["regressor"].coef_.shape[0] + 1  # parameters (coefficients + intercept)

residuals = y_test - y_pred_log                                # residuals on log scale
sigma2 = np.var(residuals, ddof=k)                             # variance estimate

log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)   # Gaussian log-likelihood

aic = -2 * log_likelihood + 2 * k                              # AIC formula
bic = -2 * log_likelihood + k * np.log(n)                      # BIC formula

print(f"AIC (log): {aic:,.2f}")                                # print AIC
print(f"BIC (log): {bic:,.2f}")                                # print BIC

# =========================================================
# 7. Actual vs Predicted Plot (log-scale)
# =========================================================

plt.figure(figsize=(10, 6))                                    # figure size
sns.scatterplot(x=y_test, y=y_pred_log, alpha=0.6)             # scatter plot (log actual vs log predicted)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--", linewidth=2)                                   # perfect prediction line
plt.title("Interaction‑Only Regression: Actual vs Predicted (LOG Salary)")  
plt.xlabel("Actual log(salary)")                               # x-axis label
plt.ylabel("Predicted log(salary)")                            # y-axis label
plt.show()                                                     # display plot

# =========================================================
# 8. Residual Plot (log-scale)
# =========================================================

plt.figure(figsize=(10, 6))                                    # figure size
sns.scatterplot(x=y_pred_log, y=residuals, alpha=0.6)          # residuals vs predicted (log)
plt.axhline(0, color="red", linestyle="--")                    # zero line
plt.title("Residuals vs Predicted (Interaction‑Only, LOG Salary)")  
plt.xlabel("Predicted log(salary)")                            # x-axis label
plt.ylabel("Residuals (log-scale)")                            # y-axis label
plt.show()                                                     # display plot

"""
PART 3 — Combined Polynomial + Interaction Model

Purpose:
    Fit a regression model using BOTH:
        - Polynomial terms (degree 2) for numeric predictors
        - Interaction terms between numeric predictors
    Includes preprocessing, model fitting, evaluation, and basic diagnostics.

Assumes df, X, y, numeric_features, categorical_features,
X_train, X_test, y_train, y_test were already defined ONCE
at the top of the script.
"""

# =========================================================
# 1. Polynomial + Interaction Preprocessing
# =========================================================

poly_interact = PolynomialFeatures(
    degree=2,                                                  # degree 2 polynomial
    include_bias=False,                                        # no bias term
    interaction_only=False                                     # include squares + interactions
)

preprocessor_poly_interact = ColumnTransformer(
    transformers=[
        ("poly_interact_numeric", poly_interact, numeric_features),  # polynomial + interaction terms
        ("categorical",
         OneHotEncoder(handle_unknown="ignore", drop="first"),
         categorical_features)                                      # one-hot encode categorical predictors
    ]
)

# =========================================================
# 2. Combined Polynomial + Interaction Pipeline
# =========================================================

poly_interact_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor_poly_interact),         # apply polynomial + interactions + encoding
        ("regressor", LinearRegression())                     # linear regression model
    ]
)

# =========================================================
# 3. Fit Model
# =========================================================

poly_interact_pipeline.fit(X_train, y_train)                  # fit combined model (log-response)

# =========================================================
# 4. Predict (log-scale)
# =========================================================

y_pred_log = poly_interact_pipeline.predict(X_test)           # predictions on log scale

# =========================================================
# 5. Metrics (log-scale)
# =========================================================

rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))    # RMSE on log scale
mae_log = mean_absolute_error(y_test, y_pred_log)             # MAE on log scale
r2_log = r2_score(y_test, y_pred_log)                         # R² on log scale

print("\n===== PART 3: Combined Polynomial + Interaction Model (LOG Response) =====")
print(f"RMSE (log): {rmse_log:,.4f}")                         # print RMSE
print(f"MAE  (log): {mae_log:,.4f}")                          # print MAE
print(f"R²   (log): {r2_log:.4f}")                            # print R²

# =========================================================
# 6. AIC / BIC (log-likelihood)
# =========================================================

n = len(y_test)                                               # number of observations
k = poly_interact_pipeline.named_steps["regressor"].coef_.shape[0] + 1  # parameters (coefficients + intercept)

residuals = y_test - y_pred_log                               # residuals on log scale
sigma2 = np.var(residuals, ddof=k)                            # variance estimate

log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)  # Gaussian log-likelihood

aic = -2 * log_likelihood + 2 * k                             # AIC formula
bic = -2 * log_likelihood + k * np.log(n)                     # BIC formula

print(f"AIC (log): {aic:,.2f}")                               # print AIC
print(f"BIC (log): {bic:,.2f}")                               # print BIC

# =========================================================
# 7. Actual vs Predicted Plot (log-scale)
# =========================================================

plt.figure(figsize=(10, 6))                                   # figure size
sns.scatterplot(x=y_test, y=y_pred_log, alpha=0.6)            # scatter plot (log actual vs log predicted)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--", linewidth=2)                                  # perfect prediction line
plt.title("Combined Polynomial + Interaction: Actual vs Predicted (LOG Salary)")  
plt.xlabel("Actual log(salary)")                              # x-axis label
plt.ylabel("Predicted log(salary)")                           # y-axis label
plt.show()                                                    # display plot

# =========================================================
# 8. Residual Plot (log-scale)
# =========================================================

plt.figure(figsize=(10, 6))                                   # figure size
sns.scatterplot(x=y_pred_log, y=residuals, alpha=0.6)         # residuals vs predicted (log)
plt.axhline(0, color="red", linestyle="--")                   # zero line
plt.title("Residuals vs Predicted (Polynomial + Interaction, LOG Salary)")  
plt.xlabel("Predicted log(salary)")                           # x-axis label
plt.ylabel("Residuals (log-scale)")                           # y-axis label
plt.show()                                                    # display plot

"""
PART 4 — Model Selection using AIC/BIC (Forward + Backward Only) Based on File 1.5

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
"""

# =========================================================
# PART 4 — AIC/BIC Model Selection Using stepAIC
# Based strictly on final_lm_log predictors
# =========================================================

df["log_salary"] = np.log(df["salary_in_usd"])

import statsmodels.formula.api as smf
from itertools import combinations

# final numeric predictors (from final_lm_log)
num_vars = ["remote_ratio", "work_year_cat", "data_age"]

# final categorical predictors (from final_lm_log)
cat_vars = ["experience_level", "employment_type", "job_title"]

# categorical terms
cat_terms = " + ".join([f"C({v})" for v in cat_vars])

# numeric main effects
num_terms = " + ".join(num_vars)

# polynomial-only formula (degree 2)
poly_formula = (
    "log_salary ~ "
    + cat_terms + " + "
    + num_terms + " + "
    + " + ".join([f"I({v}**2)" for v in num_vars])
)

# interaction-only formula (numeric × numeric)
inter_formula = (
    "log_salary ~ "
    + cat_terms + " + "
    + num_terms + " + "
    + " + ".join([f"{a}:{b}" for a, b in combinations(num_vars, 2)])
)

# polynomial + interaction formula (full quadratic)
poly_inter_formula = (
    "log_salary ~ "
    + cat_terms + " + "
    + num_terms + " + "
    + " + ".join([f"I({v}**2)" for v in num_vars])
    + " + "
    + " + ".join([f"{a}:{b}" for a, b in combinations(num_vars, 2)])
)

# =========================================================
# 2. Fit initial models on df (USA-only)
# =========================================================

poly_model_full = smf.ols(poly_formula, data=df).fit()
inter_model_full = smf.ols(inter_formula, data=df).fit()
poly_inter_model_full = smf.ols(poly_inter_formula, data=df).fit()

# =========================================================
# 3. stepAIC function (forward + backward)
# =========================================================

def stepAIC(model):
    current_model = model
    current_aic = model.aic

    improved = True
    while improved:
        improved = False

        # current terms (exclude intercept)
        terms = [t for t in current_model.model.exog_names if t != "Intercept"]

        # all possible terms from full model
        full_terms = [t for t in model.model.exog_names if t != "Intercept"]

        # ---------- backward elimination ----------
        backward_candidates = []
        for term in terms:
            reduced_terms = [t for t in terms if t != term]
            formula = "log_salary ~ " + " + ".join(reduced_terms)
            try:
                m = smf.ols(formula, data=df).fit()
                backward_candidates.append((m.aic, m))
            except:
                pass

        # ---------- forward selection ----------
        forward_candidates = []
        for term in set(full_terms) - set(terms):
            new_terms = terms + [term]
            formula = "log_salary ~ " + " + ".join(new_terms)
            try:
                m = smf.ols(formula, data=df).fit()
                forward_candidates.append((m.aic, m))
            except:
                pass

        candidates = backward_candidates + forward_candidates
        if not candidates:
            break

        best_aic, best_model_candidate = min(candidates, key=lambda x: x[0])

        if best_aic < current_aic:
            current_model = best_model_candidate
            current_aic = best_aic
            improved = True

    return current_model

# =========================================================
# 4. Run stepAIC on each model
# =========================================================

best_poly_model = stepAIC(poly_model_full)
best_inter_model = stepAIC(inter_model_full)
best_poly_inter_model = stepAIC(poly_inter_model_full)

# =========================================================
# 5. Collect AIC/BIC + Adjusted R²
# =========================================================

def model_stats(model):
    return model.aic, model.bic, model.rsquared, model.rsquared_adj

poly_aic, poly_bic, poly_r2, poly_adj_r2 = model_stats(best_poly_model)
inter_aic, inter_bic, inter_r2, inter_adj_r2 = model_stats(best_inter_model)
pi_aic, pi_bic, pi_r2, pi_adj_r2 = model_stats(best_poly_inter_model)

# =========================================================
# 6. Comparison table
# =========================================================

results_aicbic = pd.DataFrame([
    {"Model": "Polynomial Only", "AIC": poly_aic, "BIC": poly_bic,
     "R²": poly_r2, "Adj_R²": poly_adj_r2},
    {"Model": "Interaction Only", "AIC": inter_aic, "BIC": inter_bic,
     "R²": inter_r2, "Adj_R²": inter_adj_r2},
    {"Model": "Polynomial + Interaction", "AIC": pi_aic, "BIC": pi_bic,
     "R²": pi_r2, "Adj_R²": pi_adj_r2}
])

print("\n===== PART 4: AIC/BIC Model Selection =====")
print(results_aicbic)

# =========================================================
# 7. Select best model by AIC
# =========================================================

best_row = results_aicbic.loc[results_aicbic["AIC"].idxmin()]
best_model_name = best_row["Model"]

print(f"\nBest Model Based on AIC: {best_model_name}")

# store best model object for Part 5
if best_model_name == "Polynomial Only":
    best_model = best_poly_model
elif best_model_name == "Interaction Only":
    best_model = best_inter_model
else:
    best_model = best_poly_inter_model

# Best Model Based on AIC: Polynomial Only
# Model           AIC           BIC             R²          Adj_R²
# Polynomial Only  43567.569925  46185.380626  0.280948  0.276037

"""
PART 5 — Final Model Fit (Based on AIC/BIC Selection)

Purpose:
    Best model determined in Part 4: 
        - Polynomial-only
        
    Model Comparison: 
        - Polynomial > (with rounding =>) Polynomial + Interaction > Interaction

    Then:
        - Refit the best model cleanly
        - Generate predictions and residuals
        - Store everything needed for Part 6 diagnostics
"""

# =========================================================
# 1. Build Design Matrix for Polynomial-only Model
# =========================================================

poly = PolynomialFeatures(degree=2, include_bias=False)
pre_poly = ColumnTransformer(
    [("poly", poly, numeric_features),
     ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)]
)

# Build matrix
X_poly = pre_poly.fit_transform(X_train)

# Convert sparse to dense BEFORE add_constant
if hasattr(X_poly, "toarray"):
    X_poly = X_poly.toarray()

# Add intercept
X_poly = sm.add_constant(X_poly)

# =========================================================
# 2. Utility: Fit OLS (LOG Response)
# =========================================================

def fit_ols(Xmat, yvec):
    return sm.OLS(yvec, Xmat).fit()

# =========================================================
# 3. Compute AIC/BIC for All Models (Optional — already done)
# =========================================================

# (You can keep or remove this section; it no longer affects the final model.)

# =========================================================
# 4. Best Model (Already Known)
# =========================================================

best_model_name = "Polynomial Only"
print("\n===== BEST MODEL SELECTED (Based on AIC) =====")
print(best_model_name)

# =========================================================
# 5. Refit Best Model Cleanly (Polynomial Only)
# =========================================================

final_preprocessor = pre_poly
X_train_final = X_poly
final_model = fit_ols(X_train_final, y_train)

# =========================================================
# 6. Prepare Test Matrix for Final Model
# =========================================================

X_test_final = final_preprocessor.transform(X_test)
X_test_final = sm.add_constant(X_test_final)

# =========================================================
# 7. Predictions + Residuals (LOG Response)
# =========================================================

y_pred_log = final_model.predict(X_test_final)
residuals_log = y_test - y_pred_log

rmse_log = np.sqrt(np.mean(residuals_log**2))
mae_log = np.mean(np.abs(residuals_log))
r2_log = final_model.rsquared

print("\n===== FINAL MODEL PERFORMANCE (LOG Response) =====")
print(f"RMSE (log): {rmse_log:,.4f}")
print(f"MAE  (log): {mae_log:,.4f}")
print(f"R²   (log): {r2_log:.4f}")

# =========================================================
# 8. Save Outputs for Part 6 Diagnostics
# =========================================================

print("\nFinal model, predictions, and residuals are ready for Part 6 diagnostics.")

"""
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
# 1. Build Polynomial-only Design Matrix
# =========================================================

poly = PolynomialFeatures(degree=2, include_bias=False)
pre_poly = ColumnTransformer(
    [("poly", poly, numeric_features),
     ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)]
)

X_poly = pre_poly.fit_transform(X_train)
X_poly = sm.add_constant(X_poly)

# =========================================================
# 2. Utility: Fit OLS (LOG Response)
# =========================================================

def fit_ols(Xmat, yvec):
    return sm.OLS(yvec, Xmat).fit()

# =========================================================
# 3. Fit Final Model (Polynomial-only)
# =========================================================

final_preprocessor = pre_poly
X_train_final = X_poly
final_model = fit_ols(X_train_final, y_train)

print("\n===== FINAL MODEL: Polynomial Only (LOG Response) =====")
print(final_model.summary())

# =========================================================
# 4. Prepare Test Matrix
# =========================================================

X_test_final = final_preprocessor.transform(X_test)
X_test_final = sm.add_constant(X_test_final)

# =========================================================
# 5. Predictions + Residuals (LOG Response)
# =========================================================

y_pred_log = final_model.predict(X_test_final)
residuals_log = y_test - y_pred_log

# =========================================================
# 6. Performance Table (LOG Response)
# =========================================================

rmse_log = np.sqrt(np.mean(residuals_log**2))
mae_log = np.mean(np.abs(residuals_log))
r2_log = final_model.rsquared

perf_df = pd.DataFrame({
    "Metric": ["RMSE (log)", "MAE (log)", "R² (log)"],
    "Value": [rmse_log, mae_log, r2_log]
})

print("\n===== PERFORMANCE METRICS (LOG Response) =====")
print(perf_df)

# =========================================================
# 7. Actual vs Predicted Plot (LOG Response)
# =========================================================

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_log, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--", linewidth=2)
plt.title("Polynomial Only: Actual vs Predicted (LOG Salary)")
plt.xlabel("Actual log(salary)")
plt.ylabel("Predicted log(salary)")
plt.show()

# =========================================================
# 8. Residual vs Fitted Plot (LOG Response)
# =========================================================

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_log, y=residuals_log, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.title("Polynomial Only: Residuals vs Predicted (LOG Salary)")
plt.xlabel("Predicted log(salary)")
plt.ylabel("Residuals (log-scale)")
plt.show()

# =========================================================
# 9. QQ Plot (Normality Check)
# =========================================================

plt.figure(figsize=(8, 6))
sm.qqplot(residuals_log, line="45", fit=True)
plt.title("QQ Plot of Residuals (LOG Response)")
plt.show()

# =========================================================
# 10. Breusch–Pagan Test (Homoscedasticity)
# =========================================================

bp_test = het_breuschpagan(residuals_log, X_test_final)

bp_labels = ["LM Statistic", "LM p-value", "F Statistic", "F p-value"]
bp_results = pd.Series(bp_test, index=bp_labels)

print("\n===== BREUSCH–PAGAN TEST =====")
print(bp_results)

# =========================================================
# 11. Coefficient Table
# =========================================================

coef_df = pd.DataFrame({
    "Coefficient": final_model.params
})

print("\n===== COEFFICIENT TABLE =====")
print(coef_df)

# =========================================================
# 12. Full Model Summary
# =========================================================

print("\n===== FINAL MODEL SUMMARY =====")
print(final_model.summary())

# =========================================================
# EXPORTS FOR LEADERBOARD (FROM PART 4 ONLY)
# =========================================================

inter_pred = best_inter_model.predict(df)
inter_rmse = np.sqrt(np.mean((df["log_salary"] - inter_pred)**2))
inter_mae = np.mean(np.abs(df["log_salary"] - inter_pred))

pi_pred = best_poly_inter_model.predict(df)
pi_rmse = np.sqrt(np.mean((df["log_salary"] - pi_pred)**2))
pi_mae = np.mean(np.abs(df["log_salary"] - pi_pred))


# ---------- Polynomial-only ----------
poly_pred = best_poly_model.predict(df)
poly_rmse = np.sqrt(np.mean((df["log_salary"] - poly_pred)**2))
poly_mae = np.mean(np.abs(df["log_salary"] - poly_pred))

best_poly_rmse = poly_rmse
best_poly_mae = poly_mae
best_poly_r2 = poly_r2
best_poly_adj_r2 = poly_adj_r2

# ---------- Interaction-only ----------
inter_pred = best_inter_model.predict(df)
inter_rmse = np.sqrt(np.mean((df["log_salary"] - inter_pred)**2))
inter_mae = np.mean(np.abs(df["log_salary"] - inter_pred))

best_inter_rmse = inter_rmse
best_inter_mae = inter_mae
best_inter_r2 = inter_r2
best_inter_adj_r2 = inter_adj_r2

# ---------- Polynomial + Interaction ----------
pi_pred = best_poly_inter_model.predict(df)
pi_rmse = np.sqrt(np.mean((df["log_salary"] - pi_pred)**2))
pi_mae = np.mean(np.abs(df["log_salary"] - pi_pred))

best_poly_inter_rmse = pi_rmse
best_poly_inter_mae = pi_mae
best_poly_inter_r2 = pi_r2
best_poly_inter_adj_r2 = pi_adj_r2
