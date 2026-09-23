# -*- coding: utf-8 -*-

# =========================================================
'''
Author:     Aaron Niecestro
Project:    DS Salaries – Production-Ready EDA for Regression Models

Created:    September 22 ,2026
Last Edit:  September 23, 2026
Progress:   Completed

Description - Linear Regression Pipeline (Multi‑Part):

Purpose:
    End‑to‑end salary prediction workflow using standard linear regression.
    Includes data loading, preprocessing, train/test split, model fitting,
    evaluation, diagnostic plots, and assumption checks.

What it covers:
    • USA‑only dataset filtering
    • R‑style summary + structure overview
    • Predictor/response setup (x_pred, y_resp)
    • Scaling + one‑hot encoding via ColumnTransformer
    • Linear regression model pipeline
    • Performance metrics (R², RMSE, MAE)
    • Residual, linearity, variance, and BP tests

Use:
    Run as a full regression analysis script. 
    Each section is labeled and modular for clarity.
'''

# =========================================================

# --------------------------------------------------------------------------
# Linear Regression - DSS2025
# --------------------------------------------------------------------------

# =========================================================
# 1. Import Packages
# =========================================================

import pandas as pd                 # data manipulation
import numpy as np                  # numerical operations
import seaborn as sns               # visualization
import matplotlib.pyplot as plt     # plotting

from sklearn.linear_model import LinearRegression          # linear regression model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # model metrics
from sklearn.model_selection import train_test_split       # train/test split
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # scaling + encoding
from sklearn.compose import ColumnTransformer              # preprocessing transformer
from sklearn.pipeline import Pipeline                      # ML pipeline

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set(style="whitegrid", context="talk")                 # sets seaborn visual style

# =========================================================
# 2. Load Data
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
dss2025 = pd.read_csv(file_path)                           # loads dataset from CSV

print("----- DATA SHAPE (MASTER) -----")
print(f"Rows: {dss2025.shape[0]}, Columns: {dss2025.shape[1]}\n")  # prints dataset size

# =========================================================
# 3. USA-Only Dataset
# =========================================================

usa_df = dss2025.query("company_location == 'United States'").copy()  # filters to USA rows

print("----- USA DATASET -----")
print(f"Rows: {usa_df.shape[0]}, Columns: {usa_df.shape[1]}\n")       # prints USA dataset size

dss2025 = usa_df                                                     # replaces main dataset with USA-only

# =========================================================
# 4. Column Type Setup
# =========================================================

numeric_cols = dss2025.select_dtypes(include=["int64", "float64"]).columns.tolist()  # numeric columns
categorical_cols = dss2025.select_dtypes(include=["object", "category"]).columns.tolist()  # categorical columns

print("----- NUMERIC COLUMNS -----")
print(numeric_cols)

print("\n----- CATEGORICAL COLUMNS -----")
print(categorical_cols)

print("\n----- ALL COLUMNS -----")
print(dss2025.columns.tolist())                                      # prints all column names

# =========================================================
# 5. R-style Summary Function
# =========================================================

def summary_r(df):                                         # defines R-style summary function
    print("----- R-style Summary -----")                   # header for summary
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")# prints dataset dimensions

    for col in df.columns:                                 # loops through each column
        print(f"--- {col} ---")                            # prints column name
        if pd.api.types.is_numeric_dtype(df[col]):         # checks if column is numeric
            desc = df[col].describe()                      # gets numeric summary stats
            print(f"Min:      {desc['min']}")              # prints minimum
            print(f"1st Qu.:  {desc['25%']}")              # prints 25th percentile
            print(f"Median:   {desc['50%']}")              # prints median
            print(f"Mean:     {desc['mean']}")             # prints mean
            print(f"3rd Qu.:  {desc['75%']}")              # prints 75th percentile
            print(f"Max:      {desc['max']}")              # prints maximum
        else:
            desc = df[col].describe()                      # gets categorical summary
            print("Type:     Categorical")                 # marks column as categorical
            print(f"Unique:   {desc['unique']}")           # number of unique categories
            print(f"Top:      {desc['top']}")              # most frequent category
            print(f"Freq:     {desc['freq']}")             # frequency of top category
        print()                                            # blank line for readability

summary_r(dss2025)                                         # runs R-style summary on dataset

print("\n----- SUMMARY (R-style table) -----")
print(dss2025.describe(include="all"))                     # full summary including categorical

print("\n----- NUMERIC SUMMARY -----")
print(dss2025.describe())                                  # numeric-only summary

print("\n----- CATEGORICAL SUMMARY -----")
print(dss2025.describe(include=["category", "object"]))    # categorical-only summary

# =========================================================
# 6. Define Response and Predictor Variables (LOG RESPONSE)
# =========================================================

y_resp = dss2025["salary_in_usd"]                          # original salary in USD
log_y_resp = np.log(y_resp)                                # log-transformed salary

x_pred = dss2025[[
    'experience_level', 'employment_type', 'job_title',
    'employee_residence', 'remote_ratio',
    'company_location', 'company_size', 'data_age',
    'work_year_cat', 'remote_work_cat', 'job_title_group'
]]                                          # same predictors as before

print("----- TARGET VARIABLE (log(y_resp)) -----")
print(y_resp.head())                             # show first few log-salary values

# =========================================================
# 7. Check Missing Values
# =========================================================

missing_values = dss2025.isnull().sum()                     # counts missing values per column

print("\n----- MISSING VALUES -----")
print(missing_values[missing_values > 0]                    # prints only columns with missing values
      .sort_values(ascending=False))                        # sorts missing values from most to least

# =========================================================
# 8. Distribution of salary_in_usd
# =========================================================

plt.figure(figsize=(10, 6))                                 # sets figure size

sns.histplot(y_resp, kde=True)                              # plots histogram + KDE of salary

plt.title("Distribution of Salary in USD")                  # chart title
plt.xlabel("Salary in USD")                                 # x-axis label
plt.ylabel("Count")                                         # y-axis label

plt.show()                                                  # displays plot

# =========================================================
# 9. Create X and y (LOG RESPONSE)
# =========================================================

X = x_pred.copy()                                          # predictor matrix
y = y_resp.copy()                                          # original salary response

print("\n----- X AND y SHAPES -----")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# =========================================================
# 10. Identify Numeric and Categorical Predictors
# =========================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()                                          # numeric predictor columns

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()                                          # categorical predictor columns

print("\n----- NUMERIC PREDICTORS -----")
print(numeric_features)                                     # prints numeric predictor list

print("\n----- CATEGORICAL PREDICTORS -----")
print(categorical_features)                                 # prints categorical predictor list

# =========================================================
# 11. Train/Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=72018
)                                                          # splits data into 70% train / 30% test

print("\n----- TRAIN / TEST SPLIT -----")
print(f"X_train: {X_train.shape}")                         # prints training predictor shape
print(f"X_test:  {X_test.shape}")                          # prints testing predictor shape
print(f"y_train: {y_train.shape}")                         # prints training response size
print(f"y_test:  {y_test.shape}")                          # prints testing response size

# =========================================================
# 12. Preprocessing (Scaling + One-Hot Encoding)
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),    # scales numeric predictors
        ("categorical",
         OneHotEncoder(handle_unknown="ignore", drop="first"),
         categorical_features)                              # encodes categorical predictors
    ]
)

# =========================================================
# 13. Linear Regression Pipeline
# =========================================================

linear_regression = Pipeline(
    steps=[
        ("preprocessor", preprocessor),                     # applies preprocessing
        ("regressor", LinearRegression())                   # fits linear regression model
    ]
)

# =========================================================
# 14. Fit Model (LOG-LINEAR REGRESSION)
# =========================================================

linear_regression.fit(X_train, y_train)            # fit pipeline using log-salary

print("\n----- MODEL FITTED (LOG RESPONSE) -----")
print("Log-linear regression model successfully fitted.")  # confirmation

# =========================================================
# 15. Predict on Test Data (LOG SCALE)
# =========================================================

y_pred_log = linear_regression.predict(X_test)          # predictions in log(salary)

# OPTIONAL: back-transform only for inspection (with clipping to avoid overflow)
y_pred = np.exp(np.clip(y_pred_log, a_min=None, a_max=20))  # cap log-salary at 20 (~$4.85e8)

# =========================================================
# 16. Model Evaluation (BACK-TRANSFORMED)
# =========================================================

log_y_test = np.log(y_test)                             # log-transform true salaries

r2 = r2_score(log_y_test, y_pred_log)                   # R² on log scale
rmse = np.sqrt(mean_squared_error(log_y_test, y_pred_log))  # RMSE on log scale
mae = mean_absolute_error(log_y_test, y_pred_log)       # MAE on log scale

print("\n----- LOG-LINEAR REGRESSION RESULTS (LOG SCALE) -----")
print("\n----- LOG-LINEAR REGRESSION RESULTS -----")
print(f"R^2:   {r2:.4f}")                          # print R²
print(f"RMSE: ${rmse:,.2f}")                       # print RMSE
print(f"MAE:  ${mae:,.2f}")                        # print MAE

# =========================================================
# 17. Actual vs Predicted Salary Plot
# =========================================================

plt.figure(figsize=(10, 7))                                   # sets figure size

sns.scatterplot(x=y_test, y=y_pred)                           # scatterplot of actual vs predicted

min_salary = min(y_test.min(), y_pred.min())                  # minimum value for diagonal line
max_salary = max(y_test.max(), y_pred.max())                  # maximum value for diagonal line

plt.plot([min_salary, max_salary],                            # x-coordinates of perfect line
         [min_salary, max_salary],                            # y-coordinates of perfect line
         color="red", linestyle="--", label="Perfect Prediction")  # draws perfect prediction line

plt.title("Actual vs Predicted Salary")                       # chart title
plt.xlabel("Actual Salary in USD")                            # x-axis label
plt.ylabel("Predicted Salary in USD")                         # y-axis label
plt.legend()                                                  # shows legend

plt.show()                                                    # displays plot

# =========================================================
# 18. Residuals (LOG MODEL)
# =========================================================

residuals = y_test - y_pred                        # residuals in original salary scale

print("\n----- RESIDUAL SUMMARY (LOG MODEL) -----")
print(pd.Series(residuals).describe())             # summary of residuals

# =========================================================
# 19. Linearity Check (Residuals vs Predicted)
# =========================================================

plt.figure(figsize=(10, 6))                                   # sets figure size

sns.residplot(x=y_pred, y=residuals, lowess=True,             # residual plot with LOWESS smoothing
              line_kws={"color": "red", "linewidth": 2})      # styling for LOWESS line

plt.axhline(y=0, color="black", linestyle="--")               # horizontal zero line

plt.title("Linearity Check: Residuals vs Predicted Salary")   # chart title
plt.xlabel("Predicted Salary in USD")                         # x-axis label
plt.ylabel("Residuals")                                       # y-axis label

plt.show()                                                    # displays plot

# =========================================================
# 20. Constant Variance (Homoscedasticity) Check
# =========================================================

plt.figure(figsize=(10, 6))                                   # sets figure size

sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)             # scatterplot of residuals vs predicted

plt.axhline(y=0, color="red", linestyle="--")                 # horizontal zero line

plt.title("Constant Variance Check")                          # chart title
plt.xlabel("Predicted Salary in USD")                         # x-axis label
plt.ylabel("Residuals")                                       # y-axis label

plt.show()                                                    # displays plot

# =========================================================
# 21. Breusch-Pagan Test (LOG MODEL, LOG-SCALE RESIDUALS)
# =========================================================

# Transform predictors using preprocessing
X_test_transformed = linear_regression.named_steps["preprocessor"].transform(X_test)

# Convert sparse to dense if needed
if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

# Add intercept
X_test_bp = sm.add_constant(X_test_transformed)

# Compute LOG-SCALE residuals (correct for log-linear model)
log_y_test = np.log(y_test)                 # true log-salary
residuals_log = log_y_test - y_pred_log     # residuals in log space

# Convert to numpy array
residuals_bp = np.array(residuals_log)

# Run Breusch-Pagan test on LOG residuals
bp_test = het_breuschpagan(residuals_bp, X_test_bp)

# Label results
bp_results = pd.Series(bp_test, index=[
    "LM Statistic", "LM-Test p-value", "F Statistic", "F-Test p-value"
])

print("\n----- BREUSCH-PAGAN TEST (LOG MODEL, LOG SCALE) -----")
print(bp_results)


# =========================================================
# 22. Prepare Data for AIC/BIC Model Selection (LOG RESPONSE)
# =========================================================

# Transform full predictor matrix using fitted preprocessing
X_transformed = linear_regression.named_steps["preprocessor"].fit_transform(X)

# Convert sparse matrix to dense if needed
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

# Add intercept column
X_transformed = sm.add_constant(X_transformed)

# Use LOG-TRANSFORMED response for AIC/BIC
y_array = np.array(log_y_resp)        # <-- FIXED: must use log response

# =========================================================
# 23. Utility Function: Fit OLS Model (LOG RESPONSE)
# =========================================================

def fit_ols(Xmat, yvec):
    """
    Fits an OLS model using statsmodels on a numeric matrix.
    Used ONLY for the final reduced predictor matrix.
    """
    return sm.OLS(yvec, Xmat).fit()

# =========================================================
# 24. Stepwise Selection (AIC/BIC) — R-style
# =========================================================

import statsmodels.formula.api as smf
import numpy as np

def stepAIC(formula, data, direction="both", criterion="AIC"):
    """
    Performs R-style stepwise AIC/BIC model selection using statsmodels formula API.
    - Works with categorical variables (factors) exactly like R.
    - Avoids one-hot encoding explosion.
    - Supports forward, backward, and both directions.
    """

    # Split formula into response and predictor list
    response, predictors = formula.split("~")
    response = response.strip()
    predictors = [p.strip() for p in predictors.split("+")]

    # Fit a model given a list of predictors
    def fit_model(pred_list):
        f = response + " ~ " + " + ".join(pred_list)
        model = smf.ols(f, data=data).fit()
        score = model.aic if criterion == "AIC" else model.bic
        return model, score

    # Initialize predictor set depending on direction
    if direction == "backward":
        current_predictors = predictors.copy()   # start with full model
    elif direction == "forward":
        current_predictors = []                  # start with empty model
    else:  # both directions
        current_predictors = predictors.copy()

    # Fit initial model
    best_model, best_score = fit_model(current_predictors)
    improved = True

    # Stepwise loop
    while improved:
        improved = False
        candidate_models = []

        # BACKWARD elimination
        if direction in ["backward", "both"] and len(current_predictors) > 1:
            for p in current_predictors:
                new_preds = [x for x in current_predictors if x != p]
                model, score = fit_model(new_preds)
                candidate_models.append((model, score, new_preds))

        # FORWARD selection
        if direction in ["forward", "both"]:
            remaining = [p for p in predictors if p not in current_predictors]
            for p in remaining:
                new_preds = current_predictors + [p]
                model, score = fit_model(new_preds)
                candidate_models.append((model, score, new_preds))

        # Evaluate all candidate models
        for model, score, pred_list in candidate_models:
            if score < best_score:              # AIC/BIC improvement
                best_model = model
                best_score = score
                current_predictors = pred_list
                improved = True

    return best_model

# =========================================================
# Run stepAIC on LOG-LINEAR MODEL
# =========================================================

formula_log = """
np.log(salary_in_usd) ~ experience_level + employment_type + job_title +
employee_residence + remote_ratio + company_location + company_size +
data_age + work_year_cat + remote_work_cat + job_title_group
"""

best_log = stepAIC(formula_log, dss2025, direction="both", criterion="AIC")

print(best_log.summary())
print("AIC:", best_log.aic)
print("BIC:", best_log.bic)

# =========================================================
# 25. Fit Final Model (LOG RESPONSE) — Matrix Version
# =========================================================

# =========================================================
# 25.1. Fit Final Log-Linear Model
# =========================================================

final_lm_log = """
np.log(salary_in_usd) ~ experience_level + employment_type + job_title +
remote_ratio + work_year_cat
"""

final_model = smf.ols(final_lm_log, data=dss2025).fit()
print(final_model.summary())

# =========================================================
# 25.2. Basic Metrics: R², Adjusted R², RMSE, MAE, MAPE
# =========================================================

y_true = np.log(dss2025["salary_in_usd"])
y_pred = final_model.fittedvalues
residuals = final_model.resid

r2 = final_model.rsquared
adj_r2 = final_model.rsquared_adj
rmse = np.sqrt(np.mean((y_true - y_pred)**2))
mae = np.mean(np.abs(y_true - y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("\n=== MODEL PERFORMANCE ===")
print("R²:", r2)
print("Adjusted R²:", adj_r2)
print("RMSE:", rmse)
print("MAE:", mae)
print("MAPE (%):", mape)

# =========================================================
# 25.3. Constant Variance (Breusch-Pagan Test)
# =========================================================

bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, final_model.model.exog)
print("\n=== BREUSCH-PAGAN TEST ===")
print("BP p-value:", bp_pvalue)

# =========================================================
# 25.4. Multicollinearity (VIF)
# =========================================================

X = final_model.model.exog
vif_df = pd.DataFrame({
    "variable": final_model.model.exog_names,
    "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])]
})

print("\n=== VARIANCE INFLATION FACTORS (VIF) ===")
print(vif_df)

# =========================================================
# 25.5. Normality of Residuals (QQ Plot)
# =========================================================

sm.qqplot(residuals, line='45')
plt.title("QQ Plot of Residuals")
plt.show()

# =========================================================
# 6. Linearity Check (Residuals vs Fitted)
# =========================================================

plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(0, color='red')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.show()

# =========================================================
# 25.7. Influential Points (Cook's Distance)
# =========================================================

influence = final_model.get_influence()
cooks = influence.cooks_distance[0]

plt.stem(cooks, markerfmt=",")
plt.title("Cook's Distance")
plt.show()

# =========================================================
# Export Best Metrics for Linear Regression (Final Log Model)
# =========================================================

best_ols_rmse = rmse
best_ols_mae  = mae
best_ols_r2   = r2
best_ols_adj_r2 = adj_r2   # <-- NEW: Adjusted R² added

__all__ = [
    "best_ols_rmse",
    "best_ols_mae",
    "best_ols_r2",
    "best_ols_adj_r2"       
]

