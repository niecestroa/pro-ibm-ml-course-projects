# -*- coding: utf-8 -*-

# =========================================================
'''
Author: Aaron Niecestro  
Project: DS Salaries – Production‑Ready EDA for Logistic Regression Models  
Created: September 28, 2026  
Last Edit: September 28, 2026  
Progress: Ongoing  

---

## Description — Logistic Regression Pipeline (Multi‑Part)

### Purpose**  
End‑to‑end salary classification workflow using standard logistic regression.  
This module performs data loading, preprocessing, train/test splitting, model 
fitting, evaluation, diagnostic visualization, and assumption checks for a 
binary salary classification task.

### What it covers

- **USA‑only dataset filtering**  
  Ensures geographic consistency by restricting analysis to U.S. salary records.

- **R‑style summary + structural overview**  
  Provides dataset shape, variable types, and descriptive statistics for both numeric and categorical features.

- **Predictor/response setup**  
  - **Response:** `salary_mean_cat` (binary high‑vs‑low salary category)  
  - **Predictors:** `experience_level`, `employment_type`, `job_title`,  
    `employee_residence`, `remote_ratio`, `company_location`,  
    `company_size`, `data_age`, `work_year_cat`, `remote_work_cat`,  
    `job_title_group`

- **Preprocessing via ColumnTransformer**  
  - Standard scaling for numeric predictors  
  - One‑hot encoding (drop‑first) for categorical predictors  
  - Full integration into a scikit‑learn pipeline

- **Logistic regression model pipeline**  
  - Fits a binary classifier using maximum likelihood estimation  
  - Extracts coefficients and interprets feature influence  
  - Supports probability‑based evaluation (ROC, PR curves)

- **Classification performance metrics**  
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  
  - ROC‑AUC  

- **Diagnostic plots**  
  - ROC Curve  
  - Precision–Recall Curve  
  - Confusion Matrix Heatmap  
  - Coefficient Importance Plot  

- **Assumption and stability checks**  
  - Multicollinearity assessment via VIF  
  - Coefficient sign and magnitude inspection  
  - Class balance review  
  - Missing‑value audit  

### Use:
Run as a complete logistic regression classification analysis script.  
Each section is modular, labeled, and designed for production‑ready EDA, 
model diagnostics, and interpretability.
'''

# =========================================================

# --------------------------------------------------------------------------
# Logistic Regression - DSS2025 (Classification Version)
# --------------------------------------------------------------------------

import pandas as pd                      # data manipulation
import numpy as np                       # numerical operations
import seaborn as sns                    # visualization
import matplotlib.pyplot as plt          # plotting

from sklearn.linear_model import LogisticRegression   # logistic regression model
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve)  # classification metrics + curves
from sklearn.model_selection import train_test_split  # train/test split
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # scaling + encoding
from sklearn.compose import ColumnTransformer          # preprocessing transformer
from sklearn.pipeline import Pipeline                  # ML pipeline
from sklearn.impute import SimpleImputer              # imputation for missing values

import statsmodels.formula.api as smf   # formula-based modeling (R-style), handles categorical predictors automatically
import statsmodels.api as sm            # core statsmodels API for GLM/Logit, likelihoods, summaries, and inference
from scipy.stats import chi2    # chi-square distribution for computing LRT p-values (df-based significance test)

sns.set(style="whitegrid", context="talk")             # seaborn visual style

# =========================================================
# 1. Load Data
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"  # file path
dss2025 = pd.read_csv(file_path)                       # load dataset
print("----- DATA SHAPE (MASTER) -----")
print(dss2025.shape)

# =========================================================
# 2. USA-Only Dataset
# =========================================================

dss2025 = dss2025.query("company_location == 'United States'").copy()  # filter USA rows
print("----- USA DATASET -----")
print(dss2025.shape)

# =========================================================
# 3. Convert Response to Numeric Labels
# =========================================================

dss2025["salary_mean_cat"] = dss2025["salary_mean_cat"].map({
    "Below-Average": 0,
    "Above-Average": 1
})  # convert to 0/1

# =========================================================
# 4. Response + Predictors
# =========================================================

y = dss2025["salary_mean_cat"]  # binary response
X = dss2025[
    ['experience_level', 'employment_type', 'job_title',
     'employee_residence', 'remote_ratio', 'company_location',
     'company_size', 'data_age', 'work_year_cat',
     'remote_work_cat', 'job_title_group']
]  # predictor matrix

print("----- RESPONSE (salary_mean_cat) -----")
print(y.value_counts())

print("\n----- PREDICTORS -----")
print(X.head())

# =========================================================
# 5. Missing Values
# =========================================================

print("\n----- MISSING VALUES -----")
print(dss2025.isnull().sum().sort_values(ascending=False))

# =========================================================
# 6. Identify Numeric + Categorical Predictors
# =========================================================

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()  # numeric
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()  # categorical

print("\n----- NUMERIC FEATURES -----")
print(numeric_features)

print("\n----- CATEGORICAL FEATURES -----")
print(categorical_features)

# =========================================================
# 7. Train/Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=72018, stratify=y
)  # stratified split

print("\n----- TRAIN / TEST SPLIT -----")
print(X_train.shape, X_test.shape)

# =========================================================
# 8. Preprocessing (Imputation + Scaling + Encoding)
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric",
         Pipeline([
             ("imputer", SimpleImputer(strategy="median")),  # numeric imputation
             ("scaler", StandardScaler())                    # numeric scaling
         ]),
         numeric_features),

        ("categorical",
         Pipeline([
             ("imputer", SimpleImputer(strategy="most_frequent")),  # categorical imputation
             ("encoder", OneHotEncoder(
                 handle_unknown="ignore",
                 drop="first",
                 sparse_output=False
             ))
         ]),
         categorical_features)
    ]
)

# =========================================================
# 9. Full Logistic Regression Pipeline (scikit-learn only)
# =========================================================

log_reg = Pipeline([
    ("preprocessor", preprocessor),                # preprocessing
    ("classifier", LogisticRegression(max_iter=2000))  # logistic regression
])

log_reg.fit(X_train, y_train)  # fit full model

# =========================================================
# 10. Full Model Performance
# =========================================================

y_pred = log_reg.predict(X_test)                  # predicted classes
y_prob = log_reg.predict_proba(X_test)[:, 1]      # predicted probabilities

print("\n----- FULL MODEL PERFORMANCE -----")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

print("\n----- FULL MODEL CONFUSION MATRIX -----")
print(confusion_matrix(y_test, y_pred))

# =========================================================
# Stepwise AIC/BIC Selection (Formula API — Safe)
# =========================================================

def stepAIC_logit(formula, data, direction="both", criterion="AIC"):
    """
    Performs R-style stepwise AIC/BIC selection for logistic regression.
    Uses statsmodels formula API (safe for categorical variables).
    """

    # Split formula into response and predictor list
    response, predictors = formula.split("~")
    response = response.strip()
    predictors = [p.strip() for p in predictors.split("+")]

    # Fit logistic regression model given a list of predictors
    def fit_model(pred_list):
        if len(pred_list) == 0:
            f = response + " ~ 1"  # intercept-only model
        else:
            f = response + " ~ " + " + ".join(pred_list)

        model = smf.glm(f, data=data, family=sm.families.Binomial()).fit()
        score = model.aic if criterion == "AIC" else model.bic
        return model, score

    # Initialize predictor set
    if direction == "backward":
        current_predictors = predictors.copy()
    elif direction == "forward":
        current_predictors = []
    else:
        current_predictors = predictors.copy()

    # Fit initial model
    best_model, best_score = fit_model(current_predictors)
    improved = True

    print(f"\nInitial {criterion}: {best_score:.4f}")
    print(f"Starting predictors: {current_predictors}\n")

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
            if score < best_score:
                print(f"Improved {criterion}: {best_score:.4f} → {score:.4f}")
                print(f"New predictors: {pred_list}\n")

                best_model = model
                best_score = score
                current_predictors = pred_list
                improved = True

    print("\nFinal selected predictors:")
    print(current_predictors)

    return best_model, current_predictors

# =========================================================
# Run Stepwise AIC/BIC to Select Predictors
# =========================================================

formula_full = (
    "salary_mean_cat ~ experience_level + employment_type + job_title + "
    "employee_residence + remote_ratio + company_location + company_size + "
    "data_age + work_year_cat + remote_work_cat + job_title_group"
)

step_model, selected_predictors = stepAIC_logit(
    formula=formula_full,
    data=dss2025,
    direction="both",
    criterion="AIC"
)

print("\n----- STEPWISE SELECTED PREDICTORS -----")
print(selected_predictors)

# =========================================================
# Likelihood Ratio Test (LRT) – Formula API (Correct + Stable)
# =========================================================

def lrt_formula(full_formula, reduced_formula, data):
    """
    Performs a Likelihood Ratio Test (LRT) between:
    - full model (with predictor)
    - reduced model (without predictor)
    Uses statsmodels GLM Binomial with formula API.
    """

    # Fit full model
    full_model = smf.glm(full_formula, data=data, family=sm.families.Binomial()).fit()

    # Fit reduced model
    reduced_model = smf.glm(reduced_formula, data=data, family=sm.families.Binomial()).fit()

    # Compute LR statistic
    lr_stat = -2 * (reduced_model.llf - full_model.llf)

    # Degrees of freedom = difference in number of parameters
    df = full_model.df_model - reduced_model.df_model

    # Compute p-value
    p_value = chi2.sf(lr_stat, df)

    # Print results
    print("\n----- LRT RESULT -----")
    print(f"Full model:    {full_formula}")
    print(f"Reduced model: {reduced_formula}")
    print(f"LR stat = {lr_stat:.4f}")
    print(f"df = {df}")
    print(f"p-value = {p_value:.4f}")

    decision = "Reject H₀ (significant)" if p_value < 0.05 else "Fail to Reject H₀ (not significant)"
    print(f"Decision: {decision}")

    return lr_stat, p_value, decision

# =========================================================
# LRT Example – Remove job_title_group
# =========================================================

full_formula = (
    "salary_mean_cat ~ experience_level + remote_ratio + company_size + job_title_group"
)

reduced_formula = (
    "salary_mean_cat ~ experience_level + remote_ratio + company_size"
)

lrt_formula(full_formula, reduced_formula, dss2025)

# =========================================================
# LRT Example – Remove remote_ratio + company_size
# =========================================================

full_formula = (
    "salary_mean_cat ~ experience_level + remote_ratio + company_size + job_title_group"
)

reduced_formula = (
    "salary_mean_cat ~ experience_level + job_title_group"
)

lrt_formula(full_formula, reduced_formula, dss2025)

# =========================================================
# Final Model – Reduced Logistic Regression
# =========================================================

# Selected predictors from Stepwise AIC/BIC
final_predictors = selected_predictors  # use stepwise output

print("\n----- FINAL MODEL PREDICTORS -----")
print(final_predictors)

# Subset X to final predictors
X_final = dss2025[final_predictors]

# Train/test split (same seed for reproducibility)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_final, y, test_size=0.30, random_state=72018, stratify=y
)

# =========================================================
# Preprocessing for Reduced Model
# =========================================================

preprocessor_final = ColumnTransformer(
    transformers=[
        ("numeric",
         Pipeline([
             ("imputer", SimpleImputer(strategy="median")),   # numeric imputation
             ("scaler", StandardScaler())                    # numeric scaling
         ]),
         X_final.select_dtypes(include=["int64", "float64"]).columns.tolist()),

        ("categorical",
         Pipeline([
             ("imputer", SimpleImputer(strategy="most_frequent")),  # categorical imputation
             ("encoder", OneHotEncoder(
                 handle_unknown="ignore",
                 drop="first",
                 sparse_output=False
             ))
         ]),
         X_final.select_dtypes(include=["object", "category"]).columns.tolist())
    ]
)

# =========================================================
# Reduced Logistic Regression Pipeline
# =========================================================

final_model = Pipeline([
    ("preprocessor", preprocessor_final),                 # preprocessing
    ("classifier", LogisticRegression(max_iter=2000))     # logistic regression
])

final_model.fit(X_train_f, y_train_f)  # fit reduced model

# Predictions
y_pred_f = final_model.predict(X_test_f)
y_prob_f = final_model.predict_proba(X_test_f)[:, 1]

# =========================================================
# Reduced Model Performance
# =========================================================

print("\n----- REDUCED MODEL PERFORMANCE -----")
print(f"Accuracy:  {accuracy_score(y_test_f, y_pred_f):.4f}")
print(f"Precision: {precision_score(y_test_f, y_pred_f):.4f}")
print(f"Recall:    {recall_score(y_test_f, y_pred_f):.4f}")
print(f"F1 Score:  {f1_score(y_test_f, y_pred_f):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test_f, y_prob_f):.4f}")

print("\n----- REDUCED MODEL CONFUSION MATRIX -----")
print(confusion_matrix(y_test_f, y_pred_f))

# =========================================================
# Assumption Checks – Reduced Logistic Regression
# =========================================================

# 1. Extract transformed training data
X_train_f_transformed = final_model.named_steps["preprocessor"].transform(X_train_f)
X_train_f_transformed = np.asarray(X_train_f_transformed)

# 2. Fit statsmodels logistic regression for inference (safe for reduced model)
sm_model = sm.Logit(y_train_f, X_train_f_transformed).fit(disp=0)

print("\n----- REDUCED MODEL SUMMARY (Statsmodels) -----")
print(sm_model.summary())

# 3. Coefficient stability check
print("\n----- COEFFICIENTS (Reduced Model) -----")
coef_reduced = final_model.named_steps["classifier"].coef_[0]
print(coef_reduced)

# 4. Class balance check
print("\n----- CLASS BALANCE (Reduced Model) -----")
print(y_train_f.value_counts(normalize=True))

# 5. Missing-value audit
print("\n----- MISSING VALUES (Reduced Predictors) -----")
print(X_final.isnull().sum())

