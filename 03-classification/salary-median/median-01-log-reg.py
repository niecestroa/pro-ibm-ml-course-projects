# -*- coding: utf-8 -*-

# =========================================================
'''
**Author:** Aaron Niecestro  
**Project:** DS Salaries – Production‑Ready EDA for Logistic Regression Models  
**Created:** September 28, 2026  
**Last Edit:** September 28, 2026  
**Progress:** Paused (To be completed after mean salary classification models are completd)

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
  - **Response:** `salary_median_cat` (binary high‑vs‑low salary category)  
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

"""
Logistic Regression Pipeline – Median Salary Classification
Converted from salary_mean_cat → salary_median_cat
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import chi2

sns.set(style="whitegrid", context="talk")

# =========================================================
# 1. Load Data
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
dss2025 = pd.read_csv(file_path)
print("----- DATA SHAPE (MASTER) -----")
print(dss2025.shape)

# =========================================================
# 2. USA-Only Dataset
# =========================================================

dss2025 = dss2025.query("company_location == 'United States'").copy()
print("----- USA DATASET -----")
print(dss2025.shape)

# =========================================================
# 3. Convert Response to Numeric Labels (MEDIAN)
# =========================================================

dss2025["salary_median_cat"] = dss2025["salary_median_cat"].map({
    "Below-Median": 0,
    "Above-Median": 1
})

# =========================================================
# 4. Response + Predictors
# =========================================================

y = dss2025["salary_median_cat"]   # <<<<<< CHANGED HERE

X = dss2025[
    ['experience_level', 'employment_type','employee_residence',
     'remote_ratio', 'company_location', 'company_size', 'data_age',
     'work_year_cat', 'remote_work_cat', 'job_title_group']
]

print("----- RESPONSE (salary_median_cat) -----")
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

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("\n----- NUMERIC FEATURES -----")
print(numeric_features)

print("\n----- CATEGORICAL FEATURES -----")
print(categorical_features)

# =========================================================
# 7. Train/Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=72018, stratify=y
)

print("\n----- TRAIN / TEST SPLIT -----")
print(X_train.shape, X_test.shape)

# =========================================================
# 8. Preprocessing (Imputation + Scaling + Encoding)
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric",
         Pipeline([
             ("imputer", SimpleImputer(strategy="median")),
             ("scaler", StandardScaler())
         ]),
         numeric_features),

        ("categorical",
         Pipeline([
             ("imputer", SimpleImputer(strategy="most_frequent")),
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
# 9. Full Logistic Regression Pipeline
# =========================================================

log_reg = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

log_reg.fit(X_train, y_train)

# =========================================================
# 10. Full Model Performance
# =========================================================

y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]

print("\n----- FULL MODEL PERFORMANCE (Median Salary) -----")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

print("\n----- FULL MODEL CONFUSION MATRIX -----")
print(confusion_matrix(y_test, y_pred))

# =========================================================
# Likelihood Ratio Test (LRT) – Formula API (MEDIAN)
# =========================================================

def lrt_formula(full_formula, reduced_formula, data):
    full_model = smf.glm(full_formula, data=data, family=sm.families.Binomial()).fit()
    reduced_model = smf.glm(reduced_formula, data=data, family=sm.families.Binomial()).fit()
    lr_stat = -2 * (reduced_model.llf - full_model.llf)
    df = full_model.df_model - reduced_model.df_model
    p_value = chi2.sf(lr_stat, df)

    print("\n----- LRT RESULT -----")
    print(f"Full model:    {full_formula}")
    print(f"Reduced model: {reduced_formula}")
    print(f"LR stat = {lr_stat:.4f}")
    print(f"df = {df}")
    print(f"p-value = {p_value:.4f}")
    print("Decision:", "Reject H0 (significant)" if p_value < 0.05 else "Fail to Reject H0 (not significant)")

    return lr_stat, p_value

# =========================================================
# LRT Example – Remove job_title_group (MEDIAN)
# =========================================================

full_formula = (
    "salary_median_cat ~ experience_level + employment_type + job_title_group + "
    "employee_residence + remote_ratio + company_location + company_size + "
    "data_age + work_year_cat + remote_work_cat"
)

reduced_formula = (
    "salary_median_cat ~ experience_level + employment_type + employee_residence + "
    "remote_ratio + company_location + company_size + data_age + work_year_cat + remote_work_cat"
)

lrt_formula(full_formula, reduced_formula, dss2025)

# =========================================================
# LRT Example – Remove remote_ratio + company_size (MEDIAN)
# =========================================================

full_formula = (
    "salary_median_cat ~ experience_level + employment_type + job_title_group + "
    "employee_residence + remote_ratio + company_location + company_size + "
    "data_age + work_year_cat + remote_work_cat"
)

reduced_formula = (
    "salary_median_cat ~ experience_level + employment_type + job_title_group + "
    "employee_residence + remote_ratio + company_location + company_size + "
    "work_year_cat + remote_work_cat"
)

lrt_formula(full_formula, reduced_formula, dss2025)

# =========================================================
# Final Model – Reduced Logistic Regression (Median)
# =========================================================

final_predictors = [
    'experience_level', 'employment_type', 'job_title_group',
    'employee_residence', 'remote_ratio', 'company_location',
    'company_size', 'data_age', 'work_year_cat', 'remote_work_cat'
]

print("\n----- FINAL MODEL PREDICTORS (Median Salary) -----")
print(final_predictors)

# =========================================================
# IMPORTANT FIX — USE CLEANED DATA (NO MISSING VALUES)
# =========================================================

X_final = X[final_predictors]

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
             ("imputer", SimpleImputer(strategy="median")),
             ("scaler", StandardScaler())
         ]),
         X_final.select_dtypes(include=["int64", "float64"]).columns.tolist()),

        ("categorical",
         Pipeline([
             ("imputer", SimpleImputer(strategy="most_frequent")),
             ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
         ]),
         X_final.select_dtypes(include=["object", "category"]).columns.tolist())
    ]
)

# =========================================================
# CREATE TRANSFORMED MATRIX BEFORE LRT
# =========================================================

preprocessor_final.fit(X_train_f)
X_train_f_transformed = np.asarray(preprocessor_final.transform(X_train_f))

# =========================================================
# LRT LOOP (Grouped Predictors)
# =========================================================

def get_feature_names(preprocessor, X_final):
    num_features = preprocessor.named_transformers_["numeric"].named_steps["imputer"].get_feature_names_out(
        X_final.select_dtypes(include=["int64", "float64"]).columns
    )
    cat_features = preprocessor.named_transformers_["categorical"].named_steps["encoder"].get_feature_names_out(
        X_final.select_dtypes(include=["object", "category"]).columns
    )
    return np.concatenate([num_features, cat_features])

def group_columns_by_predictor(feature_names):
    groups = {}
    for name in feature_names:
        base = name.split("_")[0]
        groups.setdefault(base, []).append(name)
    return groups

def lrt_grouped(X, y, feature_groups, feature_names):
    full_model = sm.Logit(y, X).fit(disp=0)
    full_ll = full_model.llf
    results = []

    for predictor, cols in feature_groups.items():
        drop_idx = [i for i, name in enumerate(feature_names) if name in cols]
        reduced_X = np.delete(X, drop_idx, axis=1)

        try:
            reduced_model = sm.Logit(y, reduced_X).fit(disp=0)
            reduced_ll = reduced_model.llf
            lr_stat = -2 * (reduced_ll - full_ll)
            df = len(cols)
            p_value = chi2.sf(lr_stat, df)
            results.append((predictor, lr_stat, df, p_value))
            print(f"{predictor}: LR={lr_stat:.4f}, df={df}, p={p_value:.4f}")
        except Exception as e:
            results.append((predictor, None, None, None))
            print(f"{predictor}: model failed ({e})")

    return results

all_feature_names = get_feature_names(preprocessor_final, X_final)
feature_groups = group_columns_by_predictor(all_feature_names)

results = lrt_grouped(X_train_f_transformed, y_train_f, feature_groups, all_feature_names)

# =========================================================
# Fit Final Statsmodels Model (Median)
# =========================================================

sm_model = sm.Logit(y_train_f, X_train_f_transformed).fit(disp=0)
print("\n----- REDUCED MODEL SUMMARY (Median Salary) -----")
print(sm_model.summary())

# =========================================================
# Reduced Logistic Regression Pipeline
# =========================================================

final_model = Pipeline([
    ("preprocessor", preprocessor_final),
    ("classifier", LogisticRegression(max_iter=2000))
])

final_model.fit(X_train_f, y_train_f)

y_pred_f = final_model.predict(X_test_f)
y_prob_f = final_model.predict_proba(X_test_f)[:, 1]

# =========================================================
# Reduced Model Performance (Median)
# =========================================================

print("\n----- REDUCED MODEL PERFORMANCE (Median Salary) -----")
print(f"Accuracy:  {accuracy_score(y_test_f, y_pred_f):.4f}")
print(f"Precision: {precision_score(y_test_f, y_pred_f):.4f}")
print(f"Recall:    {recall_score(y_test_f, y_pred_f):.4f}")
print(f"F1 Score:  {f1_score(y_test_f, y_pred_f):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test_f, y_prob_f):.4f}")

print("\n----- REDUCED MODEL CONFUSION MATRIX -----")
print(confusion_matrix(y_test_f, y_pred_f))

# =========================================================
# Assumption Checks – Reduced Logistic Regression (Median)
# =========================================================

X_train_f_transformed = np.asarray(final_model.named_steps["preprocessor"].transform(X_train_f))

sm_model = sm.Logit(y_train_f, X_train_f_transformed).fit(disp=0)
print("\n----- REDUCED MODEL SUMMARY (Median Salary) -----")
print(sm_model.summary())

print("\n----- COEFFICIENTS (Reduced Model) -----")
print(final_model.named_steps["classifier"].coef_[0])

print("\n----- CLASS BALANCE (Reduced Model) -----")
print(y_train_f.value_counts(normalize=True))

print("\n----- MISSING VALUES (Reduced Predictors) -----")
print(X_final.isnull().sum())
