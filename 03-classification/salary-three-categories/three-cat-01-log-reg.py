# -*- coding: utf-8 -*-

# =========================================================
'''
**Author:** Aaron Niecestro  
**Project:** DS Salaries – Production‑Ready EDA for Logistic Regression Models  
**Created:** September 28, 2026  
**Last Edit:** September 28, 2026  
**Progress:** Paused (To be completed after mean and median salary classification models are completd)

---

## Description — Logistic Regression Pipeline (Multi‑Part)

### Purpose**  
End‑to‑end salary classification workflow using standard logistic regression.  
This module performs data loading, preprocessing, train/test splitting, model 
fitting, evaluation, diagnostic visualization, and assumption checks for 
three salary classification task.

### What it covers

- **USA‑only dataset filtering**  
  Ensures geographic consistency by restricting analysis to U.S. salary records.

- **R‑style summary + structural overview**  
  Provides dataset shape, variable types, and descriptive statistics for both numeric and categorical features.

- **Predictor/response setup**  
  - **Response:** `salary_mean_cat` (three high-vs-average-vs-low salary category)  
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

"""
Logistic Regression Pipeline – THREE Salary Categories (Multinomial)
Converted from binary salary_mean_cat → salary_3cat (Low / Middle / High)
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import statsmodels.api as sm
import statsmodels.formula.api as smf

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
# 3. Convert Response to Numeric Labels (THREE CLASSES)
# =========================================================

dss2025["salary_3cat"] = dss2025["salary_3cat"].map({
    "Low-Salary": 0,
    "Middle-Salary": 1,
    "High-Salary": 2
})

# =========================================================
# 4. Response + Predictors
# =========================================================

y = dss2025["salary_3cat"]   # <<<<<< THREE-CLASS RESPONSE

X = dss2025[
    ['experience_level', 'employment_type','employee_residence',
     'remote_ratio', 'company_location', 'company_size', 'data_age',
     'work_year_cat', 'remote_work_cat', 'job_title_group']
]

print("----- RESPONSE (salary_3cat) -----")
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
# 9. Multinomial Logistic Regression Pipeline
# =========================================================

log_reg = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=3000, multi_class="multinomial"))
])

log_reg.fit(X_train, y_train)

# =========================================================
# 10. Full Model Performance (Multiclass)
# =========================================================

y_pred = log_reg.predict(X_test)

print("\n----- FULL MODEL PERFORMANCE (Three Categories) -----")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print("\n----- CLASSIFICATION REPORT -----")
print(classification_report(y_test, y_pred))

print("\n----- CONFUSION MATRIX -----")
print(confusion_matrix(y_test, y_pred))

# =========================================================
# 11. Multinomial GLM (Statsmodels) for Inference
# =========================================================

# Create dummy matrix for statsmodels
X_sm = pd.get_dummies(X_train, drop_first=True)
X_sm = sm.add_constant(X_sm)

mn_model = sm.MNLogit(y_train, X_sm).fit(method="newton", maxiter=200)
print("\n----- MULTINOMIAL MODEL SUMMARY -----")
print(mn_model.summary())

# =========================================================
# 12. Pseudo-LRT (Multinomial) – Full vs Reduced Model
# =========================================================

def pseudo_lrt(full_formula, reduced_formula, data):
    full = smf.mnlogit(full_formula, data=data).fit(method="newton", maxiter=200)
    reduced = smf.mnlogit(reduced_formula, data=data).fit(method="newton", maxiter=200)

    lr_stat = -2 * (reduced.llf - full.llf)
    df = full.df_model - reduced.df_model
    p_value = chi2.sf(lr_stat, df)

    print("\n----- MULTINOMIAL PSEUDO-LRT -----")
    print(f"LR stat = {lr_stat:.4f}")
    print(f"df = {df}")
    print(f"p-value = {p_value:.4f}")

    return lr_stat, p_value

# =========================================================
# LRT Example – Remove job_title_group (Three Categories)
# =========================================================

full_formula = (
    "salary_3cat ~ experience_level + employment_type + job_title_group + "
    "employee_residence + remote_ratio + company_location + company_size + "
    "data_age + work_year_cat + remote_work_cat"
)

reduced_formula = (
    "salary_3cat ~ experience_level + employment_type + employee_residence + "
    "remote_ratio + company_location + company_size + data_age + work_year_cat + remote_work_cat"
)

pseudo_lrt(full_formula, reduced_formula, dss2025)

# =========================================================
# 13. Final Reduced Model (Three Categories)
# =========================================================

final_predictors = [
    'experience_level', 'employment_type', 'job_title_group',
    'employee_residence', 'remote_ratio', 'company_location',
    'company_size', 'data_age', 'work_year_cat', 'remote_work_cat'
]

print("\n----- FINAL MODEL PREDICTORS (Three Categories) -----")
print(final_predictors)

X_final = X[final_predictors]

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_final, y, test_size=0.30, random_state=72018, stratify=y
)

# =========================================================
# 14. Reduced Multinomial Pipeline
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

final_model = Pipeline([
    ("preprocessor", preprocessor_final),
    ("classifier", LogisticRegression(max_iter=3000, multi_class="multinomial"))
])

final_model.fit(X_train_f, y_train_f)

y_pred_f = final_model.predict(X_test_f)

# =========================================================
# 15. Reduced Model Performance (Three Categories)
# =========================================================

print("\n----- REDUCED MODEL PERFORMANCE (Three Categories) -----")
print(f"Accuracy:  {accuracy_score(y_test_f, y_pred_f):.4f}")
print("\n----- CLASSIFICATION REPORT -----")
print(classification_report(y_test_f, y_pred_f))

print("\n----- CONFUSION MATRIX -----")
print(confusion_matrix(y_test_f, y_pred_f))

# =========================================================
# 16. Statsmodels Multinomial Inference (Reduced)
# =========================================================

X_sm_f = pd.get_dummies(X_train_f, drop_first=True)
X_sm_f = sm.add_constant(X_sm_f)

mn_model_reduced = sm.MNLogit(y_train_f, X_sm_f).fit(method="newton", maxiter=200)
print("\n----- REDUCED MULTINOMIAL MODEL SUMMARY -----")
print(mn_model_reduced.summary())

# =========================================================
# 17. Diagnostics
# =========================================================

print("\n----- CLASS BALANCE (Three Categories) -----")
print(y_train_f.value_counts(normalize=True))

print("\n----- MISSING VALUES (Reduced Predictors) -----")
print(X_final.isnull().sum())
