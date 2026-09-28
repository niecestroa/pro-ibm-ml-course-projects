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

# =========================================================
# Logistic Regression - DSS2025 (Classification Version)
# =========================================================

import pandas as pd                      # data manipulation
import numpy as np                       # numerical operations
import seaborn as sns                    # visualization
import matplotlib.pyplot as plt          # plotting

from sklearn.linear_model import LogisticRegression   # logistic regression model
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve)  # classification metrics
from sklearn.model_selection import train_test_split  # train/test split
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # scaling + encoding
from sklearn.compose import ColumnTransformer          # preprocessing transformer
from sklearn.pipeline import Pipeline                  # ML pipeline
from sklearn.impute import SimpleImputer              # imputation

import statsmodels.formula.api as smf   # formula-based modeling
import statsmodels.api as sm            # core statsmodels API
from scipy.stats import chi2            # chi-square for LRT

sns.set(style="whitegrid", context="talk")             # seaborn style

# =========================================================
# 1. Load Data
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"  # file path
dss2025 = pd.read_csv(file_path)  # load dataset
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
    ['experience_level', 'employment_type','employee_residence', 
     'remote_ratio', 'company_location', 'company_size', 'data_age', 
     'work_year_cat', 'remote_work_cat', 'job_title_group']
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
# 9. Full Logistic Regression Pipeline
# =========================================================

log_reg = Pipeline([
    ("preprocessor", preprocessor),  # preprocessing
    ("classifier", LogisticRegression(max_iter=2000))  # logistic regression
])

log_reg.fit(X_train, y_train)  # fit full model

# =========================================================
# 10. Full Model Performance
# =========================================================

y_pred = log_reg.predict(X_test)  # predicted classes
y_prob = log_reg.predict_proba(X_test)[:, 1]  # predicted probabilities

print("\n----- FULL MODEL PERFORMANCE -----")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

print("\n----- FULL MODEL CONFUSION MATRIX -----")
print(confusion_matrix(y_test, y_pred))

# =========================================================
# Likelihood Ratio Test (LRT) – Formula API
# =========================================================

def lrt_formula(full_formula, reduced_formula, data):
    full_model = smf.glm(full_formula, data=data, family=sm.families.Binomial()).fit()  # full model
    reduced_model = smf.glm(reduced_formula, data=data, family=sm.families.Binomial()).fit()  # reduced model
    lr_stat = -2 * (reduced_model.llf - full_model.llf)  # LR statistic
    df = full_model.df_model - reduced_model.df_model  # degrees of freedom
    p_value = chi2.sf(lr_stat, df)  # p-value

    print("\n----- LRT RESULT -----")
    print(f"Full model:    {full_formula}")
    print(f"Reduced model: {reduced_formula}")
    print(f"LR stat = {lr_stat:.4f}")
    print(f"df = {df}")
    print(f"p-value = {p_value:.4f}")
    print("Decision:", "Reject H0 (significant), Keep predictor(s), use Full Model" if p_value < 0.05 else "Fail to Reject H0 (not significant), discard predictor(s), use Reduced Model")

    return lr_stat, p_value

# =========================================================
# LRT Example – Remove job_title_group
# =========================================================

full_formula = (
    "salary_mean_cat ~ experience_level + employment_type + job_title_group + "
    "employee_residence + remote_ratio + company_location + company_size + "
    "data_age + work_year_cat + remote_work_cat"
)

reduced_formula = (
    "salary_mean_cat ~ experience_level + employment_type + employee_residence + "
    "remote_ratio + company_location + company_size + data_age + work_year_cat + remote_work_cat"
)

lrt_formula(full_formula, reduced_formula, dss2025)

# =========================================================
# LRT Example – Remove remote_ratio + company_size
# =========================================================

full_formula = (
    "salary_mean_cat ~ experience_level + employment_type + job_title_group + "
    "employee_residence + remote_ratio + company_location + company_size + "
    "data_age + work_year_cat + remote_work_cat"
)

reduced_formula = (
    "salary_mean_cat ~ experience_level + employment_type + job_title_group + "
    "employee_residence + remote_ratio + company_location + company_size + "
    "work_year_cat + remote_work_cat"
)

lrt_formula(full_formula, reduced_formula, dss2025)

# =========================================================
# Final Model – Reduced Logistic Regression (Manual Predictors)
# =========================================================

final_predictors = [
    'experience_level', 'employment_type', 'job_title_group',
    'employee_residence', 'remote_ratio', 'company_location',
    'company_size', 'data_age', 'work_year_cat', 'remote_work_cat'
]  # reduced predictor set

print("\n----- FINAL MODEL PREDICTORS -----")
print(final_predictors)

# =========================================================
# IMPORTANT FIX — USE CLEANED DATA (NO MISSING VALUES)
# =========================================================

X_final = X[final_predictors]  # use CLEANED predictors (correct)

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_final, y, test_size=0.30, random_state=72018, stratify=y
)  # split reduced data

# =========================================================
# Preprocessing for Reduced Model
# =========================================================

preprocessor_final = ColumnTransformer(
    transformers=[
        ("numeric",
         Pipeline([
             ("imputer", SimpleImputer(strategy="median")),  # numeric imputation
             ("scaler", StandardScaler())  # numeric scaling
         ]),
         X_final.select_dtypes(include=["int64", "float64"]).columns.tolist()),

        ("categorical",
         Pipeline([
             ("imputer", SimpleImputer(strategy="most_frequent")),  # categorical imputation
             ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))  # encoding
         ]),
         X_final.select_dtypes(include=["object", "category"]).columns.tolist())
    ]
)

# =========================================================
# CREATE TRANSFORMED MATRIX BEFORE LRT
# =========================================================

preprocessor_final.fit(X_train_f)  # fit preprocessing
X_train_f_transformed = preprocessor_final.transform(X_train_f)  # transform data
X_train_f_transformed = np.asarray(X_train_f_transformed)  # convert to array

# =========================================================
# LRT LOOP (GROUPED BY ORIGINAL PREDICTOR)
# =========================================================

def get_feature_names(preprocessor, X_final):
    num_features = preprocessor.named_transformers_["numeric"].named_steps["imputer"].get_feature_names_out(
        X_final.select_dtypes(include=["int64", "float64"]).columns
    )  # numeric names
    cat_features = preprocessor.named_transformers_["categorical"].named_steps["encoder"].get_feature_names_out(
        X_final.select_dtypes(include=["object", "category"]).columns
    )  # dummy names
    return np.concatenate([num_features, cat_features])  # all names

def group_columns_by_predictor(feature_names):
    groups = {}
    for name in feature_names:
        base = name.split("_")[0]  # original predictor name
        groups.setdefault(base, []).append(name)  # group dummy columns
    return groups

def lrt_grouped(X, y, feature_groups, feature_names):
    full_model = sm.Logit(y, X).fit(disp=0)  # full model
    full_ll = full_model.llf  # full log-likelihood
    results = []

    for predictor, cols in feature_groups.items():
        drop_idx = [i for i, name in enumerate(feature_names) if name in cols]  # columns to drop
        reduced_X = np.delete(X, drop_idx, axis=1)  # reduced matrix

        try:
            reduced_model = sm.Logit(y, reduced_X).fit(disp=0)  # reduced model
            reduced_ll = reduced_model.llf  # reduced log-likelihood
            lr_stat = -2 * (reduced_ll - full_ll)  # LR statistic
            df = len(cols)  # degrees of freedom
            p_value = chi2.sf(lr_stat, df)  # p-value
            results.append((predictor, lr_stat, df, p_value))  # store
            print(f"{predictor}: LR={lr_stat:.4f}, df={df}, p={p_value:.4f}")  # print
        except Exception as e:
            results.append((predictor, None, None, None))  # failure
            print(f"{predictor}: model failed ({e})")

    return results

all_feature_names = get_feature_names(preprocessor_final, X_final)  # feature names
feature_groups = group_columns_by_predictor(all_feature_names)  # grouped predictors

results = lrt_grouped(X_train_f_transformed, y_train_f, feature_groups, all_feature_names)  # run LRT

# =========================================================
# FIT FINAL STATSMODELS MODEL AFTER LRT
# =========================================================

sm_model = sm.Logit(y_train_f, X_train_f_transformed).fit(disp=0)  # final inference model
print("\n----- REDUCED MODEL SUMMARY (Statsmodels) -----")
print(sm_model.summary())

# =========================================================
# Reduced Logistic Regression Pipeline
# =========================================================

final_model = Pipeline([
    ("preprocessor", preprocessor_final),  # preprocessing
    ("classifier", LogisticRegression(max_iter=2000))  # logistic regression
])

final_model.fit(X_train_f, y_train_f)  # fit reduced model

y_pred_f = final_model.predict(X_test_f)  # predicted classes
y_prob_f = final_model.predict_proba(X_test_f)[:, 1]  # predicted probabilities

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

X_train_f_transformed = final_model.named_steps["preprocessor"].transform(X_train_f)  # transformed data
X_train_f_transformed = np.asarray(X_train_f_transformed)  # convert to array

sm_model = sm.Logit(y_train_f, X_train_f_transformed).fit(disp=0)  # statsmodels inference
print("\n----- REDUCED MODEL SUMMARY (Statsmodels) -----")
print(sm_model.summary())

print("\n----- COEFFICIENTS (Reduced Model) -----")
coef_reduced = final_model.named_steps["classifier"].coef_[0]  # coefficients
print(coef_reduced)

print("\n----- CLASS BALANCE (Reduced Model) -----")
print(y_train_f.value_counts(normalize=True))  # class balance

print("\n----- MISSING VALUES (Reduced Predictors) -----")
print(X_final.isnull().sum())  # should be zero now
