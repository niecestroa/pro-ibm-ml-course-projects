# -*- coding: utf-8 -*-

# =========================================================
'''
**Author:** Aaron Niecestro  
**Project:** DS Salaries – Production‑Ready EDA for Logistic Regression Models  
**Created:** September 28, 2026  
**Last Edit:** September 28, 2026  
**Progress:** Ongoing  

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
dss2025 = pd.read_csv(file_path)                       # read CSV into dataframe
print("----- DATA SHAPE (MASTER) -----")               # print header
print(dss2025.shape)                                   # show dataset dimensions

# =========================================================
# 2. USA-Only Dataset
# =========================================================

dss2025 = dss2025.query("company_location == 'United States'").copy()  # filter USA rows
print("----- USA DATASET -----")                                       # print header
print(dss2025.shape)                                                   # show USA-only dimensions

# =========================================================
# 3. Convert Response to Numeric Labels
# =========================================================

dss2025["salary_mean_cat"] = dss2025["salary_mean_cat"].map({          # convert labels to 0/1
    "Below-Average": 0,
    "Above-Average": 1
})

# =========================================================
# 4. Response + Predictors
# =========================================================

y = dss2025["salary_median_cat"]                 # binary response variable (0/1)
X = dss2025[                                   # predictor matrix
    ['experience_level', 'employment_type', 'job_title',
     'employee_residence', 'remote_ratio',
     'company_location', 'company_size', 'data_age',
     'work_year_cat', 'remote_work_cat', 'job_title_group']
]

print("----- RESPONSE (salary_median_cat) -----")  # print header
print(y.value_counts())                          # show class distribution

print("\n----- PREDICTORS -----")                # print header
print(X.head())                                  # preview predictors

# =========================================================
# 5. Missing Values
# =========================================================

print("\n----- MISSING VALUES -----")            # print header
print(dss2025.isnull().sum().sort_values(ascending=False))  # list missing values

# =========================================================
# 6. Identify Numeric + Categorical Predictors
# =========================================================

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()  # numeric predictors
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()  # categorical predictors

print("\n----- NUMERIC FEATURES -----")          # print header
print(numeric_features)                          # show numeric columns

print("\n----- CATEGORICAL FEATURES -----")       # print header
print(categorical_features)                       # show categorical columns

# =========================================================
# 7. Train/Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(  # split data
    X, y, test_size=0.30, random_state=72018, stratify=y
)

print("\n----- TRAIN / TEST SPLIT -----")         # print header
print(X_train.shape, X_test.shape)                # show split sizes
