# -*- coding: utf-8 -*-


'''
Author: Aaron Niecestro  
Project: DS Salaries – LDA and QDA
Created: September 29, 2026  
Last Edit: September 29, 2026  
Progress: Completed  

---

Description:
# **Brief Description — LDA & QDA Models**

### **Linear Discriminant Analysis (LDA)**  
LDA is a classification method that assumes each class follows a 
**Gaussian (normal) distribution** and that all classes share a 
**common covariance matrix**.  
This shared‑covariance assumption medians LDA learns **linear decision boundaries**, 
making it effective when classes are well‑separated but have similar spread.  
It is computationally efficient and often performs well on high‑dimensional 
data after preprocessing.

### **Quadratic Discriminant Analysis (QDA)**  
QDA also assumes Gaussian class distributions but **allows each class to have 
its own covariance matrix**.  
This flexibility produces **quadratic decision boundaries**, enabling QDA 
to model more complex class shapes and interactions.  
QDA is more expressive than LDA but requires more data and can overfit 
when sample sizes are small.

This is a masters copy of BOTH: 1. Full Model and 2. The Reduced Model
'''

# =========================================================
# Linear & Quadratic Discriminant Analysis (Median Salary)
# =========================================================

# -*- coding: utf-8 -*-  # encoding

# =========================================================
# Regularized Logistic Regression – Median Salary Classification
# =========================================================

import pandas as pd                      # data handling
import numpy as np                       # numerical ops
import seaborn as sns                    # visualization
import matplotlib.pyplot as plt          # plotting

from sklearn.linear_model import LogisticRegression  # logistic models
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)  # metrics
from sklearn.model_selection import train_test_split  # train/test split
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # scaling + encoding
from sklearn.compose import ColumnTransformer          # preprocessing
from sklearn.pipeline import Pipeline                  # ML pipeline
from sklearn.impute import SimpleImputer              # imputation

sns.set(style="whitegrid", context="talk")            # seaborn style

# =========================================================
# 1. Load Data
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"  # dataset path
dss2025 = pd.read_csv(file_path)                      # load dataset
dss2025 = dss2025.query("company_location == 'United States'").copy()  # USA-only

# =========================================================
# 2. Encode Median Salary Category
# =========================================================

dss2025["salary_median_cat"] = dss2025["salary_median_cat"].map({
    "Below-Median": 0,
    "Above-Median": 1
})                                              # convert to 0/1

# =========================================================
# 3. Define Response + Predictors
# =========================================================

y = dss2025["salary_median_cat"]                         # binary response
X = dss2025[
    ['experience_level', 'employment_type','employee_residence',
     'remote_ratio', 'company_size', 'data_age',
     'work_year_cat', 'remote_work_cat', 'job_title_group']
]                                                      # predictor matrix

# =========================================================
# 4. Train/Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=72018, stratify=y
)                                                      # stratified split

# =========================================================
# 5. Preprocessing Pipeline
# =========================================================

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()  # numeric cols
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()  # categorical cols

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
             ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))  # encoding
         ]),
         categorical_features)
    ]
)    

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis  # LDA/QDA

# ---------------------------------------------------------
# 6A. Build LDA Pipeline (requires numeric-only input)
# ---------------------------------------------------------

lda_model_full = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LinearDiscriminantAnalysis())
])

# ---------------------------------------------------------
# 6B. Build QDA Pipeline (requires numeric-only input)
# ---------------------------------------------------------

qda_model_full = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", QuadraticDiscriminantAnalysis())
])

# ---------------------------------------------------------
# 7. Fit LDA & QDA Models
# ---------------------------------------------------------

lda_model_full.fit(X_train, y_train) # fit LDA
qda_model_full.fit(X_train, y_train) # fit QDA

# ---------------------------------------------------------
# 8. Evaluation Helper (same as logistic models)
# ---------------------------------------------------------

def eval_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)                     # predicted classes
    y_prob = model.predict_proba(X_test)[:, 1]         # predicted probabilities
    return {                                           # return metrics
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob)
    }

# ---------------------------------------------------------
# 9. Build LDA/QDA Comparison Table
# ---------------------------------------------------------

lda_results = eval_model("LDA", lda_model_full, X_test, y_test)   # LDA metrics
qda_results = eval_model("QDA", qda_model_full, X_test, y_test)   # QDA metrics

lda_qda_df = pd.DataFrame([lda_results, qda_results])        # combine results

print("\n----- LDA vs QDA (MEDIAN SALARY - Full Model) -----")
print(lda_qda_df)                                            # show comparison table

# ---------------------------------------------------------
# 10. ROC Curves for LDA & QDA
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))                                   # figure size

for name, model in [("LDA", lda_model_full), ("QDA", qda_model_full)]:
    y_prob = model.predict_proba(X_test)[:, 1]               # predicted probabilities
    fpr, tpr, _ = roc_curve(y_test, y_prob)                  # ROC curve
    auc = roc_auc_score(y_test, y_prob)                      # AUC score
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")      # plot curve

plt.plot([0, 1], [0, 1], "k--", label="Random")              # random baseline
plt.xlabel("False Positive Rate")                            # x-axis label
plt.ylabel("True Positive Rate")                             # y-axis label
plt.title("ROC Curves – LDA vs QDA (Median Salary)")           # title
plt.legend()                                                 # legend
plt.tight_layout()                                           # layout
plt.show()                                                   # display plot

# =========================================================
# Linear & Quadratic Discriminant Analysis – Reduced Final Model
# =========================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

sns.set(style="whitegrid", context="talk")

# =========================================================
# 1. Load Data
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
dss2025 = pd.read_csv(file_path)
dss2025 = dss2025.query("company_location == 'United States'").copy()

# =========================================================
# 2. Encode Median Salary Category
# =========================================================

dss2025["salary_median_cat"] = dss2025["salary_median_cat"].map({
    "Below-Median": 0,
    "Above-Median": 1
})   

# =========================================================
# 3. Define Response + REDUCED Predictors
# =========================================================

final_predictors = [
    'experience_level', 'employment_type', 'job_title_group',
    'employee_residence', 'company_size', 'work_year_cat', 'remote_work_cat'
]

X = dss2025[final_predictors]
y = dss2025["salary_median_cat"]

# =========================================================
# 4. Train/Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=72018, stratify=y
)

# =========================================================
# 5. Preprocessing Pipeline (same as logistic models)
# =========================================================

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

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
             ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
         ]),
         categorical_features)
    ]
)

# =========================================================
# 6A. LDA Pipeline
# =========================================================

lda_model_reduced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LinearDiscriminantAnalysis())
])

# =========================================================
# 6B. QDA Pipeline
# =========================================================

qda_model_reduced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", QuadraticDiscriminantAnalysis())
])

# =========================================================
# 7. Fit LDA & QDA Models
# =========================================================

lda_model_reduced.fit(X_train, y_train)
qda_model_reduced.fit(X_train, y_train)

# =========================================================
# 8. Evaluation Helper
# =========================================================

def eval_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob)
    }

# =========================================================
# 9. Build LDA/QDA Comparison Table
# =========================================================

lda_results = eval_model("LDA (Reduced)", lda_model_reduced, X_test, y_test)
qda_results = eval_model("QDA (Reduced)", qda_model_reduced, X_test, y_test)

lda_qda_df = pd.DataFrame([lda_results, qda_results])

print("\n----- LDA vs QDA (REDUCED FINAL MODEL) -----")
print(lda_qda_df)

# =========================================================
# 10. ROC Curves for LDA & QDA
# =========================================================

plt.figure(figsize=(8, 6))

for name, model in [("LDA", lda_model_reduced), ("QDA", qda_model_reduced)]:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – LDA vs QDA (Reduced Final Model)")
plt.legend()
plt.tight_layout()
plt.show()
