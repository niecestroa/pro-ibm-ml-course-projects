# -*- coding: utf-8 -*-
'''
Author: Aaron Niecestro  
Project: DS Salaries - Support Vector Macines 
Created: September 29, 2026  
Last Edit: September 29, 2026  
Progress: Completed  

---

# **Support Vector Machines (SVM) — Brief Notes**

### **What SVMs Do**
Support Vector Machines classify data by finding the **optimal separating hyperplane** between classes. They maximize the **margin**, which is the distance between the decision boundary and the closest data points (support vectors).  
Kernel SVMs transform the feature space to learn **nonlinear boundaries** when linear separation is not possible.

### **Why Use SVMs**
- Perform well on **medium‑sized datasets**  
- Effective when classes are **not linearly separable**  
- **RBF and polynomial kernels** capture nonlinear patterns  
- Margin‑based decision boundaries help **reduce overfitting**  
- Robust to high‑dimensional feature spaces (especially linear SVM)

### **Models Included**
| Model | Description |
|-------|-------------|
| **Linear SVM** | Fast, high‑dimensional linear classifier; good baseline |
| **RBF SVM** | Nonlinear kernel; captures complex relationships |
| **Polynomial SVM** | Models interaction effects and curved boundaries |

### **Why Company Location Is Excluded**
Since the dataset is filtered to **USA‑only**, `company_location` provides no additional information and is removed to avoid redundant encoding.  
It can be re‑introduced when modeling **global or multi‑country** data.

Support Vector Machines take significantly longer to train in this project 
because the dataset is high‑dimensional after encoding, kernel methods require 
expensive pairwise computations, and enabling probability outputs adds 
additional logistic calibration. Linear SVM is faster, but RBF and polynomial 
kernels can be slow on datasets of this size.

This is for BOTH 1. The Final Model and 2. The Reduced Model
'''

# =========================================================
# Support Vector Machines – Full Model (USA-only, no company_location)
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

from sklearn.svm import SVC

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

# 3. Convert Median-Based Response to Numeric Labels
dss2025["salary_median_cat"] = dss2025["salary_median_cat"].map({
    "Below-Median": 0,
    "Above-Median": 1
})

# =========================================================
# 3. Define Response + FULL Predictors (company_location excluded)
# =========================================================

svm_predictors = [
    'experience_level', 'employment_type', 'employee_residence',
    'remote_ratio', 'company_size', 'data_age',
    'work_year_cat', 'remote_work_cat', 'job_title_group'
]

X = dss2025[svm_predictors]
y = dss2025["salary_median_cat"]

# =========================================================
# 4. Train/Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=72018, stratify=y
)

# =========================================================
# 5. Preprocessing Pipeline
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
# 6. SVM Models (Linear, RBF, Polynomial)
# =========================================================

linear_svm_full = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="linear", probability=True, random_state=72018))
])

rbf_svm_full = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale",
                       probability=True, random_state=72018))
])

poly_svm_full = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="poly", degree=3, C=1.0,
                       probability=True, random_state=72018))
])

linear_svm_full.fit(X_train, y_train)
rbf_svm_full.fit(X_train, y_train)
poly_svm_full.fit(X_train, y_train)

# =========================================================
# 7. Fit All SVM Models
# =========================================================

linear_svm_full.fit(X_train, y_train)
rbf_svm_full.fit(X_train, y_train)
poly_svm_full.fit(X_train, y_train)

# =========================================================
# 8. Evaluation Helper
# =========================================================

def eval_svm(name, model, X_test, y_test):
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
# 9. Build Comparison Table
# =========================================================

svm_results = []
svm_results.append(eval_svm("Linear SVM", linear_svm_full, X_test, y_test))
svm_results.append(eval_svm("RBF SVM", rbf_svm_full, X_test, y_test))
svm_results.append(eval_svm("Polynomial SVM", poly_svm_full, X_test, y_test))

svm_df = pd.DataFrame(svm_results)

print("\n----- SUPPORT VECTOR MACHINES (FULL MODEL, USA-only) -----")
print(svm_df)

# =========================================================
# 10. ROC Curves for All SVM Models
# =========================================================

plt.figure(figsize=(10, 7))

for name, model in [
    ("Linear SVM", linear_svm_full),
    ("RBF SVM", rbf_svm_full),
    ("Polynomial SVM", poly_svm_full)
]:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – Support Vector Machines (Full Model, USA-only)")
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# Support Vector Machines – Reduced Final Model
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

from sklearn.svm import SVC

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

# 3. Convert Median-Based Response to Numeric Labels
dss2025["salary_median_cat"] = dss2025["salary_median_cat"].map({
    "Below-Median": 0,
    "Above-Median": 1
})

# =========================================================
# 3. Define Response + REDUCED Predictors
# =========================================================

reduced_predictors = [
    'experience_level', 'employment_type', 'job_title_group',
    'employee_residence', 'company_size', 'work_year_cat', 'remote_work_cat'
]

X = dss2025[reduced_predictors]
y = dss2025["salary_median_cat"]

# =========================================================
# 4. Train/Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=72018, stratify=y
)

# =========================================================
# 5. Preprocessing Pipeline
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
# 6. SVM Models (Linear, RBF, Polynomial)
# =========================================================

linear_svm_reduced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="linear", probability=True, random_state=72018))
])

rbf_svm_reduced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale",
                       probability=True, random_state=72018))
])

poly_svm_reduced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="poly", degree=3, C=1.0,
                       probability=True, random_state=72018))
])

linear_svm_reduced.fit(X_train, y_train)
rbf_svm_reduced.fit(X_train, y_train)
poly_svm_reduced.fit(X_train, y_train)


# =========================================================
# 7. Fit All SVM Models
# =========================================================

linear_svm_reduced.fit(X_train, y_train)
rbf_svm_reduced.fit(X_train, y_train)
poly_svm_reduced.fit(X_train, y_train)

# =========================================================
# 8. Evaluation Helper
# =========================================================

def eval_svm(name, model, X_test, y_test):
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
# 9. Build Comparison Table
# =========================================================

svm_results = []
svm_results.append(eval_svm("Linear SVM (Reduced)", linear_svm_reduced, X_test, y_test))
svm_results.append(eval_svm("RBF SVM (Reduced)", rbf_svm_reduced, X_test, y_test))
svm_results.append(eval_svm("Polynomial SVM (Reduced)", poly_svm_reduced, X_test, y_test))

svm_df = pd.DataFrame(svm_results)

print("\n----- SUPPORT VECTOR MACHINES (REDUCED FINAL MODEL) -----")
print(svm_df)

# =========================================================
# 10. ROC Curves for All SVM Models
# =========================================================

plt.figure(figsize=(10, 7))

for name, model in [
    ("Linear SVM", linear_svm_reduced),
    ("RBF SVM", rbf_svm_reduced),
    ("Polynomial SVM", poly_svm_reduced)
]:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – SVM Models (Reduced Final Model)")
plt.legend()
plt.tight_layout()
plt.show()
