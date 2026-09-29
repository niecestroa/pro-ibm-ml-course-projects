'''
Author: Aaron Niecestro  
Project: DS Salaries - Support Vector Macines 
Created: September 29, 2026  
Last Edit: September 29, 2026  
Progress: Completed  

---

# **Brief Description — Instance‑Based / Distance‑Based Models**

### **What These Models Do**
Instance‑based models classify new observations by comparing them directly to 
existing samples. Instead of learning parameters, they rely on 
**distance metrics** (e.g., Euclidean, Manhattan) to find the 
most similar training points and use their labels to make predictions.

### **Why Use Them**
- Capture **irregular, highly nonlinear boundaries**  
- Useful when relationships are **local** rather than global  
- Simple, intuitive, and non‑parametric  
- Perform well when data is **properly scaled**  
- Provide a strong **sanity‑check baseline** for comparison  

### **Model Included**
| Model | Description |
|-------|-------------|
| **K‑Nearest Neighbors (KNN)** | Classifies based on the labels of the k most similar samples |

### **Full vs Reduced Model**
- **Full Model:** Uses all predictors except `company_location` 
(excluded because dataset is USA‑only).  
- **Reduced Final Model:** Uses the optimized 7‑predictor set selected 
through GLM inference and model diagnostics.
'''

# **Instance‑Based Models — FULL MODEL (USA‑only, company_location excluded)**

# =========================================================
# K-Nearest Neighbors – Full Model (USA-only, no company_location)
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

from sklearn.neighbors import KNeighborsClassifier

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
# 3. FULL Predictors (company_location excluded)
# =========================================================

full_predictors = [
    'experience_level', 'employment_type','employee_residence',
    'remote_ratio', 'company_size', 'data_age',
    'work_year_cat', 'remote_work_cat', 'job_title_group'
]

X = dss2025[full_predictors]
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
# Find Best k for KNN
# =========================================================

k_values = range(1, 100)  # search k = 1 to 100
k_results_full = []

for k in k_values:
    knn_temp = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=k))
    ])
    knn_temp.fit(X_train, y_train)
    y_pred = knn_temp.predict(X_test)
    y_prob = knn_temp.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    k_results_full.append((k, auc))

# Pick best k
best_k_full, best_auc_full = max(k_results_full, key=lambda x: x[1])
print(f"\nBest k = {best_k_full}  (AUC = {best_auc_full:.4f})")

# =========================================================
# 6. KNN Model
# =========================================================

knn_model_full = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=best_k_full))
])

knn_model_full.fit(X_train, y_train)

# =========================================================
# 7. Fit Model
# =========================================================

knn_model_full.fit(X_train, y_train)

# =========================================================
# 8. Evaluation Helper
# =========================================================

def eval_knn(name, model, X_test, y_test):
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
# 9. Results Table
# =========================================================

knn_results_full = eval_knn(f"KNN (k={best_k_full})", knn_model_full, X_test, y_test)
print("\n----- INSTANCE-BASED MODEL (FULL MODEL) -----")
print(pd.DataFrame([knn_results_full]))

# =========================================================
# 10. ROC Curve
# =========================================================

plt.figure(figsize=(8, 6))
y_prob = knn_model_full.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f"KNN (k={best_k_full}, AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – KNN (Optimized k)")
plt.legend()
plt.tight_layout()
plt.show()

# **Instance‑Based Models — REDUCED FINAL MODEL (7 predictors)**

# =========================================================
# K-Nearest Neighbors – Reduced Final Model
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

from sklearn.neighbors import KNeighborsClassifier

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
# 3. REDUCED Final Predictors
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
# Find Best k for KNN
# =========================================================

k_values = range(1, 31)  # search k = 1 to 30
k_results_reduced = []

for k in k_values:
    knn_temp = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=k))
    ])
    knn_temp.fit(X_train, y_train)
    y_pred = knn_temp.predict(X_test)
    y_prob = knn_temp.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    k_results_reduced.append((k, auc))

# Pick best k
best_k_reduced, best_auc_reduced = max(k_results_reduced, key=lambda x: x[1])
print(f"\nBest k = {best_k_reduced}  (AUC = {best_auc_reduced:.4f})")

# =========================================================
# 6. KNN Model
# =========================================================

knn_model_reduced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=best_k_reduced))
])

# =========================================================
# 7. Fit Model
# =========================================================

knn_model_reduced.fit(X_train, y_train)

# =========================================================
# 8. Evaluation Helper
# =========================================================

def eval_knn(name, model, X_test, y_test):
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
# 9. Results Table
# =========================================================

knn_results = eval_knn(f"KNN (k={best_k_reduced})", knn_model_reduced, X_test, y_test)
print("\n----- INSTANCE-BASED MODEL (REDUCED FINAL MODEL) -----")
print(pd.DataFrame([knn_results]))

# =========================================================
# 10. ROC Curve
# =========================================================

plt.figure(figsize=(8, 6))
y_prob = knn_model_reduced.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f"KNN (k={best_k_reduced}, AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – KNN (Optimized k)")
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# Combined FULL vs REDUCED KNN Comparison Table
# =========================================================

knn_full_results = {
    "Model": f"KNN (Full, k={best_k_full})",
    "Accuracy": accuracy_score(y_test, knn_model_full.predict(X_test)),
    "Precision": precision_score(y_test, knn_model_full.predict(X_test)),
    "Recall": recall_score(y_test, knn_model_full.predict(X_test)),
    "F1": f1_score(y_test, knn_model_full.predict(X_test)),
    "ROC_AUC": roc_auc_score(y_test, knn_model_full.predict_proba(X_test)[:, 1])
}

knn_reduced_results = {
    "Model": f"KNN (Reduced, k={best_k_reduced})",
    "Accuracy": accuracy_score(y_test, knn_model_reduced.predict(X_test)),
    "Precision": precision_score(y_test, knn_model_reduced.predict(X_test)),
    "Recall": recall_score(y_test, knn_model_reduced.predict(X_test)),
    "F1": f1_score(y_test, knn_model_reduced.predict(X_test)),
    "ROC_AUC": roc_auc_score(y_test, knn_model_reduced.predict_proba(X_test)[:, 1])
}

knn_compare_df = pd.DataFrame([knn_full_results, knn_reduced_results])

print("\n----- KNN FULL vs REDUCED COMPARISON TABLE -----")
print(knn_compare_df)

