'''
Author: Aaron Niecestro  
Project: DS Salaries - Support Vector Macines 
Created: September 29, 2026  
Last Edit: September 29, 2026  
Progress: Completed  

---

# **Probabilistic Models — Brief Notes**

### **What These Models Do**
Probabilistic models classify observations by computing the 
**likelihood of each class** given the input features. They rely on 
**Bayes’ theorem** and assume specific statistical distributions 
(e.g., Gaussian, multinomial, Bernoulli). Instead of learning complex 
boundaries, they estimate how probable each class is under the model’s assumptions.

### **Why Use Them**
- Extremely **fast and lightweight**, even on large datasets  
- Provide **calibrated class probabilities**  
- Useful when features are **independent**, categorical, or count‑based  
- Serve as strong **baseline models** for comparison  
- Helpful for understanding **uncertainty** in predictions  

### **Models Included**
| Model | Description |
|-------|-------------|
| **Gaussian Naive Bayes** | Assumes continuous features follow a normal distribution |
| **Multinomial Naive Bayes** | Designed for count‑based or frequency features |
| **Bernoulli Naive Bayes** | Works with binary or indicator features |

### **Full vs Reduced Model**
- **Full Model:** Uses all predictors except `company_location` 
(excluded because dataset is USA‑only).  
- **Reduced Final Model:** Uses the optimized 7‑predictor set selected 
through GLM inference and model diagnostics.
'''

# =========================================================
# Probabilistic Models – Full Model (USA-only, no company_location)
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

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

sns.set(style="whitegrid", context="talk")

# =========================================================
# 1. Load Data
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
dss2025 = pd.read_csv(file_path)
dss2025 = dss2025.query("company_location == 'United States'").copy()

# =========================================================
# 2. Encode Mean Salary Category
# =========================================================

dss2025["salary_mean_cat"] = dss2025["salary_mean_cat"].map({
    "Below-Average": 0,
    "Above-Average": 1
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
y = dss2025["salary_mean_cat"]

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
# 6. Probabilistic Models
# =========================================================

gnb_model_full = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GaussianNB())
])

mnb_model_full = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", MultinomialNB())
])

bnb_model_full = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", BernoulliNB())
])

gnb_model_full.fit(X_train, y_train)
mnb_model_full.fit(X_train, y_train)
bnb_model_full.fit(X_train, y_train)

# =========================================================
# 7. Fit Models
# =========================================================

gnb_model_full.fit(X_train, y_train)
mnb_model_full.fit(X_train, y_train)
bnb_model_full.fit(X_train, y_train)

# =========================================================
# 8. Evaluation Helper
# =========================================================

def eval_nb(name, model, X_test, y_test):
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
# 9. Comparison Table
# =========================================================

nb_results = []
nb_results.append(eval_nb("Gaussian NB (Full)", gnb_model_full, X_test, y_test))
nb_results.append(eval_nb("Multinomial NB (Full)", mnb_model_full, X_test, y_test))
nb_results.append(eval_nb("Bernoulli NB (Full)", bnb_model_full, X_test, y_test))

nb_df = pd.DataFrame(nb_results)

print("\n----- PROBABILISTIC MODELS (FULL MODEL) -----")
print(nb_df)

# =========================================================
# 10. ROC Curves
# =========================================================

plt.figure(figsize=(10, 7))

for name, model in [
    ("Gaussian NB", gnb_model_full),
    ("Multinomial NB", mnb_model_full),
    ("Bernoulli NB", bnb_model_full)
]:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – Probabilistic Models (Full Model)")
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# Probabilistic Models – Reduced Final Model
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

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

sns.set(style="whitegrid", context="talk")

# =========================================================
# 1. Load Data
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
dss2025 = pd.read_csv(file_path)
dss2025 = dss2025.query("company_location == 'United States'").copy()

# =========================================================
# 2. Encode Mean Salary Category
# =========================================================

dss2025["salary_mean_cat"] = dss2025["salary_mean_cat"].map({
    "Below-Average": 0,
    "Above-Average": 1
})

# =========================================================
# 3. REDUCED Final Predictors
# =========================================================

reduced_predictors = [
    'experience_level', 'employment_type', 'job_title_group',
    'employee_residence', 'company_size', 'work_year_cat', 'remote_work_cat'
]

X = dss2025[reduced_predictors]
y = dss2025["salary_mean_cat"]

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
# 6. Probabilistic Models
# =========================================================

gnb_model_reduced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GaussianNB())
])

mnb_model_reduced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", MultinomialNB())
])

bnb_model_reduced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", BernoulliNB())
])

gnb_model_reduced.fit(X_train, y_train)
mnb_model_reduced.fit(X_train, y_train)
bnb_model_reduced.fit(X_train, y_train)

# =========================================================
# 7. Fit Models
# =========================================================

gnb_model_reduced.fit(X_train, y_train)
mnb_model_reduced.fit(X_train, y_train)
bnb_model_reduced.fit(X_train, y_train)

# =========================================================
# 8. Evaluation Helper
# =========================================================

def eval_nb(name, model, X_test, y_test):
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
# 9. Comparison Table
# =========================================================

nb_results = []
nb_results.append(eval_nb("Gaussian NB (Reduced)", gnb_mode_reducedl, X_test, y_test))
nb_results.append(eval_nb("Multinomial NB (Reduced)", mnb_model_reduced, X_test, y_test))
nb_results.append(eval_nb("Bernoulli NB (Reduced)", bnb_model_reduced, X_test, y_test))

nb_df = pd.DataFrame(nb_results)

print("\n----- PROBABILISTIC MODELS (REDUCED FINAL MODEL) -----")
print(nb_df)

# =========================================================
# 10. ROC Curves
# =========================================================

plt.figure(figsize=(10, 7))

for name, model in [
    ("Gaussian NB", gnb_model_reduced),
    ("Multinomial NB", mnb_model_reduced),
    ("Bernoulli NB", bnb_model_reduced)
]:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – Probabilistic Models (Reduced Final Model)")
plt.legend()
plt.tight_layout()
plt.show()
