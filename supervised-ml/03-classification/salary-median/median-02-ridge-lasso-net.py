# -*- coding: utf-8 -*-
# =========================================================
'''
Author: Aaron Niecestro  
Project: Data Science Salaries - Ridge, LASSO, Elastic Net 
Created: September 29, 2026  
Last Edit: September 29, 2026  
Progress: Completed  

---

Description:

This script performs regularized logistic regression on the median‑salary 
binary classification task using the cleaned DSS2025 dataset. 
It filters the dataset to USA‑only records, encodes the salary target 
(salary_median_cat), selects relevant predictors, and applies a train/test split.

A unified preprocessing pipeline is built using a ColumnTransformer that applies:
- Median imputation and standard scaling to numeric features
- Most‑frequent imputation and one‑hot encoding (drop‑first) to categorical features
- Three regularized logistic regression models are trained:
- Ridge (L2) — shrinks coefficients smoothly
- Lasso (L1) — performs feature selection by driving coefficients to zero
- Elastic Net (L1 + L2) — balances shrinkage and sparsity

Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC‑AUC

A comparison table summarizes all metrics side‑by‑side.

The script also extracts model coefficients after preprocessing and generates 
a coefficient shrinkage plot to visualize how Ridge, Lasso, and Elastic Net 
differ in regularization strength.

Finally, ROC curves for all three models are plotted together to compare 
classification performance visually.

This is a masters copy of BOTH: 1. Full Model and 2. The Reduced Model
'''

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

# 3. Convert Median-Based Response to Numeric Labels
dss2025["salary_median_cat"] = dss2025["salary_median_cat"].map({
    "Below-Median": 0,
    "Above-Median": 1
})                                                 
# convert to 0/1

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
)                                                      # preprocessing transformer

# =========================================================
# 6. Ridge, Lasso, Elastic Net Models
# =========================================================

ridge_model_full = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=2000))
])

lasso_model_full = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        penalty="l1", C=1.0, solver="liblinear", max_iter=2000))
])

elastic_model_full = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        penalty="elasticnet", C=1.0, l1_ratio=0.5,
        solver="saga", max_iter=3000))
])

ridge_model_full.fit(X_train, y_train)
lasso_model_full.fit(X_train, y_train)
elastic_model_full.fit(X_train, y_train)


# =========================================================
# 7. Fit All Models
# =========================================================

ridge_model_full.fit(X_train, y_train) # fit ridge
lasso_model_full.fit(X_train, y_train) # fit lasso
elastic_model_full.fit(X_train, y_train) # fit elastic net

# =========================================================
# 8. Evaluation Helper
# =========================================================

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

# =========================================================
# 9. Build Comparison Table
# =========================================================

results = []                                           # results list

results.append(eval_model("Ridge (Full)", ridge_model_full, X_test, y_test))
results.append(eval_model("Lasso (Full)", lasso_model_full, X_test, y_test))
results.append(eval_model("Elastic Net (Full)", elastic_model_full, X_test, y_test))


results_df = pd.DataFrame(results)                     # convert to dataframe
print("\n----- REGULARIZED LOGISTIC MODELS (MEDIAN SALARY) -----")
print(results_df)                                      # show table

# =========================================================
# 10. Extract Coefficients for Shrinkage Plot
# =========================================================

def get_coefficients(model, X_train, name):
    num_features = preprocessor.named_transformers_["numeric"].named_steps["imputer"].get_feature_names_out(
        X_train.select_dtypes(include=["int64", "float64"]).columns
    )                                                  # numeric names
    cat_features = preprocessor.named_transformers_["categorical"].named_steps["encoder"].get_feature_names_out(
        X_train.select_dtypes(include=["object", "category"]).columns
    )                                                  # categorical names
    feature_names = np.concatenate([num_features, cat_features])  # all names
    X_train_trans = model.named_steps["preprocessor"].transform(X_train)  # transformed matrix
    coefs = model.named_steps["classifier"].coef_[0]   # coefficients
    return pd.DataFrame({"feature": feature_names, "coef": coefs, "model": name})  # dataframe

coef_ridge = get_coefficients(ridge_model_full, X_train, "Ridge (Full)")
coef_lasso = get_coefficients(lasso_model_full, X_train, "Lasso (Full)")
coef_elastic = get_coefficients(elastic_model_full, X_train, "ElasticNet (Full)")

coef_all = pd.concat([coef_ridge, coef_lasso, coef_elastic], ignore_index=True)  # combine

# =========================================================
# 11. Coefficient Shrinkage Plot
# =========================================================

plt.figure(figsize=(12, 10))                           # figure size
sns.barplot(data=coef_all, x="coef", y="feature", hue="model")  # barplot
plt.title("Coefficient Shrinkage – Ridge vs Lasso vs Elastic Net (Median Salary)")  # title
plt.tight_layout()                                     # layout
plt.show()                                             # display

# =========================================================
# 12. ROC Curves for All Models (Full)
# =========================================================

plt.figure(figsize=(8, 6)) # figure size

for name, model in [
    ("Ridge (Full)", ridge_model_full),
    ("Lasso (Full)", lasso_model_full),
    ("Elastic Net (Full)", elastic_model_full)
]:
    y_prob = model.predict_proba(X_test)[:, 1] # predicted probabilities
    fpr, tpr, _ = roc_curve(y_test, y_prob) # ROC curve
    auc = roc_auc_score(y_test, y_prob) # AUC score
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})") # plot curve

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – Regularized Logistic Models (Full Model)")
plt.legend()
plt.tight_layout()
plt.show()



# =========================================================
# Regularized Logistic Regression – Reduced Final Model
# =========================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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
# 6. Ridge, Lasso, Elastic Net Models (Reduced Model)
# =========================================================

ridge_model_reduced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=2000))
])

lasso_model_reduced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        penalty="l1", C=1.0, solver="liblinear", max_iter=2000))
])

elastic_model_reduced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        penalty="elasticnet", C=1.0, l1_ratio=0.5,
        solver="saga", max_iter=3000))
])

# =========================================================
# 7. Fit All Models
# =========================================================

ridge_model_reduced.fit(X_train, y_train)
lasso_model_reduced.fit(X_train, y_train)
elastic_model_reduced.fit(X_train, y_train)

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
# 9. Build Comparison Table
# =========================================================

results = []
results.append(eval_model("Ridge (Reduced)", ridge_model_reduced, X_test, y_test))
results.append(eval_model("Lasso (Reduced)", lasso_model_reduced, X_test, y_test))
results.append(eval_model("Elastic Net (Reduced)", elastic_model_reduced, X_test, y_test))

results_df = pd.DataFrame(results)
print("\n----- REGULARIZED LOGISTIC MODELS (REDUCED FINAL MODEL) -----")
print(results_df)

# =========================================================
# 10. Extract Coefficients for Shrinkage Plot
# =========================================================

def get_coefficients(model, X_train, name):
    num_features = preprocessor.named_transformers_["numeric"].named_steps["imputer"].get_feature_names_out(
        X_train.select_dtypes(include=["int64", "float64"]).columns
    )
    cat_features = preprocessor.named_transformers_["categorical"].named_steps["encoder"].get_feature_names_out(
        X_train.select_dtypes(include=["object", "category"]).columns
    )
    feature_names = np.concatenate([num_features, cat_features])
    coefs = model.named_steps["classifier"].coef_[0]
    return pd.DataFrame({"feature": feature_names, "coef": coefs, "model": name})

coef_ridge = get_coefficients(ridge_model_reduced, X_train, "Ridge (Reduced)")
coef_lasso = get_coefficients(lasso_model_reduced, X_train, "Lasso (Reduced)")
coef_elastic = get_coefficients(elastic_model_reduced, X_train, "ElasticNet (Reduced)")

coef_all = pd.concat([coef_ridge, coef_lasso, coef_elastic], ignore_index=True)

# =========================================================
# 11. Coefficient Shrinkage Plot
# =========================================================

plt.figure(figsize=(12, 10))
sns.barplot(data=coef_all, x="coef", y="feature", hue="model")
plt.title("Coefficient Shrinkage – Ridge vs Lasso vs Elastic Net (Reduced Model)")
plt.tight_layout()
plt.show()

# =========================================================
# 12. ROC Curves for All Models (Reduced)
# =========================================================

plt.figure(figsize=(8, 6))

for name, model in [
    ("Ridge (Reduced)", ridge_model_reduced),
    ("Lasso (Reduced)", lasso_model_reduced),
    ("Elastic Net (Reduced)", elastic_model_reduced)
]:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – Regularized Logistic Models (Reduced Final Model)")
plt.legend()
plt.tight_layout()
plt.show()

