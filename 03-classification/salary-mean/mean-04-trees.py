# -*- coding: utf-8 -*-

'''
Author: Aaron Niecestro  
Project: DS Salaries – Tree based Methods
Created: September 29, 2026  
Last Edit: September 29, 2026  
Progress: Completed

---

Description:
This section introduces a suite of tree‑based classification models for 
predicting mean salary category. These models split the data into 
hierarchical decision regions, allowing them to capture nonlinear relationships, 
feature interactions, and complex boundaries that linear models may miss. 
Because tree algorithms rely on threshold‑based splits rather than distance 
calculations, they naturally handle mixed numeric and categorical features 
without requiring scaling. The block trains and evaluates five major 
tree‑based methods — Decision Tree, Random Forest, Extra Trees, Gradient Boosting, 
and AdaBoost — and compares their performance using accuracy, precision, recall, 
F1 score, and ROC‑AUC. ROC curves are plotted to visualize model discrimination 
ability across all tree‑based approaches.

This is a masters copy of BOTH: 1. Full Model and 2. The Reduced Model
'''

# -*- coding: utf-8 -*-  # encoding

# =========================================================
# Mean Salary Classification – Steps 1 to 6 (Tree-Based Models Included)
# =========================================================

import pandas as pd                      # data handling
import numpy as np                       # numerical ops
import seaborn as sns                    # visualization
import matplotlib.pyplot as plt          # plotting

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)  # metrics
from sklearn.model_selection import train_test_split              # train/test split
from sklearn.preprocessing import StandardScaler, OneHotEncoder   # scaling + encoding
from sklearn.compose import ColumnTransformer                     # preprocessing
from sklearn.pipeline import Pipeline                             # ML pipeline
from sklearn.impute import SimpleImputer                         # imputation

sns.set(style="whitegrid", context="talk")                        # seaborn style

# =========================================================
# 1. Load Data
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"  # dataset path
dss2025 = pd.read_csv(file_path)                        # load dataset
dss2025 = dss2025.query("company_location == 'United States'").copy()  # USA-only filter

# =========================================================
# 2. Encode Mean Salary Category
# =========================================================

dss2025["salary_mean_cat"] = dss2025["salary_mean_cat"].map({
    "Below-Average": 0,
    "Above-Average": 1
})                                                       # convert to 0/1

# =========================================================
# 3. Define Response + Predictors
# =========================================================

y = dss2025["salary_mean_cat"]                           # binary response
X = dss2025[
    ['experience_level', 'employment_type','employee_residence',
     'remote_ratio', 'company_size', 'data_age',
     'work_year_cat', 'remote_work_cat', 'job_title_group']
]                                                        # predictor matrix

# =========================================================
# 4. Train/Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=72018, stratify=y
)                                                        # stratified split

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
)                                                        # preprocessing transformer

# =========================================================
# 6. Tree-Based Models (Decision Tree, RF, Extra Trees, GB, AdaBoost)
# =========================================================

from sklearn.tree import DecisionTreeClassifier                     # decision tree
from sklearn.ensemble import RandomForestClassifier                 # random forest
from sklearn.ensemble import ExtraTreesClassifier                   # extra trees
from sklearn.ensemble import GradientBoostingClassifier             # gradient boosting
from sklearn.ensemble import AdaBoostClassifier                     # adaboost

# ---------------------------------------------------------
# Build all tree-based pipelines
# ---------------------------------------------------------

tree_model_full = Pipeline([
    ("preprocessor", preprocessor),                                 # preprocessing
    ("classifier", DecisionTreeClassifier(random_state=72018))      # decision tree
])

rf_model_full = Pipeline([
    ("preprocessor", preprocessor),                                 # preprocessing
    ("classifier", RandomForestClassifier(
        n_estimators=300, random_state=72018))                       # random forest
])

extra_model_full = Pipeline([
    ("preprocessor", preprocessor),                                 # preprocessing
    ("classifier", ExtraTreesClassifier(
        n_estimators=300, random_state=72018))                       # extra trees
])

gb_model_full = Pipeline([
    ("preprocessor", preprocessor),                                 # preprocessing
    ("classifier", GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, random_state=72018))   # gradient boosting
])

ada_model_full = Pipeline([
    ("preprocessor", preprocessor),                                 # preprocessing
    ("classifier", AdaBoostClassifier(
        n_estimators=300, learning_rate=0.05, random_state=72018))   # adaboost
])

# ---------------------------------------------------------
# Fit all tree-based models
# ---------------------------------------------------------

tree_model_full.fit(X_train, y_train)                                    # fit decision tree
rf_model_full.fit(X_train, y_train)                                      # fit random forest
extra_model_full.fit(X_train, y_train)                                   # fit extra trees
gb_model_full.fit(X_train, y_train)                                      # fit gradient boosting
ada_model_full.fit(X_train, y_train)                                     # fit adaboost

# ---------------------------------------------------------
# Evaluation helper function
# ---------------------------------------------------------

def eval_tree(name, model, X_test, y_test):
    y_pred = model.predict(X_test)                                  # predicted classes
    y_prob = model.predict_proba(X_test)[:, 1]                      # predicted probabilities
    return {                                                        # return metrics
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob)
    }

# ---------------------------------------------------------
# Build comparison table
# ---------------------------------------------------------

tree_results = []                                                   # results list
tree_results.append(eval_tree("Decision Tree", tree_model_full, X_test, y_test))
tree_results.append(eval_tree("Random Forest", rf_model_full, X_test, y_test))
tree_results.append(eval_tree("Extra Trees", extra_model_full, X_test, y_test))
tree_results.append(eval_tree("Gradient Boosting", gb_model_full, X_test, y_test))
tree_results.append(eval_tree("AdaBoost", ada_model_full, X_test, y_test))

tree_df = pd.DataFrame(tree_results)                                # convert to dataframe

print("\n----- TREE-BASED MODELS (MEAN SALARY) -----")
print(tree_df)                                                      # show comparison table

# ---------------------------------------------------------
# ROC curves for all tree-based models
# ---------------------------------------------------------

plt.figure(figsize=(10, 7))                                         # figure size

for name, model in [
    ("Decision Tree", tree_model_full),
    ("Random Forest", rf_model_full),
    ("Extra Trees", extra_model_full),
    ("Gradient Boosting", gb_model_full),
    ("AdaBoost", ada_model_full)
]:
    y_prob = model.predict_proba(X_test)[:, 1]                      # predicted probabilities
    fpr, tpr, _ = roc_curve(y_test, y_prob)                         # ROC curve
    auc = roc_auc_score(y_test, y_prob)                             # AUC score
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")             # plot curve

plt.plot([0, 1], [0, 1], "k--", label="Random")                     # random baseline
plt.xlabel("False Positive Rate")                                   # x-axis label
plt.ylabel("True Positive Rate")                                    # y-axis label
plt.title("ROC Curves – Tree-Based Models (Mean Salary)")           # title
plt.legend()                                                        # legend
plt.tight_layout()                                                  # layout
plt.show()                                                          # display plot

# =========================================================
# Feature Importance – Random Forest / Extra Trees / Gradient Boosting
# =========================================================

def plot_feature_importance(model, model_name):
    # get feature names from preprocessor
    num_features = preprocessor.named_transformers_["numeric"].named_steps["imputer"].get_feature_names_out(
        X_train.select_dtypes(include=["int64", "float64"]).columns
    )
    cat_features = preprocessor.named_transformers_["categorical"].named_steps["encoder"].get_feature_names_out(
        X_train.select_dtypes(include=["object", "category"]).columns
    )
    feature_names = np.concatenate([num_features, cat_features])

    # extract importance values
    importances = model.named_steps["classifier"].feature_importances_

    # build dataframe
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    # plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=imp_df.head(20), x="importance", y="feature")
    plt.title(f"Top Feature Importances – {model_name}")
    plt.tight_layout()
    plt.show()

# Plot for Random Forest
plot_feature_importance(rf_model_full, "Random Forest")

# Plot for Extra Trees
plot_feature_importance(extra_model_full, "Extra Trees")

# Plot for Gradient Boosting
plot_feature_importance(gb_model_full, "Gradient Boosting")

# =========================================================
# 7. SHAP Interpretability (Final Robust Version)
# =========================================================

import shap                                                # SHAP library
shap.initjs()                                              # initialize JS

# ---------------------------------------------------------
# Prepare transformed training data
# ---------------------------------------------------------

X_train_transformed = preprocessor.transform(X_train)      # transformed matrix

feature_names = np.concatenate([
    preprocessor.named_transformers_["numeric"].named_steps["imputer"].get_feature_names_out(
        X_train.select_dtypes(include=["int64", "float64"]).columns
    ),
    preprocessor.named_transformers_["categorical"].named_steps["encoder"].get_feature_names_out(
        X_train.select_dtypes(include=["object", "category"]).columns
    )
])                                                         # combined feature names

# ---------------------------------------------------------
# Use a smaller background sample (prevents stalling)
# ---------------------------------------------------------

background = shap.sample(X_train_transformed, 200, random_state=72018)

# ---------------------------------------------------------
# Use Gradient Boosting for SHAP (much faster than RF)
# ---------------------------------------------------------

gb_clf = gb_model_full.named_steps["classifier"]                # extract GB classifier
explainer = shap.TreeExplainer(gb_clf, model_output="raw") # raw output avoids mismatch
shap_values = explainer.shap_values(background)            # compute SHAP values

# ---------------------------------------------------------
# Handle binary vs multiclass SHAP output
# ---------------------------------------------------------

# Case 1: binary classifier → shap_values is 2D
if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
    shap_matrix = shap_values                              # use directly

# Case 2: multiclass classifier → shap_values is list of arrays
elif isinstance(shap_values, list):
    shap_matrix = shap_values[1]                           # class 1 = Above-Average salary

# Case 3: SHAP returns 3D array (rare)
elif shap_values.ndim == 3:
    shap_matrix = shap_values[:, :, :-1]                   # drop bias column if present

# ---------------------------------------------------------
# SHAP Summary Plot (global importance)
# ---------------------------------------------------------

shap.summary_plot(shap_matrix, background, feature_names=feature_names)
plt.show()

# ---------------------------------------------------------
# SHAP Beeswarm Plot (distribution)
# ---------------------------------------------------------

shap.summary_plot(shap_matrix, background, feature_names=feature_names, plot_type="dot")
plt.show()

# ---------------------------------------------------------
# SHAP Bar Plot (mean absolute importance)
# ---------------------------------------------------------

shap.summary_plot(shap_matrix, background, feature_names=feature_names, plot_type="bar")
plt.show()


# =========================================================
# Tree-Based Models – Reduced Final Model
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
# 3. Define Response + REDUCED Predictors
# =========================================================

final_predictors = [
    'experience_level', 'employment_type', 'job_title_group',
    'employee_residence', 'company_size', 'work_year_cat', 'remote_work_cat'
]

X = dss2025[final_predictors]
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
# 6. Tree-Based Models (Decision Tree, RF, Extra Trees, GB, AdaBoost)
# =========================================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

tree_model_reduced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=72018))
])

rf_model_reduced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300, random_state=72018))
])

extra_model_reduced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", ExtraTreesClassifier(
        n_estimators=300, random_state=72018))
])

gb_model_reduced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, random_state=72018))
])

ada_model_reduced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", AdaBoostClassifier(
        n_estimators=300, learning_rate=0.05, random_state=72018))
])

# =========================================================
# 7. Fit All Tree-Based Models
# =========================================================

tree_model_reduced.fit(X_train, y_train)
rf_model_reduced.fit(X_train, y_train)
extra_model_reduced.fit(X_train, y_train)
gb_model_reduced.fit(X_train, y_train)
ada_model_reduced.fit(X_train, y_train)

# =========================================================
# 8. Evaluation Helper
# =========================================================

def eval_tree(name, model, X_test, y_test):
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

tree_results = []
tree_results.append(eval_tree("Decision Tree", tree_model_reduced, X_test, y_test))
tree_results.append(eval_tree("Random Forest", rf_model_reduced, X_test, y_test))
tree_results.append(eval_tree("Extra Trees", extra_model_reduced, X_test, y_test))
tree_results.append(eval_tree("Gradient Boosting", gb_model_reduced, X_test, y_test))
tree_results.append(eval_tree("AdaBoost", ada_model_reduced, X_test, y_test))

tree_df = pd.DataFrame(tree_results)

print("\n----- TREE-BASED MODELS (REDUCED FINAL MODEL) -----")
print(tree_df)

# =========================================================
# 10. ROC Curves for All Tree-Based Models
# =========================================================

plt.figure(figsize=(10, 7))

for name, model in [
    ("Decision Tree", tree_model_reduced),
    ("Random Forest", rf_model_reduced),
    ("Extra Trees", extra_model_reduced),
    ("Gradient Boosting", gb_model_reduced),
    ("AdaBoost", ada_model_reduced)
]:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – Tree-Based Models (Reduced Final Model)")
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# 11. Feature Importance Plots (RF, Extra Trees, GB)
# =========================================================

def plot_feature_importance(model, model_name):
    num_features = preprocessor.named_transformers_["numeric"].named_steps["imputer"].get_feature_names_out(
        X_train.select_dtypes(include=["int64", "float64"]).columns
    )
    cat_features = preprocessor.named_transformers_["categorical"].named_steps["encoder"].get_feature_names_out(
        X_train.select_dtypes(include=["object", "category"]).columns
    )
    feature_names = np.concatenate([num_features, cat_features])

    importances = model.named_steps["classifier"].feature_importances_

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=imp_df.head(20), x="importance", y="feature")
    plt.title(f"Top Feature Importances – {model_name}")
    plt.tight_layout()
    plt.show()

plot_feature_importance(rf_model_reduced, "Random Forest")
plot_feature_importance(extra_model_reduced, "Extra Trees")
plot_feature_importance(gb_model_reduced, "Gradient Boosting")

# =========================================================
# 12. SHAP Interpretability (Reduced Model)
# =========================================================

import shap
shap.initjs()

X_train_transformed = preprocessor.transform(X_train)

feature_names = np.concatenate([
    preprocessor.named_transformers_["numeric"].named_steps["imputer"].get_feature_names_out(
        X_train.select_dtypes(include=["int64", "float64"]).columns
    ),
    preprocessor.named_transformers_["categorical"].named_steps["encoder"].get_feature_names_out(
        X_train.select_dtypes(include=["object", "category"]).columns
    )
])

background = shap.sample(X_train_transformed, 200, random_state=72018)

gb_clf = gb_model_reduced.named_steps["classifier"]
explainer = shap.TreeExplainer(gb_clf, model_output="raw")
shap_values = explainer.shap_values(background)

if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
    shap_matrix = shap_values
elif isinstance(shap_values, list):
    shap_matrix = shap_values[1]
elif shap_values.ndim == 3:
    shap_matrix = shap_values[:, :, :-1]

shap.summary_plot(shap_matrix, background, feature_names=feature_names)
plt.show()

shap.summary_plot(shap_matrix, background, feature_names=feature_names, plot_type="dot")
plt.show()

shap.summary_plot(shap_matrix, background, feature_names=feature_names, plot_type="bar")
plt.show()
