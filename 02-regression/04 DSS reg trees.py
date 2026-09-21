# -*- coding: utf-8 -*-

"""
Author:     Aaron Niecestro
Project:    DS Salaries – Polynomial Regression Analysis

# Created:    September 21 ,2026
# Last Edit:  September 23, 2026
Progress:     Ongoing

Description - PART 3 — Tree‑Based Ensemble Models:

Purpose:
    Model nonlinearities and interactions automatically using tree‑based methods:
        • Decision Tree Regressor
        • Random Forest Regressor
        • Gradient Boosting Regressor
        • XGBoost Regressor
        • LightGBM Regressor

    Includes:
        • Data loading and preprocessing (USA‑only)
        • One‑hot encoding for categorical predictors
        • Train/test split
        • Model fitting and evaluation (RMSE, MAE, R²)
        • Basic comparison table and plots
"""

# =========================================================
# Imports
# =========================================================

import numpy as np                                      # numerical operations
import pandas as pd                                     # data handling
import matplotlib.pyplot as plt                         # plotting
import seaborn as sns                                   # visualization

from sklearn.model_selection import train_test_split     # train/test split
from sklearn.preprocessing import OneHotEncoder          # categorical encoding
from sklearn.compose import ColumnTransformer            # preprocessing transformer
from sklearn.pipeline import Pipeline                    # pipeline

from sklearn.tree import DecisionTreeRegressor           # decision tree
from sklearn.ensemble import RandomForestRegressor       # random forest
from sklearn.ensemble import GradientBoostingRegressor   # gradient boosting

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # metrics

# optional XGBoost
try:
    from xgboost import XGBRegressor                     # XGBoost model
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

# optional LightGBM
try:
    from lightgbm import LGBMRegressor                   # LightGBM model
    HAVE_LGBM = True
except ImportError:
    HAVE_LGBM = False

sns.set(style="whitegrid", context="talk")               # seaborn style

# =========================================================
# Load Data (USA-only)
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
df = pd.read_csv(file_path)                              # load dataset
df = df.query("company_location == 'United States'").copy()  # filter USA rows

# =========================================================
# Define Response and Predictors (your variables)
# =========================================================

y_resp = df["salary_in_usd"]                             # response variable
x_pred = df[[
    'experience_level', 'employment_type', 'job_title',
    'employee_residence', 'remote_ratio',
    'company_location', 'company_size', 'data_age',
    'work_year_cat', 'remote_work_cat', 'job_title_group'
]]                                                       # predictor matrix

# =========================================================
# Identify Numeric and Categorical Predictors
# =========================================================

numeric_features = x_pred.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()                                       # numeric predictors

categorical_features = x_pred.select_dtypes(
    include=["object", "category"]
).columns.tolist()                                       # categorical predictors

# =========================================================
# Train/Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    x_pred, y_resp, test_size=0.30, random_state=72018
)                                                        # 70/30 split

# =========================================================
# Preprocessor (One-Hot Encoding)
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_features),     # numeric passthrough
        ("categorical",
         OneHotEncoder(handle_unknown="ignore", drop="first"),
         categorical_features)                            # one-hot encode categorical
    ]
)

# =========================================================
# Metrics Utility
# =========================================================

def compute_metrics(y_true, y_pred):                      # compute RMSE, MAE, R²
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# =========================================================
# Define Models
# =========================================================

models = {}

models["Decision Tree"] = DecisionTreeRegressor(
    max_depth=None, random_state=72018
)

models["Random Forest"] = RandomForestRegressor(
    n_estimators=300, max_features="sqrt", random_state=72018, n_jobs=-1
)

models["Gradient Boosting"] = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=3, random_state=72018
)

if HAVE_XGB:
    models["XGBoost"] = XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=72018, n_jobs=-1
    )

if HAVE_LGBM:
    models["LightGBM"] = LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=72018
    )

# =========================================================
# Fit Models + Collect Metrics
# =========================================================

results = []                                              # store metrics

for name, model in models.items():                        # loop through models
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),              # preprocessing
            ("model", model)                             # tree-based model
        ]
    )

    pipe.fit(X_train, y_train)                           # fit model
    y_pred = pipe.predict(X_test)                        # predict

    rmse, mae, r2 = compute_metrics(y_test, y_pred)      # compute metrics

    results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R²": r2})

    models[name] = {"pipeline": pipe, "y_pred": y_pred}  # store pipeline + preds

# =========================================================
# Comparison Table
# =========================================================

results_df = pd.DataFrame(results)                       # metrics table
print("\n===== TREE-ENSEMBLE MODEL COMPARISON =====")
print(results_df)

# =========================================================
# Actual vs Predicted (Best Model)
# =========================================================

best_row = results_df.loc[results_df["RMSE"].idxmin()]   # best model by RMSE
best_name = best_row["Model"]                            # model name
best_pred = models[best_name]["y_pred"]                  # predictions

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=best_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", linewidth=2)
plt.title(f"{best_name}: Actual vs Predicted")
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.tight_layout()
plt.show()

# =========================================================
# Residual Plot (Best Model)
# =========================================================

residuals = y_test - best_pred                           # residuals

plt.figure(figsize=(10, 6))
sns.scatterplot(x=best_pred, y=residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.title(f"{best_name}: Residuals vs Predicted")
plt.xlabel("Predicted Salary")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()

# =========================================================
# Feature Name Extraction
# =========================================================

def get_feature_names(preprocessor):                      # extract feature names
    feature_names = []
    feature_names.extend(preprocessor.transformers_[0][2])  # numeric
    ohe = preprocessor.transformers_[1][1]                  # OHE encoder
    cat_cols = preprocessor.transformers_[1][2]             # categorical cols
    ohe_names = ohe.get_feature_names_out(cat_cols)         # encoded names
    feature_names.extend(ohe_names)
    return feature_names

# =========================================================
# Random Forest Feature Importance
# =========================================================

if "Random Forest" in models:
    rf_pipe = models["Random Forest"]["pipeline"]
    rf_model = rf_pipe.named_steps["model"]
    rf_features = get_feature_names(preprocessor)

    rf_importance = pd.Series(
        rf_model.feature_importances_, index=rf_features
    ).sort_values(ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=rf_importance.values, y=rf_importance.index, palette="viridis")
    plt.title("Random Forest — Top 20 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

# =========================================================
# XGBoost Feature Importance
# =========================================================

if HAVE_XGB and "XGBoost" in models:
    xgb_pipe = models["XGBoost"]["pipeline"]
    xgb_model = xgb_pipe.named_steps["model"]
    xgb_features = get_feature_names(preprocessor)

    xgb_importance = pd.Series(
        xgb_model.feature_importances_, index=xgb_features
    ).sort_values(ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=xgb_importance.values, y=xgb_importance.index, palette="magma")
    plt.title("XGBoost — Top 20 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

# =========================================================
# LightGBM Feature Importance
# =========================================================

if HAVE_LGBM and "LightGBM" in models:
    lgbm_pipe = models["LightGBM"]["pipeline"]
    lgbm_model = lgbm_pipe.named_steps["model"]
    lgbm_features = get_feature_names(preprocessor)

    lgbm_importance = pd.Series(
        lgbm_model.feature_importances_, index=lgbm_features
    ).sort_values(ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=lgbm_importance.values, y=lgbm_importance.index, palette="cubehelix")
    plt.title("LightGBM — Top 20 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

# =========================================================
# SHAP + Permutation Importance + Pruned Regression Tree
# =========================================================

import shap                                               # SHAP library
from sklearn.inspection import permutation_importance      # permutation importance
from sklearn.tree import DecisionTreeRegressor, plot_tree  # regression tree + plotting

# =========================================================
# 1. Choose Best Model (based on RMSE)
# =========================================================

best_row = results_df.loc[results_df["RMSE"].idxmin()]     # best model by RMSE
best_name = best_row["Model"]                              # model name
best_pipe = models[best_name]["pipeline"]                  # pipeline
best_model = best_pipe.named_steps["model"]                # underlying model

print(f"\n===== BEST MODEL FOR INTERPRETATION: {best_name} =====")

# =========================================================
# 2. Extract Feature Names
# =========================================================

def get_feature_names(preprocessor):                       # extract feature names
    feature_names = []
    feature_names.extend(preprocessor.transformers_[0][2]) # numeric names
    ohe = preprocessor.transformers_[1][1]                 # OHE encoder
    cat_cols = preprocessor.transformers_[1][2]            # categorical cols
    ohe_names = ohe.get_feature_names_out(cat_cols)        # encoded names
    feature_names.extend(ohe_names)
    return feature_names

feature_names = get_feature_names(preprocessor)            # unified feature names

# =========================================================
# 3. SHAP Explainer (TreeExplainer for tree models)
# =========================================================

explainer = shap.TreeExplainer(best_model)                 # SHAP tree explainer
X_test_transformed = best_pipe.named_steps["preprocessor"].transform(X_test)  # encoded test
shap_values = explainer.shap_values(X_test_transformed)    # SHAP values

# =========================================================
# 4. SHAP Global Summary Plot
# =========================================================

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names)
plt.title(f"SHAP Summary Plot — {best_name}")
plt.show()

# =========================================================
# 5. SHAP Local Explanation (Force Plot for one observation)
# =========================================================

idx = 0                                                   # choose first test sample
shap.initjs()                                             # JS for force plot
shap.force_plot(
    explainer.expected_value,
    shap_values[idx, :],
    X_test_transformed[idx, :],
    feature_names=feature_names
)

# =========================================================
# 6. Permutation Importance (Model-Agnostic)
# =========================================================

perm = permutation_importance(
    best_pipe, X_test, y_test, n_repeats=10, random_state=72018, n_jobs=-1
)

perm_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": perm.importances_mean
}).sort_values("Importance", ascending=False)

print("\n===== PERMUTATION IMPORTANCE (MODEL-AGNOSTIC) =====")
print(perm_df.head(20))

plt.figure(figsize=(10, 8))
sns.barplot(data=perm_df.head(20), x="Importance", y="Feature", palette="coolwarm")
plt.title(f"Permutation Importance — {best_name}")
plt.tight_layout()
plt.show()

# =========================================================
# SHAP vs Permutation Importance Comparison Table
# =========================================================

# compute mean absolute SHAP values per feature
shap_abs_mean = np.abs(shap_values).mean(axis=0)              # mean |SHAP| per feature

# build SHAP importance dataframe
shap_df = pd.DataFrame({
    "Feature": feature_names,
    "SHAP_Importance": shap_abs_mean
}).sort_values("SHAP_Importance", ascending=False)

# merge SHAP and permutation importance
compare_df = shap_df.merge(
    perm_df.rename(columns={"Importance": "Permutation_Importance"}),
    on="Feature",
    how="inner"
)

# sort by SHAP importance
compare_df = compare_df.sort_values("SHAP_Importance", ascending=False)

print(f"\n===== SHAP vs Permutation Importance — {best_name} =====")
print(compare_df.head(25))                                   # show top 25 features

# plot comparison (top 20)
plt.figure(figsize=(12, 10))
sns.scatterplot(
    data=compare_df.head(20),
    x="SHAP_Importance",
    y="Permutation_Importance",
    hue="Feature",
    palette="tab20",
    s=120
)
plt.title(f"SHAP vs Permutation Importance — {best_name}")
plt.xlabel("Mean |SHAP Value|")
plt.ylabel("Permutation Importance")
plt.tight_layout()
plt.show()

# =========================================================
# 7. Final Pruned Regression Tree (Cost-Complexity Pruning)
# =========================================================

# fit a large tree for pruning
tree_full = DecisionTreeRegressor(random_state=72018)
tree_full.fit(X_train_transformed := best_pipe.named_steps["preprocessor"].transform(X_train), y_train)

# compute pruning path
path = tree_full.cost_complexity_pruning_path(X_train_transformed, y_train)
ccp_alphas = path.ccp_alphas

# fit pruned trees for each alpha
trees = []
for alpha in ccp_alphas:
    t = DecisionTreeRegressor(random_state=72018, ccp_alpha=alpha)
    t.fit(X_train_transformed, y_train)
    trees.append(t)

# evaluate each pruned tree
prune_results = []
for alpha, t in zip(ccp_alphas, trees):
    y_pred_t = t.predict(best_pipe.named_steps["preprocessor"].transform(X_test))
    rmse_t = np.sqrt(mean_squared_error(y_test, y_pred_t))
    prune_results.append({"alpha": alpha, "RMSE": rmse_t})

prune_df = pd.DataFrame(prune_results).sort_values("RMSE")

print("\n===== PRUNED TREE PERFORMANCE TABLE =====")
print(prune_df.head())

# best pruned tree
best_alpha = prune_df.iloc[0]["alpha"]
best_pruned_tree = DecisionTreeRegressor(random_state=72018, ccp_alpha=best_alpha)
best_pruned_tree.fit(X_train_transformed, y_train)

print(f"\nBest pruned tree alpha: {best_alpha}")

# =========================================================
# 8. Plot Final Pruned Tree
# =========================================================

plt.figure(figsize=(20, 12))
plot_tree(
    best_pruned_tree,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Final Pruned Regression Tree")
plt.show()
