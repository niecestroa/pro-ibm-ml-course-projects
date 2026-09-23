# -*- coding: utf-8 -*-

"""
Author:     Aaron Niecestro
Project:    DS Salaries – Regression Tree Based Methods

# Created:    September 21 ,2026
# Last Edit:  September 23, 2026
Progress:     Completed

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

import numpy as np                          # numerical operations
import pandas as pd                         # data handling
import matplotlib.pyplot as plt             # plotting
import seaborn as sns                       # visualization

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
    HAVE_XGB = True                                      # flag if XGBoost available
except ImportError:
    HAVE_XGB = False                                     # flag if XGBoost missing

# optional LightGBM
try:
    from lightgbm import LGBMRegressor                   # LightGBM model
    HAVE_LGBM = True                                     # flag if LightGBM available
except ImportError:
    HAVE_LGBM = False                                    # flag if LightGBM missing

sns.set(style="whitegrid", context="talk")               # seaborn style

# =========================================================
# Load Data (USA-only)
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
                                                # dataset path
df = pd.read_csv(file_path)                              # load dataset
df = df.query("company_location == 'United States'").copy()
                                                # filter USA rows

# =========================================================
# Define Response and Predictors
# =========================================================

y_resp = df["salary_in_usd"]                             # response variable
x_pred = df[[                                            # predictor matrix
    'experience_level', 'employment_type', 'job_title',
    'employee_residence', 'remote_ratio',
    'company_location', 'company_size', 'data_age',
    'work_year_cat', 'remote_work_cat', 'job_title_group'
]]

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
# Metrics Utility (RMSE, MAE, R², Adjusted R²)
# =========================================================

def compute_metrics(y_true, y_pred, p):                   # compute metrics including adjusted R²
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))    # RMSE
    mae = mean_absolute_error(y_true, y_pred)             # MAE
    r2 = r2_score(y_true, y_pred)                         # R²
    n = len(y_true)                                       # sample size
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))       # adjusted R² formula
    return rmse, mae, r2, adj_r2                          # return all metrics

# =========================================================
# Define Models
# =========================================================

models = {}                                               # dictionary of models

models["Decision Tree"] = DecisionTreeRegressor(
    max_depth=None, random_state=72018                    # unpruned decision tree
)

models["Random Forest"] = RandomForestRegressor(
    n_estimators=300, max_features="sqrt", random_state=72018, n_jobs=-1
)                                                        # random forest settings

models["Gradient Boosting"] = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=3, random_state=72018
)                                                        # gradient boosting settings

if HAVE_XGB:                                             # add XGBoost if available
    models["XGBoost"] = XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=72018, n_jobs=-1
    )

if HAVE_LGBM:                                            # add LightGBM if available
    models["LightGBM"] = LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=72018
    )

# =========================================================
# Fit Models + Collect Metrics
# =========================================================

results = []                                              # store metrics
models_store = {}                                         # store pipelines + predictions

for name, model in models.items():                        # loop through models
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),              # preprocessing step
            ("model", model)                             # tree-based model
        ]
    )

    pipe.fit(X_train, y_train)                           # fit model
    y_pred = pipe.predict(X_test)                        # predict on test set

    X_test_t = pipe.named_steps["preprocessor"].transform(X_test)
                                                        # transformed test predictors
    X_test_t = X_test_t.toarray() if hasattr(X_test_t, "toarray") else X_test_t
                                                        # convert sparse to dense if needed
    p = X_test_t.shape[1]                               # number of encoded predictors

    rmse, mae, r2, adj_r2 = compute_metrics(y_test, y_pred, p)
                                                        # compute metrics including adjusted R²

    results.append({
        "Model": name,                                  # model name
        "RMSE": rmse,                                   # RMSE
        "MAE": mae,                                     # MAE
        "R²": r2,                                       # R²
        "Adj R²": adj_r2                                # adjusted R²
    })

    models_store[name] = {"pipeline": pipe, "y_pred": y_pred}
                                                        # store pipeline and predictions

# =========================================================
# Comparison Table
# =========================================================

results_df = pd.DataFrame(results)                       # metrics table
print("\n===== TREE-ENSEMBLE MODEL COMPARISON =====")     # header
print(results_df)                                        # print metrics

# =========================================================
# Actual vs Predicted (Best Model)
# =========================================================

best_row = results_df.loc[results_df["RMSE"].idxmin()]   # select model with lowest RMSE
best_name = best_row["Model"]                            # extract model name
best_pred = models_store[best_name]["y_pred"]            # retrieve predictions

plt.figure(figsize=(10, 6))                              # set figure size
sns.scatterplot(x=y_test, y=best_pred, alpha=0.6)        # scatter plot of actual vs predicted
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", linewidth=2)
                                                         # 45-degree reference line
plt.title(f"{best_name}: Actual vs Predicted")           # plot title
plt.xlabel("Actual Salary")                              # x-axis label
plt.ylabel("Predicted Salary")                           # y-axis label
plt.tight_layout()                                       # adjust layout
plt.show()                                               # display plot

# =========================================================
# Residual Plot (Best Model)
# =========================================================

residuals = y_test - best_pred                           # compute residuals

plt.figure(figsize=(10, 6))                              # set figure size
sns.scatterplot(x=best_pred, y=residuals, alpha=0.6)     # residuals vs predicted plot
plt.axhline(0, color="red", linestyle="--")              # horizontal zero line
plt.title(f"{best_name}: Residuals vs Predicted")        # plot title
plt.xlabel("Predicted Salary")                           # x-axis label
plt.ylabel("Residuals")                                  # y-axis label
plt.tight_layout()                                       # adjust layout
plt.show()                                               # display plot

# =========================================================
# Feature Name Extraction
# =========================================================

def get_feature_names(preprocessor):                      # function to extract feature names
    feature_names = []                                    # list to store names
    feature_names.extend(preprocessor.transformers_[0][2])# add numeric feature names
    ohe = preprocessor.transformers_[1][1]                # retrieve OHE encoder
    cat_cols = preprocessor.transformers_[1][2]           # categorical columns
    ohe_names = ohe.get_feature_names_out(cat_cols)       # encoded categorical names
    feature_names.extend(ohe_names)                       # append encoded names
    return feature_names                                  # return full feature list

# =========================================================
# Random Forest Feature Importance
# =========================================================

if "Random Forest" in models_store:                      # check if RF model exists
    rf_pipe = models_store["Random Forest"]["pipeline"]  # retrieve RF pipeline
    rf_model = rf_pipe.named_steps["model"]              # retrieve RF model
    rf_features = get_feature_names(preprocessor)        # get feature names

    rf_importance = pd.Series(
        rf_model.feature_importances_, index=rf_features
    ).sort_values(ascending=False).head(20)              # top 20 important features

    plt.figure(figsize=(10, 8))                          # set figure size
    sns.barplot(x=rf_importance.values, y=rf_importance.index, palette="viridis")
                                                         # bar plot of importances
    plt.title("Random Forest — Top 20 Feature Importances")  # plot title
    plt.xlabel("Importance")                             # x-axis label
    plt.ylabel("Feature")                                # y-axis label
    plt.tight_layout()                                   # adjust layout
    plt.show()                                           # display plot

# =========================================================
# XGBoost Feature Importance
# =========================================================

if HAVE_XGB and "XGBoost" in models_store:               # check if XGB model exists
    xgb_pipe = models_store["XGBoost"]["pipeline"]       # retrieve XGB pipeline
    xgb_model = xgb_pipe.named_steps["model"]            # retrieve XGB model
    xgb_features = get_feature_names(preprocessor)       # get feature names

    xgb_importance = pd.Series(
        xgb_model.feature_importances_, index=xgb_features
    ).sort_values(ascending=False).head(20)              # top 20 important features

    plt.figure(figsize=(10, 8))                          # set figure size
    sns.barplot(x=xgb_importance.values, y=xgb_importance.index, palette="magma")
                                                         # bar plot of importances
    plt.title("XGBoost — Top 20 Feature Importances")    # plot title
    plt.xlabel("Importance")                             # x-axis label
    plt.ylabel("Feature")                                # y-axis label
    plt.tight_layout()                                   # adjust layout
    plt.show()                                           # display plot

# =========================================================
# LightGBM Feature Importance
# =========================================================

if HAVE_LGBM and "LightGBM" in models_store:             # check if LGBM model exists
    lgbm_pipe = models_store["LightGBM"]["pipeline"]     # retrieve LGBM pipeline
    lgbm_model = lgbm_pipe.named_steps["model"]          # retrieve LGBM model
    lgbm_features = get_feature_names(preprocessor)      # get feature names

    lgbm_importance = pd.Series(
        lgbm_model.feature_importances_, index=lgbm_features
    ).sort_values(ascending=False).head(20)              # top 20 important features

    plt.figure(figsize=(10, 8))                          # set figure size
    sns.barplot(x=lgbm_importance.values, y=lgbm_importance.index, palette="cubehelix")
                                                         # bar plot of importances
    plt.title("LightGBM — Top 20 Feature Importances")   # plot title
    plt.xlabel("Importance")                             # x-axis label
    plt.ylabel("Feature")                                # y-axis label
    plt.tight_layout()                                   # adjust layout
    plt.show()                                           # display plot

# =========================================================
# SHAP + Permutation Importance + Pruned Regression Tree
# =========================================================

import shap                                               # SHAP library for model explanations
from sklearn.inspection import permutation_importance      # permutation importance function
from sklearn.tree import DecisionTreeRegressor, plot_tree  # regression tree + plotting utilities

# =========================================================
# 1. Choose Best Model (based on RMSE)
# =========================================================

best_row = results_df.loc[results_df["RMSE"].idxmin()]     # select model with lowest RMSE
best_name = best_row["Model"]                              # extract model name
best_pipe = models_store[best_name]["pipeline"]            # retrieve pipeline for best model
best_model = best_pipe.named_steps["model"]                # retrieve underlying model

print(f"\n===== BEST MODEL FOR INTERPRETATION: {best_name} =====")  # display best model name

# =========================================================
# 2. Extract Feature Names
# =========================================================

feature_names = get_feature_names(preprocessor)            # unified feature names list

# =========================================================
# 3. SHAP Explainer (TreeExplainer for tree models)
# =========================================================

explainer = shap.TreeExplainer(best_model)                 # SHAP tree-based explainer
X_test_transformed = best_pipe.named_steps["preprocessor"].transform(X_test)
                                                         # transform test predictors
shap_values = explainer.shap_values(X_test_transformed)    # compute SHAP values

# =========================================================
# 4. SHAP Global Summary Plot
# =========================================================

plt.figure(figsize=(10, 8))                                # set figure size
shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names)
                                                         # SHAP summary plot
plt.title(f"SHAP Summary Plot — {best_name}")             # plot title
plt.show()                                               # display plot

# =========================================================
# 5. SHAP Local Explanation (Force Plot for one observation)
# =========================================================

idx = 0                                                   # index of sample to explain
shap.initjs()                                             # initialize JS for force plot
shap.force_plot(
    explainer.expected_value,                             # baseline expected value
    shap_values[idx, :],                                  # SHAP values for sample
    X_test_transformed[idx, :],                           # encoded sample values
    feature_names=feature_names                           # feature names
)

# =========================================================
# 6. Permutation Importance (Model-Agnostic)
# =========================================================

perm = permutation_importance(
    best_pipe, X_test, y_test, n_repeats=10, random_state=72018, n_jobs=-1
)                                                        # compute permutation importance

perm_df = pd.DataFrame({
    "Feature": feature_names,                             # feature names
    "Importance": perm.importances_mean                   # mean importance scores
}).sort_values("Importance", ascending=False)             # sort descending

print("\n===== PERMUTATION IMPORTANCE (MODEL-AGNOSTIC) =====")  # header
print(perm_df.head(20))                                   # display top 20 features

plt.figure(figsize=(10, 8))                               # set figure size
sns.barplot(data=perm_df.head(20), x="Importance", y="Feature", palette="coolwarm")
                                                         # bar plot of permutation importance
plt.title(f"Permutation Importance — {best_name}")        # plot title
plt.tight_layout()                                       # adjust layout
plt.show()                                               # display plot

# =========================================================
# SHAP vs Permutation Importance Comparison Table
# =========================================================

shap_abs_mean = np.abs(shap_values).mean(axis=0)          # compute mean absolute SHAP values

shap_df = pd.DataFrame({
    "Feature": feature_names,                             # feature names
    "SHAP_Importance": shap_abs_mean                      # SHAP importance values
}).sort_values("SHAP_Importance", ascending=False)        # sort descending

compare_df = shap_df.merge(
    perm_df.rename(columns={"Importance": "Permutation_Importance"}),
    on="Feature",
    how="inner"
)                                                        # merge SHAP + permutation importance

compare_df = compare_df.sort_values("SHAP_Importance", ascending=False)
                                                         # sort by SHAP importance

print(f"\n===== SHAP vs Permutation Importance — {best_name} =====")  # header
print(compare_df.head(25))                               # display top 25 features

plt.figure(figsize=(12, 10))                             # set figure size
sns.scatterplot(
    data=compare_df.head(20),                            # top 20 features
    x="SHAP_Importance",                                 # SHAP importance
    y="Permutation_Importance",                          # permutation importance
    hue="Feature",                                       # color by feature
    palette="tab20",                                     # color palette
    s=120                                                # marker size
)
plt.title(f"SHAP vs Permutation Importance — {best_name}")  # plot title
plt.xlabel("Mean |SHAP Value|")                           # x-axis label
plt.ylabel("Permutation Importance")                      # y-axis label
plt.tight_layout()                                        # adjust layout
plt.show()                                                # display plot

# =========================================================
# 7. Final Pruned Regression Tree (Cost-Complexity Pruning)
# =========================================================

# transform training data for pruning
X_train_transformed = best_pipe.named_steps["preprocessor"].transform(X_train)
                                                         # encoded training predictors

# fit a large unpruned tree for pruning path
tree_full = DecisionTreeRegressor(random_state=72018)     # full decision tree
tree_full.fit(X_train_transformed, y_train)               # fit full tree

# compute pruning path (list of effective alphas)
path = tree_full.cost_complexity_pruning_path(X_train_transformed, y_train)
                                                         # compute pruning path
ccp_alphas = path.ccp_alphas                             # extract list of alphas

# fit pruned trees for each alpha
trees = []                                                # list to store pruned trees
for alpha in ccp_alphas:                                  # loop through alphas
    t = DecisionTreeRegressor(random_state=72018, ccp_alpha=alpha)
                                                         # create pruned tree
    t.fit(X_train_transformed, y_train)                  # fit pruned tree
    trees.append(t)                                      # store tree

# evaluate each pruned tree
prune_results = []                                       # list to store RMSE results
for alpha, t in zip(ccp_alphas, trees):                  # loop through trees
    X_test_transformed = best_pipe.named_steps["preprocessor"].transform(X_test)
                                                         # transform test predictors
    y_pred_t = t.predict(X_test_transformed)             # predict test set
    rmse_t = np.sqrt(mean_squared_error(y_test, y_pred_t))
                                                         # compute RMSE
    prune_results.append({"alpha": alpha, "RMSE": rmse_t})
                                                         # store alpha + RMSE

# convert pruning results to DataFrame
prune_df = pd.DataFrame(prune_results).sort_values("RMSE")
                                                         # sort by RMSE ascending

print("\n===== PRUNED TREE PERFORMANCE TABLE =====")       # header
print(prune_df.head())                                   # display top-performing pruned trees

# best pruned tree alpha
best_alpha = prune_df.iloc[0]["alpha"]                   # extract best alpha
best_pruned_tree = DecisionTreeRegressor(
    random_state=72018, ccp_alpha=best_alpha             # create best pruned tree
)
best_pruned_tree.fit(X_train_transformed, y_train)       # fit best pruned tree

print(f"\nBest pruned tree alpha: {best_alpha}")         # print best alpha

# =========================================================
# 8. Plot Final Pruned Regression Tree
# =========================================================

plt.figure(figsize=(20, 12))                             # set figure size
plot_tree(
    best_pruned_tree,                                    # pruned tree model
    feature_names=feature_names,                         # feature names
    filled=True,                                         # color nodes by impurity
    rounded=True,                                        # rounded node boxes
    fontsize=8                                           # font size
)
plt.title("Final Pruned Regression Tree")                # plot title
plt.show()                                               # display plot

# =========================================================
# Export Best Metrics for Each Tree-Based Model (with Adjusted R²)
# =========================================================

# Helper to safely extract metrics including adjusted R²
def get_best(model_name):                                # function to extract metrics
    row = results_df.loc[results_df["Model"] == model_name].iloc[0]
                                                         # select row for model
    return (
        row["RMSE"],                                     # RMSE
        row["MAE"],                                      # MAE
        row["R²"],                                       # R²
        row["Adj R²"]                                    # Adjusted R²
    )

# Random Forest
if "Random Forest" in results_df["Model"].values:        # check RF exists
    best_rf_rmse, best_rf_mae, best_rf_r2, best_rf_adj_r2 = get_best("Random Forest")

# XGBoost
if "XGBoost" in results_df["Model"].values:              # check XGB exists
    best_xgb_rmse, best_xgb_mae, best_xgb_r2, best_xgb_adj_r2 = get_best("XGBoost")

# LightGBM
if "LightGBM" in results_df["Model"].values:             # check LGBM exists
    best_lgbm_rmse, best_lgbm_mae, best_lgbm_r2, best_lgbm_adj_r2 = get_best("LightGBM")

# Decision Tree
if "Decision Tree" in results_df["Model"].values:        # check DT exists
    best_tree_rmse, best_tree_mae, best_tree_r2, best_tree_adj_r2 = get_best("Decision Tree")

# Gradient Boosting
if "Gradient Boosting" in results_df["Model"].values:    # check GB exists
    best_gb_rmse, best_gb_mae, best_gb_r2, best_gb_adj_r2 = get_best("Gradient Boosting")

# =========================================================
# Export variables for import
# =========================================================

__all__ = [
    "best_rf_rmse", "best_rf_mae", "best_rf_r2", "best_rf_adj_r2",
    "best_xgb_rmse", "best_xgb_mae", "best_xgb_r2", "best_xgb_adj_r2",
    "best_lgbm_rmse", "best_lgbm_mae", "best_lgbm_r2", "best_lgbm_adj_r2",
    "best_tree_rmse", "best_tree_mae", "best_tree_r2", "best_tree_adj_r2",
    "best_gb_rmse", "best_gb_mae", "best_gb_r2", "best_gb_adj_r2"
]
