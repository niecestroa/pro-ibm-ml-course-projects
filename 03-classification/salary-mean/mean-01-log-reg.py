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

y = dss2025["salary_mean_cat"]                 # binary response variable (0/1)
X = dss2025[                                   # predictor matrix
    ['experience_level', 'employment_type', 'job_title',
     'employee_residence', 'remote_ratio',
     'company_location', 'company_size', 'data_age',
     'work_year_cat', 'remote_work_cat', 'job_title_group']
]

print("----- RESPONSE (salary_mean_cat) -----")  # print header
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

# =========================================================
# 8. Preprocessing (Imputation + Scaling + Encoding)
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric",
         Pipeline([
             ("imputer", SimpleImputer(strategy="median")),     # fill numeric NaNs
             ("scaler", StandardScaler())                       # scale numeric features
         ]),
         numeric_features),

        ("categorical",
         Pipeline([
             ("imputer", SimpleImputer(strategy="most_frequent")),  # fill categorical NaNs
             ("encoder", OneHotEncoder(
                 handle_unknown="ignore",
                 drop="first",
                 sparse_output=False      # force dense output for VIF + DataFrame
             ))
         ]),
         categorical_features)
    ]
)


# =========================================================
# 9. Logistic Regression Pipeline
# =========================================================

log_reg = Pipeline(                               # full ML pipeline
    steps=[
        ("preprocessor", preprocessor),            # apply preprocessing
        ("classifier", LogisticRegression(max_iter=2000))  # logistic regression model
    ]
)

log_reg.fit(X_train, y_train)                     # fit model on training data

# =========================================================
# 10. Predictions + Metrics
# =========================================================

y_pred = log_reg.predict(X_test)                  # predicted classes
y_prob = log_reg.predict_proba(X_test)[:, 1]      # predicted probabilities

print("\n----- MODEL PERFORMANCE -----")          # print header
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")   # accuracy
print(f"Precision: {precision_score(y_test, y_pred):.4f}")  # precision
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")     # recall
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")         # F1 score
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")    # ROC-AUC

print("\n----- CONFUSION MATRIX -----")           # print header
print(confusion_matrix(y_test, y_pred))           # confusion matrix

# =========================================================
# 11. ROC Curve Plot (with labels)
# =========================================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)     # compute ROC curve values

plt.figure(figsize=(8, 6))                           # set figure size
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.3f}")  # plot ROC
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # diagonal reference line
plt.xlabel("False Positive Rate")                    # x-axis label
plt.ylabel("True Positive Rate")                     # y-axis label
plt.title("ROC Curve – Salary Classification (Above vs Below Average)")  # title
plt.legend()                                         # show legend
plt.show()                                           # display plot

# =========================================================
# 12. Precision–Recall Curve Plot (with labels)
# =========================================================

precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)  # compute PR curve

plt.figure(figsize=(8, 6))                           # set figure size
plt.plot(recall, precision, color="blue")            # plot PR curve
plt.xlabel("Recall")                                 # x-axis label
plt.ylabel("Precision")                              # y-axis label
plt.title("Precision–Recall Curve – Salary Classification")  # title
plt.show()                                           # display plot

# =========================================================
# 13. Confusion Matrix Heatmap (with labels)
# =========================================================

cm = confusion_matrix(y_test, y_pred)                # compute confusion matrix

plt.figure(figsize=(6, 5))                           # set figure size
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",   # heatmap with counts
            xticklabels=["Below-Average", "Above-Average"],  # predicted labels
            yticklabels=["Below-Average", "Above-Average"])  # actual labels
plt.xlabel("Predicted Label")                        # x-axis label
plt.ylabel("Actual Label")                           # y-axis label
plt.title("Confusion Matrix – Salary Classification")  # title
plt.show()                                           # display plot

# =========================================================
# 14. Coefficient Plot
# =========================================================

ohe = log_reg.named_steps["preprocessor"].named_transformers_["categorical"]  # get OHE object
encoded_cat_names = ohe.get_feature_names_out(categorical_features)           # encoded feature names

all_feature_names = numeric_features + list(encoded_cat_names)                # full feature list
coefficients = log_reg.named_steps["classifier"].coef_[0]                     # logistic regression coefficients

coef_df = pd.DataFrame({                                                      # coefficient table
    "Feature": all_feature_names,
    "Coefficient": coefficients
}).sort_values("Coefficient", ascending=False)

plt.figure(figsize=(10, 12))                         # set figure size
sns.barplot(
    data=coef_df,
    x="Coefficient",
    y="Feature",
    hue="Feature",          # assign hue to avoid deprecation warning
    dodge=False,            # keep bars aligned
    legend=False,           # hide legend
    palette="viridis"       # color palette
)
plt.title("Logistic Regression Coefficients – Salary Classification")  # title
plt.xlabel("Coefficient Value")                      # x-axis label
plt.ylabel("Feature")                                # y-axis label
plt.show()                                           # display plot

# =========================================================
# Stepwise AIC/BIC Selection for Logistic Regression (R-style)
# =========================================================

def stepAIC_logit(formula, data, direction="both", criterion="AIC"):
    """
    Performs R-style stepwise AIC/BIC selection for logistic regression.
    Supports forward, backward, and both directions.
    Uses statsmodels formula API (handles categorical variables automatically).
    """

    # Split formula into response and predictor list
    response, predictors = formula.split("~")
    response = response.strip()
    predictors = [p.strip() for p in predictors.split("+")]

    # Fit a logistic regression model given a list of predictors
    def fit_model(pred_list):
        if len(pred_list) == 0:
            f = response + " ~ 1"   # intercept-only model
        else:
            f = response + " ~ " + " + ".join(pred_list)

        model = smf.glm(f, data=data, family=sm.families.Binomial()).fit()

        score = model.aic if criterion == "AIC" else model.bic
        return model, score

    # Initialize predictor set
    if direction == "backward":
        current_predictors = predictors.copy()
    elif direction == "forward":
        current_predictors = []
    else:  # both directions
        current_predictors = predictors.copy()

    # Fit initial model
    best_model, best_score = fit_model(current_predictors)
    improved = True

    print(f"\nInitial {criterion}: {best_score:.4f}")
    print(f"Starting predictors: {current_predictors}\n")

    # Stepwise loop
    while improved:
        improved = False
        candidate_models = []

        # BACKWARD elimination
        if direction in ["backward", "both"] and len(current_predictors) > 1:
            for p in current_predictors:
                new_preds = [x for x in current_predictors if x != p]
                model, score = fit_model(new_preds)
                candidate_models.append((model, score, new_preds))

        # FORWARD selection
        if direction in ["forward", "both"]:
            remaining = [p for p in predictors if p not in current_predictors]
            for p in remaining:
                new_preds = current_predictors + [p]
                model, score = fit_model(new_preds)
                candidate_models.append((model, score, new_preds))

        # Evaluate all candidate models
        for model, score, pred_list in candidate_models:
            if score < best_score:  # improvement found
                print(f"Improved {criterion}: {best_score:.4f} → {score:.4f}")
                print(f"New predictors: {pred_list}\n")

                best_model = model
                best_score = score
                current_predictors = pred_list
                improved = True

    print("\nFinal selected predictors:")
    print(current_predictors)

    return best_model

# =========================================================
# Likelihood Ratio Test (LRT) for Each Predictor
# =========================================================

# Transform training data using your pipeline
X_train_transformed = log_reg.named_steps["preprocessor"].transform(X_train)
X_train_transformed = np.asarray(X_train_transformed)  # ensure dense matrix

# Fit full logistic regression model using statsmodels
full_model = sm.Logit(y_train, X_train_transformed).fit(disp=0)

print("\n----- LIKELIHOOD RATIO TESTS FOR EACH PREDICTOR -----")

# Loop through each predictor
for i, feature in enumerate(all_feature_names):

    # Remove one predictor column
    reduced_X = np.delete(X_train_transformed, i, axis=1)

    # Fit reduced model
    reduced_model = sm.Logit(y_train, reduced_X).fit(disp=0)

    # Compute LR statistic
    lr_stat = -2 * (reduced_model.llf - full_model.llf)

    # Compute p-value (df = 1)
    p_value = chi2.sf(lr_stat, df=1)

    # Decision rule
    decision = "Reject H₀ (significant)" if p_value < 0.05 else "Fail to Reject H₀ (not significant)"

    # Print result
    print(f"{feature}: LR stat = {lr_stat:.4f}, p = {p_value:.4f} → {decision}")
    
# =========================================================
# LRT for a Single Predictor
# =========================================================

def lrt_single_predictor(feature_name, X_train_transformed, y_train, all_feature_names):
    """
    Computes the Likelihood Ratio Test (LRT) for a single predictor
    in a logistic regression model.
    """

    # Fit full model
    full_model = sm.Logit(y_train, X_train_transformed).fit(disp=0)

    # Find index of the predictor
    idx = all_feature_names.index(feature_name)

    # Remove that predictor column
    reduced_X = np.delete(X_train_transformed, idx, axis=1)

    # Fit reduced model
    reduced_model = sm.Logit(y_train, reduced_X).fit(disp=0)

    # Compute LR statistic
    lr_stat = -2 * (reduced_model.llf - full_model.llf)

    # Compute p-value (df = 1)
    p_value = chi2.sf(lr_stat, df=1)

    # Decision rule
    decision = "Reject H₀ (significant)" if p_value < 0.05 else "Fail to Reject H₀ (not significant)"

    # Print result
    print(f"\nPredictor: {feature_name}")
    print(f"LR stat = {lr_stat:.4f}")
    print(f"p-value = {p_value:.4f}")
    print(f"Decision: {decision}")

# Transform training data using your pipeline
X_train_transformed = log_reg.named_steps["preprocessor"].transform(X_train)
X_train_transformed = np.asarray(X_train_transformed)

# Run LRT for one predictor
lrt_single_predictor("job_title_group", X_train_transformed, y_train, all_feature_names)

# =========================================================
# LRT for a Set of Predictors (1, 2, 3, or more)
# =========================================================

def lrt_predictor_set(feature_list, X_train_transformed, y_train, all_feature_names):
    """
    Computes the Likelihood Ratio Test (LRT) for a set of predictors
    in a logistic regression model.
    Removes ALL predictors listed in feature_list at once.
    """

    # Fit full model
    full_model = sm.Logit(y_train, X_train_transformed).fit(disp=0)

    # Find indices of predictors to remove
    idx_list = [all_feature_names.index(f) for f in feature_list]

    # Remove those columns
    reduced_X = np.delete(X_train_transformed, idx_list, axis=1)

    # Fit reduced model
    reduced_model = sm.Logit(y_train, reduced_X).fit(disp=0)

    # Compute LR statistic
    lr_stat = -2 * (reduced_model.llf - full_model.llf)

    # Degrees of freedom = number of removed predictors
    df = len(idx_list)

    # Compute p-value
    p_value = chi2.sf(lr_stat, df=df)

    # Decision rule
    decision = "Reject H₀ (significant)" if p_value < 0.05 else "Fail to Reject H₀ (not significant)"

    # Print result
    print("\n----- LRT FOR PREDICTOR SET -----")
    print(f"Removed predictors: {feature_list}")
    print(f"LR stat = {lr_stat:.4f}")
    print(f"Degrees of freedom = {df}")
    print(f"p-value = {p_value:.4f}")
    print(f"Decision: {decision}")

lrt_predictor_set(
    ["remote_ratio", "company_size_M"],
    X_train_transformed,
    y_train,
    all_feature_names
)


# =========================================================
# Final Model – Reduced Logistic Regression
# =========================================================

# Selected predictors based on LRT / Stepwise / AIC/BIC
final_predictors = [
    "experience_level",
    "remote_ratio",
    "company_size",
    "job_title_group"
]

# Subset X to final predictors
X_final = dss2025[final_predictors]

# Train/test split (same seed for reproducibility)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_final, y, test_size=0.30, random_state=72018, stratify=y
)

# =========================================================
# Preprocessing for Reduced Model
# =========================================================

preprocessor_final = ColumnTransformer(
    transformers=[
        ("numeric",
         Pipeline([
             ("imputer", SimpleImputer(strategy="median")),   # numeric imputation
             ("scaler", StandardScaler())                    # numeric scaling
         ]),
         X_final.select_dtypes(include=["int64", "float64"]).columns.tolist()),

        ("categorical",
         Pipeline([
             ("imputer", SimpleImputer(strategy="most_frequent")),  # categorical imputation
             ("encoder", OneHotEncoder(
                 handle_unknown="ignore",
                 drop="first",
                 sparse_output=False
             ))
         ]),
         X_final.select_dtypes(include=["object", "category"]).columns.tolist())
    ]
)

# =========================================================
# Reduced Logistic Regression Pipeline
# =========================================================

final_model = Pipeline([
    ("preprocessor", preprocessor_final),
    ("classifier", LogisticRegression(max_iter=2000))
])

# Fit reduced model
final_model.fit(X_train_f, y_train_f)

# Predictions
y_pred_f = final_model.predict(X_test_f)
y_prob_f = final_model.predict_proba(X_test_f)[:, 1]

# =========================================================
# Final Model Performance
# =========================================================

print("\n----- FINAL MODEL PERFORMANCE -----")
print(f"Accuracy:  {accuracy_score(y_test_f, y_pred_f):.4f}")
print(f"Precision: {precision_score(y_test_f, y_pred_f):.4f}")
print(f"Recall:    {recall_score(y_test_f, y_pred_f):.4f}")
print(f"F1 Score:  {f1_score(y_test_f, y_pred_f):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test_f, y_prob_f):.4f}")

print("\n----- FINAL MODEL CONFUSION MATRIX -----")
print(confusion_matrix(y_test_f, y_pred_f))

# =========================================================
# Assumption Checks – Reduced Logistic Regression
# =========================================================

# 1. Extract transformed training data
X_train_f_transformed = final_model.named_steps["preprocessor"].transform(X_train_f)
X_train_f_transformed = np.asarray(X_train_f_transformed)

# 2. Fit statsmodels logistic regression for inference
sm_model = sm.Logit(y_train_f, X_train_f_transformed).fit(disp=0)

print("\n----- REDUCED MODEL SUMMARY (Statsmodels) -----")
print(sm_model.summary())

# 3. Check for separation or unstable coefficients
print("\n----- COEFFICIENTS (Reduced Model) -----")
coef_reduced = final_model.named_steps["classifier"].coef_[0]
print(coef_reduced)

# 4. Check class balance (important for logistic regression)
print("\n----- CLASS BALANCE (Reduced Model) -----")
print(y_train_f.value_counts(normalize=True))

# 5. Missing-value audit (reduced predictors)
print("\n----- MISSING VALUES (Reduced Predictors) -----")
print(X_final.isnull().sum())
