# README — Regression Models

Created: 2026‑07‑20  
Last Edited: 2026‑07‑20  
Author: Aaron Niecestro 

---

## Overview  
This directory contains the **modeling phase** of the *Data Science Salaries 2023* project.  
The preliminary analysis (Python, R, SAS) established clean data, labeled categorical variables, exploratory visualizations, and early insights into salary drivers.

This folder now focuses on **predictive modeling**, using two distinct outcomes:

1. **Continuous Salary Prediction (Regression)**  
2. **Categorical Salary Classification (Classification)**  

Both outcomes use consistent feature engineering across Python, R, and SAS.

---

## Modeling Outcomes

### **1. Regression Outcome — Continuous Salary Prediction**  
**Target:** `salary_in_usd`  
**Goal:** Predict the numeric salary value.

**Models included:**
- Linear Regression  
- Lasso / Ridge / Elastic Net  
- Random Forest Regressor  
- Gradient Boosting / XGBoost  
- R GLM / tidymodels regression  
- SAS PROC GLMSELECT / PROC FOREST  

**Evaluation Metrics:**
- RMSE  
- MAE  
- R²  
- Cross‑validation error  

---

### **2. Classification Outcome — Salary Category Prediction**  
**Target:** `salary_category` (engineered variable)

**Example binning strategy:**
- **Low Salary** — bottom 33%  
- **Medium Salary** — middle 33%  
- **High Salary** — top 33%  

**Models included:**
- Logistic Regression  
- Random Forest Classifier  
- Gradient Boosting / XGBoost Classifier  
- R tidymodels classification  
- SAS PROC LOGISTIC / PROC FOREST  

**Evaluation Metrics:**
- Accuracy  
- F1 Score  
- Precision / Recall  
- ROC‑AUC  
- Confusion Matrix  

---

## Feature Engineering (Shared Across Both Outcomes)

- `experience_label`  
- `employment_label`  
- `company_size_label`  
- `remote_label`  
- `job_title_grouped`  
- `region`  
- `work_year`  
- One‑hot encoding  
- Scaling (if required)  
- Train/test split  

Feature engineering scripts are kept consistent across Python, R, and SAS.

## Deliverables

- **model_results.md** — consolidated comparison across regression and classification  
- Modeling scripts:
  - Python: `.py` files (and optional `.ipynb` notebooks)
  - R: `.R` scripts
  - SAS: `.sas` programs
- Feature importance plots  
- Residual diagnostics (regression)  
- Confusion matrices (classification)  
- Summary of final recommended models  
