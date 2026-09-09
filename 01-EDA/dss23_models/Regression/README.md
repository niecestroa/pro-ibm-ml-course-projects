# README — Regression Models

Created: 2026‑07‑21  
Last Edited: 2026‑07‑21  
Author: Aaron Niecestro  

---

## Overview  
This folder contains all regression models used to predict the continuous outcome:

salary_in_usd

The goal of this modeling track is to estimate salary as a numeric value using engineered features derived from the Data Science Salaries 2023 dataset.  
Regression modeling is implemented in Python, R, and SAS, with consistent feature engineering across all three.

---

## Objective  
Build and evaluate models that predict salary_in_usd using:

- experience level  
- employment type  
- job title grouping  
- company size  
- remote ratio  
- geographic variables  
- engineered features  

---

## Feature Engineering  
Shared across Python, R, and SAS:

- experience_label  
- employment_label  
- company_size_label  
- remote_label  
- job_title_grouped  
- region  
- work_year  
- one‑hot encoding  
- scaling (if needed)  
- train/test split  

---

## Models Included

### Python
- Linear Regression  
- Lasso / Ridge / Elastic Net  
- Random Forest Regressor  
- Gradient Boosting / XGBoost  

### R
- GLM  
- tidymodels regression workflow  
- Random Forest  
- Gradient Boosting  

### SAS
- PROC GLMSELECT  
- PROC REG  
- PROC FOREST  

---

## Evaluation Metrics
- RMSE  
- MAE  
- R²  
- Cross‑validation error  
- Residual diagnostics  

---

## Deliverables
- Python (.py), R (.R), and SAS (.sas) regression scripts  
- Feature importance outputs  
- Residual analysis  
- Model comparison summaries  
