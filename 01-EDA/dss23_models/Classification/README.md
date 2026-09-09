# README — Classification Models

Created: 2026‑07‑22  
Last Edited: 2026‑07‑22  
Author: Aaron Niecestro  

---

## Overview  
This folder contains all classification models used to predict the engineered categorical outcome:

salary_category

This modeling track converts salary into discrete tiers (Low, Medium, High) and builds classification models across Python, R, and SAS.

---

## Objective  
Build and evaluate models that classify salary into categories based on:

- experience level  
- employment type  
- job title grouping  
- company size  
- remote ratio  
- geographic variables  
- engineered features  

---

## Salary Category Definition  
Example tertile‑based binning:

- Low Salary — bottom 33%  
- Medium Salary — middle 33%  
- High Salary — top 33%  

(Exact binning code is included in each language folder.)

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

## Folder Structure

```
classification/
│
├── python/        # .py scripts and optional .ipynb notebooks
├── r/             # .R scripts
└── sas/           # .sas programs
```

---

## Models Included

### Python
- Logistic Regression  
- Random Forest Classifier  
- Gradient Boosting / XGBoost Classifier  

### R
- Logistic Regression  
- tidymodels classification workflow  
- Random Forest  
- Gradient Boosting  

### SAS
- PROC LOGISTIC  
- PROC FOREST  

---

## Evaluation Metrics
- Accuracy  
- F1 Score  
- Precision / Recall  
- ROC‑AUC  
- Confusion Matrix  

---

## Deliverables
- Python (.py), R (.R), and SAS (.sas) classification scripts  
- Confusion matrices  
- ROC curves  
- Feature importance outputs  
- Model comparison summaries  
