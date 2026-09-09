# Data Science Job Salaries ML Pipeline (EDA, Regression, Classification)
Professional machine learning pipelines derived from IBM/Coursera Machine Learning Courses 1–3, rebuilt into modular, reproducible, production‑aligned workflows using Python and R.

## Timeline
- Started on:     September 10, 2026
- Last Edited on: September 2026
- Progress:        On-Going

---

# **Overview**

This repository contains **professional, production‑ready machine learning projects** based on the first three courses of the IBM/Coursera Machine Learning Certificate:

1. **Exploratory Data Analysis (EDA)**  
2. **Supervised Machine Learning: Regression**  
3. **Supervised Machine Learning: Classification**

Each academic project was fully refactored into a **professional pipeline**, emphasizing:

- modular code structure  
- reproducible workflows  
- robust evaluation  
- feature engineering  
- model comparison  
- production‑aligned ML practices  

The repo includes multiple datasets, including the **Latest Data Science Job Salaries 2024** dataset from Kaggle, used for EDA and supervised learning tasks.

---

# **Dataset: Data Science Job Salaries 2024**

Source: Kaggle  
[Latest Data Science Job Salaries 2020 - 2025](https://www.kaggle.com/datasets/saurabhbadole/latest-data-science-job-salaries-2024/data)

This dataset contains **global data science job roles**, salaries, experience levels, employment types, company sizes, and locations. It is used across EDA, regression, and classification pipelines to demonstrate:

- salary trend analysis  
- feature engineering  
- regression modeling  
- classification of job categories  
- cross‑validated evaluation  

---

# **Project Structure**

```
pro-ibm-ml-course1-3-projects/
│
├── 01_eda/
│   ├── notebooks/
│   ├── scripts/
│   ├── visuals/
│   └── reports/
│
├── 02_regression/
│   ├── feature_engineering/
│   ├── models/
│   ├── evaluation/
│   └── pipeline/
│
├── 03_classification/
│   ├── preprocessing/
│   ├── models/
│   ├── evaluation/
│   └── pipeline/
│
└── docs/
    ├── project_overview.md
    ├── methods.md
    └── future_work.md
```

---

# **Professional Pipeline Design**

## **1. Exploratory Data Analysis (EDA)**  
Focus: understanding structure, distributions, correlations, and anomalies.

### **Key Components**
- missing‑value profiling  
- univariate and multivariate analysis  
- salary distribution analysis  
- geographic and role‑based segmentation  
- correlation heatmaps  
- outlier detection  
- feature quality assessment  

### **Tools**
- pandas  
- seaborn / matplotlib  
- scipy  
- R tidyverse (optional)

---

## **2. Supervised ML — Regression**  
Focus: predicting continuous outcomes (e.g., salary).

### **Models Implemented**
- Linear Regression  
- Ridge / Lasso  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  

### **Engineering Features**
- train/test split  
- k‑fold cross‑validation  
- hyperparameter tuning (GridSearchCV)  
- feature scaling  
- pipeline automation  
- model comparison (RMSE, MAE, R²)

---

## **3. Supervised ML — Classification**  
Focus: predicting categorical outcomes (e.g., job role, experience level).

### **Models Implemented**
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Gradient Boosting Classifier  
- Support Vector Machine (SVM)  
- K‑Nearest Neighbors (KNN)

### **Evaluation Metrics**
- accuracy  
- precision / recall  
- F1 score  
- ROC‑AUC  
- confusion matrix  
- cross‑validated performance  

---

# **Cross‑Language Implementation**

| Component | Python | R |
|----------|--------|---|
| EDA | pandas, seaborn | tidyverse, ggplot2 |
| Regression | scikit‑learn | caret |
| Classification | scikit‑learn | caret |
| Pipelines | sklearn Pipeline | custom modular scripts |
| Evaluation | sklearn metrics | caret metrics |

This dual‑language approach demonstrates **engineering maturity** and **cross‑platform reproducibility**, consistent with your professional survival analysis and DC property modeling repos.

---

# **Professional Engineering Standards**

- modular scripts for each ML stage  
- reproducible outputs  
- version‑controlled notebooks  
- standardized folder structure  
- clear separation of data, models, and evaluation  
- production‑aligned pipeline design  
- consistent documentation  

---

# **Future Work**

- integrate unsupervised learning (Course 4)  
- add deep learning modules (Course 5)  
- build a unified ML pipeline across all 6 IBM courses  
- add model explainability (SHAP, LIME)  
- deploy selected models via Flask/FastAPI  
- add MLOps components (Course 6)

---

# **Credits**

This repository is based on coursework from:

- **[IBM Machine Learning Professional Certificate](https://www.coursera.org/professional-certificates/ibm-machine-learning)**  
- **Coursera Machine Learning Specialization**

Dataset credit:

- **Latest Data Science Job Salaries 2024**  
  [Kaggle dataset by *Saurabh Badole*](https://www.kaggle.com/datasets/saurabhbadole/latest-data-science-job-salaries-2024/data)

