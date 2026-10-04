# Data Science Job Salaries ML Pipeline (EDA, Regression, Classification)
Professional machine learning pipelines derived from IBM/Coursera Machine Learning Courses 1–3, rebuilt into modular, reproducible, production‑aligned workflows using Python.

---

## **Project Timeline**

### **00 — Cleaning Data**  
- **Started:** September 9, 2026  
- **Last Edited:** September 28, 2026  
- **Progress:** **Completed**  
- **Notes:** All datasets merged, normalized, deduplicated, and exported.

---

### **01 — Exploratory Data Analysis (EDA)**  
*(Regression & Classification)*  
- **Started:** September 9, 2026  
- **Last Edited:** September 14, 2026  
- **Progress:** **Completed**  
- **Notes:**  
  - Numeric + categorical summaries  
  - Regression Outcome: Salary in USD
  - Classification Outcome: Salary in USD category creation (mean, median, 3‑band)  
  - Class imbalance checks  
  - Leakage detection  
  - Feature importance & histograms in progress

---

### **03 - Supervised Machine Learning: Regression**  
- **Started:** September 17, 2026  
- **Last Edited:** September 23, 2026  
- **Progress:** **Completed**  
- **Notes:**  
  - Planned models: Linear Regression, Random Forest, Gradient Boosting  
  - Target: `salary_in_usd`  
  - Numeric predictors: `data_age`, `remote_ratio`, `experience_level_num`, etc.

---

### **03 - Supervised Machine Learning: Classification**  
- **Started:** September 28, 2026  
- **Last Edited:** October 3, 2026  
- **Progress:** **Completed**  
- **Notes:**  
  - Targets: `salary_mean_cat` & `salary_median_cat` 

---

### **04 — Unsupervised Machine Learning**  
- **Started:** October __, 2026  
- **Last Edited:** October __, 2026  
- **Progress:** **No Plan Yet**  
- **Notes:**  
  - Need to complete 03 - Supervised Machine Learning: Classification first.
    
---

### **05 — Deep Learning and Reinforcement Learning**  
- **Started:** October __, 2026  
- **Last Edited:** October __, 2026  
- **Progress:** **No Plan Yet**  
- **Notes:**
   - Need to complete 04 — Unsupervised Machine Learning first.

---

### **06 — Machine Learning Capstone**  
- **Started:** October __, 2026  
- **Last Edited:** October __, 2026  
- **Progress:** **No Plan Yet**  
- **Notes:**
   - Need to complete 05 — Deep Learning and Reinforcement Learning first.

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

Here’s a clean, GitHub‑ready **Dataset Credit** section you can drop directly into your README. It includes all the Kaggle sources you listed, formatted consistently and with proper attribution.

---

## Dataset Credits

This project uses multiple publicly available datasets from Kaggle. Full credit to the original dataset authors:

- **Latest Data Science Job Salaries 2024**  
  *Author:* Saurabh Badole  
  *Source:* [Latest Data Science Job Salaries 2024](https://www.bing.com/search?q="https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fsaurabhbadole%2Flatest-data-science-job-salaries-2024")

- **Data Science Salaries 2023**  
  *Author:* Arnab Chaki  
  *Source:* [https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023](https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023)

- **Data Science Salaries 2024**  
  *Author:* Yusuf Delikkaya  
  *Source:* [https://www.kaggle.com/datasets/yusufdelikkaya/datascience-salaries-2024](https://www.kaggle.com/datasets/yusufdelikkaya/datascience-salaries-2024)

- **Data Science Salaries and Fields**  
  *Author:* Josia Given  
  *Source:* [https://www.kaggle.com/datasets/josiagiven/data-science-salaries-and-fields](https://www.kaggle.com/datasets/josiagiven/data-science-salaries-and-fields)

- **Data Science Salaries**  
  *Author:* Sazid  
  *Source:* [https://www.kaggle.com/datasets/sazidthe1/data-science-salaries](https://www.kaggle.com/datasets/sazidthe1/data-science-salaries)

- **Data Science Salaries**  
  *Author:* Zain  
  *Source:* [https://www.kaggle.com/datasets/zain280/data-science-salaries](https://www.kaggle.com/datasets/zain280/data-science-salaries)

- **Data Science Salary Landscape**  
  *Author:* Lai Nguy  
  *Source:* `https://www.kaggle.com/datasets/lainguyn123/data-science-salary-landscape` [(kaggle.com in Bing)](https://www.bing.com/search?q="https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Flainguyn123%2Fdata-science-salary-landscape")
