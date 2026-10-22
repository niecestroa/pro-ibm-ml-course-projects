# 📘 Professional IBM Machine Learning Course Projects  

- **Created:** August 25, 2026  
- **Updated:** October 22, 2026  
- **Progress:** Ongoing  

This repository contains **professional, production‑ready machine learning pipelines** derived from the IBM/Coursera Machine Learning Professional Certificate (Courses 1–6). Each academic assignment has been rebuilt into a **modular, reproducible, engineering‑aligned workflow** using Python (and optional R).

The repo includes:

- Exploratory Data Analysis (EDA)  
- Supervised ML: Regression  
- Supervised ML: Classification  
- Unsupervised ML  
- Deep Learning & Reinforcement Learning  
- Capstone Project  

All projects follow a consistent structure emphasizing:

- modular scripts  
- reproducible workflows  
- robust evaluation  
- feature engineering  
- model comparison  
- production‑aligned ML practices  

---

# 📑 Table of Contents

1. [Project Timeline](#project-timeline)  
   - [00 — Cleaning Data](#00--cleaning-data)  
   - [01 — Exploratory Data Analysis (EDA)](#01--exploratory-data-analysis-eda)  
   - [02 — Supervised ML: Regression](#03---supervised-machine-learning-regression)  
   - [03 — Supervised ML: Classification](#03---supervised-machine-learning-classification)  
   - [04 — Unsupervised Machine Learning](#04--unsupervised-machine-learning)  
   - [05 — Deep Learning & Reinforcement Learning](#05--deep-learning-and-reinforcement-learning)  
   - [06 — Machine Learning Capstone](#06--machine-learning-capstone)  
2. [Overview](#overview)  
3. [Datasets & Citations](#datasets--citations)  
4. [Professional Pipeline Design](#professional-pipeline-design)  
5. [Cross‑Language Implementation](#cross-language-implementation)  
6. [Professional Engineering Standards](#professional-engineering-standards)  
7. [Future Work](#future-work)  
8. [Credits](#credits)

---

# **Project Timeline**

## **00 — Cleaning Data**  
- **Started:** September 9, 2026  
- **Last Edited:** September 28, 2026  
- **Progress:** **Completed**  
- **Notes:**  
  - All Kaggle salary datasets merged, normalized, deduplicated, and exported.  
  - Foundation for Projects 01–03.

---

## **01 — Exploratory Data Analysis (EDA)**  
*(Regression & Classification)*  
- **Started:** September 9, 2026  
- **Last Edited:** September 14, 2026  
- **Progress:** **Completed**  
- **Datasets Used:** Kaggle Data Science Job Salaries (multiple sources)  
- **Notes:**  
  - Numeric + categorical summaries  
  - Regression target: `salary_in_usd`  
  - Classification targets: mean/median salary categories  
  - Class imbalance checks  
  - Leakage detection  
  - Feature importance visualizations  

---

## **03 — Supervised Machine Learning: Regression**  
- **Started:** September 17, 2026  
- **Last Edited:** September 23, 2026  
- **Progress:** **Completed**  
- **Datasets Used:** Kaggle Data Science Job Salaries  
- **Notes:**  
  - Models: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting  
  - Target: `salary_in_usd`  
  - Full numeric pipeline with CV + hyperparameter tuning  

---

## **03 — Supervised Machine Learning: Classification**  
- **Started:** September 28, 2026  
- **Last Edited:** October 3, 2026  
- **Progress:** **Completed**  
- **Datasets Used:** Kaggle Data Science Job Salaries  
- **Notes:**  
  - Targets: `salary_mean_cat`, `salary_median_cat`  
  - Models: Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN  
  - Full evaluation suite (accuracy, F1, ROC‑AUC, confusion matrix)  

---

## **04 — Unsupervised Machine Learning**  
- **Started:** October 19, 2026  
- **Last Edited:** October 19, 2026  
- **Progress:** **In Progress**  
- **Dataset Used:** UCI Wine Quality Dataset  
- **Notes:**  
  - Primary project: **Wine Quality Unsupervised ML Pipeline**  
  - Methods:  
    - Clustering: K‑Means, GMM, Agglomerative, DBSCAN  
    - Dimensionality Reduction: PCA, t‑SNE, UMAP  
    - Anomaly Detection: Isolation Forest, LOF  
  - Focus: latent structure discovery, cluster profiling, anomaly identification  
  - Will integrate with supervised pipelines for hybrid analysis  

---

## **05 — Deep Learning and Reinforcement Learning**  
- **Started:** October __, 2026  
- **Last Edited:** October __, 2026  
- **Progress:** **Not Started**  
- **Datasets:** TBD  
- **Notes:**  
  - Neural networks (dense, CNN, RNN)  
  - TensorFlow/Keras pipelines  
  - Reinforcement learning fundamentals (Q‑learning, policy gradients)  
  - GPU‑accelerated training workflows  
  - Will build on supervised + unsupervised pipelines  

---

## **06 — Machine Learning Capstone**  
- **Started:** October __, 2026  
- **Last Edited:** October __, 2026  
- **Progress:** **Not Started**  
- **Datasets:** TBD  
- **Notes:**  
  - Unified ML pipeline combining Courses 1–5  
  - End‑to‑end engineering: ingestion → preprocessing → modeling → evaluation → deployment  
  - Optional MLOps integration (CI/CD, model registry, monitoring)  
  - Final professional project demonstrating full ML lifecycle  

---

# **Overview**

This repository contains **professional, production‑ready machine learning projects** based on the IBM Machine Learning Certificate:

1. Exploratory Data Analysis  
2. Supervised ML: Regression  
3. Supervised ML: Classification  
4. Unsupervised ML  
5. Deep Learning & Reinforcement Learning  
6. Capstone Project  

Each project is fully refactored into a **professional pipeline** emphasizing modularity, reproducibility, and engineering best practices.

---

# **Datasets & Citations**

## **Projects 01–03: Data Science Job Salaries (Kaggle)**  
This repo uses multiple publicly available Kaggle datasets:

- **Latest Data Science Job Salaries 2024** — Saurabh Badole  
  [https://www.kaggle.com/datasets/saurabhbadole/latest-data-science-job-salaries-2024](https://www.kaggle.com/datasets/saurabhbadole/latest-data-science-job-salaries-2024)  

- **Data Science Salaries 2023** — Arnab Chaki  
  [https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023](https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023)  

- **Data Science Salaries 2024** — Yusuf Delikkaya  
  [https://www.kaggle.com/datasets/yusufdelikkaya/datascience-salaries-2024](https://www.kaggle.com/datasets/yusufdelikkaya/datascience-salaries-2024)  

- **Data Science Salaries and Fields** — Josia Given  
  [https://www.kaggle.com/datasets/josiagiven/data-science-salaries-and-fields](https://www.kaggle.com/datasets/josiagiven/data-science-salaries-and-fields)  

- **Data Science Salaries** — Sazid  
  [https://www.kaggle.com/datasets/sazidthe1/data-science-salaries](https://www.kaggle.com/datasets/sazidthe1/data-science-salaries)  

- **Data Science Salaries** — Zain  
  [https://www.kaggle.com/datasets/zain280/data-science-salaries](https://www.kaggle.com/datasets/zain280/data-science-salaries)  

- **Data Science Salary Landscape** — Lai Nguy  
  `https://www.kaggle.com/datasets/lainguyn123/data-science-salary-landscape` [(kaggle.com in Bing)](https://www.bing.com/search?q="https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Flainguyn123%2Fdata-science-salary-landscape")  

---

## **Project 04: Wine Quality Dataset (UCI)**  
Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Wine Quality* [Dataset].  
UCI Machine Learning Repository.  
[https://doi.org/10.24432/C56S3T](https://doi.org/10.24432/C56S3T)

---

# **Professional Pipeline Design**

## **1. Exploratory Data Analysis (EDA)**  
- missing‑value profiling  
- univariate & multivariate analysis  
- salary distribution analysis  
- geographic segmentation  
- correlation heatmaps  
- outlier detection  

---

## **2. Supervised ML — Regression**  
Models:  
- Linear Regression  
- Ridge / Lasso  
- Decision Tree  
- Random Forest  
- Gradient Boosting  

Engineering:  
- CV  
- hyperparameter tuning  
- scaling  
- pipeline automation  

---

## **3. Supervised ML — Classification**  
Models:  
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- SVM  
- KNN  

Evaluation:  
- accuracy  
- precision/recall  
- F1  
- ROC‑AUC  
- confusion matrix  

---

## **4. Unsupervised ML — Wine Quality Pipeline**  
- clustering  
- dimensionality reduction  
- anomaly detection  
- cluster profiling  
- PCA/t‑SNE/UMAP visualization  

---

## **5. Deep Learning & Reinforcement Learning**  
- neural networks  
- CNN/RNN architectures  
- RL fundamentals  
- TensorFlow/Keras pipelines  

---

## **6. Capstone Project**  
- unified ML pipeline  
- explainability (SHAP, LIME)  
- deployment (Flask/FastAPI/Streamlit)  
- optional MLOps  

---

# **Cross‑Language Implementation**

| Component | Python | R |
|----------|--------|---|
| EDA | pandas, seaborn | tidyverse, ggplot2 |
| Regression | scikit‑learn | caret |
| Classification | scikit‑learn | caret |
| Pipelines | sklearn Pipeline | modular scripts |
| Evaluation | sklearn metrics | caret metrics |

---

# **Professional Engineering Standards**

- modular scripts  
- reproducible outputs  
- version‑controlled notebooks  
- standardized folder structure  
- separation of data, models, evaluation  
- production‑aligned pipeline design  

---

# **Future Work**

- complete unsupervised ML  
- add deep learning modules  
- build unified pipeline across all 6 courses  
- add explainability (SHAP, LIME)  
- deploy selected models  
- integrate MLOps  

---

# **Credits**

- Based on coursework from: **IBM Machine Learning Professional Certificate**  
