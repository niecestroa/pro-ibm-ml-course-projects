# **Supervised Machine Learning — Salary Modeling Pipeline**  

This directory contains the full **Supervised Machine Learning pipeline** for the *Master Data Science Salaries (2020–2025)* dataset.  
All workflows are rebuilt from IBM/Coursera Machine Learning Courses 1–3 into **modular, reproducible, production‑aligned pipelines** using Python.

The supervised‑ML module includes:

- **Data Engineering & Integration**  
- **Exploratory Data Analysis (EDA)**  
- **Regression Modeling (Numeric‑Only Phase)**  
- **Classification Modeling (Binary Targets)**  

Each stage is documented with its own README and follows a consistent engineering structure.

- **Modules:** Data • EDA • Regression • Classification  
- **Status:** Completed  
- **Last Updated:** October 2, 2026  

---

# 📘 **Dataset: Master Data Science Salaries (2020–2025)**

This supervised‑ML pipeline uses a unified dataset created by merging **seven Kaggle datasets**, downloaded via `kagglehub`:

1. **Latest Data Science Job Salaries 2024** — Saurabh Badole  
2. **Data Science Salaries 2023** — Arnab Chaki  
3. **DataScience Salaries 2024** — Yusuf Delikkaya  
4. **Data Science Salaries and Fields** — Josia Given  
5. **Data Science Salaries** — Sazidthe1  
6. **Data Science Salaries** — Zain280  
7. **Data Science Salary Landscape** — Lain Guyn  

The unified dataset contains:

- salary  
- salary_in_usd  
- job_title  
- experience_level  
- employment_type  
- remote_ratio  
- company_size  
- employee_residence  
- company_location  
- work_year  

Final merged dataset: **`dss2025_final.csv`**

---

# 📅 **Project Timeline**

### **00 — Data Cleaning & Integration**  
- **Started:** September 9, 2026  
- **Completed:** September 28, 2026  
- **Highlights:**  
  - All seven datasets merged  
  - Columns normalized  
  - Duplicates removed  
  - Final unified dataset exported  

---

### **01 — Exploratory Data Analysis (EDA)**  
- **Started:** September 9, 2026  
- **Completed:** September 14, 2026  
- **Highlights:**  
  - Numeric + categorical summaries  
  - Salary distribution analysis  
  - Class imbalance checks  
  - Leakage detection  
  - Feature quality assessment  
  - Regression target: `salary_in_usd`  
  - Classification targets: mean‑based, median‑based salary categories  

---

### **02 — Regression Modeling (Numeric‑Only Phase)**  
- **Started:** September 17, 2026  
- **Completed:** September 23, 2026  
- **Highlights:**  
  - Target transformed to `log_salary`  
  - Models: Linear, Polynomial, Interaction, Ridge, Lasso, Elastic Net, Tree‑Based, SVR  
  - Unified evaluation: RMSE, MAE, R²  
  - Full leaderboard generated  

---

### **03 — Classification Modeling**  
- **Started:** September 28, 2026  
- **Completed:** October 2, 2026  
- **Highlights:**  
  - Targets: `salary_mean_class`, `salary_median_class`  
  - Models: Logistic, LDA/QDA, Trees, SVM, Naive Bayes, KNN  
  - Imbalance handling: class weights  
  - Full evaluation suite: Accuracy, Precision, Recall, F1, ROC‑AUC  

---

# 🔍 **Module Summaries**

## **1. Data Engineering**
- Kaggle dataset ingestion  
- Schema normalization  
- Deduplication  
- Unified dataset creation  
- Export of `dss2025_final.csv`  

---

## **2. Exploratory Data Analysis (EDA)**
Focus: understanding structure, distributions, correlations, and anomalies.

### Key Components
- Missing‑value profiling  
- Salary distribution analysis  
- Geographic segmentation  
- Role‑based segmentation  
- Correlation heatmaps  
- Outlier detection  
- Feature quality assessment  

### Tools
- pandas  
- seaborn / matplotlib  
- scipy  

---

## **3. Regression Modeling (Numeric‑Only Phase)**  
Focus: predicting continuous salary values.

### Models Implemented
- Linear Regression  
- Polynomial Regression  
- Interaction Regression  
- Ridge / Lasso / Elastic Net  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- XGBoost / LightGBM  
- Support Vector Regression (SVR)

### Engineering Features
- Log transformation (`log_salary`)  
- Train/test split  
- Cross‑validation  
- Hyperparameter tuning  
- Pipeline automation  
- Unified leaderboard (RMSE, MAE, R²)

---

## **4. Classification Modeling**  
Focus: predicting salary categories.

### Targets
- `salary_mean_class`  
- `salary_median_class`

### Models Implemented
- Logistic Regression  
- Ridge/Lasso/Elastic Net Logistic  
- LDA / QDA  
- Decision Tree  
- Random Forest  
- Extra Trees  
- Gradient Boosting  
- AdaBoost  
- SVM (Linear, RBF, Polynomial)  
- Gaussian/Bernoulli Naive Bayes  
- K‑Nearest Neighbors (KNN)

### Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC‑AUC  
- Confusion Matrix  
- Cross‑validated performance  

---

# 🧩 **Engineering Standards**

- Modular scripts for each ML stage  
- Reproducible outputs  
- Version‑controlled notebooks  
- Standardized folder structure  
- Clear separation of data, models, and evaluation  
- Production‑aligned pipeline design  
- Consistent documentation across modules  

---

# 🚀 **Status**

- **Integrate unsupervised learning (Course 4) Status:** Ongoing <br>
- **Add deep learning modules (Course 5) Status:** Not Started <br>
- **Build unified ML pipeline across all 6 IBM courses Status:** Not Started <br>

---

# 🏷 **Credits**

### **Coursework**
This supervised‑ML pipeline is based on coursework from:

- **IBM Machine Learning Professional Certificate**  
- **Coursera Machine Learning Specialization**

### **Datasets (Merged Into the Master Data Science Salaries 2020–2025 Dataset)**  
All raw datasets were sourced from Kaggle and merged using `kagglehub`:

1. **Latest Data Science Job Salaries 2024** — *Saurabh Badole*  
2. **Data Science Salaries 2023** — *Arnab Chaki*  
3. **DataScience Salaries 2024** — *Yusuf Delikkaya*  
4. **Data Science Salaries and Fields** — *Josia Given*  
5. **Data Science Salaries** — *Sazidthe1*  
6. **Data Science Salaries** — *Zain280*  
7. **Data Science Salary Landscape** — *Lain Guyn*  

These datasets collectively form the unified dataset used across EDA, regression, and classification pipelines.
