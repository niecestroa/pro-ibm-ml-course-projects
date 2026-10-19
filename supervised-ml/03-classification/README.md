# **03 — Salary Classification Modeling Suite**  

- **Supervised Machine Learning Module**  
- **Status:** Completed  
- **Last Updated:** October 2, 2026  

This folder contains the full classification modeling workflow for the **Master Data Science Salaries (2020–2025)** dataset.  
This phase reframes salary prediction as a **categorical classification problem**, extending the numeric‑only regression work completed earlier.

Classification models are trained across **two target formulations**:

### **Classification Targets**
1. **Mean‑Based Binary Classification**  
2. **Median‑Based Binary Classification**

These targets allow exploration of how different statistical definitions of salary influence model performance, interpretability, and decision boundaries.

---

## 👤 **Author**
**Aaron Niecestro**  
Data Scientist & Biostatistician  
Lead Analyst and Sole Contributor

---

## 📘 **Dataset Source**

All classification modeling uses the unified dataset created in:

```
supervised-ml/00-data/
```

Target variables are engineered from `salary_in_usd`:

### **Mean‑Based Binary Classification**
```python
mean_salary = df['salary_in_usd'].mean()
df['salary_mean_class'] = (df['salary_in_usd'] >= mean_salary).astype(int)
```

### **Median‑Based Binary Classification**
```python
median_salary = df['salary_in_usd'].median()
df['salary_median_class'] = (df['salary_in_usd'] >= median_salary).astype(int)
```

---

# 📌 **Project Overview**

Salary can be modeled not only as a continuous variable (regression) but also as a categorical one.  
This classification suite evaluates **five major machine‑learning families**, each offering different strengths for tabular salary data:

1. Linear & Generalized Linear Models  
2. Tree‑Based Models  
3. Support Vector Machines  
4. Probabilistic Models  
5. Instance‑Based / Distance‑Based Models  

Each model is trained and evaluated on both binary targets.

---

# 🎯 **Core Research Questions**

### **1. What features most strongly predict high‑salary vs. low‑salary categories?**
- Which variables drive classification decisions?  
- How do numeric vs. categorical features contribute?

### **2. How do job characteristics influence salary category?**
- Seniority  
- Employment type  
- Job title group (added in later phases)

### **3. How does remote work affect salary category?**
- Does remote_ratio shift class boundaries?  
- Are hybrid roles more likely to fall above the median?

### **4. How does company size relate to salary class?**
- Are large companies more likely to pay above the median?  
- Are effects linear or stepwise?

### **5. How do different statistical definitions of salary (mean vs. median) change model behavior?**
- Does one target produce more stable decision boundaries?  
- Does class imbalance differ?

---

# 🧪 **Classification Models Implemented**

Below are the five modeling families, each with **what the method does** and **why it is used**, rewritten for clarity and consistency.

---

## **1. Linear & Generalized Linear Models**

### **What they do**
Learn weighted combinations of features to separate classes using linear decision boundaries.

### **Why use them**
- Fast and interpretable  
- Strong baselines  
- Useful for understanding feature directionality  
- Provide clear coefficient‑based insights  

### **Models Included**
- Logistic Regression  
- Ridge Logistic Regression  
- Lasso Logistic Regression  
- Elastic Net Logistic Regression  
- Linear Discriminant Analysis (LDA)  
- Quadratic Discriminant Analysis (QDA)

---

## **2. Tree‑Based Models**

### **What they do**
Split data into hierarchical decision regions, capturing nonlinear relationships and interactions.

### **Why use them**
- Excellent for tabular data  
- Handle mixed numeric + categorical features  
- Provide feature importance  
- Robust to outliers and skew  

### **Models Included**
- Decision Tree  
- Random Forest  
- Extra Trees  
- Gradient Boosting  
- AdaBoost  

---

## **3. Support Vector Machines**

### **What they do**
Find optimal separating hyperplanes; kernel SVMs capture nonlinear boundaries.

### **Why use them**
- Strong performance on medium‑sized datasets  
- Effective when classes are not linearly separable  
- RBF and polynomial kernels capture complex patterns  

### **Models Included**
- Linear SVM  
- RBF SVM  
- Polynomial SVM  

---

## **4. Probabilistic Models**

### **What they do**
Compute class likelihoods using statistical distributions and Bayes’ theorem.

### **Why use them**
- Extremely fast  
- Provide calibrated probabilities  
- Strong baselines  
- Useful when features are independent or categorical  

### **Models Included**
- Gaussian Naive Bayes  
- Bernoulli Naive Bayes  

---

## **5. Instance‑Based / Distance‑Based Models**

### **What they do**
Classify samples based on similarity to neighbors using distance metrics.

### **Why use them**
- Capture irregular, highly nonlinear boundaries  
- Simple and intuitive  
- Strong performance when data is well‑scaled  

### **Models Included**
- K‑Nearest Neighbors (KNN)

---

# 🔧 **Modeling Workflow**

### **1. Preprocessing**
- Stratified train/test split  
- Scaling for linear/SVM/KNN models  
- One‑hot encoding for categorical features  
- Optional feature selection  

### **2. Training**
Each model is trained on both binary targets:
- `salary_mean_class`  
- `salary_median_class`  

### **3. Evaluation Metrics**

#### **Binary Classification**
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC‑AUC  
- Confusion Matrix  

### **4. Model Comparison**
- Performance tables  
- Ranking by metric  
- Hyperparameter tuning  
- Cross‑validation  
- Feature importance (tree models)  
- SHAP analysis (tree models only)  
- Ensemble stacking  

---

# ✅ **Classification Status**
Classification modeling is **complete**.  
Results from this phase feed directly into:

- Mixed‑feature modeling  
- SHAP interpretability  
- Salary category dashboards  
- Final supervised‑ML model comparison
