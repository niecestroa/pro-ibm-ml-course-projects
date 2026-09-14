# **Data Science Salary Modeling — Regression Analysis (Numeric‑Only Phase)**

This repository contains the first phase of a multi‑stage machine learning project focused on modeling and understanding salary dynamics in the data science workforce.  
In this phase, we perform **supervised regression modeling** using **numeric‑only features**, with the response variable:

### **Target Variable:** `salary_in_usd`

Categorical variables will be incorporated later in a separate **classification modeling phase**.

---

## **Core Research Questions**

This project is driven by a set of foundational research questions that explore how salary varies across roles, experience, remote work, company characteristics, and time. These questions guide model selection, feature engineering, and interpretation.

### **1. What factors most strongly influence salary levels in the data science workforce?**
- Which numeric variables matter most?  
- How large are their effects?  
- Are relationships linear or nonlinear?

### **2. How do job‑related characteristics shape salary outcomes?**  
*(Categorical variables will be added later)*  
- Seniority vs. job title  
- Full‑time vs. contract roles  

### **3. How does remote work affect salary?**
- Does remote_ratio correlate with higher or lower salary?  
- Does hybrid work behave differently?  
- Does remote work reduce salary inequality?

### **4. How does company size relate to salary outcomes?**
- Do larger companies pay more?  
- Are effects linear or stepwise?  
- Does company size interact with experience level?

### **5. How has salary changed over time?**
- Salary inflation across work_year  
- Does experience level moderate salary growth?  
- Does data_age capture meaningful temporal trends?

### **6. Are there interaction effects between job characteristics, remote work, and time?**
- Numeric × numeric interactions  
- Nonlinear relationships captured by tree‑based models  

---

## **The Five Big Research Themes**

These themes summarize the overarching goals of the regression phase:

1. **Determinants of salary**  
2. **Remote‑work salary dynamics**  
3. **Job‑level & experience‑level salary structure**  
4. **Temporal salary trends & inflation**  
5. **Predictive modeling for salary estimation**

---

# **Supervised Machine Learning Regression Models (Numeric‑Only)**

Since this phase uses **only one numeric predictors**, all models below operate without categorical encoding.  
Categorical variables will be introduced later for classification tasks in classification folder.

---

## **1. Linear & Regularized Models**
These provide interpretability and establish baseline performance.

- **Linear Regression (OLS)**  
- **Ridge Regression**  
- **Lasso Regression**  
- **Elastic Net Regression**

---

## **2. Polynomial & Interaction Models**
Capture nonlinear salary patterns using numeric transformations.

- **Polynomial Regression (degree 2 or 3)**  
- **Interaction Regression (numeric × numeric)**  

---

## **3. Tree‑Based Ensemble Models**
Model nonlinearities and interactions automatically.

- **Decision Tree Regressor**  
- **Random Forest Regressor**  
- **Gradient Boosting Regressor**  
- **XGBoost Regressor**  
- **LightGBM Regressor**

---

## **4. Support Vector Regression**
Kernel‑based nonlinear modeling.

- **SVR with RBF kernel**

---

# **Next Steps**
This repository represents **Phase 1: Numeric‑Only Regression Models**.  
Future phases will include:

- **Categorical feature encoding**  
- **Classification modeling**  
- **Salary category prediction**  
- **SHAP interpretability across mixed feature types**  
- **Full model comparison dashboard**

