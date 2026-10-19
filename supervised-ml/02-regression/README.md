# **02 — Regression Modeling (Numeric‑Only Phase)**  
**Supervised Machine Learning Module**  
**Status:** Completed  
**Last Updated:** September 28, 2026  

This folder contains the first phase of supervised machine learning modeling for the **Master Data Science Salaries (2020–2025)** dataset.  
This phase focuses exclusively on **numeric‑only regression models**, using:

### **Target Variable:**  
`salary_in_usd` → transformed to `log_salary`

Categorical variables (job title, experience level, company size, etc.) will be incorporated in the **classification module**.

---

## 📌 **Core Research Questions**

### **1. Determinants of Salary**
- Which numeric features most strongly influence salary?  
- Are relationships linear or nonlinear?  
- How large are the effects?

### **2. Remote‑Work Salary Dynamics**
- Does remote_ratio correlate with higher or lower salary?  
- Do hybrid roles behave differently?  
- Does remote work reduce salary inequality?

### **3. Job‑Level & Experience‑Level Structure**  
*(Categorical variables added later)*  
- Seniority vs. job title  
- Full‑time vs. contract roles  

### **4. Company Characteristics**
- Do larger companies pay more?  
- Are effects linear or stepwise?  
- Do company attributes interact with experience level?

### **5. Temporal Salary Trends**
- Salary inflation across work_year  
- Does experience level moderate salary growth?  
- Does data_age capture meaningful temporal trends?

### **6. Interaction Effects**
- Numeric × numeric interactions  
- Nonlinear relationships captured by tree‑based models  

---

## 🎯 **The Five Big Research Themes**

1. Determinants of salary  
2. Remote‑work salary dynamics  
3. Job‑level & experience‑level salary structure  
4. Temporal salary trends & inflation  
5. Predictive modeling for salary estimation  

---

# **Why Log Transformation Was Required**

The original target variable:

### `salary_in_usd`

was **highly right‑skewed**, causing violations of OLS assumptions:

- heteroscedasticity  
- non‑normal residuals  
- nonlinear mean–variance relationship  
- high‑influence outliers  

### ✔ Log Transformation Applied  
`log_salary = log(salary_in_usd)`

### ✔ Improvements After Log Transform
- Residuals more symmetric  
- Variance stabilized  
- QQ‑plots closer to normality  
- Reduced funneling in residual vs. fitted plots  
- Outliers exerted less influence  

### **Final Decision:**  
All regression models in this numeric‑only phase use:

### **`log_salary` as the response variable**

This ensures:

- more reliable inference  
- stable coefficient estimates  
- better AIC/BIC model selection  
- interpretable multiplicative effects (percentage salary changes)

---

# **Regression Models Implemented (Numeric‑Only)**

## **1. Linear Models**
All trained using `log_salary`:

- Ordinary Least Squares (OLS)  
- Polynomial Regression (degree 2–3)  
- Interaction‑Only Regression  
- Polynomial + Interaction Regression  

### Why log helps linear models:
- stabilizes variance  
- improves normality  
- reduces outlier influence  
- improves diagnostic plots  
- yields interpretable percentage effects  

---

## **2. Regularized Models**
All trained using `log_salary`:

- Ridge Regression  
- Lasso Regression  
- Elastic Net Regression  

### Why log helps regularization:
- Ridge stabilizes coefficients under heteroscedasticity  
- Lasso performs cleaner feature selection  
- Elastic Net handles correlated predictors more effectively  

---

## **3. Polynomial & Interaction Models**
All trained using `log_salary`:

- Polynomial Regression (degree 2–3)  
- Interaction‑Only Regression  
- Polynomial + Interaction Regression  

### Why log helps polynomial models:
- reduces variance inflation  
- improves residual structure  
- stabilizes polynomial terms  
- improves AIC/BIC selection  
- yields clearer nonlinear effects  

---

## **4. Tree‑Based Ensemble Models**
Although tree models do not require log transformation, they were trained on `log_salary` for consistency:

- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- XGBoost Regressor  
- LightGBM Regressor  

### Benefits of using log:
- unified RMSE/MAE comparison  
- consistent leaderboard  
- smoother transition into mixed‑feature modeling  
- multiplicative interpretation of predictions  

---

## **5. Support Vector Regression (SVR)**
SVR with RBF kernel trained on `log_salary`.

### Benefits:
- more stable margins  
- smoother kernel behavior  
- improved comparability with other nonlinear models  

---

# **Model Evaluation**

All models were evaluated using:

- **RMSE (log scale)**  
- **MAE (log scale)**  
- **R²**  
- **Adjusted R²**  

This ensures a fair, unified comparison across linear, nonlinear, regularized, and ensemble models.

---

# **Status**

This folder represents **Phase 1: Numeric‑Only Regression Models**.

Completed:
- Numeric‑only regression suite  
- Log‑transformed modeling pipeline  
- Full model comparison leaderboard  

Next phases (in separate folders):
- **Categorical encoding**  
- **Classification modeling**  
- **Salary category prediction**  
- **SHAP interpretability across mixed feature types**  
- **Full model comparison dashboard**

---
