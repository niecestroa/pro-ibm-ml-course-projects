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

## **1. Linear Models**

Linear Regression (OLS) serves as the baseline model for the numeric‑only phase.  
During initial modeling, the response variable was:

### **Original Target Variable:**  
`salary_in_usd`

However, the raw salary distribution was **highly right‑skewed**, and the OLS diagnostic plots showed clear violations of key regression assumptions:

- **Non‑constant variance (heteroscedasticity)**  
- **Non‑normal residuals**  
- **Nonlinear mean–variance relationship**  
- **High‑influence outliers**

To address these issues, the response variable was transformed using:

### **Log Transformation:**  
`log_salary = log(salary_in_usd)`

### **Why Log Transformation Was Necessary**
Even after initial preprocessing, the OLS assumptions were **not met** using the original salary scale.  
After applying the log transformation:

- Residuals became **more symmetric**  
- Variance stabilized across fitted values  
- QQ‑plots showed **closer alignment to normality**  
- Residual vs. fitted plots showed **reduced funneling**  
- Extreme salaries exerted less influence on the model  

Although the assumptions were **not perfectly satisfied**, the log‑transformed model exhibited **substantially improved diagnostic behavior** compared to the untransformed model.

### **Final Decision**
All subsequent regression models in this numeric‑only phase use:

**`log_salary` as the response variable.**

This includes:

- Linear Regression (OLS)  
- Polynomial Regression  
- Interaction‑Only Regression  
- Polynomial + Interaction Regression  
- Ridge Regression  
- Lasso Regression  
- Elastic Net Regression  

Using the log‑transformed response ensures:

- More reliable inference  
- More stable coefficient estimates  
- Better comparability across modeling families  
- Improved AIC/BIC model selection  
- More interpretable multiplicative effects (percentage salary changes)

---

## **2. Regularized Models**

Regularization was applied **after** the log transformation to stabilize coefficients and reduce overfitting in high‑dimensional numeric expansions.

All regularized models use:

**`log_salary` as the response variable.**

Models included:

- **Ridge Regression**  
- **Lasso Regression**  
- **Elastic Net Regression**

These models benefit from the log transformation because:

- Ridge stabilizes coefficients under heteroscedasticity  
- Lasso performs cleaner feature selection when variance is stabilized  
- Elastic Net handles correlated numeric predictors more effectively  

The log transformation improved residual structure and reduced the influence of extreme salary values, making regularization more effective and interpretable.

---

## **3. Polynomial & Interaction Models**

Polynomial and interaction models also use the **log‑transformed response**.

### **Why?**

Polynomial expansions (degree 2 or 3) and numeric × numeric interactions can amplify:

- heteroscedasticity  
- skewness  
- outlier influence  

Using `log_salary`:

- reduces variance inflation  
- improves normality of residuals  
- stabilizes polynomial terms  
- improves AIC/BIC model selection  
- yields more interpretable nonlinear effects  

Models included:

- **Polynomial Regression (degree 2 or 3)**  
- **Interaction‑Only Regression**  
- **Polynomial + Interaction Regression**

All diagnostics (residual plots, QQ plots, BP tests) showed **clear improvement** after log transformation.

---

## **4. Tree‑Based Ensemble Models**

Tree‑based models do not require log transformation for assumptions, but for **consistency across the regression suite**, they were also trained on:

**`log_salary`**

This allows:

- consistent RMSE/MAE comparison  
- unified leaderboard evaluation  
- easier interpretation of multiplicative effects  
- smoother transition into Phase 2 (mixed‑feature modeling)

Models included:

- **Decision Tree Regressor**  
- **Random Forest Regressor**  
- **Gradient Boosting Regressor**  
- **XGBoost Regressor**  
- **LightGBM Regressor**

---

## **5. Support Vector Regression**

SVR with RBF kernel also uses:

**`log_salary`**

This improves:

- margin stability  
- kernel behavior  
- prediction smoothness  
- comparability with other nonlinear models

---

## **6. Model Comparison**

All models — linear, polynomial, interaction, regularized, tree‑based, and SVR — are compared using:

- RMSE (log scale)  
- MAE (log scale)  
- R²  
- Adjusted R²  

This ensures a fair, unified evaluation across the entire numeric‑only modeling phase.

---

# **Next Steps**

This repository represents **Phase 1: Numeric‑Only Regression Models**.  
Future phases will include:

- **Categorical feature encoding**  
- **Classification modeling**  
- **Salary category prediction**  
- **SHAP interpretability across mixed feature types**  
- **Full model comparison dashboard**
