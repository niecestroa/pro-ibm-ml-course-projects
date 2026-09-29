# **Median‑Based Binary Classification (README Section)**

This section documents the **median‑based binary classification target**, the second of three target‑engineering strategies used in the Salary Classification Modeling Suite. It explains the motivation, construction, modeling approach, and evaluation framework for this target.

---

## **1. Motivation**

While the mean salary provides a useful benchmark, it can be influenced by extreme values—especially in datasets where compensation varies widely across roles, industries, or seniority levels. The **median salary** offers a more robust central tendency measure.

Using the median allows us to ask:

**Does a data professional earn at or above the median salary?**

This formulation is particularly valuable in skewed salary distributions, where the median better represents the “typical” salary.

---

## **2. Target Construction**

The median salary is computed from the dataset, and each observation is labeled based on whether its salary meets or exceeds that median.

```python
median_salary = df['salary'].median()
df['salary_median_class'] = (df['salary'] >= median_salary).astype(int)
```

### **Class Definitions**
- **0 — Below‑Median Salary**  
- **1 — Above‑Median Salary**

This creates a binary classification problem similar to the mean‑based target but with a more robust threshold.

---

## **3. Why Median‑Based Classification?**

### **Advantages**
- More robust to outliers than the mean  
- Better reflects the “typical” salary in skewed distributions  
- Often produces a more balanced class split  
- Useful for fairness‑oriented or distribution‑aware modeling  
- Aligns with compensation benchmarking practices (median is widely used in HR analytics)

### **Use Cases**
- Market‑rate classification  
- Compensation equity analysis  
- Salary benchmarking dashboards  
- Predicting whether a role is “above typical pay”  
- Complementary analysis alongside mean‑based classification

---

## **4. Models Applied**

All five ML families are trained and evaluated on the median‑based target:

### **Linear / GLM Models**
- Logistic Regression  
- Ridge / Lasso / Elastic Net  
- LDA / QDA  

### **Tree‑Based Models**
- Decision Tree  
- Random Forest  
- Extra Trees  
- Gradient Boosting  
- AdaBoost  

### **Support Vector Machines**
- Linear SVM  
- RBF SVM  
- Polynomial SVM  

### **Probabilistic Models**
- Gaussian Naive Bayes  
- Bernoulli Naive Bayes  

### **Instance‑Based Models**
- K‑Nearest Neighbors (with tuned k)

This ensures a complete benchmarking suite across linear, nonlinear, probabilistic, and distance‑based paradigms.

---

## **5. Preprocessing Pipeline**

To ensure consistency across models:

- **Stratified train/test split**  
- **Median imputation** for numeric features  
- **Mode imputation** for categorical features  
- **StandardScaler** for numeric features  
- **OneHotEncoder(drop="first")** for categorical features  
- **ColumnTransformer** to unify preprocessing  

This produces a fully reproducible and model‑agnostic workflow identical to the mean‑based target.

---

## **6. Evaluation Metrics**

### **Primary Metrics**
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1 Score**  
- **ROC‑AUC**

### **Diagnostic Tools**
- Confusion Matrix  
- ROC Curves for all models  
- SHAP analysis for Extra Trees  
- GLM coefficient interpretation  

These metrics provide both predictive and interpretive insight.

---

## **7. Outputs Generated**

For the median‑based target, the modeling suite produces:

- Full model comparison table  
- Reduced model comparison table  
- ROC curves for all models  
- SHAP summary plots (full + reduced)  
- GLM inference summaries  
- Feature importance plots  
- Final model selection summary  

This mirrors the outputs of the mean‑based target, enabling direct comparison between the two formulations.

---

## **8. Summary**

The median‑based binary classification target provides a robust, distribution‑aware way to evaluate salary prediction models. It complements the mean‑based target by reducing sensitivity to outliers and offering a more representative threshold for typical salary levels. Together, the mean‑ and median‑based targets form a comprehensive binary classification framework for salary analytics.
