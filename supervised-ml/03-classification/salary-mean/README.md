# **Mean‑Based Binary Classification (README Section)**

This section documents the **mean‑based binary classification target**, the first of three target‑engineering strategies used in the Salary Classification Modeling Suite. It explains the motivation, construction, modeling approach, and evaluation framework for this target.

---

## **1. Motivation**

Salary prediction can be reframed as a binary decision problem:  
**Does a data professional earn above the average salary?**

This formulation is useful because:

- It converts a continuous salary variable into a clear, interpretable class label  
- It aligns with real‑world compensation questions (e.g., “Is this role above market rate?”)  
- It enables comparison across multiple ML families using a unified binary target  
- It provides a balanced, statistically meaningful threshold based on the dataset’s central tendency  

---

## **2. Target Construction**

The mean salary is computed from the dataset, and each observation is labeled based on whether its salary meets or exceeds that mean.

```python
mean_salary = df['salary'].mean()
df['salary_mean_class'] = (df['salary'] >= mean_salary).astype(int)
```

### **Class Definitions**
- **0 — Below‑Average Salary**  
- **1 — Above‑Average Salary**

This creates a binary classification problem suitable for logistic models, SVMs, tree ensembles, Naive Bayes, and KNN.

---

## **3. Why Mean‑Based Classification?**

### **Advantages**
- Uses a statistically grounded threshold  
- Produces a reasonably balanced class distribution  
- Works well with ROC‑AUC and other binary metrics  
- Provides intuitive interpretation for stakeholders  
- Avoids arbitrary or domain‑specific cutoffs  

### **Use Cases**
- Compensation benchmarking  
- Salary‑tier prediction  
- HR analytics dashboards  
- Market‑rate classification  
- Exploratory modeling before regression or multiclass tiers  

---

## **4. Models Applied**

All five ML families are trained and evaluated on this target:

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

This produces a fully reproducible and model‑agnostic workflow.

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

For the mean‑based target, the modeling suite produces:

- Full model comparison table  
- Reduced model comparison table  
- ROC curves for all models  
- SHAP summary plots (full + reduced)  
- GLM inference summaries  
- Feature importance plots  
- Final model selection summary  

This creates a complete, end‑to‑end classification analysis for the mean‑based target.

---

## **8. Summary**

The mean‑based binary classification target provides a clean, interpretable, and statistically grounded way to evaluate salary prediction models. It serves as the foundation for comparing model families, understanding feature influence, and establishing baseline performance before exploring median‑based or tiered multiclass targets.
