# **Salary Classification Modeling Suite**

**Author:** Aaron Niecestro  
**Created on:** Not started yet (Estimated to start on September 25, 2026)  
**Edited on:** Not started yet (Estimated to start on September 25, 2026)  
**Progress:** Not started yet (Estimated to start on September 25, 2026)

---

A comprehensive machine learning project exploring multiple classification approaches for predicting salary categories using the WSD dataset. This work extends prior regression modeling by reframing salary prediction as a classification problem under three different target definitions.

---

## **📌 Project Overview**
Salary can be modeled not only as a continuous variable but also as a categorical one. This project builds and compares classification models across **five major ML families**, applied to three different target formulations:

### **Classification Targets**
1. **Mean‑Based Binary Classification**  
2. **Median‑Based Binary Classification**  
3. **Tiered Salary Categories (Multiclass)**  

These targets allow exploration of how different statistical definitions of salary influence model performance and interpretability.

---

## **📁 Dataset**
- **Source:** WSD Salary Dataset  
- **Features:** Demographics, job characteristics, experience, education, etc.  
- **Targets:**  
  - `salary_mean_class`  
  - `salary_median_class`  
  - `salary_tier`

---

## **🎯 Target Engineering**
### **Mean‑Based Binary Classification**
```python
mean_salary = df['salary'].mean()
df['salary_mean_class'] = (df['salary'] >= mean_salary).astype(int)
```

### **Median‑Based Binary Classification**
```python
median_salary = df['salary'].median()
df['salary_median_class'] = (df['salary'] >= median_salary).astype(int)
```

### **Tiered Salary Categories (Multiclass)**
```python
df['salary_tier'] = pd.qcut(
    df['salary'],
    q=3,
    labels=['Low', 'Medium', 'High']
)
```

---

# **🧠 Classification Models (Methods 1–5)**

This project implements every classification model from the first five major ML families.  
Each section now includes **what the method does** and **why it is used**.

---

## **1. Linear & Generalized Linear Models**

### **What these models do**
Linear models learn a weighted combination of input features to separate classes. They assume a linear decision boundary and often provide interpretable coefficients showing how each feature influences the prediction.

### **Why use them**
- They are fast and computationally efficient  
- Provide clear interpretability  
- Serve as strong baselines  
- Perform well when relationships are approximately linear  
- Useful for understanding feature directionality (positive/negative influence)

### **Models Included**
| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear classifier; interpretable |
| Ridge Logistic Regression | L2‑regularized variant |
| Lasso Logistic Regression | L1‑regularized variant |
| Elastic Net Logistic Regression | Combined L1/L2 regularization |
| Linear Discriminant Analysis (LDA) | Assumes Gaussian class distributions |
| Quadratic Discriminant Analysis (QDA) | Allows class‑specific covariance |

---

## **2. Tree‑Based Models**

### **What these models do**
Tree‑based models split the data into decision regions using hierarchical rules. They capture nonlinear relationships, interactions, and complex boundaries without requiring feature scaling.

### **Why use them**
- Excellent performance on tabular data  
- Naturally handle mixed numeric + categorical features  
- Provide feature importance  
- Capture nonlinear patterns missed by linear models  
- Robust to outliers and skewed distributions  
- Often top performers for salary prediction tasks

### **Models Included**
| Model | Description |
|-------|-------------|
| Decision Tree | Simple, interpretable tree |
| Random Forest | Ensemble of trees; robust and stable |
| Extra Trees | Highly randomized trees; fast |
| Gradient Boosting | Sequential boosting; strong performance |
| AdaBoost | Boosting with weighted samples |

---

## **3. Support Vector Machines**

### **What these models do**
SVMs find the optimal separating hyperplane between classes by maximizing the margin. Kernel SVMs transform the feature space to capture nonlinear boundaries.

### **Why use them**
- Strong performance on medium‑sized datasets  
- Effective when classes are not linearly separable  
- RBF and polynomial kernels capture complex relationships  
- Provide margin‑based decision boundaries that reduce overfitting

### **Models Included**
| Model | Description |
|-------|-------------|
| Linear SVM | High‑dimensional linear classifier |
| RBF SVM | Nonlinear kernel; powerful |
| Polynomial SVM | Captures interaction effects |

---

## **4. Probabilistic Models**

### **What these models do**
Probabilistic models compute the likelihood of each class given the input features. They assume statistical distributions (e.g., Gaussian) and use Bayes’ theorem to classify.

### **Why use them**
- Extremely fast and lightweight  
- Provide calibrated probabilities  
- Useful when features are independent or categorical  
- Strong baselines for comparison  
- Good for understanding uncertainty in predictions

### **Models Included**
| Model | Description |
|-------|-------------|
| Gaussian Naive Bayes | Assumes normal feature distributions |
| Multinomial Naive Bayes | Useful for count‑based features |
| Bernoulli Naive Bayes | Binary feature variant |

---

## **5. Instance‑Based / Distance‑Based Models**

### **What these models do**
These models classify new samples based on similarity to existing samples. They rely on distance metrics (e.g., Euclidean) rather than learned parameters.

### **Why use them**
- Capture irregular, highly nonlinear boundaries  
- Useful when relationships are local  
- Simple and intuitive  
- Provide strong performance when data is well‑scaled  
- Good sanity‑check models for comparison

### **Models Included**
| Model | Description |
|-------|-------------|
| K‑Nearest Neighbors (KNN) | Non‑parametric, similarity‑based |
| Radius Neighbors | Uses fixed radius instead of k |

---

# **⚙️ Modeling Workflow**
### **1. Preprocessing**
- Stratified train/test split  
- Scaling for linear/SVM/KNN models  
- One‑hot encoding for categorical features  
- Optional feature selection  

### **2. Training**
Each model is trained on all three targets:
- Mean‑based binary  
- Median‑based binary  
- Tiered multiclass  

### **3. Evaluation Metrics**
#### **Binary**
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC‑AUC  
- Confusion Matrix  

#### **Multiclass**
- Macro F1  
- Weighted F1  
- One‑vs‑Rest ROC‑AUC  
- Confusion Matrix  

### **4. Model Comparison**
- Performance tables  
- Ranking by metric  
- Hyperparameter tuning  
- Cross‑validation  
- Feature importance (tree models)  
- SHAP analysis (tree models only)  
- Ensemble stacking  
- SHAP interpretability  

---

# **📂 Repository Structure**
```
classification/
│
├── 01_preprocessing.py
├── 02_target_engineering.py
│
├── 03_linear_models.py
├── 04_tree_models.py
├── 05_svm_models.py
├── 06_probabilistic_models.py
├── 07_knn_models.py
│
├── 08_model_comparison.py
└── README.md
```

