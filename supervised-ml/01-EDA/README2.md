# **01 — Exploratory Data Analysis (EDA)**  

**Supervised Machine Learning Module**  
**Status:** Completed  
**Last Updated:** September 28, 2026  

This folder contains all exploratory data analysis (EDA) work for the **Supervised Machine Learning** module using the unified **Master Data Science Salaries (2020–2025)** dataset.  
The goal of this stage is to understand the dataset’s structure, identify key patterns, detect anomalies, and prepare insights that guide feature engineering and model selection.

## 📊 **Key EDA Objectives**

The EDA process focuses on:

### **1. Understanding Salary Distributions**
- Raw salary vs. salary in USD  
- Variation across job titles, experience levels, and company size  
- Identifying skewed distributions and potential transformations  

### **2. Exploring Categorical Variables**
- Frequency counts  
- Geographic patterns (employee residence vs. company location)  
- Remote work ratios  

### **3. Detecting Outliers**
- Extreme salary values  
- Rare job titles  
- Unusual combinations of experience level and salary  

### **4. Correlation & Feature Relationships**
- Pearson/Spearman correlations  
- Heatmaps  
- Early signals for feature importance  

### **5. Preparing for Modeling**
- Identifying features requiring encoding  
- Flagging variables with low variance  
- Highlighting potential engineered features (e.g., region grouping, seniority bins)

---

## 📘 **Dataset Source**

All EDA is performed on the unified dataset created in:

```
supervised-ml/data/
```

Dataset details and citations are provided in that folder’s README.

---

## 🧪 **Reproducibility Notes**

- All figures and tables are generated programmatically from `eda.ipynb`.  
- No manual edits are applied to the dataset during EDA.  
- Outputs are version‑controlled to ensure consistent results across modeling experiments.  
- The notebook can be re‑run end‑to‑end using the cleaned dataset (`dss2025_final.csv`).

---

## ✅ **EDA Status**

Exploratory Data Analysis is **complete**.  
Insights from this stage directly inform feature engineering and model development in:

```
supervised-ml/02-regression/
supervised-ml/03-classification/
```
