# **01 — Exploratory Data Analysis (EDA)**  

- **Supervised Machine Learning Module**  
- **Status:** Completed

This folder contains the full exploratory data analysis (EDA) workflow for the **Master Data Science Salaries (2020–2025)** dataset.  
The goal of this stage is to understand compensation trends across the data science profession, evaluate feature relationships, detect anomalies, and prepare insights that guide feature engineering and model development.

EDA was performed in **Python** as the primary workflow.  
An R translation was planned but not completed due to KaggleHub import issues and data‑cleaning inconsistencies. R support may be added in a future update.

---

## 👤 **Author**
**Aaron Niecestro**  
Data Scientist & Biostatistician  
Lead Analyst and Sole Contributor

---

## 📅 **Project Timeline**
- **Data Downloaded:** June 29, 2026  
- **EDA Started:** July 19, 2026  
- **EDA Completed:** September 28, 2026  
- **Status:** EDA finalized; modeling completed in later phases  

---

## 📚 **Included Kaggle Datasets (Cited)**  
The following datasets were downloaded using `kagglehub` and used for exploratory analysis:

1. Latest Data Science Job Salaries 2024  
2. Data Science Salaries 2023  
3. DataScience Salaries 2024  
4. Data Science Salaries and Fields  
5. Data Science Salaries (Sazidthe1)  
6. Data Science Salaries (Zain280)  
7. Data Science Salary Landscape  

These datasets provide overlapping but complementary views of the global data‑science job market.

---

## 📊 **Key EDA Objectives**

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

## 🔍 **Key Questions Investigated**
- How do salaries vary by **experience level** (Entry, Mid, Senior, Executive)?  
- Which **job titles** consistently show higher compensation?  
- Does **remote work** correlate with higher or lower pay?  
- How does **company size** affect salary outcomes?  
- What global **location trends** emerge across datasets?  

---

## 🧪 **EDA Workflow Overview (Python)**

### **Phase 1 — Python EDA (Primary Workflow)**
- Import and inspect datasets  
- Clean and preprocess data  
- Conduct exploratory analysis  
- Generate visualizations (salary distributions, role comparisons, remote ratio effects, etc.)  
- Export cleaned datasets for modeling  

### **Phase 2 — (Optional Future Work) R EDA**
R translation was planned but not completed due to:
- KaggleHub import issues  
- Data‑cleaning inconsistencies  
- Time constraints during project timeline  

R support may be added in a future update for:
- Statistical deep dives  
- ggplot2 visualization comparisons  
- Cross‑language reproducibility checks  

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
