# 🍷 **Unsupervised ML Data — Wine Quality Dataset**

This directory contains all data used for the **Unsupervised Machine Learning (Course 4)** project based on the **UCI Wine Quality Dataset**.  
The data is organized into **raw** and **clean** subfolders to support a fully reproducible, production‑aligned workflow.

---

## 📁 **Folder Structure**

```
data/
│
├── raw/
│   ├── winequality-red.csv
│   ├── winequality-white.csv
│   ├── winequality.names
│
└── clean/
    ├── winequality_combined.csv
    ├── winequality_combined_clean.csv
```

### **raw/**
Contains the **original, untouched datasets** downloaded directly from the UCI Machine Learning Repository:

- `winequality-red.csv`  
- `winequality-white.csv`  
- `winequality.names` (metadata file)

These files are preserved exactly as provided by the dataset authors.

### **clean/**
Contains **processed, analysis‑ready datasets** created during the data cleaning pipeline:

- `winequality_combined.csv`  
  - Red + white datasets merged  
  - Added `color` column (`"red"` or `"white"`)

- `winequality_combined_clean.csv`  
  - Fully cleaned version  
  - Standardized column names  
  - Removed duplicates  
  - Optional transformations applied (e.g., log transforms for skewed features)  
  - Ready for clustering, PCA, t‑SNE, UMAP, and anomaly detection  

---

## 📘 **Dataset Description**

The Wine Quality dataset contains physicochemical measurements of red and white variants of Portuguese *Vinho Verde* wine.

### **Features (11 numeric variables)**  
- fixed acidity  
- volatile acidity  
- citric acid  
- residual sugar  
- chlorides  
- free sulfur dioxide  
- total sulfur dioxide  
- density  
- pH  
- sulphates  
- alcohol  

### **Targets (used only for interpretation)**  
- `quality` — integer score (0–10)  
- `color` — added manually during cleaning  

---

## 📚 **Citation**

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Wine Quality* [Dataset].  
UCI Machine Learning Repository.  
[https://doi.org/10.24432/C56S3T](https://doi.org/10.24432/C56S3T)

---

## 🔧 **How This Data Is Used in the Project**

The cleaned dataset is used for:

### **Clustering**
- K‑Means  
- Gaussian Mixture Models  
- Agglomerative Clustering  
- DBSCAN  

### **Dimensionality Reduction**
- PCA  
- t‑SNE  
- UMAP  

### **Anomaly Detection**
- Isolation Forest  
- Local Outlier Factor  

### **Interpretation**
- Cluster profiling  
- Quality distribution analysis  
- Color composition  
- PCA/t‑SNE/UMAP visualization  

---

## 🧪 **Reproducibility Notes**

- All transformations applied to raw data are documented in the `unsupervised-ml` folder.  
- Cleaned datasets are version‑controlled to ensure consistent results across experiments.  
- Raw data is never modified — all processing happens in the `clean/` folder.
