# **04 — Unsupervised Machine Learning**  

- **Started:** October 19, 2026  
- **Last Updated:** October 19, 2026  
- **Status:** In Progress  

---

## **📌 Project Overview**
This module applies **unsupervised machine learning** techniques to the combined **Red + White Wine Quality Dataset** (UCI).  
The goal is to uncover **latent structure**, **natural groupings**, and **anomalous samples** within the chemical composition of wines — without using quality labels.

This section expands the IBM ML Certificate curriculum by implementing **production‑style pipelines**, **modern clustering methods**, **dimensionality reduction**, and **anomaly detection**, with a focus on interpretability and engineering rigor.

---

## **🎯 Objectives**
- Identify **clusters** of wines with similar chemical profiles  
- Understand **drivers of cluster separation** using feature importance  
- Visualize high‑dimensional data using **PCA**, **t‑SNE**, and **UMAP**  
- Detect **outlier wines** that may indicate production inconsistencies  
- Compare clustering algorithms using quantitative metrics  
- Integrate cluster labels into supervised pipelines for hybrid analysis  

---

## **📂 Repository Structure**
```
04-unsupervised-ml/
    README.md
    data/
    notebooks/
        01_eda.ipynb
        02_clustering.ipynb
        03_dimensionality_reduction.ipynb
        04_anomaly_detection.ipynb
    src/
        load_data.py
        preprocess.py
        clustering.py
        dimred.py
        anomaly.py
        evaluate.py
        utils.py
    models/
    reports/
        figures/
        tables/
```

This structure mirrors the engineering style used in your supervised ML pipelines and ensures reproducibility, modularity, and clarity.

---

## **🧪 Methods Implemented**

### **1. Clustering**
- **K‑Means**  
- **Gaussian Mixture Models (GMM)**  
- **Agglomerative Clustering**  
- **DBSCAN**  

**Evaluation Metrics:**
- Silhouette Score  
- Davies–Bouldin Index  
- Cluster Stability  
- Visualization‑based assessment  

---

### **2. Dimensionality Reduction**
- **Principal Component Analysis (PCA)**  
- **t‑SNE**  
- **UMAP**  

Used for:
- Visualizing cluster structure  
- Understanding variance explained  
- Identifying nonlinear manifolds  

---

### **3. Anomaly Detection**
- **Isolation Forest**  
- **Local Outlier Factor (LOF)**  

Used to detect:
- Wines with unusual chemical profiles  
- Potential production anomalies  
- Samples that may distort clustering  

---

## **📊 Model Comparison**
A unified evaluation table will be included:

| Method | Silhouette | DB Index | Notes |
|--------|------------|----------|-------|
| K-Means | TBD | TBD | Strong baseline; spherical clusters |
| GMM | TBD | TBD | Handles overlapping clusters |
| Agglomerative | TBD | TBD | Reveals hierarchical structure |
| DBSCAN | N/A | N/A | Effective for noise + irregular shapes |

---

## **🔍 Cluster Interpretation**
To make clusters actionable, this project includes:

### **SHAP‑based Cluster Interpretation**
Workflow:
1. Train a classifier to predict cluster labels  
2. Use SHAP to identify chemical features driving cluster membership  

This bridges unsupervised learning with supervised interpretability.

---

## **📈 Visualizations**
All visual outputs will be stored in:

`reports/figures/`

Planned visuals:
- PCA scatterplots  
- t‑SNE / UMAP embeddings  
- Cluster overlays  
- Dendrograms  
- Anomaly score distributions  
- SHAP summary plots  

---

## **🔧 Engineering Features**
- Modular Python pipeline (`src/`)  
- Reproducible environment (`requirements.txt`)  
- Unified pipeline script (`unsupervised_pipeline.py`)  
- Lightweight unit tests (`tests/`)  
- Saved models + artifacts (`models/`)  

This section is designed to reflect real‑world ML engineering practices, not course‑style notebooks.

---

## **📘 Dataset**
This project uses the **Wine Quality Dataset** from the UCI Machine Learning Repository. The dataset contains physicochemical measurements for red and white variants of Portuguese “Vinho Verde” wine.

### **Citation**
Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Wine Quality* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C56S3T](https://doi.org/10.24432/C56S3T)

### **Data Source & Licensing**
The dataset is publicly available through the UCI Machine Learning Repository for research and educational use.  
No modifications were made beyond combining the red and white subsets and adding a `color` indicator variable.

---

## **🔗 Hybrid Analysis Preview**
Once clustering and anomaly detection are complete, this module will integrate cluster labels into supervised pipelines to examine:

- Alignment between clusters and quality scores  
- Whether cluster membership improves predictive performance  
- Whether anomalies correspond to misclassified or extreme‑quality samples  

This creates a full end‑to‑end ML workflow that blends unsupervised discovery with supervised prediction.

---

## **🚀 Next Steps**
- Finalize clustering evaluation  
- Add SHAP cluster interpretation  
- Add anomaly detection visualizations  
- Add hybrid supervised + unsupervised analysis  
- Prepare summary report for the capstone section  
