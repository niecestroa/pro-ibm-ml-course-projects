# **04 — Unsupervised Machine Learning**  
**Started:** October 19, 2026  
**Last Updated:** October 19, 2026  
**Status:** In Progress  

---

## **📌 Project Overview**
This module implements a full **unsupervised machine learning pipeline** on the combined **Red + White Wine Quality Dataset** (UCI).  
The goal is to uncover **latent structure**, **natural groupings**, **nonlinear manifolds**, and **anomalous wines** using clustering and dimensionality‑reduction techniques.

This section expands the IBM ML Certificate curriculum by incorporating **modern clustering algorithms**, **distance‑metric analysis**, **nonlinear dimensionality reduction**, and **production‑style engineering practices**.

---

## **🎯 Objectives**
- Identify **clusters** of wines with similar physicochemical profiles  
- Understand **drivers of cluster separation** using interpretable methods  
- Visualize high‑dimensional structure using PCA, t‑SNE, and UMAP  
- Detect **outlier wines** using anomaly‑detection algorithms  
- Compare clustering algorithms using quantitative metrics  
- Integrate cluster labels into supervised pipelines for hybrid modeling  

---

## **📁 Data Pipeline**
This module begins with a unified data‑cleaning script that:

1. Loads red + white wine datasets  
2. Adds a `type` indicator  
3. Combines datasets  
4. Removes duplicates  
5. Saves both:  
   - `winequality-combined.csv`  
   - `winequality-final.csv`  

All cleaned data is stored in:

```
00-data/clean-data/
```

---

## **📚 Conceptual Foundations (From Notes)**
This module is grounded in key unsupervised‑learning concepts:

### **Unsupervised Learning**
Used when no labels exist. Algorithms infer structure using geometry, distance, density, and variance.

### **Curse of Dimensionality**
High‑dimensional data becomes sparse, distances lose meaning, and clustering degrades.  
This motivates **dimensionality reduction** (PCA, t‑SNE, UMAP).

### **Distance Metrics**
Distance definitions shape cluster formation.  
This project evaluates:

- **Euclidean** — required for K‑Means, Ward linkage  
- **Manhattan** — better in high‑dimensional or sparse settings  
- **Cosine** — ideal for directional similarity  
- **Jaccard** — binary/set‑based similarity  
- **Mahalanobis** — accounts for covariance; strong for anomaly detection  

Distance choice is as important as algorithm choice.

---

## **🧪 Methods Implemented**

### **1. Clustering Algorithms**
- **K‑Means** (Euclidean; spherical clusters)  
- **Gaussian Mixture Models (GMM)** (probabilistic clusters)  
- **Agglomerative Clustering** (hierarchical structure; multiple linkages)  
- **DBSCAN** (density‑based; detects noise/outliers)  

Each algorithm is evaluated using:

- Silhouette Score  
- Davies–Bouldin Index  
- Cluster stability  
- PCA/t‑SNE/UMAP visualization  

---

### **2. Linkage Methods (Hierarchical Clustering)**
Included for interpretability and multi‑level structure:

- **Single** — preserves elongated clusters  
- **Complete** — compact clusters; noise‑resistant  
- **Average** — balanced behavior  
- **Ward** — minimizes inertia; Euclidean only  

---

### **3. Dimensionality Reduction**
Used for visualization, noise reduction, and improving clustering:

- **PCA** — linear; variance‑preserving
- **Kernel PCA** (nonlinear; RBF kernel) 
- **t‑SNE** — nonlinear; excellent for 2D/3D visualization  
- **UMAP** — nonlinear; preserves global + local structure  
- **NMF** (non‑negative data)  

---

### **4. Anomaly Detection**
To identify unusual wines:

- **Isolation Forest**  
- **Local Outlier Factor (LOF)**  

These help detect:
- Production inconsistencies  
- Rare chemical profiles  
- Samples that distort clustering  

---

## **📊 Model Comparison Table**
| Method | Silhouette | DB Index | Notes |
|--------|------------|----------|-------|
| K-Means | TBD | TBD | Fast baseline; spherical clusters |
| GMM | TBD | TBD | Soft clustering; overlapping groups |
| Agglomerative | TBD | TBD | Hierarchical structure; linkage‑dependent |
| DBSCAN | N/A | N/A | Irregular shapes; noise detection |

---

## **🔍 Cluster Interpretation**
To make clusters actionable:

### **SHAP‑Based Interpretation**
1. Train a classifier to predict cluster labels  
2. Use SHAP to identify chemical features driving cluster membership  

This bridges unsupervised discovery with supervised interpretability.

---

## **📈 Visualizations**
Stored in:

```
reports/figures/
```

Planned outputs:
- PCA scatterplots  
- t‑SNE / UMAP embeddings  
- Cluster overlays  
- Dendrograms  
- Distance‑metric comparisons  
- Anomaly score distributions  
- SHAP summary plots  

---

## **🔧 Engineering Features**
- Modular pipeline (`src/`)  
- Unified unsupervised pipeline script  
- Reproducible environment (`requirements.txt`)  
- Lightweight unit tests (`tests/`)  
- Saved models + artifacts (`models/`)  

This module is designed to reflect **real‑world ML engineering**, not notebook‑style coursework.

---

## **📘 Dataset**
UCI Wine Quality Dataset — physicochemical measurements for red and white Vinho Verde wines.

### **Citation**
Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Wine Quality* [Dataset]. UCI Machine Learning Repository.  
[https://doi.org/10.24432/C56S3T](https://doi.org/10.24432/C56S3T)

---

## **🔗 Hybrid Analysis Preview**
After clustering:

- Compare clusters with quality scores  
- Evaluate cluster‑aware supervised models  
- Examine whether anomalies correspond to extreme or misclassified samples  

This creates a full end‑to‑end ML workflow blending unsupervised + supervised learning.

---

## **🚀 Next Steps**
- Complete EDA + distance‑metric analysis  
- Implement clustering evaluation suite  
- Add SHAP cluster interpretation  
- Add anomaly detection visualizations  
- Build hybrid supervised + unsupervised pipeline  
- Prepare capstone summary report  
