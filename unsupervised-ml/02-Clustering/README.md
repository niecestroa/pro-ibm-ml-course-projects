# **02 — Clustering**

This module implements the core clustering algorithms from the IBM Machine Learning curriculum using a fully numeric dataset. The goal is to discover natural groupings, evaluate cluster quality, and visualize latent structure using multiple clustering techniques.

---

## **📌 Clustering Algorithms Implemented**

### **1. K‑Means**
- Euclidean distance  
- Assumes spherical clusters  
- Uses K‑Means++ initialization  
- Evaluated across multiple values of *k*

### **2. Gaussian Mixture Models (GMM)**
- Probabilistic clustering  
- Soft cluster assignments  
- Multiple covariance structures tested (full, diag, spherical)

### **3. Agglomerative Clustering**
- Hierarchical clustering  
- Multiple linkage methods:
  - Ward  
  - Complete  
  - Average  
- Dendrogram‑based structure exploration

### **4. DBSCAN**
- Density‑based clustering  
- Automatically detects noise/outliers  
- Sensitive to ε and min_samples tuning

---

## **📊 Evaluation Metrics**

Each clustering algorithm is evaluated using:

- **Silhouette Score**  
- **Davies–Bouldin Index**  
- **Cluster Stability Checks**  
- **PCA / t‑SNE / UMAP Visualization**

These metrics provide a consistent framework for comparing cluster quality across algorithms.

---

---

## 🏆 Clustering Results Summary

| Algorithm | Final Model | Silhouette | Davies-Bouldin | Primary Strength |
|------------|------------|-----------:|---------------:|-----------------|
| K-Means | k = 4 | 0.2428 | 1.4844 | Best wine-profile segmentation |
| GMM | 2 Components (diag) | 0.2516 | 1.7782 | Best probabilistic clustering |
| Agglomerative | Ward k = 2 | 0.2607 | 1.7164 | Best recovery of natural structure |
| DBSCAN | eps = 1.3, min_samples = 10 | 0.1505 | 3.3824 | Best outlier and boundary detection |

### Key Findings

- **Agglomerative Clustering (Ward k=2)** achieved the highest Silhouette Score and best recovered the dominant latent structure in the wine dataset.

- **K-Means (k=4)** produced the most interpretable business segmentation by identifying four chemically distinct wine profiles.

- **Gaussian Mixture Models** reproduced the red-versus-white wine distinction with approximately 95% agreement while providing a probabilistic framework.

- **DBSCAN** identified two dense wine populations and classified 26.8% of observations as transitional or sparse-region wines.

---

## 🎯 Purpose of This Module

This clustering section provides:

- a foundation for unsupervised learning
- a comparison of major clustering paradigms
- visual intuition for latent structure
- automated model evaluation and comparison
- business-oriented wine segmentation insights

Results from this module indicate that:

- Agglomerative Clustering most effectively captured the natural structure of the dataset.
- K-Means produced the most actionable segmentation for business analysis.
- DBSCAN provided useful identification of sparse and transitional observations.

