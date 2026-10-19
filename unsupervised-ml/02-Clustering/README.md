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

## **🎯 Purpose of This Module**

This clustering section provides:

- a foundation for unsupervised learning  
- a comparison of major clustering paradigms  
- visual intuition for latent structure  
- insights that feed into:
  - dimensionality reduction  
  - anomaly detection  
  - hybrid supervised + unsupervised workflows  
