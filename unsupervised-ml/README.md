# **04 — Unsupervised Machine Learning**
**Started:** October 19, 2026  
**Last Updated:** October 25, 2026   
**Status:** ✅ Complete

---

## **📌 Project Overview**

This module implements a comprehensive unsupervised machine learning pipeline using the combined **Red + White Wine Quality Dataset** from the UCI Machine Learning Repository.

The objective is to discover natural structure within wine chemistry data through clustering, dimensionality reduction, hierarchical analysis, and anomaly detection. Multiple algorithms were evaluated to understand how wines organize in feature space, identify rare observations, and uncover latent chemical relationships without relying on supervised labels.

This project expands beyond the IBM Machine Learning Certificate curriculum by incorporating modern clustering techniques, nonlinear manifold learning, anomaly-detection methods, and production-oriented engineering practices.

The project culminated in a comprehensive analytical report and presentation summarizing findings across all implemented methods.

---

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Scipy
- Matplotlib
- Seaborn
- UMAP-learn

---

## **🏆 Major Findings**

Analysis across clustering, dimensionality reduction, and anomaly-detection techniques revealed several highly consistent patterns:

### **Wine Type Dominates Dataset Structure**
Red and white wines consistently formed distinct populations regardless of the algorithm applied.

This pattern was observed across:

- Hierarchical Clustering
- K-Means
- Gaussian Mixture Models
- PCA
- Kernel PCA
- t-SNE
- UMAP

### **Quality Ratings Remain Highly Overlapping**
Wine quality did not form clearly separated groups.

Quality remained broadly distributed across:

- Clusters
- PCA components
- Kernel PCA embeddings
- t-SNE embeddings
- UMAP embeddings
- NMF latent-factor space

### **Evidence of Internal Subpopulations**
Ward linkage, Kernel PCA, t-SNE, and UMAP all suggested meaningful subgroup structure within both red and white wine populations.

This indicates that wine chemistry organization extends beyond a simple red-vs-white distinction.

### **UMAP Produced the Strongest Visualization**
Among all dimensionality-reduction methods evaluated, UMAP provided the strongest balance between:

- Population separation
- Structure preservation
- Visual interpretability

### **NMF Produced the Most Interpretable Latent Factors**
NMF identified meaningful additive chemistry components representing:

- Alcohol and acidity structure
- Sulfur, density, and preservation chemistry

### **Anomaly Structure Depends on Detection Method**
Isolation Forest and Local Outlier Factor identified substantially different anomaly populations.

- Isolation Forest detected globally unusual wines
- LOF detected locally unusual wines

Only **36.09%** of anomalies were identified by both methods.

---

## **🎯 Project Objectives**

This project aimed to:

- Identify natural wine populations using unsupervised learning
- Compare clustering algorithms using quantitative metrics
- Investigate hierarchical relationships between wines
- Visualize wine chemistry using linear and nonlinear embeddings
- Discover latent chemistry structure
- Detect unusual wine observations
- Evaluate agreement across multiple unsupervised-learning methods
- Generate interpretable chemistry-based insights

---

## **📁 Data Pipeline**

The project begins with a unified preprocessing workflow that:

1. Loads red and white wine datasets
2. Adds a `type` indicator
3. Combines datasets
4. Removes duplicate observations
5. Produces cleaned datasets

Output files:

```text
winequality-combined.csv
winequality-final.csv
```

Stored in:

```text
00-data/clean-data/
```

---

## **📚 Conceptual Foundations**

### **Unsupervised Learning**

Unsupervised learning identifies hidden structure without known target labels.

Methods rely on:

- Distance
- Density
- Variance
- Similarity
- Neighborhood structure

to discover patterns within the data.

### **Curse of Dimensionality**

As dimensionality increases:

- Distances become less informative
- Samples become sparse
- Clustering performance can degrade

This motivates the use of dimensionality reduction techniques such as:

- PCA
- Kernel PCA
- t-SNE
- UMAP
- NMF

### **Distance Metrics**

Distance selection directly influences clustering behavior.

This project examined:

- Euclidean Distance
- Manhattan Distance
- Cosine Similarity
- Jaccard Similarity
- Mahalanobis Distance

and their impact on cluster formation.

---

## **🧪 Methods Implemented**

### **1. Clustering Algorithms**

- K-Means
- Gaussian Mixture Models (GMM)
- Agglomerative Clustering
- DBSCAN

Evaluation included:

- Silhouette Score
- Davies-Bouldin Index
- Cluster stability
- Visual analysis

### **2. Hierarchical Linkage Methods**

- Single Linkage
- Complete Linkage
- Average Linkage
- Ward Linkage

Ward linkage produced the most informative and interpretable hierarchical structure.

### **3. Dimensionality Reduction**

#### Principal Component Analysis (PCA)
- Best variance explanation
- Strong component interpretation
- Explained nearly 50% of variance using two components

#### Kernel PCA
- Revealed nonlinear chemistry structure
- Improved class separation over PCA

#### t-SNE
- Produced outstanding local separation
- Revealed strong population boundaries

#### UMAP
- Produced the strongest overall embedding
- Preserved both local and global structure

#### Non-Negative Matrix Factorization (NMF)
- Identified latent chemistry factors
- Provided the strongest chemistry interpretation

### **4. Anomaly Detection**

#### Isolation Forest
- Detected globally unusual wines
- Identified 266 anomalies
- Primarily flagged red wines

#### Local Outlier Factor (LOF)
- Detected locally unusual wines
- Identified 266 anomalies
- Primarily flagged white wines

---

## **📊 Dimensionality Reduction Summary**

| Category | Best Method |
|-----------|-----------|
| Variance Explanation | PCA |
| Feature Interpretation | NMF |
| Nonlinear Structure | Kernel PCA |
| Local Separation | t-SNE |
| Global Structure | UMAP |
| Overall Visualization | UMAP |

---

## **📊 Anomaly Detection Summary**

| Metric | Result |
|----------|----------|
| Isolation Forest Anomalies | 266 |
| LOF Anomalies | 266 |
| Shared Anomalies | 96 |
| Agreement Between Methods | 36.09% |

Key findings:

- Isolation Forest emphasized globally extreme wines
- LOF emphasized locally unusual wines
- Different anomaly detectors reveal different forms of rare behavior

---

## **📈 Visualizations Generated**

Outputs include:

- PCA projections
- Kernel PCA embeddings
- t-SNE embeddings
- UMAP embeddings
- NMF component analysis
- Cluster visualizations
- Dendrograms
- Distance-metric comparisons
- Anomaly score distributions
- Anomaly visualizations in reduced-dimensional space

---

## **🔧 Engineering Features**

Project architecture follows production-oriented design principles:

- Modular workflows
- Script-based pipelines
- Reproducible processing
- Saved intermediate outputs
- Structured exports for reporting
- Artifact generation for downstream analysis

---

## **📘 Dataset**

**UCI Wine Quality Dataset**

Physicochemical measurements for red and white Portuguese Vinho Verde wines.

### **Citation**

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009).

*Wine Quality Dataset.*

UCI Machine Learning Repository.

DOI: https://doi.org/10.24432/C56S3T

---

## **✅ Overall Conclusions**

Multiple unsupervised-learning techniques were applied to understand the structure, composition, and rare behaviors present within the Wine Quality dataset.

Across nearly every analysis, wine type emerged as the dominant source of variation.

Major conclusions include:

- Red and white wines form distinct chemistry populations
- Quality ratings do not form clearly separated groups
- Additional subgroup structure exists within wine populations
- UMAP delivered the strongest overall visualization
- NMF revealed interpretable latent chemistry factors
- Isolation Forest and LOF discovered different forms of anomalous behavior
- Multiple independent methods reached highly consistent conclusions

Together, clustering, hierarchical analysis, dimensionality reduction, and anomaly detection provided a comprehensive view of wine chemistry structure and demonstrated the effectiveness of modern unsupervised machine-learning techniques for exploratory data analysis.

This project demonstrates practical application of clustering, dimensionality reduction, and anomaly detection techniques within a real-world analytical workflow. By combining multiple complementary methods, the analysis provides a robust understanding of wine chemistry structure while showcasing modern unsupervised machine learning practices using Python and Scikit-Learn.
---

## ✅ Deliverables

Completed deliverables include:

- Comprehensive analytical report
- Executive presentation deck
- Reproducible Python workflows
- Visualizations and exported artifacts
- GitHub portfolio documentation
