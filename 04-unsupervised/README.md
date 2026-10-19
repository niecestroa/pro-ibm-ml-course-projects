# 🍷 Unsupervised Machine Learning on the UCI Wine Quality Dataset  

- **Created: October 19, 2026**
- **Updated: October 19, 2026**
- **Progress: Ongoing**

This project applies a full unsupervised machine learning workflow to the combined **red + white Wine Quality dataset** from the UCI Machine Learning Repository. The goal is to discover latent wine profiles, visualize underlying structure, and identify anomalous wines using physicochemical attributes alone.

---

## 📚 Citation for Data  
Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Wine Quality* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C56S3T](https://doi.org/10.24432/C56S3T).

---

## 📘 Project Summary

This repository implements a **numeric unsupervised ML pipeline** including:

1. **Data Integration** — Combine red + white datasets; add `color` indicator.  
2. **Preprocessing** — Scaling, optional log transforms, feature matrix construction.  
3. **Exploratory Data Analysis (EDA)** — Distribution analysis, correlations, red vs white comparisons.  
4. **Clustering** — K-Means, GMM, Agglomerative, DBSCAN; internal metric evaluation.  
5. **Dimensionality Reduction** — PCA, t-SNE, UMAP for structure visualization.  
6. **Anomaly Detection** — Isolation Forest, LOF to identify unusual wines.  
7. **Interpretation** — Cluster profiles, quality distributions, color composition, anomaly insights.

This workflow is designed to reveal **latent wine styles**, **nonlinear structure**, and **outlier samples** in the physicochemical feature space.

---

## 🔢 Numeric Unsupervised ML Plan (Project Workflow)

### **1. Data Loading & Integration**
- Load red and white datasets.  
- Add `color` column (`"red"` or `"white"`).  
- Concatenate into a unified dataset.  

### **2. Preprocessing**
- Select physicochemical features (11 numeric variables).  
- Standardize features using `StandardScaler`.  
- Optional: log-transform skewed variables (residual sugar, chlorides, sulphates).  
- Construct:
  - **X** — scaled numeric features  
  - **y_quality** — quality score (for interpretation only)  
  - **y_color** — red/white indicator  

### **3. Exploratory Data Analysis**
- Summary statistics.  
- Histograms, KDEs, boxplots.  
- Correlation heatmap.  
- Red vs white comparisons.  
- Quality vs feature relationships.  

### **4. Clustering Experiments**
Algorithms:
- **K-Means**  
- **Gaussian Mixture Models (GMM)**  
- **Agglomerative Clustering (Ward)**  
- **DBSCAN**

Model selection:
- Silhouette score  
- Calinski–Harabasz  
- Davies–Bouldin  
- Elbow method (for K-Means/GMM)

Outputs:
- Cluster labels  
- Cluster centroids / means  
- Cluster composition by color & quality  

### **5. Dimensionality Reduction**
Methods:
- **PCA** — variance explained + 2D/3D visualization  
- **t-SNE** — nonlinear structure  
- **UMAP** — manifold-based embedding  

Visualization:
- Points colored by cluster  
- Optional overlays: quality score, wine color  

### **6. Anomaly Detection**
Algorithms:
- **Isolation Forest**  
- **Local Outlier Factor (LOF)**  

Outputs:
- Anomaly scores  
- Top anomalous wines  
- Comparison with extreme quality ratings  

### **7. Interpretation & Reporting**
- Cluster profiles (mean feature values).  
- Quality distribution per cluster.  
- Color distribution per cluster.  
- PCA/t-SNE/UMAP plots.  
- Anomaly tables.  

---

## 🧪 Code Skeleton (High-Level)

### Load Data
```python
from src.data import load_wine_data
df = load_wine_data()
```

### Preprocess
```python
from src.preprocessing import prepare_features
X_scaled, y_quality, y_color, scaler = prepare_features(df)
```

### Clustering
```python
from src.clustering import run_kmeans
results = run_kmeans(X_scaled, k_list=[3,4,5,6])
```

### PCA Visualization
```python
from src.dim_reduction import run_pca, plot_embedding
pca, X_pca = run_pca(X_scaled)
plot_embedding(X_pca, labels=results[4]["labels"])
```

### Cluster Profiling
```python
from src.evaluation import cluster_profile
profile = cluster_profile(df, results[4]["labels"], feature_cols, "quality", "color")
print(profile)
```

---

## 📊 Example Outputs

- PCA scatterplots colored by cluster  
- t-SNE/UMAP embeddings showing nonlinear separation  
- Cluster profile tables summarizing physicochemical signatures  
- Anomaly lists highlighting unusual wines  

---

## 🛠️ Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
umap-learn
```

---

## 🚀 Future Work

- Deep clustering (DEC, VaDE)  
- Autoencoder-based anomaly detection  
- Cluster stability analysis  
- Wine recommendation engine based on cluster proximity  
