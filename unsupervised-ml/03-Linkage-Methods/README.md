## **Linkage Methods Implemented**

### **1. Single Linkage**
- Connects clusters using the minimum pairwise distance  
- Preserves elongated or chain‑like clusters  
- Sensitive to noise and outliers  

### **2. Complete Linkage**
- Uses maximum pairwise distance  
- Produces compact, well‑separated clusters  
- More robust to noise  

### **3. Average Linkage**
- Uses mean pairwise distance  
- Balanced behavior between single and complete  
- Often yields stable, interpretable clusters  

### **4. Ward Linkage**
- Minimizes within‑cluster variance (inertia)  
- Requires Euclidean distance  
- Produces spherical, compact clusters  

---

## **📊 Evaluation & Visualization**

- Dendrograms for hierarchical structure  
- Cluster cut‑level selection  
- Silhouette Score  
- Davies–Bouldin Index  
- PCA / t‑SNE / UMAP visualization of final clusters  

---

## **🎯 Purpose of This Module**

Linkage methods provide interpretability and insight into:

- how clusters merge  
- cluster hierarchy  
- cluster shape assumptions  
- multi‑level structure not captured by K‑Means or GMM  
