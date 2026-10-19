## **Dimensionality Reduction**

This section implements linear and nonlinear dimensionality‑reduction techniques to visualize cluster structure, reduce noise, and reveal latent patterns in high‑dimensional numeric data.

---

## **📉 Methods Implemented**

### **1. PCA (Principal Component Analysis)**
- Linear  
- Variance‑preserving  
- Used for 2D/3D visualization  
- Helps identify dominant directions of variation  

### **2. Kernel PCA (RBF Kernel)**
- Nonlinear  
- Captures curved manifolds  
- Useful when PCA fails to separate clusters  

### **3. t‑SNE**
- Nonlinear  
- Excellent for local structure  
- Strong separation in 2D/3D embeddings  
- Sensitive to perplexity and learning rate  

### **4. UMAP**
- Nonlinear  
- Preserves global + local structure  
- Faster and more stable than t‑SNE  
- Great for cluster visualization  

### **5. NMF (Non‑Negative Matrix Factorization)**
- Produces additive components  
- Useful for interpretability when data is non‑negative  

---

## **📊 Evaluation & Usage**

Dimensionality‑reduction outputs are used for:

- cluster visualization  
- noise reduction  
- preprocessing for clustering  
- identifying nonlinear structure  
- comparing embedding quality  

---

## **🎯 Purpose of This Module**

This section provides visual intuition for:

- cluster separation  
- manifold structure  
- nonlinear relationships  
- embedding quality across methods  
