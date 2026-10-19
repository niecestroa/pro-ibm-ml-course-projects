## **Anomaly Detection**

This section identifies unusual samples that may distort clustering, represent rare chemical profiles, or indicate production inconsistencies.

---

## **🔍 Methods Implemented**

### **1. Isolation Forest**
- Tree‑based anomaly detection  
- Isolates points by random splits  
- Effective for high‑dimensional numeric data  
- Produces anomaly scores  

### **2. Local Outlier Factor (LOF)**
- Density‑based anomaly detection  
- Compares local neighborhood density  
- Flags points in sparse regions as anomalies  

---

## **📊 Evaluation & Visualization**

- Anomaly score distributions  
- PCA / t‑SNE / UMAP visualization with anomalies highlighted  
- Cluster distortion checks  
- Removal of anomalies before clustering  

---

## **🎯 Purpose of This Module**

Anomaly detection helps identify:

- rare chemical profiles  
- production inconsistencies  
- samples that distort cluster boundaries  
- noise that affects clustering algorithms  

---

## **Integration with Clustering**

Anomalies are analyzed both:

- **before clustering** (to remove distortions)  
- **after clustering** (to identify cluster‑specific outliers)  
