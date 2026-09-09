# **Data Science Salaries 2023 — Analysis & Insights**

This repository contains a structured and reproducible data analysis project built around the **Data Science Salaries 2023** dataset from Kaggle. The goal is to explore compensation trends across the data science profession, uncover meaningful patterns, and visualize how factors such as experience level, job role, company size, location, and remote work influence salary outcomes.

**Dataset Source:**  
Kaggle — *Data Science Salaries 2023*  
[https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023](https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023)

# Project Progress

Created: 2026‑06-29  
Last Edited: 2026‑07‑20  
Author: Aaron Niecestro  

## **Project Objectives**

- Analyze salary distributions across roles and seniority levels  
- Examine how remote work and company size impact compensation  
- Explore geographic salary differences  
- Build preprocessing pipelines for clean, reproducible workflows  
- Generate visual analytics to highlight key trends in the data science job market  

## **Key Questions Explored**

- How do salaries vary by **experience level** (Entry, Mid, Senior, Executive)?  
- Which **job titles** command the highest compensation?  
- Does **remote work** correlate with higher or lower salaries?  
- How does **company size** influence pay?  
- What global **location trends** emerge from the dataset?  

## **Author**

**Aaron Niecestro**  
Data Scientist & Biostatistician 
Project Lead and Sole Contributor

## **Project Timeline**

**Downloaded Data on:** June 29, 2026  
**Started Poject on:** July 19, 2026
**Status:** Ongoing  (Plan to be completed by July 30, 206)
**Progress:** 
- Completed the preliminary analysis first starting with Python
- Converted the preliminary analysis from Python into R and SAS
**Current Phase:** Review the preliminary analysis and decide which models to use

### **Planned Progression**
1. **Python (Phase 1)** — Build the full baseline workflow  
2. **R (Phase 2)** — Translate Python workflow into tidyverse equivalents  
3. **SAS (Phase 3)** — Replicate workflow using SAS procedures
5. **Cross‑Language Comparison (Phase 4)** — Document differences, strengths, and insights  
6. **Extensions (Phase 5)** — Modeling, dashboards, external datasets using Python first then convert into R and SAS (when possible) 

# **Project Roadmap**

A clear roadmap helps document the progression of your cross‑language analysis and shows how each stage builds on the previous one.

### **Phase 1 — Python (Primary Workflow)**
- Set up project structure and environment  
- Import and inspect the Kaggle dataset  
- Clean and preprocess data  
- Conduct exploratory data analysis (EDA)  
- Generate visualizations (salary distributions, role comparisons, remote ratio effects, etc.)  
- Document findings and code structure  
- Export cleaned datasets for cross‑language use  

### **Phase 2 — R (Translation & Statistical Deep Dive)**
- Recreate Python preprocessing using tidyverse  
- Replicate EDA and visualizations in ggplot2  
- Compare R’s statistical summaries with Python’s  
- Highlight differences in syntax, workflow, and data handling  
- Document translation challenges and advantages  
- Export R‑processed datasets for SAS validation  

### **Phase 3 — SAS (Enterprise Replication)**
- Import cleaned datasets using PROC IMPORT  
- Reproduce data cleaning steps using Data Step and PROC SQL  
- Generate SAS‑based descriptive statistics  
- Create equivalent visualizations (PROC SGPLOT, PROC SGSCATTER)  
- Compare SAS workflow efficiency vs Python/R  
- Document enterprise‑style data handling differences  

### **Phase 4 — Cross‑Language Comparison**
- Evaluate readability and workflow complexity  
- Compare visualization capabilities  
- Assess statistical output differences  
- Summarize strengths and limitations of each language  
- Produce a final report consolidating insights  

### **Phase 5 — Extensions**
- Build predictive salary models in each language  
- Add dashboards (Streamlit, Shiny, SAS VA)  
- Integrate external datasets (e.g., cost of living, job postings)  
- Publish results or write a blog post summarizing findings  

---

# **Python vs R vs SAS — Comparison Table**

A clean, high‑level comparison you can include directly in your README:

| Feature / Aspect | **Python** | **R** | **SAS** |
|------------------|------------|-------|---------|
| **Primary Strength** | General‑purpose programming + ML ecosystem | Statistical analysis + tidy data workflows | Enterprise data processing + regulatory environments |
| **Best For** | EDA, modeling, automation, visualization | Statistical modeling, academic analysis, elegant plots | Large datasets, compliance, repeatable production pipelines |
| **Learning Curve** | Moderate | Moderate | Steep (especially PROC syntax) |
| **Data Manipulation** | pandas | dplyr / tidyverse | Data Step + PROC SQL |
| **Visualization** | matplotlib, seaborn, plotly | ggplot2 (highly expressive) | PROC SGPLOT / SGSCATTER |
| **Modeling Tools** | scikit‑learn, statsmodels | caret, tidymodels | PROC REG, PROC GLM, PROC LOGISTIC |
| **Workflow Style** | Script‑based, modular | Functional, pipe‑driven | Procedure‑driven, stepwise |
| **Community & Ecosystem** | Huge, fast‑moving | Strong academic/statistical | Enterprise‑focused, slower evolution |
| **Reproducibility** | High (notebooks + scripts) | High (RMarkdown + scripts) | Very high (strict procedural workflows) |
| **Ideal Use Case** | Flexible data science projects | Statistical deep dives | Enterprise analytics & compliance environments |
