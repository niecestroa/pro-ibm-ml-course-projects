# **Data Science Salaries — Exploratory Data Analysis (EDA)**  
This folder contains the full exploratory analysis workflow for the **Data Science Salaries** datasets sourced from Kaggle. The goal of this EDA is to understand compensation trends across the data science profession and establish a clean, reproducible foundation for later modeling work.

Timelines:
Started On:    September 8, 2026
Completed on:   September 10, 2026

---

## **Author**  
**Aaron Niecestro**  
Data Scientist & Biostatistician  
Lead Analyst and Sole Contributor

---

## **Included Kaggle Datasets (Cited)**  
The following datasets were downloaded using `kagglehub` and used for exploratory analysis:

1. **Latest Data Science Job Salaries 2024**  
2. **Data Science Salaries 2023**  
3. **DataScience Salaries 2024**  
4. **Data Science Salaries and Fields**  
5. **Data Science Salaries** (Sazidthe1)  
6. **Data Science Salaries** (Zain280)  
7. **Data Science Salary Landscape**

These datasets provide overlapping but complementary views of the data science job market across roles, industries, and regions.

---

## **EDA Objectives**

- Explore salary distributions across roles and seniority levels  
- Examine how remote work, company size, and employment type influence compensation  
- Identify geographic salary patterns  
- Build clean preprocessing pipelines in **Python** and **R**  
- Produce visual analytics to highlight key trends  
- Export cleaned datasets for downstream modeling  

---

## **Key Questions Investigated**

- How do salaries vary by **experience level** (Entry, Mid, Senior, Executive)?  
- Which **job titles** consistently show higher compensation?  
- Does **remote work** correlate with higher or lower pay?  
- How does **company size** affect salary outcomes?  
- What global **location trends** emerge across datasets?  

---

# **Project Timeline (EDA Scope)**

- **Data Downloaded:** June 29, 2026  
- **EDA Started:** July 19, 2026  
- **EDA Completed:** **September 10, 2025**  
- **Status:** EDA finalized; modeling planned for later phases  

---

# **EDA Workflow Overview**

### **Phase 1 — Python EDA (Primary Workflow)**  
- Import and inspect datasets  
- Clean and preprocess data  
- Conduct exploratory data analysis  
- Generate visualizations (salary distributions, role comparisons, remote ratio effects, etc.)  
- Export cleaned datasets for R translation  

### **Phase 2 — R EDA (Translation & Statistical Deep Dive)**  
- Recreate Python preprocessing using tidyverse  
- Replicate EDA and visualizations using ggplot2  
- Compare R’s statistical summaries with Python’s  
- Document syntax and workflow differences  
- Export R‑processed datasets for modeling  

### **Phase 3 — Cross‑Language Comparison (Python vs R)**  
- Compare readability and workflow complexity  
- Evaluate visualization expressiveness  
- Summarize statistical output differences  
- Document strengths and limitations of each language  

---

# **Python vs R — EDA Comparison Table**

| Feature / Aspect | **Python** | **R** |
|------------------|------------|-------|
| **Primary Strength** | General‑purpose programming + ML ecosystem | Statistical analysis + tidy data workflows |
| **Best For** | EDA, automation, modeling | Statistical summaries, elegant visualizations |
| **Learning Curve** | Moderate | Moderate |
| **Data Manipulation** | pandas | dplyr / tidyverse |
| **Visualization** | matplotlib, seaborn, plotly | ggplot2 (highly expressive) |
| **Workflow Style** | Script‑based, modular | Functional, pipe‑driven |
| **Reproducibility** | High (notebooks + scripts) | High (RMarkdown + scripts) |
| **Ideal Use Case** | Flexible data science projects | Statistical deep dives |
