# **Master Data Science Salaries — Unified Kaggle Dataset (2020–2025)**

This module prepares a unified dataset for supervised machine learning tasks by merging multiple Kaggle salary datasets into a single, standardized schema.  
The goal is to create a clean, consistent foundation for downstream modeling, EDA, and salary‑prediction workflows.

The final merged dataset spans **six years (2020–2025)** and contains **53,566 rows × 11 standardized columns**.

---

## **Why Merge These Datasets?**
The included Kaggle datasets contain overlapping but inconsistent salary records across different years and sources.  
Merging them provides:

- A single source of truth  
- A consistent schema  
- Multi‑year coverage  
- Deduplicated, clean data for ML pipelines  

---

## **Included Kaggle Datasets (Cited)**

The following datasets were downloaded using `kagglehub`:

1. **[Latest Data Science Job Salaries 2024](https://www.bing.com/search?q="https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fsaurabhbadole%2Flatest-data-science-job-salaries-2024")** <br>
2. **[Data Science Salaries 2023](https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023)** <br>
3. **[DataScience Salaries 2024](https://www.kaggle.com/datasets/yusufdelikkaya/datascience-salaries-2024)** <br>
4. **[Data Science Salaries and Fields](https://www.kaggle.com/datasets/josiagiven/data-science-salaries-and-fields)** <br>
5. **[Data Science Salaries](https://www.kaggle.com/datasets/sazidthe1/data-science-salaries)** <br>
6. **[Data Science Salaries](https://www.kaggle.com/datasets/zain280/data-science-salaries)** <br>
7. **[Data Science Salary Landscape](https://www.kaggle.com/datasets/lainguyn123/data-science-salary-landscape)** <br>

(Links included in your original version.)

---

## **Pipeline Overview**

### **1. Download datasets**
All datasets are downloaded via:

```python
kagglehub.dataset_download()
```

### **2. Load CSV files**
- Loads all `.csv` files from each dataset folder  
- Normalizes column names  
- Removes duplicate columns

### **3. Standardize schema**
All datasets are reindexed to the following 11 required columns:

```
work_year
experience_level
employment_type
job_title
salary
salary_currency
salary_in_usd
employee_residence
remote_ratio
company_location
company_size
```

### **4. Merge datasets**
All cleaned DataFrames are concatenated.

### **5. Deduplicate**
Duplicate rows across datasets are removed.

---

## **Schema Definition**

| Column | Description |
|--------|-------------|
| work_year | Year of the salary record |
| experience_level | Junior/Mid/Senior/Executive |
| employment_type | FT/PT/Contract/Intern |
| job_title | Role title |
| salary | Raw salary amount |
| salary_currency | Currency code |
| salary_in_usd | Converted salary |
| employee_residence | Country of employee |
| remote_ratio | % remote work |
| company_location | Country of employer |
| company_size | S/M/L |

---

## **Final Output**

### **File:** `dss2025_final`

- **Rows:** 51,240  
- **Columns:** 16  
- **Years Covered:** 2020–2025  
- **Missing values:** None

### Example unique values for `work_year`:

```
[2025. 2024. 2022. 2023. 2020. 2021.   nan]
```

---

## **Data Quality Notes**
- All datasets normalized to a shared schema  
- Duplicate rows removed  
- Missing values resolved through schema alignment  
- Final dataset contains **no null values**  

---

## **Reproducibility**

Run:

```bash
python final_data.py
```

Dependencies:

```
pandas
glob
os
kagglehub
```

---

## **License & Attribution**

Each dataset is governed by its respective Kaggle license.  
This repository does **not** redistribute raw Kaggle data — only the merged CSV generated locally.

---

## **Status**
Data engineering stage completed.  
The unified dataset is ready for supervised ML modeling, EDA, and downstream analysis.  
