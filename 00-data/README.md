# Master Data Science Salaries — Unified Kaggle Dataset (2020–2025)

This repository consolidates **seven Kaggle datasets** related to data‑science salaries into a single, normalized, deduplicated master dataset.  
The final merged dataset spans **six years (2020–2025)** and contains **53,566 rows × 11 standardized columns**.

---

## Included Kaggle Datasets (Cited)

The following datasets were downloaded using `kagglehub`:

1. **Latest Data Science Job Salaries 2024**  
   `https://www.kaggle.com/datasets/saurabhbadole/latest-data-science-job-salaries-2024` [(kaggle.com in Bing)](https://www.bing.com/search?q="https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fsaurabhbadole%2Flatest-data-science-job-salaries-2024")

2. **Data Science Salaries 2023**  
   [https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023](https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023)

3. **DataScience Salaries 2024**  
   [https://www.kaggle.com/datasets/yusufdelikkaya/datascience-salaries-2024](https://www.kaggle.com/datasets/yusufdelikkaya/datascience-salaries-2024)

4. **Data Science Salaries and Fields**  
   [https://www.kaggle.com/datasets/josiagiven/data-science-salaries-and-fields](https://www.kaggle.com/datasets/josiagiven/data-science-salaries-and-fields)

5. **Data Science Salaries**  
   [https://www.kaggle.com/datasets/sazidthe1/data-science-salaries](https://www.kaggle.com/datasets/sazidthe1/data-science-salaries)

6. **Data Science Salaries**  
   [https://www.kaggle.com/datasets/zain280/data-science-salaries](https://www.kaggle.com/datasets/zain280/data-science-salaries)

7. **Data Science Salary Landscape**  
   [https://www.kaggle.com/datasets/lainguyn123/data-science-salary-landscape](https://www.kaggle.com/datasets/lainguyn123/data-science-salary-landscape)

---

## Pipeline Overview

### **1. Download datasets**
All datasets are downloaded via `kagglehub.dataset_download()`.

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

## Final Output

### **File:** `master_data_science_salaries.csv`

- **Rows:** 53,566  
- **Columns:** 11  
- **Years Covered:** **2020, 2021, 2022, 2023, 2024, 2025**  
- **Missing work_year values:** present (NaN)

### Example unique values for `work_year`:

```
[2025. 2024. 2022. 2023. 2020. 2021.   nan]
```

---

## Reproducibility

Run:

```bash
python cleaned-data.py
```

Dependencies:

```
pandas
glob
os
kagglehub
```

---

## 📜 License & Attribution

Each dataset is governed by its respective Kaggle license.  
This repository does **not** redistribute raw Kaggle data — only a merged CSV generated locally.
