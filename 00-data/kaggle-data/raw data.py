# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 03:53:27 2026
@author: aniec
"""

import kagglehub
import pandas as pd
import glob
import os

# ---------------------------------------------------------
# 1. DOWNLOAD ALL DATASETS (LINKS LEFT IN COMMENTS)
# ---------------------------------------------------------

# https://www.kaggle.com/datasets/saurabhbadole/latest-data-science-job-salaries-2024/data
paths = []
paths.append(kagglehub.dataset_download("saurabhbadole/latest-data-science-job-salaries-2024"))

# https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023
paths.append(kagglehub.dataset_download("arnabchaki/data-science-salaries-2023"))

# https://www.kaggle.com/datasets/yusufdelikkaya/datascience-salaries-2024
paths.append(kagglehub.dataset_download("yusufdelikkaya/datascience-salaries-2024"))

# https://www.kaggle.com/datasets/josiagiven/data-science-salaries-and-fields
paths.append(kagglehub.dataset_download("josiagiven/data-science-salaries-and-fields"))

# https://www.kaggle.com/datasets/sazidthe1/data-science-salaries
paths.append(kagglehub.dataset_download("sazidthe1/data-science-salaries"))

# https://www.kaggle.com/datasets/zain280/data-science-salaries
paths.append(kagglehub.dataset_download("zain280/data-science-salaries"))

# https://www.kaggle.com/datasets/lainguyn123/data-science-salary-landscape
paths.append(kagglehub.dataset_download("lainguyn123/data-science-salary-landscape"))

print("Downloaded dataset folders:")
for p in paths:
    print(p)

# ---------------------------------------------------------
# 2. LOAD ALL CSV FILES
# ---------------------------------------------------------

dfs = []

for folder in paths:
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    if len(csv_files) == 0:
        print(f"No CSV found in {folder}")
        continue

    for csv in csv_files:
        print(f"Loading: {csv}")
        df = pd.read_csv(csv)

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        # Remove duplicate columns (compatible method)
        df = df.loc[:, ~df.columns.duplicated()]

        dfs.append(df)

print(f"\nLoaded {len(dfs)} datasets.")

# ---------------------------------------------------------
# 3. DEFINE REQUIRED SCHEMA
# ---------------------------------------------------------

required_cols = [
    "work_year",
    "experience_level",
    "employment_type",
    "job_title",
    "salary",
    "salary_currency",
    "salary_in_usd",
    "employee_residence",
    "remote_ratio",
    "company_location",
    "company_size"
]

# ---------------------------------------------------------
# 4. REINDEX EACH DATASET TO REQUIRED SCHEMA
# ---------------------------------------------------------

final_dfs = []
for df in dfs:

    # Remove duplicate columns again before reindexing
    df = df.loc[:, ~df.columns.duplicated()]

    # Reindex safely
    sub = df.reindex(columns=required_cols)

    final_dfs.append(sub)

# ---------------------------------------------------------
# 5. MERGE ALL DATASETS
# ---------------------------------------------------------

df_master = pd.concat(final_dfs, ignore_index=True)
print(f"\nMerged dataset size BEFORE deduplication: {df_master.shape}")

# ---------------------------------------------------------
# 6. REMOVE DUPLICATE ROWS
# ---------------------------------------------------------

df_master = df_master.drop_duplicates()
print(f"Merged dataset size AFTER deduplication: {df_master.shape}")

# ---------------------------------------------------------
# 7. SAVE FINAL MERGED DATASET
# ---------------------------------------------------------

df_master.to_csv("master_data_science_salaries.csv", index=False)
print("\nSaved merged dataset as master_data_science_salaries.csv")


# ---------------------------------------------------------
# 8. View the completed master dataset
# ---------------------------------------------------------

df = pd.read_csv("master_data_science_salaries.csv")

# View first 10 rows
print(df.head(10))

# View dataset shape (rows, columns)
print("\nDataset shape:", df.shape)

# View column names
print("\nColumns:")
print(df.columns.tolist())

# View summary of each column
print("\nInfo:")
print(df.info())

print(df['work_year'].unique())
print(df['work_year'].isna().sum())

print(df['work_year'].unique())
print(df['work_year'].isna().sum())
