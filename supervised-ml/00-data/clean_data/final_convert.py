# %% [markdown]
# # Unified Pipeline: RAW → CLEAN → FINAL
# **Created:** 2026-09-14  
# **Author:** Aaron Niecestro  
#
# This notebook performs the full data ingestion, cleaning, and final dataset creation
# for the Master Data Science Salaries (2020–2025) dataset.

# %%
import kagglehub
import pandas as pd
import glob
import os
from ydata_profiling import ProfileReport
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Download All Kaggle Datasets

# %%
kaggle_sources = [
    "saurabhbadole/latest-data-science-job-salaries-2024",
    "arnabchaki/data-science-salaries-2023",
    "yusufdelikkaya/datascience-salaries-2024",
    "josiagiven/data-science-salaries-and-fields",
    "sazidthe1/data-science-salaries",
    "zain280/data-science-salaries",
    "lainguyn123/data-science-salary-landscape"
]

paths = [kagglehub.dataset_download(src) for src in kaggle_sources]

print("Downloaded dataset folders:")
for p in paths:
    print(p)

# %% [markdown]
# ## 2. Load All CSV Files

# %%
dfs = []
for folder in paths:
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    if not csv_files:
        print(f"No CSV found in {folder}")
        continue

    for csv in csv_files:
        print(f"Loading: {csv}")
        df = pd.read_csv(csv)
        df.columns = df.columns.str.lower().str.strip()
        df = df.loc[:, ~df.columns.duplicated()]
        dfs.append(df)

print(f"\nLoaded {len(dfs)} datasets.")

# %% [markdown]
# ## 3. Required Schema

# %%
required_cols = [
    "work_year", "experience_level", "employment_type", "job_title",
    "salary", "salary_currency", "salary_in_usd",
    "employee_residence", "remote_ratio",
    "company_location", "company_size"
]

# %% [markdown]
# ## 4. Align Schema

# %%
aligned_dfs = []
for df in dfs:
    df = df.loc[:, ~df.columns.duplicated()]
    aligned_dfs.append(df.reindex(columns=required_cols))

# %% [markdown]
# ## 5. Merge Raw Datasets

# %%
raw_master = pd.concat(aligned_dfs, ignore_index=True)
raw_master = raw_master.drop_duplicates()
raw_master.to_csv("dss_raw_data.csv", index=False)
print("Saved RAW dataset as dss_raw_data.csv")

# %% [markdown]
# ## 6. Initial Cleaning

# %%
clean_df = raw_master.copy()
clean_df = clean_df.dropna(subset=["work_year"])
clean_df.to_csv("dss_clean_data.csv", index=False)
print("Saved CLEAN dataset as dss_clean_data.csv")

# %% [markdown]
# ## 7. Full Cleaning Pipeline

# %%
df = clean_df.copy()
df["data_age"] = 2026 - df["work_year"].astype(int)

# %% [markdown]
# ### Standardize Experience Levels

# %%
experience_map = {
    "EN": "Entry-Level", "Entry": "Entry-Level", "Entry Level": "Entry-Level",
    "MI": "Middle-Level", "Mid": "Middle-Level", "Middle": "Middle-Level",
    "SE": "Senior-Level", "Senior": "Senior-Level",
    "EX": "Executive-Level", "Executive": "Executive-Level"
}

df["experience_level"] = df["experience_level"].replace(experience_map)
df["experience_level"] = pd.Categorical(
    df["experience_level"],
    categories=["Entry-Level", "Middle-Level", "Senior-Level", "Executive-Level"],
    ordered=True
)

# %% [markdown]
# ### Clean Employment Type

# %%
employment_map = {
    "FT": "Full-time", "PT": "Part-time",
    "CT": "Contract/Freelance", "FL": "Contract/Freelance"
}
df["employment_type"] = df["employment_type"].replace(employment_map)

# %% [markdown]
# ### Clean Company Size

# %%
size_map = {"M": "Medium", "L": "Large", "S": "Small"}
df["company_size"] = df["company_size"].replace(size_map)

# %% [markdown]
# ### Create Categorical Versions

# %%
df["work_year_cat"] = df["work_year"].astype(str).astype("category")

remote_map = {0: "On-Site", 50: "Hybrid", 100: "Remote"}
df["remote_work_cat"] = df["remote_ratio"].replace(remote_map)
df["remote_work_cat"] = pd.Categorical(
    df["remote_work_cat"],
    categories=["On-Site", "Hybrid", "Remote"],
    ordered=True
)

# %% [markdown]
# ### Salary Categories

# %%
mean_salary = df["salary_in_usd"].mean()
median_salary = df["salary_in_usd"].median()
q1 = df["salary_in_usd"].quantile(0.33)
q2 = df["salary_in_usd"].quantile(0.66)

df["salary_mean_cat"] = df["salary_in_usd"].apply(
    lambda x: "Above-Average" if x >= mean_salary else "Below-Average"
)

df["salary_median_cat"] = df["salary_in_usd"].apply(
    lambda x: "Above-Median" if x >= median_salary else "Below-Median"
)

def salary_band(x):
    if x < q1:
        return "Low-Salary"
    elif x < q2:
        return "Middle-Salary"
    else:
        return "High-Salary"

df["salary_3cat"] = df["salary_in_usd"].apply(salary_band)

# %% [markdown]
# ### ISO Country Mapping

# %%
iso_to_country = {
    "US": "United States", "CA": "Canada", "GB": "United Kingdom",
    "DE": "Germany", "FR": "France", "IN": "India", "CN": "China",
    "JP": "Japan", "KR": "South Korea", "BR": "Brazil", "MX": "Mexico",
    "AU": "Australia", "NZ": "New Zealand", "SG": "Singapore",
    "CH": "Switzerland", "SE": "Sweden", "NO": "Norway", "FI": "Finland",
    "DK": "Denmark", "IE": "Ireland", "IT": "Italy", "ES": "Spain",
    "PT": "Portugal", "NL": "Netherlands", "BE": "Belgium", "AT": "Austria",
    "PL": "Poland", "CZ": "Czech Republic", "RO": "Romania", "RU": "Russia"
}

df["company_location"] = df["company_location"].replace(iso_to_country)
df["employee_residence"] = df["employee_residence"].replace(iso_to_country)

# %% [markdown]
# ### Filter Locations ≥ 100

# %%
MIN_COUNT = 100

valid_locations = df["company_location"].value_counts()[lambda x: x >= MIN_COUNT].index
df = df[df["company_location"].isin(valid_locations)].copy()

valid_residence = df["employee_residence"].value_counts()[lambda x: x >= MIN_COUNT].index
df = df[df["employee_residence"].isin(valid_residence)].copy()

# %% [markdown]
# ### Collapse Job Titles into 7 Macro Categories

# %%
def collapse_job_title(title):
    title = title.lower()

    if any(k in title for k in ["machine learning", "ml", "ai", "deep learning", "nlp", "computer vision"]):
        return "ML/AI Engineering"
    if any(k in title for k in ["data scientist", "scientist", "research scientist"]):
        return "Data Science"
    if any(k in title for k in ["data analyst", "analytics", "business analyst", "reporting"]):
        return "Data Analytics"
    if any(k in title for k in ["research", "academic", "professor", "phd", "postdoc"]):
        return "Research / Academic"
    if any(k in title for k in ["bi", "business intelligence", "power bi", "tableau"]):
        return "BI / Business"
    if any(k in title for k in ["quant", "financial", "risk", "economist", "investment"]):
        return "Quant / Finance"
    return "Other Tech"

df["job_title_group"] = df["job_title"].apply(collapse_job_title)
df["job_title_group"] = df["job_title_group"].astype("category")

# %% [markdown]
# ### Convert Data Types

# %%
numeric_cols = ["salary_in_usd", "remote_ratio", "data_age"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

categorical_cols = [
    "work_year", "experience_level", "employment_type", "job_title_group",
    "employee_residence", "company_location", "company_size",
    "work_year_cat", "remote_work_cat",
    "salary_mean_cat", "salary_median_cat", "salary_3cat"
]

for col in categorical_cols:
    df[col] = df[col].astype("category")

# %% [markdown]
# ### Drop Unneeded Columns

# %%
df.drop(columns=["salary", "salary_currency"], inplace=True)

# %% [markdown]
# ### Remove Duplicates & Missing Targets

# %%
df = df.drop_duplicates()
df = df.dropna(subset=["salary_in_usd"]).reset_index(drop=True)

# %% [markdown]
# ### Fix Missing Values

# %%
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object", "category"]).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print(df.isnull().sum())

# %% [markdown]
# ## Save Final Dataset

# %%
df.to_csv("dss2025_final.csv", index=False)
print("Saved FINAL cleaned dataset as dss2025_final.csv")

# %% [markdown]
# ## Optional: Profiling Report

# %%
dss2025 = df.copy()
profile = ProfileReport(dss2025, title="DS Salaries Summary")
profile.to_file("ds_salaries_profile.html")
print("Saved profiling report to ds_salaries_profile.html")
