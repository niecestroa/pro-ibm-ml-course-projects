# -*- coding: utf-8 -*-
"""
Unified Pipeline: RAW => CLEAN => FINAL
Created: 2026-09-14
Author: Aaron Niecestro
"""

# =========================================================
# Created:    September 9, 2026
# Last Edit:  September 28, 2026
# Author:     Aaron Niecestro

'''
Description:
After downloading the 7 datasets and merging them together. I have cleaned the
data and am ready to do the Exploratory Data Analysis (EDA) in a new file. 
Doing EDA in another file because it is easier to read and check code.
'''

'''
# Master dataset (never modify)
master = dss2025

# USA-only
usa_df = master[master["company_location"] == "United States"].copy()

# Global (all countries)
global_df = master.copy()

# Non-USA (optional)
non_usa_df = master[master["company_location"] != "United States"].copy()

'''

# =========================================================

# -*- coding: utf-8 -*-  # encoding declaration

"""
Unified Pipeline: RAW => CLEAN => FINAL
Option A: Use ONLY 7 macro job-title categories
"""

import kagglehub  # Kaggle dataset downloader
import pandas as pd  # data manipulation
import glob  # file searching
import os  # path handling
from ydata_profiling import ProfileReport  # profiling report
import seaborn as sns  # visualization
import matplotlib.pyplot as plt  # plotting

# =========================================================
# 1. DOWNLOAD ALL KAGGLE DATASETS
# =========================================================

kaggle_sources = [  # list of Kaggle dataset identifiers
    "saurabhbadole/latest-data-science-job-salaries-2024",
    "arnabchaki/data-science-salaries-2023",
    "yusufdelikkaya/datascience-salaries-2024",
    "josiagiven/data-science-salaries-and-fields",
    "sazidthe1/data-science-salaries",
    "zain280/data-science-salaries",
    "lainguyn123/data-science-salary-landscape"
]

paths = [kagglehub.dataset_download(src) for src in kaggle_sources]  # download datasets

print("Downloaded dataset folders:")
for p in paths:
    print(p)

# =========================================================
# 2. LOAD ALL CSV FILES
# =========================================================

dfs = []  # list to store loaded dataframes

for folder in paths:  # iterate through downloaded folders
    csv_files = glob.glob(os.path.join(folder, "*.csv"))  # find CSV files
    if not csv_files:
        print(f"No CSV found in {folder}")
        continue

    for csv in csv_files:  # load each CSV
        print(f"Loading: {csv}")
        df = pd.read_csv(csv)  # read CSV
        df.columns = df.columns.str.lower().str.strip()  # standardize column names
        df = df.loc[:, ~df.columns.duplicated()]  # remove duplicate columns
        dfs.append(df)  # store dataframe

print(f"\nLoaded {len(dfs)} datasets.")

# =========================================================
# 3. REQUIRED SCHEMA
# =========================================================

required_cols = [  # required columns for alignment
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

# =========================================================
# 4. ALIGN SCHEMA
# =========================================================

aligned_dfs = []  # list for aligned dataframes
for df in dfs:
    df = df.loc[:, ~df.columns.duplicated()]  # remove duplicate columns
    aligned_dfs.append(df.reindex(columns=required_cols))  # align schema

# =========================================================
# 5. MERGE RAW DATASETS
# =========================================================

raw_master = pd.concat(aligned_dfs, ignore_index=True)  # merge datasets
raw_master = raw_master.drop_duplicates()  # remove duplicates
raw_master.to_csv("dss_raw_data.csv", index=False)  # save raw dataset
print("\nSaved RAW dataset as dss_raw_data.csv")

# =========================================================
# 6. INITIAL CLEANING
# =========================================================

clean_df = raw_master.copy()  # copy raw dataset
clean_df = clean_df.dropna(subset=["work_year"])  # drop missing work_year
clean_df.to_csv("dss_clean_data.csv", index=False)  # save initial clean dataset
print("Saved CLEAN dataset as dss_clean_data.csv")

# =========================================================
# 7. FULL CLEANING PIPELINE
# =========================================================

df = clean_df.copy()  # working dataframe
df["data_age"] = 2026 - df["work_year"].astype(int)  # compute data age

# ---------------------------------------------------------
# Standardize Experience Levels
# ---------------------------------------------------------

experience_map = {  # mapping for experience levels
    "EN": "Entry-Level", "Entry": "Entry-Level", "Entry Level": "Entry-Level",
    "MI": "Middle-Level", "Mid": "Middle-Level", "Middle": "Middle-Level",
    "SE": "Senior-Level", "Senior": "Senior-Level",
    "EX": "Executive-Level", "Executive": "Executive-Level"
}

df["experience_level"] = df["experience_level"].replace(experience_map)  # apply mapping
df["experience_level"] = pd.Categorical(  # convert to ordered category
    df["experience_level"],
    categories=["Entry-Level", "Middle-Level", "Senior-Level", "Executive-Level"],
    ordered=True
)

# ---------------------------------------------------------
# Clean Employment Type
# ---------------------------------------------------------

employment_map = {  # mapping for employment type
    "FT": "Full-time", "PT": "Part-time",
    "CT": "Contract/Freelance", "FL": "Contract/Freelance"
}

df["employment_type"] = df["employment_type"].replace(employment_map)  # apply mapping

# ---------------------------------------------------------
# Clean Company Size
# ---------------------------------------------------------

size_map = {"M": "Medium", "L": "Large", "S": "Small"}  # mapping for company size
df["company_size"] = df["company_size"].replace(size_map)  # apply mapping

# ---------------------------------------------------------
# Create Categorical Versions
# ---------------------------------------------------------

df["work_year_cat"] = df["work_year"].astype(str).astype("category")  # categorical year

remote_map = {0: "On-Site", 50: "Hybrid", 100: "Remote"}  # remote mapping
df["remote_work_cat"] = df["remote_ratio"].replace(remote_map)  # apply mapping
df["remote_work_cat"] = pd.Categorical(  # ordered category
    df["remote_work_cat"],
    categories=["On-Site", "Hybrid", "Remote"],
    ordered=True
)

# ---------------------------------------------------------
# Salary Categories
# ---------------------------------------------------------

mean_salary = df["salary_in_usd"].mean()  # compute mean salary
median_salary = df["salary_in_usd"].median()  # compute median salary
q1 = df["salary_in_usd"].quantile(0.33)  # lower tercile
q2 = df["salary_in_usd"].quantile(0.66)  # upper tercile

df["salary_mean_cat"] = df["salary_in_usd"].apply(  # mean-based category
    lambda x: "Above-Average" if x >= mean_salary else "Below-Average"
)

df["salary_median_cat"] = df["salary_in_usd"].apply(  # median-based category
    lambda x: "Above-Median" if x >= median_salary else "Below-Median"
)

def salary_band(x):  # 3-category salary band
    if x < q1:
        return "Low-Salary"
    elif x < q2:
        return "Middle-Salary"
    else:
        return "High-Salary"

df["salary_3cat"] = df["salary_in_usd"].apply(salary_band)  # apply salary band

# ---------------------------------------------------------
# ISO Country Mapping
# ---------------------------------------------------------

iso_to_country = {  # mapping ISO codes to country names
    "US": "United States", "CA": "Canada", "GB": "United Kingdom",
    "DE": "Germany", "FR": "France", "IN": "India", "CN": "China",
    "JP": "Japan", "KR": "South Korea", "BR": "Brazil", "MX": "Mexico",
    "AU": "Australia", "NZ": "New Zealand", "SG": "Singapore",
    "CH": "Switzerland", "SE": "Sweden", "NO": "Norway", "FI": "Finland",
    "DK": "Denmark", "IE": "Ireland", "IT": "Italy", "ES": "Spain",
    "PT": "Portugal", "NL": "Netherlands", "BE": "Belgium", "AT": "Austria",
    "PL": "Poland", "CZ": "Czech Republic", "RO": "Romania", "RU": "Russia"
}

df["company_location"] = df["company_location"].replace(iso_to_country)  # apply mapping
df["employee_residence"] = df["employee_residence"].replace(iso_to_country)  # apply mapping

# ---------------------------------------------------------
# Filter Locations ≥100
# ---------------------------------------------------------

MIN_COUNT = 100  # minimum count threshold

valid_locations = df["company_location"].value_counts()[lambda x: x >= MIN_COUNT].index  # valid company locations
df = df[df["company_location"].isin(valid_locations)].copy()  # filter dataset

valid_residence = df["employee_residence"].value_counts()[lambda x: x >= MIN_COUNT].index  # valid residences
df = df[df["employee_residence"].isin(valid_residence)].copy()  # filter dataset

# =========================================================
# Collapse Job Titles into 7 Macro Categories
# =========================================================

def collapse_job_title(title):  # function to collapse job titles
    title = title.lower()  # lowercase for matching

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

    return "Other Tech"  # default category

df["job_title_group"] = df["job_title"].apply(collapse_job_title)  # apply collapsing
df["job_title_group"] = df["job_title_group"].astype("category")  # convert to category

# =========================================================
# Convert Data Types
# =========================================================

numeric_cols = ["salary_in_usd", "remote_ratio", "data_age"]  # numeric columns
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)  # ensure numeric types

categorical_cols = [  # categorical columns
    "work_year", "experience_level", "employment_type", "job_title_group",
    "employee_residence", "company_location", "company_size",
    "work_year_cat", "remote_work_cat",
    "salary_mean_cat", "salary_median_cat", "salary_3cat"
]

for col in categorical_cols:
    df[col] = df[col].astype("category")  # convert to category

# =========================================================
# Drop salary + salary_currency
# =========================================================

df.drop(columns=["salary"], inplace=True)  # drop raw salary
df.drop(columns=["salary_currency"], inplace=True)  # drop currency column

# =========================================================
# Remove Duplicates & Missing Target
# =========================================================

df = df.drop_duplicates()  # remove duplicate rows
df = df.dropna(subset=["salary_in_usd"]).reset_index(drop=True)  # ensure salary exists

# =========================================================
# FIX MISSING VALUES (FINAL CLEANING)
# =========================================================

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns  # numeric columns
categorical_cols = df.select_dtypes(include=["object", "category"]).columns  # categorical columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())  # median imputation

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])  # mode imputation

print("\nMissing values after final fix:")
print(df.isnull().sum())

# =========================================================
# SAVE FINAL CLEANED DATASET
# =========================================================

df.to_csv("dss2025_final.csv", index=False)  # save final dataset
print("\nSaved FINAL cleaned dataset as dss2025_final.csv")

# =========================================================
# EDA SUMMARY (Optional)
# =========================================================

dss2025 = df.copy()  # copy for EDA

print("\n----- UNIQUE VALUES -----")
for col in dss2025.columns:
    print(f"\n{col}")
    print("Unique Count:", dss2025[col].nunique())

# profiling report
profile = ProfileReport(dss2025, title="DS Salaries Summary")
profile.to_file("ds_salaries_profile.html")
print("Saved profiling report to ds_salaries_profile.html")
