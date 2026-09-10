# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 21:11:26 2026

@author: aniec
"""

# =========================================================
# Created:    2026-09-09
# Last Edit:  2026-09-09
# Author:     Aaron Niecestro

'''
Description:
After downloading the 7 datasets and merging them together. I have cleaned the
data and am ready to do the Exploratory Data Analysis (EDA) in a new file. 
Doing EDA in another file because it is easier to read and check code.
'''
# =========================================================

# --------------------------------------------------------------------------
# Loading and Cleaning the Dataset Checklist
# --------------------------------------------------------------------------

# -----------------------------
# 0. Loading Packages
# -----------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 0. Loading Data
# -----------------------------

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\dss_clean_data.csv"

dss2025 = pd.read_csv(file_path)

# -----------------------------
# 1. Understanding the Data
# -----------------------------

print("----- SHAPE -----")
print(dss2025.shape)

print("\n----- COLUMN NAMES -----")
print(dss2025.columns.tolist())

print("\n----- DATA TYPES -----")
print(dss2025.dtypes)

# -----------------------------
# 2. Preview the Data
# -----------------------------

print(dss2025.head())

# -----------------------------
# 3. Convert Data Types
# -----------------------------

# -----------------------------
# Standardize Experience Levels
# -----------------------------

experience_map = {

    # Entry Level
    "EN": "Entry-Level",
    "Entry": "Entry-Level",
    "Entry Level": "Entry-Level",
    "Entry-Level": "Entry-Level",

    # Middle Level
    "MI": "Middle-Level",
    "Mid": "Middle-Level",
    "Middle": "Middle-Level",
    "Mid Level": "Middle-Level",
    "Middle Level": "Middle-Level",
    "Middle-Level": "Middle-Level",

    # Senior Level
    "SE": "Senior-Level",
    "Senior": "Senior-Level",
    "Senior Level": "Senior-Level",
    "Senior-Level": "Senior-Level",

    # Executive Level
    "EX": "Executive-Level",
    "Executive": "Executive-Level",
    "Executive Level": "Executive-Level",
    "Executive-Level": "Executive-Level"
}

dss2025["experience_level"] = (
    dss2025["experience_level"]
    .replace(experience_map)
)

# Set logical career progression order
dss2025["experience_level"] = pd.Categorical(
    dss2025["experience_level"],
    categories=[
        "Entry-Level",
        "Middle-Level",
        "Senior-Level",
        "Executive-Level"
    ],
    ordered=True
)

# ---------------------------------------------------------
# Create Categorical Versions of Numeric Variables
# ---------------------------------------------------------

# Work Year as categorical
dss2025["work_year_cat"] = (
    dss2025["work_year"]
    .astype(str)
    .astype("category")
)

# Remote Ratio as categorical labels
remote_map = {
    0: "On-Site",
    50: "Hybrid",
    100: "Remote"
}

dss2025["remote_work_cat"] = (
    dss2025["remote_ratio"]
    .replace(remote_map)
    .astype("category")
)

# Set order for remote work categories
dss2025["remote_work_cat"] = pd.Categorical(
    dss2025["remote_work_cat"],
    categories=[
        "On-Site",
        "Hybrid",
        "Remote"
    ],
    ordered=True
)

# ---------------------------------------------------------
# Verify Results
# ---------------------------------------------------------

print("\n----- WORK YEAR CATEGORY -----")
print(
    dss2025["work_year_cat"]
    .value_counts()
    .sort_index()
)

print("\n----- REMOTE WORK CATEGORY -----")
print(
    dss2025["remote_work_cat"]
    .value_counts()
)

print("\n----- NEW DATA TYPES -----")
print(
    dss2025[
        [
            "work_year_cat",
            "remote_work_cat"
        ]
    ].dtypes
)

print("\n----- SAMPLE -----")
print(
    dss2025[
        [
            "work_year",
            "work_year_cat",
            "remote_ratio",
            "remote_work_cat"
        ]
    ]
    .head()
)

# -----------------------------
# Numeric Columns
# -----------------------------

numeric_cols = [
    "work_year",
    "salary",
    "salary_in_usd",
    "remote_ratio"
]

# -----------------------------
# Categorical Columns
# -----------------------------

categorical_cols = [
    "experience_level",
    "employment_type",
    "job_title",
    "salary_currency",
    "employee_residence",
    "company_location",
    "company_size",
    "work_year_cat",
    "remote_work_cat"
]

# -----------------------------
# Convert Data Types
# -----------------------------

dss2025[numeric_cols] = (
    dss2025[numeric_cols]
    .apply(pd.to_numeric)
)

for col in categorical_cols:
    dss2025[col] = dss2025[col].astype("category")

# -----------------------------
# Verify Results
# -----------------------------

print("\n----- EXPERIENCE LEVEL COUNTS -----")
print(
    dss2025["experience_level"]
    .value_counts()
    .sort_index()
)

print("\n----- DATA TYPES -----")
print(dss2025.dtypes)

# -----------------------------
# 4. Check the Unique Values
# Unique values in each column
# -----------------------------

print("\n----- UNIQUE VALUES -----")

for col in dss2025.columns:

    print(f"\n{col}")
    print(
        "Unique Count:",
        dss2025[col].nunique()
    )

# -----------------------------
# 5. Summary Statistics
# Statistical summary columns
# -----------------------------

# -----------------------------
#  Summary (R-style)
# -----------------------------
# Data Summary - Similar to R structure

def summary_r(dss2025):
    print("----- R-style Summary -----")
    print(f"Rows: {dss2025.shape[0]}, Columns: {dss2025.shape[1]}\n")

    for col in dss2025.columns:
        print(f"--- {col} ---")
        if pd.api.types.is_numeric_dtype(dss2025[col]):
            desc = dss2025[col].describe()
            print(f"Min:      {desc['min']}")
            print(f"1st Qu.:  {desc['25%']}")
            print(f"Median:   {desc['50%']}")
            print(f"Mean:     {desc['mean']}")
            print(f"3rd Qu.:  {desc['75%']}")
            print(f"Max:      {desc['max']}")
        else:
            desc = dss2025[col].describe()
            print(f"Type:     Categorical")
            print(f"Unique:   {desc['unique']}")
            print(f"Top:      {desc['top']}")
            print(f"Freq:     {desc['freq']}")
        print()

summary_r(dss2025)

print("\n----- SUMMARY (R-style table) -----")
print(dss2025.describe(include='all'))

# -----------------------------
# (Python-style) using HTML - IMPOSSIBLY USEFUL
# Note: Best in Jupyter, since have to load HTML file manually otherwise
# -----------------------------
from ydata_profiling import ProfileReport

profile = ProfileReport(dss2025, title="DS Salaries Summary")
profile.to_notebook_iframe()

# HTML Summary of Data - Good Full Summary (Work best in Jupyter)
output_path = r"C:\Users\aniec\Desktop\ibm-ml-project\01-EDA\ds_salaries_profile.html"

profile.to_file(output_path)
print("Saved to:", output_path)

print("\n----- NUMERIC SUMMARY -----")
print(dss2025.describe())

print("\n----- CATEGORICAL SUMMARY -----")

print(dss2025.describe(include=["category", "object"]))

# -----------------------------
# 6. Check Missing Values
# Count and Percent of missing values
# -----------------------------

missing_df = pd.DataFrame({

    "Missing Count":
        dss2025.isnull().sum(),

    "Missing Percent":
        (
            dss2025.isnull().sum()
            / len(dss2025)
        ) * 100

})

print("\n----- MISSING VALUES -----")
print(missing_df)

# -----------------------------
# 7. Visualize Missing Values
# Visual the pattern of missing data
# -----------------------------

plt.figure(figsize=(12,6))

sns.heatmap(
    dss2025.isnull(),
    cbar=False,
    yticklabels=False
)

plt.title("Missing Values Heatmap")

plt.show()
          
# -----------------------------
# 8. Check Duplicates
# identify duplicate rows
# -----------------------------

duplicates = dss2025.duplicated().sum()

print("\n----- DUPLICATES -----")
print("Duplicate Rows:", duplicates)

# -----------------------------
# 9. Cleaning the Dataset
# Clean the Dataset based on steps 1-8
# -----------------------------

# Remove duplicates
dss2025 = dss2025.drop_duplicates()

# Remove rows with missing target variable
dss2025 = dss2025.dropna(
    subset=["salary_in_usd"]
)

# Reset index
dss2025 = dss2025.reset_index(
    drop=True
)

# -----------------------------
# Final Dataset Check
# -----------------------------

print("\n----- FINAL SHAPE -----")
print(dss2025.shape)

print("\n----- FINAL DATA TYPES -----")
print(dss2025.dtypes)

print("\n----- FINAL MISSING VALUES -----")
print(dss2025.isnull().sum())

print(dss2025.head())

print("\n----- SUMMARY STATISTICS -----")

print(dss2025.describe())

print("\n----- CATEGORICAL SUMMARY -----")

print(dss2025.describe(include=["category", "object"]))

# ---------------------------------------------------------
# Save Final Dataset
# ---------------------------------------------------------

final_file = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\dss2025_final.csv"

dss2025.to_csv(final_file, index=False)

print(f"Final dataset saved to:\n{final_file}")