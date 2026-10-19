# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 21:11:26 2026

@author: aniec
"""

# =========================================================
# Created:    2026-09-09
# Last Edit:  2026-14-09
# Author:     Aaron Niecestro

'''
Description:
After downloading the 7 datasets and merging them together. I have cleaned the
data and am ready to do the Exploratory Data Analysis (EDA) in a new file. 
Doing EDA in another file because it is easier to read and check code.
'''
# =========================================================

# =========================================================
# 0. Load Packages & RAW Data
# =========================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# IMPORTANT: Load RAW merged dataset, NOT the final cleaned CSV
file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
dss2025 = pd.read_csv(file_path)

# =========================================================
# 1. Standardize Experience Levels
# =========================================================

experience_map = {
    "EN": "Entry-Level", "Entry": "Entry-Level", "Entry Level": "Entry-Level",
    "Entry-Level": "Entry-Level",
    "MI": "Middle-Level", "Mid": "Middle-Level", "Middle": "Middle-Level",
    "Mid Level": "Middle-Level", "Middle Level": "Middle-Level",
    "Middle-Level": "Middle-Level",
    "SE": "Senior-Level", "Senior": "Senior-Level", "Senior Level": "Senior-Level",
    "Senior-Level": "Senior-Level",
    "EX": "Executive-Level", "Executive": "Executive-Level",
    "Executive Level": "Executive-Level", "Executive-Level": "Executive-Level"
}

dss2025["experience_level"] = dss2025["experience_level"].replace(experience_map)

dss2025["experience_level"] = pd.Categorical(
    dss2025["experience_level"],
    categories=["Entry-Level", "Middle-Level", "Senior-Level", "Executive-Level"],
    ordered=True
)

# =========================================================
# 2. Clean Employment Type
# =========================================================

employment_map = {
    "FT": "Full-time", "Full-time": "Full-time",
    "PT": "Part-time", "Part-time": "Part-time",
    "CT": "Contract", "Contract": "Contract",
    "FL": "Freelance", "Freelance": "Freelance"
}

dss2025["employment_type"] = dss2025["employment_type"].replace(employment_map)

# =========================================================
# 3. Clean Company Size
# =========================================================

size_map = {"M": "Medium", "L": "Large", "S": "Small"}
dss2025["company_size"] = dss2025["company_size"].replace(size_map)

# =========================================================
# 4. Create Categorical Versions of Numeric Variables
# =========================================================

dss2025["work_year_cat"] = dss2025["work_year"].astype(str).astype("category")

remote_map = {0: "On-Site", 50: "Hybrid", 100: "Remote"}
dss2025["remote_work_cat"] = dss2025["remote_ratio"].replace(remote_map)

dss2025["remote_work_cat"] = pd.Categorical(
    dss2025["remote_work_cat"],
    categories=["On-Site", "Hybrid", "Remote"],
    ordered=True
)

# Salary categories
mean_salary = dss2025["salary_in_usd"].mean()
median_salary = dss2025["salary_in_usd"].median()
q1 = dss2025["salary_in_usd"].quantile(0.33)
q2 = dss2025["salary_in_usd"].quantile(0.66)

dss2025["salary_mean_cat"] = dss2025["salary_in_usd"].apply(
    lambda x: "Above-Average" if x >= mean_salary else "Below-Average"
)

dss2025["salary_median_cat"] = dss2025["salary_in_usd"].apply(
    lambda x: "Above-Median" if x >= median_salary else "Below-Median"
)

def salary_band(x):
    if x < q1:
        return "Low-Salary"
    elif x < q2:
        return "Middle-Salary"
    else:
        return "High-Salary"

dss2025["salary_3cat"] = dss2025["salary_in_usd"].apply(salary_band)

# =========================================================
# 5. Clean Company Location & Employee Residence
# =========================================================

iso_to_country = {
    "US": "United States", "CA": "Canada", "GB": "United Kingdom",
    "DE": "Germany", "FR": "France", "IN": "India", "CN": "China",
    "JP": "Japan", "KR": "South Korea", "BR": "Brazil", "MX": "Mexico",
    "AU": "Australia", "NZ": "New Zealand", "SG": "Singapore",
    "CH": "Switzerland", "SE": "Sweden", "NO": "Norway", "FI": "Finland",
    "DK": "Denmark", "IE": "Ireland", "IT": "Italy", "ES": "Spain",
    "PT": "Portugal", "NL": "Netherlands", "BE": "Belgium", "AT": "Austria",
    "PL": "Poland", "CZ": "Czech Republic", "RO": "Romania", "RU": "Russia",
    "AR": "Argentina", "CL": "Chile", "CO": "Colombia", "PE": "Peru",
    "PH": "Philippines", "PK": "Pakistan", "ID": "Indonesia",
    "MY": "Malaysia", "TH": "Thailand", "TR": "Turkey",
    "SA": "Saudi Arabia", "AE": "United Arab Emirates",
    "ZA": "South Africa", "NG": "Nigeria", "KE": "Kenya", "GH": "Ghana",
    "HK": "Hong Kong", "HN": "Honduras", "MU": "Mauritius",
    "BS": "Bahamas", "XK": "Kosovo", "AM": "Armenia", "AL": "Albania",
    "LT": "Lithuania", "LV": "Latvia", "LU": "Luxembourg",
    "MD": "Moldova", "MK": "North Macedonia", "MT": "Malta",
    "SI": "Slovenia", "SK": "Slovakia", "GR": "Greece", "HR": "Croatia",
    "BG": "Bulgaria", "RS": "Serbia", "UA": "Ukraine", "VE": "Venezuela",
    "VN": "Vietnam", "IQ": "Iraq", "IR": "Iran", "IL": "Israel",
    "LB": "Lebanon", "EG": "Egypt", "DZ": "Algeria", "MA": "Morocco",
    "CF": "Central African Republic", "CD": "Democratic Republic of Congo",
    "GI": "Gibraltar", "AS": "American Samoa", "PR": "Puerto Rico",
    "UZ": "Uzbekistan", "BO": "Bolivia", "HU": "Hungary",
    "DO": "Dominican Republic", "CR": "Costa Rica", "KW": "Kuwait",
    "TN": "Tunisia", "UG": "Uganda", "GE": "Georgia"
}

dss2025["company_location"] = dss2025["company_location"].replace(iso_to_country)
dss2025["employee_residence"] = dss2025["employee_residence"].replace(iso_to_country)

# =========================================================
# 6. Filter Company Locations & Residence ≥100
# =========================================================

MIN_COUNT = 100

valid_locations = dss2025["company_location"].value_counts()[lambda x: x >= MIN_COUNT].index
dss2025 = dss2025[dss2025["company_location"].isin(valid_locations)].copy()

valid_residence = dss2025["employee_residence"].value_counts()[lambda x: x >= MIN_COUNT].index
dss2025 = dss2025[dss2025["employee_residence"].isin(valid_residence)].copy()

# =========================================================
# 7. Group Job Titles
# =========================================================

job_title_map = {
    "Data Scientist": "Data Scientist",
    "Lead Data Scientist": "Data Scientist",
    "Principal Data Scientist": "Data Scientist",
    "Staff Data Scientist": "Data Scientist",
    "Data Scientist II": "Data Scientist",
    "Data Scientist III": "Data Scientist",
    "Data Scientist Associate": "Data Scientist",
    "Data Scientist Manager": "Data Scientist",

    "Machine Learning Engineer": "Machine Learning Engineer",
    "ML Engineer": "Machine Learning Engineer",
    "AI Engineer": "Machine Learning Engineer",
    "AI Software Engineer": "Machine Learning Engineer",
    "AI Engineering Manager": "Machine Learning Engineer",

    "Data Engineer": "Data Engineer",
    "Senior Data Engineer": "Data Engineer",
    "ETL Engineer": "Data Engineer",
    "Data Platform Engineer": "Data Engineer",

    "Data Analyst": "Data Analyst",
    "Business Analyst": "Data Analyst",
    "BI Analyst": "Data Analyst",
    "Quantitative Research Analyst": "Data Analyst",
    "Financial Analyst": "Data Analyst",

    "Software Engineer": "Software Engineer",
    "Backend Engineer": "Software Engineer",
    "Full Stack Engineer": "Software Engineer",
    "Frontend Engineer": "Software Engineer",
    "Controls Engineer": "Software Engineer",

    "Customer Success Manager": "Manager",
    "Analytics Manager": "Manager",
    "Engineering Manager": "Manager",
    "Data Science Manager": "Manager",
}

dss2025["job_title_group"] = (
    dss2025["job_title"].map(job_title_map).fillna("Other")
)

dss2025["job_title_group"] = dss2025["job_title_group"].astype("category")

dss2025.drop(columns=["job_title"], inplace=True)

# =========================================================
# 8. Convert Data Types
# =========================================================

numeric_cols = ["salary_in_usd", "remote_ratio", "data_age"]

categorical_cols = [
    "work_year",
    "experience_level",
    "employment_type",
    "job_title_group",
    "employee_residence",
    "company_location",
    "company_size",
    "work_year_cat",
    "remote_work_cat",
    "salary_mean_cat",
    "salary_median_cat",
    "salary_3cat"
]

dss2025[numeric_cols] = dss2025[numeric_cols].apply(pd.to_numeric)

for col in categorical_cols:
    dss2025[col] = dss2025[col].astype("category")

# =========================================================
# Drop salary + salary_currency
# =========================================================

dss2025.drop(columns=["salary"], inplace=True)
dss2025.drop(columns=["salary_currency"], inplace=True)

# =========================================================
# 9. Remove Duplicates & Missing Target
# =========================================================

dss2025 = dss2025.drop_duplicates()
dss2025 = dss2025.dropna(subset=["salary_in_usd"]).reset_index(drop=True)

# =========================================================
# 10. Save Final Cleaned Dataset
# =========================================================

final_file = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
dss2025.to_csv(final_file, index=False)

print(f"Final cleaned dataset saved to:\n{final_file}")

# =========================================================
# SHOW OBSERVATIONS FOR EACH COLUMN
# =========================================================

cols_to_check = [
    "work_year",
    "experience_level",
    "employment_type",
    "job_title_group",
    "salary_in_usd",
    "employee_residence",
    "remote_ratio",
    "company_size",
    "work_year_cat",
    "remote_work_cat",
    "salary_mean_cat",
    "salary_median_cat",
    "salary_3cat",
    "data_age",
    "company_location"
]

for col in cols_to_check:
    print("\n" + "="*60)
    print(f"Column: {col}")
    print("="*60)

    if pd.api.types.is_numeric_dtype(dss2025[col]):
        print(dss2025[col].describe())
    else:
        print(dss2025[col].value_counts())


# =========================================================
# 11. Unique Values Per Column
# =========================================================

print("\n----- UNIQUE VALUES -----")
for col in dss2025.columns:
    print(f"\n{col}")
    print("Unique Count:", dss2025[col].nunique())

# =========================================================
# 12. Summary Statistics (R-style)
# =========================================================

def summary_r(df):
    print("----- R-style Summary -----")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

    for col in df.columns:
        print(f"--- {col} ---")
        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            print(f"Min:      {desc['min']}")
            print(f"1st Qu.:  {desc['25%']}")
            print(f"Median:   {desc['50%']}")
            print(f"Mean:     {desc['mean']}")
            print(f"3rd Qu.:  {desc['75%']}")
            print(f"Max:      {desc['max']}")
        else:
            desc = df[col].describe()
            print("Type:     Categorical")
            print(f"Unique:   {desc['unique']}")
            print(f"Top:      {desc['top']}")
            print(f"Freq:     {desc['freq']}")
        print()

summary_r(dss2025)

print("\n----- SUMMARY (describe include='all') -----")
print(dss2025.describe(include='all'))

# =========================================================
# 13. Profiling Report (HTML)
# =========================================================

from ydata_profiling import ProfileReport

profile = ProfileReport(dss2025, title="DS Salaries Summary")
profile.to_notebook_iframe()

output_path = r"C:\Users\aniec\Desktop\ibm-ml-project\01-EDA\ds_salaries_profile.html"
profile.to_file(output_path)
print("Saved profiling report to:", output_path)

# =========================================================
# 14. Numeric & Categorical Summaries
# =========================================================

print("\n----- NUMERIC SUMMARY -----")
print(dss2025.describe())

print("\n----- CATEGORICAL SUMMARY -----")
print(dss2025.describe(include=["category", "object"]))

# =========================================================
# 15. Missing Values
# =========================================================

missing_df = pd.DataFrame({
    "Missing Count": dss2025.isnull().sum(),
    "Missing Percent": (dss2025.isnull().sum() / len(dss2025)) * 100
})

print("\n----- MISSING VALUES -----")
print(missing_df)

plt.figure(figsize=(12,6))
sns.heatmap(dss2025.isnull(), cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()

# =========================================================
# 16. Duplicate Check
# =========================================================

duplicates = dss2025.duplicated().sum()
print("\n----- DUPLICATES -----")
print("Duplicate Rows:", duplicates)

# =========================================================
# 17. Final Cleaning
# =========================================================

dss2025 = dss2025.drop_duplicates()
dss2025 = dss2025.dropna(subset=["salary_in_usd"]).reset_index(drop=True)

# =========================================================
# 18. Final Dataset Check
# =========================================================

print("\n----- FINAL SHAPE -----")
print(dss2025.shape)

print("\n----- FINAL DATA TYPES -----")
print(dss2025.dtypes)

print("\n----- FINAL MISSING VALUES -----")
print(dss2025.isnull().sum())

print("\n----- HEAD -----")
print(dss2025.head())

print("\n----- FINAL SUMMARY STATISTICS -----")
print(dss2025.describe())

print("\n----- FINAL CATEGORICAL SUMMARY -----")
print(dss2025.describe(include=["category", "object"]))

# =========================================================
# 19. Save Final Cleaned Dataset
# =========================================================

final_file = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
dss2025.to_csv(final_file, index=False)

print(f"Final cleaned dataset saved to:\n{final_file}")

# =========================================================
# SHOW OBSERVATIONS FOR EACH COLUMN (ONE BY ONE)
# =========================================================

cols_to_check = [
    "work_year",
    "experience_level",
    "employment_type",
    "job_title_group",
    "salary_in_usd",
    "employee_residence",
    "remote_ratio",
    "company_size",
    "work_year_cat",
    "remote_work_cat",
    "salary_mean_cat",
    "salary_median_cat",
    "salary_3cat",
    "data_age",
    "company_location"
]

for col in cols_to_check:
    print("\n" + "="*60)
    print(f"Column: {col}")
    print("="*60)

    if pd.api.types.is_numeric_dtype(dss2025[col]):
        print(dss2025[col].describe())
    else:
        print(dss2025[col].value_counts())
