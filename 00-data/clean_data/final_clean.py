# -*- coding: utf-8 -*-
"""
Unified Pipeline: RAW => CLEAN => FINAL
Created: 2026-09-14
Author: Aaron Niecestro
"""

# =========================================================
# Created:    September 9, 2026
# Last Edit:  September 14, 2026
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

import kagglehub
import pandas as pd
import glob
import os
from ydata_profiling import ProfileReport
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# =========================================================
# 1. DOWNLOAD ALL KAGGLE DATASETS
# =========================================================

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

# =========================================================
# 2. LOAD ALL CSV FILES
# =========================================================

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

# =========================================================
# 3. REQUIRED SCHEMA
# =========================================================

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

# =========================================================
# 4. ALIGN SCHEMA
# =========================================================

aligned_dfs = []
for df in dfs:
    df = df.loc[:, ~df.columns.duplicated()]
    aligned_dfs.append(df.reindex(columns=required_cols))

# =========================================================
# 5. MERGE RAW DATASETS → dss_raw_data.csv
# =========================================================

raw_master = pd.concat(aligned_dfs, ignore_index=True)
raw_master = raw_master.drop_duplicates()

raw_path = "dss_raw_data.csv"
raw_master.to_csv(raw_path, index=False)
print(f"\nSaved RAW dataset as {raw_path}")

# =========================================================
# 6. INITIAL CLEANING → dss_clean_data.csv
# =========================================================

clean_df = raw_master.copy()
clean_df = clean_df.dropna(subset=["work_year"])

clean_path = "dss_clean_data.csv"
clean_df.to_csv(clean_path, index=False)
print(f"Saved CLEAN dataset as {clean_path}")

# =========================================================
# 7. FULL CLEANING PIPELINE → dss2025_final.csv
# =========================================================

df = clean_df.copy()
df["data_age"] = 2026 - df["work_year"].astype(int)

# ---------------------------------------------------------
# Standardize Experience Levels
# ---------------------------------------------------------

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

df["experience_level"] = df["experience_level"].replace(experience_map)
df["experience_level"] = pd.Categorical(
    df["experience_level"],
    categories=["Entry-Level", "Middle-Level", "Senior-Level", "Executive-Level"],
    ordered=True
)

# ---------------------------------------------------------
# Clean Employment Type
# ---------------------------------------------------------

employment_map = {
    "FT": "Full-time", "Full-time": "Full-time",
    "PT": "Part-time", "Part-time": "Part-time",
    "CT": "Contract/Freelance", "Contract": "Contract/Freelance",
    "FL": "Contract/Freelance", "Freelance": "Contract/Freelance"
}

df["employment_type"] = df["employment_type"].replace(employment_map)

# ---------------------------------------------------------
# Clean Company Size
# ---------------------------------------------------------

size_map = {"M": "Medium", "L": "Large", "S": "Small"}
df["company_size"] = df["company_size"].replace(size_map)

# ---------------------------------------------------------
# Create Categorical Versions
# ---------------------------------------------------------

df["work_year_cat"] = df["work_year"].astype(str).astype("category")

remote_map = {0: "On-Site", 50: "Hybrid", 100: "Remote"}
df["remote_work_cat"] = df["remote_ratio"].replace(remote_map)
df["remote_work_cat"] = pd.Categorical(
    df["remote_work_cat"],
    categories=["On-Site", "Hybrid", "Remote"],
    ordered=True
)

# Salary categories
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

# ---------------------------------------------------------
# ISO Country Mapping
# ---------------------------------------------------------

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

df["company_location"] = df["company_location"].replace(iso_to_country)
df["employee_residence"] = df["employee_residence"].replace(iso_to_country)

# ---------------------------------------------------------
# Filter Locations ≥100
# ---------------------------------------------------------

MIN_COUNT = 100

valid_locations = df["company_location"].value_counts()[lambda x: x >= MIN_COUNT].index
df = df[df["company_location"].isin(valid_locations)].copy()

valid_residence = df["employee_residence"].value_counts()[lambda x: x >= MIN_COUNT].index
df = df[df["employee_residence"].isin(valid_residence)].copy()

# ---------------------------------------------------------
# Group Job Titles
# ---------------------------------------------------------

job_title_map = {

    # ============================
    # CORE DATA ROLES
    # ============================
    "Data Scientist": "Data Scientist",
    "Applied Data Scientist": "Data Scientist",
    "Data Science Analyst": "Data Scientist",
    "Data Science Consultant": "Data Scientist",
    "Data Science Engineer": "Data Scientist",
    "Data Science Lead": "Data Scientist",
    "Data Science Practitioner": "Data Scientist",
    "Data Science Director": "Data Scientist",
    "Managing Director Data Science": "Data Scientist",
    "Director of Data Science": "Data Scientist",

    "Data Engineer": "Data Engineer",
    "BI Data Engineer": "Data Engineer",
    "Cloud Data Engineer": "Data Engineer",
    "Azure Data Engineer": "Data Engineer",
    "Big Data Engineer": "Data Engineer",
    "Data DevOps Engineer": "Data Engineer",
    "Data Pipeline Engineer": "Data Engineer",
    "Databricks Engineer": "Data Engineer",
    "DataOps Engineer": "Data Engineer",
    "Software Data Engineer": "Data Engineer",

    "Data Analyst": "Data Analyst",
    "BI Data Analyst": "Data Analyst",
    "Business Data Analyst": "Data Analyst",
    "Finance Data Analyst": "Data Analyst",
    "Financial Data Analyst": "Data Analyst",
    "Marketing Data Analyst": "Data Analyst",
    "Pricing Analyst": "Data Analyst",
    "Research Analyst": "Data Analyst",
    "Insight Analyst": "Data Analyst",
    "Compliance Data Analyst": "Data Analyst",
    "Data Reporting Analyst": "Data Analyst",
    "Data Visualization Analyst": "Data Analyst",

    # ============================
    # MACHINE LEARNING / AI ROLES
    # ============================
    "Machine Learning Engineer": "Machine Learning Engineer",
    "Applied Machine Learning Engineer": "Machine Learning Engineer",
    "Machine Learning Developer": "Machine Learning Engineer",
    "Machine Learning Software Engineer": "Machine Learning Engineer",
    "Machine Learning Research Engineer": "Machine Learning Engineer",
    "Machine Learning Infrastructure Engineer": "Machine Learning Engineer",
    "Machine Learning Platform Engineer": "Machine Learning Engineer",
    "Machine Learning Quality Engineer": "Machine Learning Engineer",
    "Machine Learning Operations Engineer": "Machine Learning Engineer",
    "ML Ops Engineer": "Machine Learning Engineer",
    "MLOps Engineer": "Machine Learning Engineer",
    "ML Infrastructure Engineer": "Machine Learning Engineer",
    "Machine Vision Engineer": "Machine Learning Engineer",

    "AI Engineer": "Machine Learning Engineer",
    "AI Developer": "Machine Learning Engineer",
    "AI Programmer": "Machine Learning Engineer",
    "AI Software Development Engineer": "Machine Learning Engineer",
    "AI Machine Learning Engineer": "Machine Learning Engineer",
    "Artificial Intelligence Engineer": "Machine Learning Engineer",

    "AI Scientist": "Machine Learning Scientist",
    "AI Research Scientist": "Machine Learning Scientist",
    "Machine Learning Scientist": "Machine Learning Scientist",
    "Applied Machine Learning Scientist": "Machine Learning Scientist",
    "Deep Learning Researcher": "Machine Learning Scientist",
    "Deep Learning Engineer": "Machine Learning Scientist",

    # ============================
    # SOFTWARE ENGINEERING ROLES
    # ============================
    "Software Engineer": "Software Engineer",
    "Full Stack Developer": "Software Engineer",
    "Backend Developer": "Software Engineer",
    "Frontend Developer": "Software Engineer",
    "Java Developer": "Software Engineer",
    "Python Developer": "Software Engineer",
    "Developer": "Software Engineer",
    "Developer Advocate": "Software Engineer",
    "SAS Developer": "Software Engineer",
    "ETL Developer": "Software Engineer",
    "Cloud Developer": "Software Engineer",
    "Robotics Software Engineer": "Software Engineer",

    # ============================
    # ENGINEER ROLES (GENERAL)
    # ============================
    "Engineer": "Engineer",
    "Infrastructure Engineer": "Engineer",
    "Platform Engineer": "Engineer",
    "System Engineer": "Engineer",
    "Systems Engineer": "Engineer",
    "Security Engineer": "Engineer",
    "Site Reliability Engineer": "Engineer",
    "QA Engineer": "Engineer",
    "Solution Engineer": "Engineer",
    "Solutions Engineer": "Engineer",
    "Technical Specialist": "Engineer",
    "Technology Integrator": "Engineer",

    # ============================
    # ANALYST ROLES (GENERAL)
    # ============================
    "Analyst": "Analyst",
    "Lead Analyst": "Analyst",
    "Research Assistant": "Analyst",
    "Research Associate": "Analyst",
    "Risk Analyst": "Analyst",
    "Quantitative Analyst": "Analyst",
    "Quantitative Developer": "Analyst",
    "Quantitative Researcher": "Analyst",

    # ============================
    # MANAGER ROLES
    # ============================
    "Manager": "Manager",
    "Analytics Manager": "Manager",
    "Engineering Manager": "Manager",
    "Data Science Manager": "Manager",
    "Data Engineering Manager": "Manager",
    "Data Operations Manager": "Manager",
    "Data Governance Manager": "Manager",
    "Business Intelligence Manager": "Manager",
    "Marketing Analytics Manager": "Manager",
    "Machine Learning Manager": "Manager",

    # ============================
    # ARCHITECT ROLES
    # ============================
    "AI Architect": "Architect",
    "GenAI Architect": "Architect",
    "Big Data Architect": "Architect",
    "Cloud Data Architect": "Architect",
    "Solution Architect": "Architect",
    "Software Architect": "Architect",
    "Principal Software Architect": "Architect",
    "Power BI Architect": "Architect",

    # ============================
    # SPECIALIST ROLES
    # ============================
    "AI Specialist": "Specialist",
    "Analytics Specialist": "Specialist",
    "Data Specialist": "Specialist",
    "Data Quality Specialist": "Specialist",
    "Data Reporting Specialist": "Specialist",
    "Data Visualization Specialist": "Specialist",
    "Business Intelligence Specialist": "Specialist",
    "Master Data Specialist": "Specialist",
    "Safety Data Management Specialist": "Specialist",

    # ============================
    # SCIENTIST ROLES (NON-ML)
    # ============================
    "Research Scientist": "Research Scientist",
    "Applied Research Scientist": "Research Scientist",
    "Computational Scientist": "Research Scientist",
    "Bioinformatics Scientist": "Research Scientist",
    "Principal Researcher": "Research Scientist",
    "Postdoctoral Researcher": "Research Scientist",
    "Postdoctoral Fellow": "Research Scientist",
}

df["job_title_group"] = df["job_title"].map(job_title_map).fillna("Other")
df["job_title_group"] = df["job_title_group"].astype("category")

# =========================================================
# NLP CLUSTERING FOR JOB TITLES (TF-IDF + KMeans)
# =========================================================

other_df = df[df["job_title_group"] == "Other"].copy()
titles = other_df["job_title"].astype(str).tolist()

# ---------------------------------------------------------
# 1. TF-IDF Vectorization
# ---------------------------------------------------------

tfidf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

X = tfidf.fit_transform(titles)

# ---------------------------------------------------------
# 2. Choose number of clusters (k)
# ---------------------------------------------------------

k = 12

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

other_df["nlp_cluster"] = clusters

# ---------------------------------------------------------
# 3. Automatic Cluster Naming
# ---------------------------------------------------------

def get_top_terms(cluster_id, n_terms=8):
    centroid = kmeans.cluster_centers_[cluster_id]
    top_indices = centroid.argsort()[::-1][:n_terms]
    return [tfidf.get_feature_names_out()[i] for i in top_indices]

cluster_names = {}

for cid in range(k):
    terms = get_top_terms(cid)
    cluster_names[cid] = ", ".join(terms[:3])

other_df["nlp_cluster_name"] = other_df["nlp_cluster"].map(cluster_names)

# ---------------------------------------------------------
# 4. Merge NLP clusters back into main df
# ---------------------------------------------------------

df["job_title_group"] = df["job_title_group"].astype("object")

df.loc[df["job_title_group"] == "Other", "job_title_group"] = \
    other_df["nlp_cluster_name"].values

df["job_title_group"] = df["job_title_group"].astype("category")

# ---------------------------------------------------------
# 5. Show cluster summary
# ---------------------------------------------------------

print("\n===== NLP CLUSTER SUMMARY =====")
print(df["job_title_group"].value_counts())

# ---------------------------------------------------------
# 6. Show top titles per NLP cluster
# ---------------------------------------------------------

print("\n===== TOP TITLES PER NLP CLUSTER =====")
for cid in range(k):
    cname = cluster_names[cid]
    subset = other_df[other_df["nlp_cluster"] == cid]
    print(f"\nCluster {cid} — {cname}")
    print(subset["job_title"].value_counts().head(15))
    
# ---------------------------------------------------------
# 7. Cleaning the NLP cluster job titles group names
# ---------------------------------------------------------

nlp_clean_map = {
    "data, consultant, engineer": "Data Consulting & Engineering",
    "business, intelligence, business intelligence": "Business Intelligence / BI",
    "architect, data architect, data": "Data Architecture",
    "analytics engineer, analytics, engineer": "Analytics Engineering",
    "product manager, product, manager": "Product Management",
    "developer, software, software developer": "Software Development",
    "research engineer, research, engineer": "Research Engineering",
    "associate, team lead, team": "Team Lead / Associate Roles",
    "data manager, manager, data": "Data Management",
    "scientist, applied scientist, applied": "Applied Science",
    "solutions architect, solutions, architect": "Solutions Architecture",
    "devops engineer, devops, engineer": "DevOps Engineering"
}

# Apply clean human-readable names to NLP clusters
df["job_title_group"] = df["job_title_group"].replace(nlp_clean_map)

# ---------------------------------------------------------
# Convert Data Types
# ---------------------------------------------------------

numeric_cols = ["salary_in_usd", "remote_ratio", "data_age"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

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

for col in categorical_cols:
    df[col] = df[col].astype("category")

# ---------------------------------------------------------
# Drop salary + salary_currency
# ---------------------------------------------------------

df.drop(columns=["salary"], inplace=True)
df.drop(columns=["salary_currency"], inplace=True)

# ---------------------------------------------------------
# Remove Duplicates & Missing Target
# ---------------------------------------------------------

df = df.drop_duplicates()
df = df.dropna(subset=["salary_in_usd"]).reset_index(drop=True)

# =========================================================
# 8. SAVE FINAL CLEANED DATASET
# =========================================================

final_path = "dss2025_final.csv"
df.to_csv(final_path, index=False)
print(f"\nSaved FINAL cleaned dataset as {final_path}")

# =========================================================
# 9. EDA SUMMARY
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

    if pd.api.types.is_numeric_dtype(df[col]):
        print(df[col].describe())
    else:
        print(df[col].value_counts())

# =========================================================
# 10. Profiling Report
# =========================================================

profile = ProfileReport(df, title="DS Salaries Summary")
profile.to_file("ds_salaries_profile.html")
print("Saved profiling report to ds_salaries_profile.html")

# =========================================================
# 11. Prepare Final Dataset for EDA
# =========================================================

dss2025 = df.copy()

# =========================================================
# 12. Unique Values Per Column
# =========================================================

print("\n----- UNIQUE VALUES -----")
for col in dss2025.columns:
    print(f"\n{col}")
    print("Unique Count:", dss2025[col].nunique())

# =========================================================
# 13. Summary Statistics (R-style)
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
# 14. Profiling Report (HTML)
# =========================================================

profile = ProfileReport(dss2025, title="DS Salaries Summary")
output_path = r"C:\Users\aniec\Desktop\ibm-ml-project\01-EDA\ds_salaries_profile.html"
profile.to_file(output_path)
print("Saved profiling report to:", output_path)

# =========================================================
# 15. Numeric & Categorical Summaries
# =========================================================

print("\n----- NUMERIC SUMMARY -----")
print(dss2025.describe())

print("\n----- CATEGORICAL SUMMARY -----")
print(dss2025.describe(include=["category", "object"]))

# =========================================================
# 16. Missing Values
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
# 16B. Imputation Functions (Experience + Remote)
# =========================================================

def impute_experience_level(df):
    df["job_title"] = df["job_title"].astype(str)

    senior_map = {
        "Executive-Level": ["executive", "vp", "vice president", "chief", "cto", "ceo", "cso", "head of"],
        "Senior-Level": ["senior", "sr", "lead", "principal", "staff", "director", "manager"],
        "Middle-Level": ["mid", "intermediate", "associate", "specialist", "analyst ii", "analyst 2"],
        "Entry-Level": ["junior", "jr", "entry", "intern", "assistant", "analyst i", "analyst 1"]
    }

    def infer_level(title):
        t = title.lower()
        for level, keywords in senior_map.items():
            if any(k in t for k in keywords):
                return level
        return "Middle-Level"

    df["experience_level"] = df["experience_level"].fillna(
        df["job_title"].apply(infer_level)
    )

    df["experience_level"] = pd.Categorical(
        df["experience_level"],
        categories=["Entry-Level", "Middle-Level", "Senior-Level", "Executive-Level"],
        ordered=True
    )

    return df


def impute_remote_ratio(df):
    remote_mask = (
        df["remote_ratio"].isna() &
        (df["employee_residence"] != df["company_location"])
    )

    df.loc[remote_mask, "remote_ratio"] = 100
    df["remote_ratio"] = df["remote_ratio"].fillna(50)

    remote_map = {0: "On-Site", 50: "Hybrid", 100: "Remote"}
    df["remote_work_cat"] = df["remote_ratio"].replace(remote_map)

    df["remote_work_cat"] = pd.Categorical(
        df["remote_work_cat"],
        categories=["On-Site", "Hybrid", "Remote"],
        ordered=True
    )

    return df


# =========================================================
# 16C. BEFORE/AFTER MISSING VALUES REPORT
# =========================================================

print("\n===== MISSING VALUES BEFORE IMPUTATION =====")
print(dss2025.isnull().sum())

# Apply imputations
dss2025 = impute_experience_level(dss2025)
dss2025 = impute_remote_ratio(dss2025)

print("\n===== MISSING VALUES AFTER IMPUTATION =====")
print(dss2025.isnull().sum())

# =========================================================
# 17. Duplicate Check
# =========================================================

duplicates = dss2025.duplicated().sum()
print("\n----- DUPLICATES -----")
print("Duplicate Rows:", duplicates)

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
# 20. SHOW OBSERVATIONS FOR EACH COLUMN
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

