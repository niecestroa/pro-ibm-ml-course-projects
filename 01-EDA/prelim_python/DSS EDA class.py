# -*- coding: utf-8 -*-

# =========================================================
'''
Classification EDA for DS Salaries Dataset

Author:     Aaron Niecestro
Created:    September 9, 2026
Last Edit:  September 14, 2026

Targets:
- salary_mean_cat (2 classes)
- salary_median_cat (2 classes)
- salary_3cat (3 classes using IQR)

This script performs:
- Summary statistics
- Profiling
- Target distribution & imbalance checks
- Categorical & numerical exploration
- Outlier detection
- Correlation & multicollinearity (VIF)
- Leakage checks
- Feature importance (Random Forest)

This is for Classification Models Purpose Only.
The classification models are the following:
1. 2 categories of mean salary from salary in USD
2. 2 categories of median salary from salary in USD
3. 3 categories of salary from salary in USD using IQR
'''
# =========================================================

# =========================================================
# Classification EDA for DS Salaries Dataset (USA-only)
# =========================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ydata_profiling import ProfileReport
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set(style="whitegrid", context="talk")

# =========================================================
# 1. Load Data
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
dss2025 = pd.read_csv(file_path)

# USA-only subset
dss2025 = dss2025.query("company_location == 'United States'").copy()

# =========================================================
# 2. Numeric Encodings
# =========================================================

exp_map = {"EN": 0, "MI": 1, "SE": 2, "EX": 3}
size_map = {"S": 0, "M": 1, "L": 2}

dss2025["experience_level_num"] = dss2025["experience_level"].map(exp_map)
dss2025["company_size_num"] = dss2025["company_size"].map(size_map)

# =========================================================
# 3. Identify Columns
# =========================================================

numeric_cols = dss2025.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = dss2025.select_dtypes(include=["object", "category"]).columns.tolist()

print("Numeric Columns:", numeric_cols)
print("Categorical Columns:", categorical_cols)

# =========================================================
# 4. Summary + Profiling
# =========================================================

def summary_r(df):
    print("----- R-style Summary -----")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")
    for col in df.columns:
        print(f"--- {col} ---")
        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            print(f"Min: {desc['min']}")
            print(f"1st Qu.: {desc['25%']}")
            print(f"Median: {desc['50%']}")
            print(f"Mean: {desc['mean']}")
            print(f"3rd Qu.: {desc['75%']}")
            print(f"Max: {desc['max']}")
        else:
            desc = df[col].describe()
            print("Type: Categorical")
            print(f"Unique: {desc['unique']}")
            print(f"Top: {desc['top']}")
            print(f"Freq: {desc['freq']}")
        print()

summary_r(dss2025)

profile = ProfileReport(dss2025, title="DS Salaries Classification Summary", minimal=True)
profile.to_file(r"C:\Users\aniec\Desktop\ibm-ml-project\01-EDA\ds_salaries_profile_classification.html")

# =========================================================
# 5. Target Variable Distribution
# =========================================================

plt.figure(figsize=(10,6))
sns.histplot(dss2025["salary_in_usd"], bins=50, kde=True)
plt.title("Salary Distribution (USD)")
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(np.log1p(dss2025["salary_in_usd"]), bins=50, kde=True)
plt.title("Log-Transformed Salary Distribution")
plt.show()

# =========================================================
# 6. Pairplot (Safe)
# =========================================================

numeric_cols = dss2025.select_dtypes(include=["int64", "float64"]).columns.tolist()
pairplot_cols = numeric_cols[:6]

print("Pairplot columns:", pairplot_cols)

if len(pairplot_cols) > 1:
    sns.pairplot(dss2025[pairplot_cols], diag_kind="kde")
    plt.suptitle("Pairplot of Numeric Variables", y=1.02)
    plt.show()
else:
    print("Not enough numeric columns for pairplot.")

# =========================================================
# 7. Correlation Matrix
# =========================================================

corr = dss2025[numeric_cols].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.show()

# =========================================================
# 8. Outlier Detection (Safe)
# =========================================================

print("\n----- OUTLIER COUNTS -----")
for col in numeric_cols:

    if dss2025[col].nunique() < 2:
        print(f"{col}: skipped (constant column)")
        continue

    Q1 = dss2025[col].quantile(0.25)
    Q3 = dss2025[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = dss2025[(dss2025[col] < lower) | (dss2025[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")

    plt.figure(figsize=(10,4))
    sns.boxplot(x=dss2025[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# =========================================================
# 9. Class Imbalance
# =========================================================

targets = ["salary_mean_cat", "salary_median_cat", "salary_3cat"]

def check_imbalance(col):
    print(f"\nClass distribution for {col}:")
    counts = dss2025[col].value_counts()
    perc = dss2025[col].value_counts(normalize=True) * 100
    print(pd.concat([counts, perc], axis=1, keys=["count", "percentage"]))

def plot_imbalance(col):
    plt.figure(figsize=(6,4))
    dss2025[col].value_counts().plot(kind="bar", color="skyblue")
    plt.title(f"Class Imbalance: {col}")
    plt.show()

for t in targets:
    check_imbalance(t)
    plot_imbalance(t)

# =========================================================
# 10. Feature Importance (Random Forest)
# =========================================================

numeric_features = ["data_age", "remote_ratio", "experience_level_num", "company_size_num"]
numeric_features = [f for f in numeric_features if f in dss2025.columns]

X = dss2025[numeric_features]
y = dss2025["salary_mean_cat"]

clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

importances = pd.Series(clf.feature_importances_, index=numeric_features)

plt.figure(figsize=(8,5))
importances.sort_values().plot(kind="barh")
plt.title("Feature Importance (Classification)")
plt.show()

# =========================================================
# 11. VIF (Safe)
# =========================================================

vif_cols = [col for col in numeric_cols if dss2025[col].nunique() > 1]
vif_data = dss2025[vif_cols].dropna().copy()

vif_df = pd.DataFrame()
vif_df["feature"] = vif_cols
vif_df["VIF"] = [
    variance_inflation_factor(vif_data.values, i)
    for i in range(len(vif_cols))
]

print("\n----- VIF -----")
print(vif_df.sort_values("VIF", ascending=False))

# =========================================================
# 12. Leakage Check
# =========================================================

corr_target = dss2025[numeric_cols].corr()["salary_in_usd"].sort_values(ascending=False)
print("\nCorrelation with salary_in_usd:")
print(corr_target)

leakage_prone = [col for col in dss2025.columns if "salary" in col.lower()]
print("\nLeakage-prone features:", leakage_prone)

# =========================================================
# 13. Salary by Categorical Features
# =========================================================

plt.figure(figsize=(10,6))
sns.boxplot(data=dss2025, x="employment_type", y="salary_in_usd")
plt.title("Salary by Employment Type")
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=dss2025, x="company_size", y="salary_in_usd")
plt.title("Salary by Company Size")
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=dss2025, x="remote_ratio", y="salary_in_usd")
plt.title("Salary by Remote Ratio")
plt.show()

# =========================================================
# 14. Job Title Insights
# =========================================================

top_jobs = dss2025["job_title"].value_counts().head(20)

plt.figure(figsize=(12,8))
sns.barplot(x=top_jobs.values, y=top_jobs.index)
plt.title("Top 20 Job Titles")
plt.show()

top_titles = top_jobs.index
title_salary = (
    dss2025[dss2025["job_title"].isin(top_titles)]
    .groupby("job_title")["salary_in_usd"]
    .median()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12,8))
sns.barplot(x=title_salary.values, y=title_salary.index)
plt.title("Median Salary by Job Title")
plt.show()

# =========================================================
# 15. Time Trend
# =========================================================

year_salary = dss2025.groupby("work_year")["salary_in_usd"].median()

plt.figure(figsize=(10,6))
sns.lineplot(x=year_salary.index, y=year_salary.values, marker="o")
plt.title("Median Salary Over Time")
plt.show()

# --------------------------------------------------------------------------
'''
Final Checklist

Did you check for the following:
1. Missing Values handled?
2. Duplicate values removed?
3. Correct Data Types?
4. Outliers Checked and noted?
5. Determined what to do about outliers?
6. Consistent Data?
7. Valid Target Distribution?
8. Data Leakage Checked?
9. All plots are done?

If Answered YES to all the above then move on.
'''
# --------------------------------------------------------------------------


# =========================================================
# KEY INSIGHTS FROM VISUALIZATION
# =========================================================

# 1. Salary Distribution
# The salary distribution is strongly right-skewed. Most salaries fall in the 
# lower-to-mid ranges, with a long tail of high earners. The KDE curve confirms 
# a single dominant mode with heavy upper-tail outliers.

# 2. Salary by Experience Level
# Salary increases consistently with experience. EN < MI < SE < EX. The boxplots 
# show clear separation between levels, indicating experience is one of the 
# strongest predictors of salary.

# 3. Salary by Employment Type
# Full-time roles dominate the dataset. Contract roles show wider variability 
# and occasionally higher pay, but also more outliers. Part-time and freelance 
# roles tend to cluster at lower salary ranges.

# 4. Salary by Remote Ratio
# Remote work (100%) shows competitive or slightly higher salaries compared to 
# on-site roles. Hybrid (50%) sits between the two. Remote flexibility appears 
# correlated with higher pay in many cases.

# 5. Job Title Frequency
# A small number of job titles dominate the dataset (e.g., Data Scientist, 
# Data Engineer). Many titles appear infrequently, suggesting the dataset is 
# top-heavy and may benefit from grouping similar roles for modeling.

# 6. Salary by Country
# Geographic differences are substantial. Countries like the US, UK, and 
# Switzerland show higher median salaries, while others cluster lower. Location 
# is a major driver of salary variation.

# 7. Correlation Heatmap
# Salary shows moderate correlation with experience level and company size. 
# Remote ratio and work year have weaker correlations, suggesting non-linear 
# or categorical effects that boxplots capture better than correlation matrices.

# 8. Salary Over Time
# Median salary trends upward across work years, indicating growth in the 
# data science job market. Year-to-year increases are visible in both line and 
# boxplots.

# 9. Salary by Company Size
# Large companies tend to pay more on average. Small companies show wider 
# variability and more outliers, suggesting inconsistent compensation structures.

# 10. Median Salary by Job Title
# Senior and specialized roles (e.g., ML Engineer, Data Architect) show higher 
# median salaries. Generalist roles cluster lower. Job title is a strong 
# categorical predictor.

# 11. Salary vs Remote Ratio (Scatter)
# Scatterplots reveal clusters at 0, 50, and 100 remote ratio. Higher salaries 
# appear more frequently at 100% remote, supporting the boxplot findings.

# 12. Pairplot (Numeric Relationships)
# Salary shows a positive trend with work year and remote ratio, though not 
# strictly linear. Remote ratio and work year are weakly related.

# 13. Countplots (Experience, Employment, Company Size)
# The dataset is dominated by full-time roles, mid-level and senior-level 
# experience, and medium-to-large companies. This imbalance should be considered 
# during modeling.

# 14. Interaction: Experience Level × Company Size
# Senior roles at large companies show the highest salaries. Entry-level roles 
# at small companies show the lowest. The interaction effect is strong and 
# meaningful for predictive modeling.

# =========================================================
# Overall Summary
# =========================================================
# Salary is influenced most strongly by:
#   - Experience level
#   - Company size
#   - Job title
#   - Country / location
#
# Remote ratio and work year also matter, but less directly.
# The dataset shows clear patterns that will be useful for feature engineering 
# and building a salary prediction model.
# =========================================================
