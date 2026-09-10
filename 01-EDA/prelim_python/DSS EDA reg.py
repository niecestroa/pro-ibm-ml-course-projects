# -*- coding: utf-8 -*-

# =========================================================
'''
Author:     Aaron Niecestro
Project:    DS Salaries – Production-Ready EDA for Regression Models

# Created:    2026-09-09
# Last Edit:  2026-09-10
# Author:     Aaron Niecestro

Description:
Preliminary EDA 7 Kaggle datasets combined into one useable data.
Dataset has data ranging from 2020 to 2025.
Includes data loading, structure checks, missing values,
summary statistics, categorical exploration, and visualization.

This is for Regression Models Purpose Only.

Author:     Aaron Niecestro
Project:    DS Salaries – Production-Ready EDA for Regression Models

Description:
Preliminary EDA on combined Kaggle DS salary datasets (2020–2025).
Includes:
- Data loading
- Structural checks
- Summary statistics
- Categorical & numerical exploration
- Outlier analysis
- Correlation & multicollinearity (VIF)
- Feature importance (Random Forest)
- Skewness & log-transform check
- Leakage check
- Target-wise feature distributions
- Key business insights

This is for Regression Models Purpose Only.
'''

# =========================================================

# --------------------------------------------------------------------------
# Explatory Data Analysis
# --------------------------------------------------------------------------

# =========================================================
# 0. Imports & Configuration
# =========================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ydata_profiling import ProfileReport
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set(style="whitegrid", context="talk")

# =========================================================
# 1. Load Data
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
dss2025 = pd.read_csv(file_path)

print("----- DATA SHAPE -----")
print(f"Rows: {dss2025.shape[0]}, Columns: {dss2025.shape[1]}\n")

# =========================================================
# 2. Column Type Setup (Numeric & Categorical)
# =========================================================

# Explicitly define numeric and categorical columns for consistency
numeric_cols = dss2025.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = dss2025.select_dtypes(include=["object", "category"]).columns.tolist()

print("----- NUMERIC COLUMNS -----")
print(numeric_cols)

print("\n----- CATEGORICAL COLUMNS -----")
print(categorical_cols)

# =========================================================
# 3. R-style Summary + Profiling
# =========================================================

def summary_r(df):
    """R-style summary for quick structure + distribution overview."""
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

print("\n----- SUMMARY (R-style table) -----")
print(dss2025.describe(include="all"))

# HTML profiling (best in Jupyter)
profile = ProfileReport(dss2025, title="DS Salaries Summary", minimal=True)
profile.to_notebook_iframe()

output_path = r"C:\Users\aniec\Desktop\ibm-ml-project\01-EDA\ds_salaries_profile.html"
profile.to_file(output_path)
print("Saved profiling report to:", output_path)

print("\n----- NUMERIC SUMMARY -----")
print(dss2025.describe())

print("\n----- CATEGORICAL SUMMARY -----")
print(dss2025.describe(include=["category", "object"]))

# =========================================================
# 4. Target Variable Exploration (salary_in_usd)
# =========================================================

plt.figure(figsize=(10, 6))
sns.histplot(dss2025["salary_in_usd"], bins=50, kde=True)
plt.title("Salary in USD Distribution")
plt.xlabel("Salary (USD)")
plt.ylabel("Count")
plt.show()

# Log-transform check for skewness
plt.figure(figsize=(10, 6))
sns.histplot(np.log1p(dss2025["salary_in_usd"]), bins=50, kde=True)
plt.title("Log-Transformed Salary in USD Distribution (log1p)")
plt.xlabel("log1p(Salary in USD)")
plt.ylabel("Count")
plt.show()

# =========================================================
# 5. Univariate Categorical Exploration
# =========================================================

plt.figure(figsize=(8, 5))
sns.countplot(data=dss2025, x="experience_level", order=dss2025["experience_level"].value_counts().index)
plt.title("Experience Level Distribution")
plt.xlabel("Experience Level")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=dss2025, x="work_year", order=sorted(dss2025["work_year"].unique()))
plt.title("Work Year Distribution")
plt.xlabel("Work Year")
plt.ylabel("Count")
plt.show()

# =========================================================
# 6. Pairplot for Numeric Variables
# =========================================================

# Use a subset of numeric columns if many exist
pairplot_cols = [col for col in numeric_cols if col not in ["salary_in_usd"]][:6]
if "salary_in_usd" not in pairplot_cols:
    pairplot_cols.append("salary_in_usd")

sns.pairplot(dss2025[pairplot_cols], diag_kind="kde")
plt.suptitle("Pairplot of Key Numeric Variables", y=1.02)
plt.show()

# =========================================================
# 7. Correlation Analysis
# =========================================================

corr = dss2025[numeric_cols].corr()

print("\n----- CORRELATION MATRIX (NUMERIC) -----")
print(corr)

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
plt.title("Correlation Matrix – Numeric Features")
plt.show()

# =========================================================
# 8. Outlier Detection & Visualization (IQR Method)
# =========================================================

print("\n----- OUTLIER COUNTS (IQR Method) -----")
for col in numeric_cols:
    Q1 = dss2025[col].quantile(0.25)
    Q3 = dss2025[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = dss2025[(dss2025[col] < lower) | (dss2025[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")

    # Boxplot for visualization
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=dss2025[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# =========================================================
# 9. Basic Consistency Checks
# =========================================================

print("\nNegative Salary Values:", (dss2025["salary_in_usd"] < 0).sum())

print("\nInvalid Remote Ratios:")
print(
    dss2025[~dss2025["remote_ratio"].isin([0, 50, 100])]["remote_ratio"]
    .value_counts()
)

# =========================================================
# 10. Categorical Feature Exploration & Cardinality
# =========================================================

print("\n----- CATEGORICAL DISTRIBUTIONS (Top 15) -----")
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=dss2025,
        y=col,
        order=dss2025[col].value_counts().index[:15]
    )
    plt.title(f"Distribution of {col} (Top 15)")
    plt.ylabel(col)
    plt.xlabel("Count")
    plt.show()

print("\n----- CARDINALITY -----")
for col in categorical_cols:
    print(f"{col}: {dss2025[col].nunique()} unique values")

# =========================================================
# 11. Feature Importance (Random Forest – Numeric Only)
# =========================================================

# Select numeric features (example set – adjust as needed)
numeric_features = [
    "data_age",
    "remote_ratio",
    "experience_level_num",
    "company_size_num"
]

numeric_features = [f for f in numeric_features if f in dss2025.columns]

X = dss2025[numeric_features]
y = dss2025["salary_in_usd"]

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

importances = pd.Series(model.feature_importances_, index=numeric_features)

plt.figure(figsize=(8, 5))
importances.sort_values().plot(kind="barh")
plt.title("Feature Importance (Random Forest Regression)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# =========================================================
# 12. Skewness Check
# =========================================================

print("\n----- SKEWNESS (NUMERIC FEATURES) -----")
print(
    dss2025[numeric_cols]
    .skew()
    .sort_values(ascending=False)
)

# =========================================================
# 13. Multicollinearity – VIF
# =========================================================

print("\n----- VARIANCE INFLATION FACTOR (VIF) -----")

# Drop columns that may cause singular matrix (e.g., perfectly correlated)
vif_cols = [col for col in numeric_cols if dss2025[col].nunique() > 1]

vif_df = pd.DataFrame()
vif_df["feature"] = vif_cols
vif_df["VIF"] = [
    variance_inflation_factor(dss2025[vif_cols].values, i)
    for i in range(len(vif_cols))
]

print(vif_df.sort_values("VIF", ascending=False))

# =========================================================
# 14. Data Leakage Check
# =========================================================

corr_target = dss2025[numeric_cols].corr()["salary_in_usd"].sort_values(ascending=False)
print("\n----- CORRELATION WITH TARGET (salary_in_usd) -----")
print(corr_target)

leakage_prone = []
for col in dss2025.columns:
    if "salary" in col.lower():
        leakage_prone.append(col)
    if col.endswith("_cat") and "salary" in col.lower():
        leakage_prone.append(col)

print("\nLeakage-prone features:", leakage_prone)

# =========================================================
# 15. Feature Distributions by Target
# =========================================================

plt.figure(figsize=(10, 6))
sns.boxplot(data=dss2025, x="employment_type", y="salary_in_usd")
plt.title("Salary by Employment Type")
plt.xlabel("Employment Type")
plt.ylabel("Salary (USD)")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=dss2025, x="company_size", y="salary_in_usd")
plt.title("Salary by Company Size")
plt.xlabel("Company Size")
plt.ylabel("Salary (USD)")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=dss2025, x="remote_ratio", y="salary_in_usd")
plt.title("Salary by Remote Ratio")
plt.xlabel("Remote Ratio")
plt.ylabel("Salary (USD)")
plt.show()

# =========================================================
# 16. Job Title & Location Insights
# =========================================================

# Top 20 job titles
top_jobs = dss2025["job_title"].value_counts().head(20)

plt.figure(figsize=(12, 8))
sns.barplot(x=top_jobs.values, y=top_jobs.index)
plt.title("Top 20 Most Common Job Titles")
plt.xlabel("Count")
plt.ylabel("Job Title")
plt.show()

# Median salary by top job titles
top_titles = top_jobs.index
title_salary = (
    dss2025[dss2025["job_title"].isin(top_titles)]
    .groupby("job_title")["salary_in_usd"]
    .median()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 8))
sns.barplot(x=title_salary.values, y=title_salary.index)
plt.title("Median Salary by Job Title (Top 20)")
plt.xlabel("Median Salary (USD)")
plt.ylabel("Job Title")
plt.show()

# Salary by employee residence (top 20 countries)
top_countries = dss2025["employee_residence"].value_counts().head(20).index

plt.figure(figsize=(12, 8))
sns.boxplot(
    data=dss2025[dss2025["employee_residence"].isin(top_countries)],
    x="employee_residence",
    y="salary_in_usd"
)
plt.xticks(rotation=45)
plt.title("Salary by Employee Residence (Top 20)")
plt.xlabel("Employee Residence")
plt.ylabel("Salary (USD)")
plt.show()

# Median salary by company location
country_salary = (
    dss2025.groupby("company_location")["salary_in_usd"]
    .median()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 8))
sns.barplot(x=country_salary.values, y=country_salary.index)
plt.title("Median Salary by Company Location")
plt.xlabel("Median Salary (USD)")
plt.ylabel("Company Location")
plt.show()

# Experience × Company Size interaction
plt.figure(figsize=(12, 8))
sns.boxplot(
    data=dss2025,
    x="experience_level",
    y="salary_in_usd",
    hue="company_size"
)
plt.title("Salary by Experience Level and Company Size")
plt.xlabel("Experience Level")
plt.ylabel("Salary (USD)")
plt.legend(title="Company Size")
plt.show()

# =========================================================
# 17. Time Trend – Median Salary Over Years
# =========================================================

year_salary = (
    dss2025.groupby("work_year")["salary_in_usd"]
    .median()
    .sort_values()
)

plt.figure(figsize=(10, 6))
sns.lineplot(x=year_salary.index, y=year_salary.values, marker="o")
plt.title("Median Salary Over Time")
plt.xlabel("Work Year")
plt.ylabel("Median Salary (USD)")
plt.show()

# =========================================================
# 18. Model Readiness Snapshot (No Split Here)
# =========================================================

print("\n----- MODEL READINESS SNAPSHOT -----")
print("Total missing values (handled elsewhere):", dss2025.isnull().sum().sum())
print("Total duplicates (handled elsewhere):", dss2025.duplicated().sum())
print("Most skewed features:")
print(
    dss2025[numeric_cols]
    .skew()
    .sort_values(ascending=False)
    .head()
)

print("\nHigh VIF features (VIF > 10):")
print(vif_df[vif_df["VIF"] > 10])

print("\nLeakage-prone features:", leakage_prone)

# =========================================================
# KEY INSIGHTS FROM VISUALIZATION
# =========================================================
'''
1. Salary Distribution
The salary distribution is strongly right-skewed. Most salaries fall in the 
lower-to-mid ranges, with a long tail of high earners. The KDE curve confirms 
a single dominant mode with heavy upper-tail outliers.

2. Salary by Experience Level
Salary increases consistently with experience. EN < MI < SE < EX. The boxplots 
show clear separation between levels, indicating experience is one of the 
strongest predictors of salary.

3. Salary by Employment Type
Full-time roles dominate the dataset. Contract roles show wider variability 
and occasionally higher pay, but also more outliers. Part-time and freelance 
roles tend to cluster at lower salary ranges.

4. Salary by Remote Ratio
Remote work (100%) shows competitive or slightly higher salaries compared to 
on-site roles. Hybrid (50%) sits between the two. Remote flexibility appears 
correlated with higher pay in many cases.

5. Job Title Frequency
A small number of job titles dominate the dataset (e.g., Data Scientist, 
Data Engineer). Many titles appear infrequently, suggesting the dataset is 
top-heavy and may benefit from grouping similar roles for modeling.

6. Salary by Country
Geographic differences are substantial. Countries like the US, UK, and 
Switzerland show higher median salaries, while others cluster lower. Location 
is a major driver of salary variation.

7. Correlation Heatmap
Salary shows moderate correlation with experience level and company size. 
Remote ratio and work year have weaker correlations, suggesting non-linear 
or categorical effects that boxplots capture better than correlation matrices.

8. Salary Over Time
Median salary trends upward across work years, indicating growth in the 
data science job market. Year-to-year increases are visible in both line and 
boxplots.

9. Salary by Company Size
Large companies tend to pay more on average. Small companies show wider 
variability and more outliers, suggesting inconsistent compensation structures.

10. Median Salary by Job Title
Senior and specialized roles (e.g., ML Engineer, Data Architect) show higher 
median salaries. Generalist roles cluster lower. Job title is a strong 
categorical predictor.

11. Salary vs Remote Ratio (Scatter)
Scatterplots reveal clusters at 0, 50, and 100 remote ratio. Higher salaries 
appear more frequently at 100% remote, supporting the boxplot findings.

12. Pairplot (Numeric Relationships)
Salary shows a positive trend with work year and remote ratio, though not 
strictly linear. Remote ratio and work year are weakly related.

13. Countplots (Experience, Employment, Company Size)
The dataset is dominated by full-time roles, mid-level and senior-level 
experience, and medium-to-large companies. This imbalance should be considered 
during modeling.

14. Interaction: Experience Level × Company Size
Senior roles at large companies show the highest salaries. Entry-level roles 
at small companies show the lowest. The interaction effect is strong and 
meaningful for predictive modeling.
'''

# =========================================================
# 19. Key Insights (Business-Level)
# =========================================================

'''
The salary distribution is strongly right-skewed. Most salaries fall in the
lower-to-mid ranges, with a long tail of high earners.
Salary increases consistently with experience. EN < MI < SE < EX, and senior
roles at large companies show the highest salaries.

Overall:
- Strong drivers: experience_level, company_size, job_title, location
- Secondary drivers: remote_ratio, work_year
- Clear structure for feature engineering and downstream regression modeling.
'''

# =========================================================
# Overall Summary
# =========================================================
'''
Salary is influenced most strongly by:
- Experience level
- Company size
- Job title
- Country / location
Remote ratio and work year also matter, but less directly.
The dataset shows clear patterns that will be useful for feature engineering 
and building a salary prediction model.
'''
# =========================================================

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

