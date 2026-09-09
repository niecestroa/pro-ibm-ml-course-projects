# =========================================================
# Created:    2026-06-29
# Last Edit:  2026-07-20
# Author:     Aaron Niecestro
#
# Description:
# Preliminary EDA for the Data Science Salaries 2023 dataset.
# Includes data loading, structure checks, missing values,
# summary statistics, categorical exploration, and visualization.
# =========================================================

# -----------------------------
# 1. Importing Libraries
# -----------------------------

# Loading the Data directly from Kaggle

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np

# -----------------------------
# 2. Loading Data
# -----------------------------

# Set the path to the file you'd like to load
file_path = ""

# Load the latest version
dss23 = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "arnabchaki/data-science-salaries-2023",
    "ds_salaries.csv"
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", dss23.head())

pd.set_option("display.max_columns", None)
print(dss23.head())

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

dss23.head()

# ==============================
# Preliminary Work - Data Summary
# ==============================

# Experience level mapping (clean labels)
exp_map = {
    'EN': 'Entry',
    'MI': 'Mid',
    'SE': 'Senior',
    'EX': 'Executive'
}

emp_map = {
    'FT': 'Full-Time',
    'PT': 'Part-Time',
    'CT': 'Contract',
    'FL': 'Freelance'
}

# Create a labeled version for display only
dss23['experience_label'] = dss23['experience_level'].map(exp_map)
dss23['employment_label'] = dss23['employment_type'].map(emp_map)

# -----------------------------
# 1. Basic structure
# -----------------------------
print("----- SHAPE -----")
print(dss23.shape)

print("\n----- COLUMN NAMES -----")
print(dss23.columns.tolist())

print("\n----- DATA TYPES -----")
print(dss23.dtypes)

# -----------------------------
# 2. Missing values
# -----------------------------
print("\n----- MISSING VALUES (COUNT) -----")
print(dss23.isna().sum())

print("\n----- MISSING VALUES (PERCENT) -----")
missing_pct = (dss23.isna().sum() / len(dss23)) * 100
print(missing_pct)

# -----------------------------
# 3. Summary (R-style)
# -----------------------------
# Data Summary - Similar to R structure

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
            print(f"Type:     Categorical")
            print(f"Unique:   {desc['unique']}")
            print(f"Top:      {desc['top']}")
            print(f"Freq:     {desc['freq']}")
        print()

summary_r(dss23)

print("\n----- SUMMARY (R-style table) -----")
print(dss23.describe(include='all'))

# -----------------------------
# 3. Summary (Python-style) using HTML - IMPOSSIBLY USEFUL
#    Note: Best in Jupyter, since have to load HTML file manually otherwise
# -----------------------------
from ydata_profiling import ProfileReport

profile = ProfileReport(dss23, title="DS Salaries Summary")
profile.to_notebook_iframe()

# HTML Summary of Data - Good Full Summary (Work best in Jupyter)
output_path = r"C:\Users\aniec\OneDrive\Documents\Research Projects\Data Science Salary\ds_salaries_profile.html"

profile.to_file(output_path)
print("Saved to:", output_path)

# -----------------------------
# 4. Unique values for categorical columns
# -----------------------------
print("\n----- UNIQUE VALUES (CATEGORICAL) -----")
cat_cols = dss23.select_dtypes(include='object').columns
for col in cat_cols:
    print(f"\n{col}:")
    print(dss23[col].unique()[:20])  # show first 20 unique values

# -----------------------------
# 5. Numeric distribution check
# -----------------------------
print("\n----- NUMERIC SUMMARY -----")
print(dss23.describe())

# -----------------------------
# 6. Duplicated rows
# -----------------------------
print("\n----- DUPLICATED ROWS -----")
print(dss23.duplicated().sum())

# -----------------------------
# 7. Categorical value counts
# -----------------------------
print("\n----- VALUE COUNTS (TOP 10 PER COLUMN) -----")
for col in cat_cols:
    print(f"\n{col}:")
    print(dss23[col].value_counts().head(10))

# -----------------------------
# 8. Correlation matrix (numeric only)
# -----------------------------
print("\n----- CORRELATION MATRIX -----")
print(dss23.corr(numeric_only=True))

# -----------------------------
# 9. Optional: convert categorical columns to category dtype
# -----------------------------
dss23[cat_cols] = dss23[cat_cols].astype('category')
print("\nConverted categorical columns to 'category' dtype.")


# =========================================================
# MOVING FROM SUMMARY TO VISUALIZATION
# =========================================================
# The summary section helped us understand the dataset's structure:
#   - data types
#   - missing values
#   - numeric ranges and quartiles
#   - category frequencies
#
# Now we move into the visualization stage, where graphs help reveal:
#   - distribution shapes
#   - group differences
#   - category patterns
#   - correlations
#   - geographic salary trends
#
# Visuals allow us to see patterns that summary statistics alone cannot.
# =========================================================


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="whitegrid", palette="deep")

# ---------------------------------------------------------
# 0. Histograms and Boxplots
# ---------------------------------------------------------

# Identify continuous (numeric) columns
num_cols = dss23.select_dtypes(include=['int64','float64']).columns

# Histograms for all numeric columns
for col in num_cols:
    plt.figure(figsize=(10,6))
    sns.histplot(dss23[col], kde=True, bins=40)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

# Boxplots for all numeric columns
for col in num_cols:
    plt.figure(figsize=(10,6))
    sns.boxplot(x=dss23[col])
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.show()

# ---------------------------------------------------------
# 1. Salary Distribution (Histogram + KDE)
# ---------------------------------------------------------
plt.figure(figsize=(10,6))
sns.histplot(dss23['salary_in_usd'], kde=False, bins=40)
plt.title("Salary Distribution (USD)")
plt.xlabel("Salary in USD")
plt.ylabel("Count")
plt.show()

# KDE (Kernel Density Estimate) is a smooth curve that shows the shape of a numeric
# distribution. It's great when you have many data points and want to see skew,
# peaks, and tails more clearly than a histogram. Avoid KDE for small datasets,
# categorical variables, or discrete values where smoothing is misleading.

plt.figure(figsize=(10,6))
sns.histplot(dss23['salary_in_usd'], kde=True, bins=40)
plt.title("Salary Distribution (USD)")
plt.xlabel("Salary in USD")
plt.ylabel("Count")
plt.show()

# ---------------------------------------------------------
# 2. Salary by Experience Level (Boxplot)
# ---------------------------------------------------------
plt.figure(figsize=(10,6))
sns.boxplot(data=dss23, x='experience_level', y='salary_in_usd')
plt.title("Salary by Experience Level")
plt.xlabel("Experience Level")
plt.ylabel("Salary in USD")
plt.show()

# ---------------------------------------------------------
# 3. Salary by Employment Type (Boxplot)
# ---------------------------------------------------------
plt.figure(figsize=(10,6))
sns.boxplot(data=dss23, x='employment_type', y='salary_in_usd')
plt.title("Salary by Employment Type")
plt.xlabel("Employment Type")
plt.ylabel("Salary in USD")
plt.show()

# ---------------------------------------------------------
# 4. Salary by Remote Ratio (Boxplot)
# ---------------------------------------------------------
plt.figure(figsize=(10,6))
sns.boxplot(data=dss23, x='remote_ratio', y='salary_in_usd')
plt.title("Salary by Remote Ratio")
plt.xlabel("Remote Ratio (%)")
plt.ylabel("Salary in USD")
plt.show()

# ---------------------------------------------------------
# 5. Job Title Frequency (Top 20)
# ---------------------------------------------------------
plt.figure(figsize=(12,8))
top_jobs = dss23['job_title'].value_counts().head(20)
sns.barplot(x=top_jobs.values, y=top_jobs.index)
plt.title("Top 20 Most Common Job Titles")
plt.xlabel("Count")
plt.ylabel("Job Title")
plt.show()

# ---------------------------------------------------------
# 6. Salary by Country (Employee Residence)
# ---------------------------------------------------------
plt.figure(figsize=(12,8))
top_countries = dss23['employee_residence'].value_counts().head(20).index
sns.boxplot(data=dss23[dss23['employee_residence'].isin(top_countries)],
            x='employee_residence', y='salary_in_usd')
plt.xticks(rotation=45)
plt.title("Salary by Employee Residence (Top 20 Countries)")
plt.xlabel("Country")
plt.ylabel("Salary in USD")
plt.show()

# ---------------------------------------------------------
# 7. Correlation Heatmap (Numeric Only)
# ---------------------------------------------------------
plt.figure(figsize=(10,6))
corr = dss23.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# ---------------------------------------------------------
# 8. Geographic Salary Plot (Company Location)
# ---------------------------------------------------------
plt.figure(figsize=(12,8))
country_salary = dss23.groupby('company_location')['salary_in_usd'].median().sort_values(ascending=False)
sns.barplot(x=country_salary.values, y=country_salary.index)
plt.title("Median Salary by Company Location")
plt.xlabel("Median Salary (USD)")
plt.ylabel("Company Location")
plt.show()

# =========================================================
# ADDITIONAL GRAPHS FOR A COMPLETE EDA
# =========================================================
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", palette="deep")

# ---------------------------------------------------------
# 9. Salary Over Time (Work Year)
# ---------------------------------------------------------
plt.figure(figsize=(10,6))
year_salary = dss23.groupby('work_year')['salary_in_usd'].median()
sns.lineplot(x=year_salary.index, y=year_salary.values, marker='o')
plt.title("Median Salary Over Time")
plt.xlabel("Work Year")
plt.ylabel("Median Salary (USD)")
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=dss23, x='work_year', y='salary_in_usd')
plt.title("Salary Distribution by Work Year")
plt.xlabel("Work Year")
plt.ylabel("Salary in USD")
plt.show()

# ---------------------------------------------------------
# 10. Salary by Company Size
# ---------------------------------------------------------
plt.figure(figsize=(10,6))
sns.boxplot(data=dss23, x='company_size', y='salary_in_usd')
plt.title("Salary by Company Size")
plt.xlabel("Company Size")
plt.ylabel("Salary in USD")
plt.show()

# ---------------------------------------------------------
# 11. Median Salary by Job Title (Top 20)
# ---------------------------------------------------------
plt.figure(figsize=(12,8))
top_titles = dss23['job_title'].value_counts().head(20).index
title_salary = (dss23[dss23['job_title'].isin(top_titles)]
                .groupby('job_title')['salary_in_usd']
                .median()
                .sort_values(ascending=False))

sns.barplot(x=title_salary.values, y=title_salary.index)
plt.title("Median Salary by Job Title (Top 20)")
plt.xlabel("Median Salary (USD)")
plt.ylabel("Job Title")
plt.show()

# ---------------------------------------------------------
# 12. Salary vs Remote Ratio (Scatter)
# ---------------------------------------------------------
plt.figure(figsize=(10,6))
sns.scatterplot(data=dss23, x='remote_ratio', y='salary_in_usd', alpha=0.6)
plt.title("Salary vs Remote Ratio")
plt.xlabel("Remote Ratio (%)")
plt.ylabel("Salary in USD")
plt.show()

# ---------------------------------------------------------
# 13. Pairplot (Numeric Relationships)
# ---------------------------------------------------------
numeric_cols = ['salary_in_usd', 'remote_ratio', 'work_year']
sns.pairplot(dss23[numeric_cols], diag_kind='kde')
plt.suptitle("Pairplot of Key Numeric Variables", y=1.02)
plt.show()

# ---------------------------------------------------------
# 14. Countplots for Key Categorical Variables
# ---------------------------------------------------------
plt.figure(figsize=(10,6))
sns.countplot(data=dss23, x='experience_level')
plt.title("Count of Experience Levels")
plt.xlabel("Experience Level")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(data=dss23, x='employment_type')
plt.title("Count of Employment Types")
plt.xlabel("Employment Type")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(data=dss23, x='company_size')
plt.title("Count of Company Sizes")
plt.xlabel("Company Size")
plt.ylabel("Count")
plt.show()

# ---------------------------------------------------------
# 15. Salary by Experience Level + Company Size (Interaction)
# ---------------------------------------------------------
plt.figure(figsize=(12,8))
sns.boxplot(data=dss23, x='experience_level', y='salary_in_usd', hue='company_size')
plt.title("Salary by Experience Level and Company Size")
plt.xlabel("Experience Level")
plt.ylabel("Salary in USD")
plt.legend(title="Company Size")
plt.show()


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
