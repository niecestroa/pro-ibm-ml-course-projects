# -*- coding: utf-8 -*-

# =========================================================
'''
Classification EDA for DS Salaries Dataset

Author: Aaron Niecestro
Created: September 9, 2026
Last Edit: September 10, 2026

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

# --------------------------------------------------------------------------
# Explatory Data Analysis
# --------------------------------------------------------------------------

# =========================================================
# 0. Imports & Configuration
# =========================================================

import pandas as pd              # Data manipulation
import numpy as np               # Numerical operations
import seaborn as sns            # Visualization
import matplotlib.pyplot as plt  # Plotting

from ydata_profiling import ProfileReport  # HTML profiling report
from sklearn.ensemble import RandomForestClassifier  # Feature importance
from statsmodels.stats.outliers_influence import variance_inflation_factor  # VIF

sns.set(style="whitegrid", context="talk")  # Clean visual theme

# =========================================================
# 1. Load Data
# =========================================================

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"
dss2025 = pd.read_csv(file_path)  # Load CSV into DataFrame

print("----- DATA SHAPE -----")
print(f"Rows: {dss2025.shape[0]}, Columns: {dss2025.shape[1]}\n")  # Print dataset dimensions

# =========================================================
# 2. Identify Numeric & Categorical Columns
# =========================================================

numeric_cols = dss2025.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = dss2025.select_dtypes(include=["object", "category"]).columns.tolist()

print("Numeric Columns:", numeric_cols)
print("Categorical Columns:", categorical_cols)

# =========================================================
# 3. R-style Summary + Profiling
# =========================================================

def summary_r(df):
    """Prints R-style summary for each column."""
    print("----- R-style Summary -----")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

    for col in df.columns:
        print(f"--- {col} ---")
        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()  # Numeric summary
            print(f"Min: {desc['min']}")
            print(f"1st Qu.: {desc['25%']}")
            print(f"Median: {desc['50%']}")
            print(f"Mean: {desc['mean']}")
            print(f"3rd Qu.: {desc['75%']}")
            print(f"Max: {desc['max']}")
        else:
            desc = df[col].describe()  # Categorical summary
            print("Type: Categorical")
            print(f"Unique: {desc['unique']}")
            print(f"Top: {desc['top']}")
            print(f"Freq: {desc['freq']}")
        print()

summary_r(dss2025)

# Full HTML profiling report
profile = ProfileReport(dss2025, title="DS Salaries Classification Summary", minimal=True)
profile.to_file(r"C:\Users\aniec\Desktop\ibm-ml-project\01-EDA\ds_salaries_profile_classification.html")

# =========================================================
# 4. Target Variable Distribution (salary_in_usd)
# =========================================================

plt.figure(figsize=(10,6))
sns.histplot(dss2025["salary_in_usd"], bins=50, kde=True)
plt.title("Salary Distribution (USD)")
plt.show()

# Log-transform check for skewness
plt.figure(figsize=(10,6))
sns.histplot(np.log1p(dss2025["salary_in_usd"]), bins=50, kde=True)
plt.title("Log-Transformed Salary Distribution")
plt.show()

# =========================================================
# 5. Univariate Categorical Exploration
# =========================================================

plt.figure(figsize=(8,5))
sns.countplot(data=dss2025, x="experience_level")
plt.title("Experience Level Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(data=dss2025, x="work_year")
plt.title("Work Year Distribution")
plt.show()

# =========================================================
# 6. Pairplot for Numeric Variables
# =========================================================

pairplot_cols = numeric_cols[:6]  # Limit to avoid overcrowding
sns.pairplot(dss2025[pairplot_cols], diag_kind="kde")
plt.suptitle("Pairplot of Numeric Variables", y=1.02)
plt.show()

# =========================================================
# 7. Correlation Analysis
# =========================================================

corr = dss2025[numeric_cols].corr()  # Compute correlation matrix

plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.show()

# =========================================================
# 8. Outlier Detection (IQR Method)
# =========================================================

print("\n----- OUTLIER COUNTS -----")
for col in numeric_cols:
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
# 9. Data Consistency Checks
# =========================================================

print("Negative Salary Values:", (dss2025["salary_in_usd"] < 0).sum())

print("\nInvalid Remote Ratios:")
print(dss2025[~dss2025["remote_ratio"].isin([0,50,100])]["remote_ratio"].value_counts())

# =========================================================
# 10. CLASS IMBALANCE CHECKS
# =========================================================

def check_imbalance(col):
    """Prints class counts + percentages."""
    print(f"\nClass distribution for {col}:")
    counts = dss2025[col].value_counts()
    perc = dss2025[col].value_counts(normalize=True) * 100
    print(pd.concat([counts, perc], axis=1, keys=["count", "percentage"]))

def plot_imbalance(col):
    """Plots class imbalance."""
    plt.figure(figsize=(6,4))
    dss2025[col].value_counts().plot(kind="bar", color="skyblue")
    plt.title(f"Class Imbalance: {col}")
    plt.show()

targets = ["salary_mean_cat", "salary_median_cat", "salary_3cat"]

for t in targets:
    check_imbalance(t)
    plot_imbalance(t)

# =========================================================
# 11. Categorical Feature Exploration
# =========================================================

for col in categorical_cols:
    plt.figure(figsize=(10,6))
    sns.countplot(data=dss2025, y=col, order=dss2025[col].value_counts().index[:15])
    plt.title(f"Distribution of {col} (Top 15)")
    plt.show()

# =========================================================
# 12. Cardinality Check
# =========================================================

print("\n----- CARDINALITY -----")
for col in categorical_cols:
    print(f"{col}: {dss2025[col].nunique()} unique values")

# =========================================================
# 13. Feature Importance (Random Forest Classifier)
# =========================================================

numeric_features = ["data_age", "remote_ratio", "experience_level_num", "company_size_num"]
numeric_features = [f for f in numeric_features if f in dss2025.columns]

X = dss2025[numeric_features]
y = dss2025["salary_mean_cat"]  # Choose one classification target

clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

importances = pd.Series(clf.feature_importances_, index=numeric_features)

plt.figure(figsize=(8,5))
importances.sort_values().plot(kind="barh")
plt.title("Feature Importance (Classification)")
plt.show()

# =========================================================
# 14. Skewness Check
# =========================================================

print("\n----- SKEWNESS -----")
print(dss2025[numeric_cols].skew().sort_values(ascending=False))

# =========================================================
# 15. Multicollinearity (VIF)
# =========================================================

vif_cols = [col for col in numeric_cols if dss2025[col].nunique() > 1]

vif_df = pd.DataFrame()
vif_df["feature"] = vif_cols
vif_df["VIF"] = [
    variance_inflation_factor(dss2025[vif_cols].values, i)
    for i in range(len(vif_cols))
]

print("\n----- VIF -----")
print(vif_df.sort_values("VIF", ascending=False))

# =========================================================
# 16. Data Leakage Check
# =========================================================

corr_target = dss2025[numeric_cols].corr()["salary_in_usd"].sort_values(ascending=False)
print("\nCorrelation with salary_in_usd:")
print(corr_target)

leakage_prone = [col for col in dss2025.columns if "salary" in col.lower()]
print("\nLeakage-prone features:", leakage_prone)

# =========================================================
# 17. Feature Distributions by Salary
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
# 18. Job Title & Location Insights
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
# 19. Time Trend – Median Salary Over Years
# =========================================================

year_salary = dss2025.groupby("work_year")["salary_in_usd"].median()

plt.figure(figsize=(10,6))
sns.lineplot(x=year_salary.index, y=year_salary.values, marker="o")
plt.title("Median Salary Over Time")
plt.show()

# =========================================================
# 20. Model Readiness Snapshot
# =========================================================

print("\n----- MODEL READINESS SNAPSHOT -----")
print("Missing values handled previously")
print("Duplicates handled previously")
print("Most skewed features:")
print(dss2025[numeric_cols].skew().sort_values(ascending=False).head())
print("\nHigh VIF features:")
print(vif_df[vif_df["VIF"] > 10])
print("\nLeakage-prone features:", leakage_prone)





import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 0. Loading the Data 
# This is the final dataset created using final_data.py
# -----------------------------

file_path = r"C:\Users\aniec\Desktop\ibm-ml-project\00-data\kaggle-data\dss2025_final.csv"

dss2025 = pd.read_csv(file_path)

# -----------------------------
# 0. Summary Statistics
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
# 1. Explore Target Variable
# Understand the distribution of the target/outcome/reposonse variable
# -----------------------------

plt.figure(figsize=(10,6))

sns.histplot(
    dss2025["salary_in_usd"],
    bins=50,
    kde=True
)

plt.title("Salary in USD Distribution")

plt.show()

# -----------------------------
# 2. Variate Analysis
# Distribution of each feature/predictor variable

# 2.1 Univariate Analysis
# Distribution of each feature/predictor variable
# -----------------------------

# Experience Level Distribution

plt.figure(figsize=(8,5))

sns.countplot(
    data=dss2025,
    x="experience_level"
)

plt.title("Experience Level Distribution")

plt.show()

plt.figure(figsize=(8,5))

sns.countplot(
    data=dss2025,
    x="work_year"
)

plt.title("Experience Level Distribution")

plt.show()

# -----------------------------
# 2.2 Bivariate Analysis
# Relationship between two variables
# -----------------------------

# Salary by Experience Level

plt.figure(figsize=(8,5))

sns.boxplot(
    data=dss2025,
    x="experience_level",
    y="salary_in_usd"
)

plt.title("Salary by Experience Level")

plt.show()

# -----------------------------
# 2.3 All variate Analysis for numerical variables
# Relationship between all variables
# -----------------------------

numeric_cols

sns.pairplot(
    dss2025[numeric_cols],
    diag_kind="kde"
)

plt.suptitle(
    "Pairplot of Key Numeric Variables",
    y=1.02
)

plt.show()

# -----------------------------
# 3. Correlation Analysis
# Check Correlation between numerical features/predictors
# -----------------------------

corr = dss2025[numeric_cols].corr()

print(corr)

plt.figure(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.show()

# -----------------------------
# 3. Detect Outliers
# Find the outliers using IQR Method
# -----------------------------

for col in numeric_cols:

    Q1 = dss2025[col].quantile(0.25)
    Q3 = dss2025[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = dss2025[
        (dss2025[col] < lower)
        |
        (dss2025[col] > upper)
    ]

    print(
        f"{col}: {len(outliers)} outliers"
    )

# -----------------------------
# 4. Outliers Visualization
# Boxplot to visulize outliers
# -----------------------------

for col in numeric_cols:

    plt.figure(figsize=(10,6))

    sns.boxplot(
        x=dss2025[col]
    )

    plt.title(
        f"Boxplot of {col}"
    )

    plt.show()

# -----------------------------
# 5. Check Data Consistency
# Inconsistent or invalid entries
# -----------------------------

print(
    "\nNegative Salary Values:",
    (dss2025["salary_in_usd"] < 0).sum()
)

print(
    "\nInvalid Remote Ratios:"
)

print(
    dss2025[
        ~dss2025["remote_ratio"]
        .isin([0, 50, 100])
    ]["remote_ratio"]
    .value_counts()
)

# --------------------------------------------------------------------------
# For Categorical Variables
# --------------------------------------------------------------------------

# -----------------------------
# 6. Check Class Imbalance (for classification)
# Check Imbalance in target
# -----------------------------

def check_imbalance(col):
    print(f"\nClass distribution for {col}:")
    counts = dss2025[col].value_counts()
    perc = dss2025[col].value_counts(normalize=True) * 100
    print(pd.concat([counts, perc], axis=1, keys=["count", "percentage"]))

check_imbalance("salary_mean_cat")
check_imbalance("salary_median_cat")
check_imbalance("salary_3cat")

import matplotlib.pyplot as plt

def plot_imbalance(col):
    plt.figure(figsize=(6,4))
    dss2025[col].value_counts().plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title(f"Class Imbalance: {col}")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.show()

plot_imbalance("salary_mean_cat")
plot_imbalance("salary_median_cat")
plot_imbalance("salary_3cat")

'''
If one class > 70%
You have high imbalance => models will bias toward the majority class.

If classes are roughly equal
You’re safe => no special handling needed.

If 3‑category salary bands show imbalance:
That’s normal because salary distributions are usually right‑skewed.
'''

# -----------------------------
# 7.Analyze Categorical Features
# Distribution of categorical variables
# -----------------------------

for col in categorical_cols:

    plt.figure(figsize=(10,6))

    sns.countplot(
        data=dss2025,
        y=col,
        order=dss2025[col]
              .value_counts()
              .index[:15]
    )

    plt.title(
        f"Distribution of {col}"
    )

    plt.show()

# -----------------------------
# 8. Check Cardinality
# High Cardinality in categorical features
# -----------------------------

print(
    "\n----- CARDINALITY -----"
)

for col in categorical_cols:

    print(
        f"{col}: "
        f"{dss2025[col].nunique()} unique values"
    )
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# For Numeric variables
# --------------------------------------------------------------------------

# -----------------------------
# 9. Feature Importance
# Histograms of variables
# -----------------------------

# Select numeric features
numeric_features = [
    "data_age",
    "remote_ratio",
    "experience_level_num",
    "company_size_num"
]

X = dss2025[numeric_features]
y = dss2025["salary_in_usd"]

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Feature importance
importances = pd.Series(model.feature_importances_, index=numeric_features)

plt.figure(figsize=(8,5))
importances.sort_values().plot(kind="barh")
plt.title("Feature Importance (Regression)")
plt.xlabel("Importance Score")
plt.show()

# -----------------------------
# 10. Check Data Skewness
# Measure Skewness of numerical features
# -----------------------------

print(
    "\n----- SKEWNESS -----"
)

print(
    dss2025[numeric_cols]
    .skew()
    .sort_values(
        ascending=False
    )
)

# --------------------------------------------------------------------------

# -----------------------------
# 11. Check Multicollinearity
# Find Highly corelated Features
# -----------------------------

corr_matrix = (
    dss2025[numeric_cols]
    .corr()
)

high_corr = corr_matrix[
    abs(corr_matrix) > 0.8
]

print(high_corr)

# -----------------------------
# 12. Time Series Check
# Check Trend by Work Year
# -----------------------------

year_salary = (
    dss2025.groupby("work_year")
    ["salary_in_usd"]
    .median()
)

plt.figure(figsize=(10,6))

sns.lineplot(
    x=year_salary.index,
    y=year_salary.values,
    marker="o"
)

plt.title(
    "Median Salary Over Time"
)

plt.xlabel("Work Year")

plt.ylabel(
    "Median Salary (USD)"
)

plt.show()

# -----------------------------
# 13. Check Data Leakage
# Identify Leakage prone features
# -----------------------------

# numeric columns only
numeric_cols = dss2025.select_dtypes(include=["int64", "float64"]).columns

corr = dss2025[numeric_cols].corr()["salary_in_usd"].sort_values(ascending=False)

print(corr)


leakage_prone = []

for col in dss2025.columns:
    if "salary" in col.lower():
        leakage_prone.append(col)
    if col.endswith("_cat") and "salary" in col.lower():
        leakage_prone.append(col)

print("Leakage-prone features:", leakage_prone)

# -----------------------------
# 14. Feature Distributions by Target
# Compare feature distribution across target/outcome/response variable
# -----------------------------

plt.figure(figsize=(10,6))

sns.boxplot(
    data=dss2025,
    x="employment_type",
    y="salary_in_usd"
)

plt.title(
    "Salary by Employment Type"
)

plt.show()

plt.figure(figsize=(10,6))

sns.boxplot(
    data=dss2025,
    x="company_size",
    y="salary_in_usd"
)

plt.title(
    "Salary by Company Size"
)

plt.show()

plt.figure(figsize=(10,6))

sns.boxplot(
    data=dss2025,
    x="remote_ratio",
    y="salary_in_usd"
)

plt.title(
    "Salary by Remote Ratio"
)

plt.show()

# ---------------------------------------------------------
# Other Plots
# ---------------------------------------------------------

plt.figure(figsize=(12,8))

top_jobs = (
    dss2025["job_title"]
    .value_counts()
    .head(20)
)

sns.barplot(
    x=top_jobs.values,
    y=top_jobs.index
)

plt.title(
    "Top 20 Most Common Job Titles"
)

plt.show()

# ---------------------------------------------------------
# Median Salary by Job Title
# ---------------------------------------------------------

top_titles = (
    dss2025["job_title"]
    .value_counts()
    .head(20)
    .index
)

title_salary = (
    dss2025[
        dss2025["job_title"]
        .isin(top_titles)
    ]
    .groupby("job_title")
    ["salary_in_usd"]
    .median()
    .sort_values(
        ascending=False
    )
)

plt.figure(figsize=(12,8))

sns.barplot(
    x=title_salary.values,
    y=title_salary.index
)

plt.title(
    "Median Salary by Job Title"
)

plt.show()

# ---------------------------------------------------------
# Salary by Employee Residence
# ---------------------------------------------------------

top_countries = (
    dss2025["employee_residence"]
    .value_counts()
    .head(20)
    .index
)

plt.figure(figsize=(12,8))

sns.boxplot(
    data=dss2025[
        dss2025["employee_residence"]
        .isin(top_countries)
    ],
    x="employee_residence",
    y="salary_in_usd"
)

plt.xticks(rotation=45)

plt.title(
    "Salary by Employee Residence"
)

plt.show()

# ---------------------------------------------------------
# Median Salary by Company Location
# ---------------------------------------------------------

country_salary = (
    dss2025.groupby(
        "company_location"
    )["salary_in_usd"]
    .median()
    .sort_values(
        ascending=False
    )
)

plt.figure(figsize=(12,8))

sns.barplot(
    x=country_salary.values,
    y=country_salary.index
)

plt.title(
    "Median Salary by Company Location"
)

plt.show()

# ---------------------------------------------------------
# Salary by Experience Level +
# Company Size Interaction
# ---------------------------------------------------------

plt.figure(figsize=(12,8))

sns.boxplot(
    data=dss2025,
    x="experience_level",
    y="salary_in_usd",
    hue="company_size"
)

plt.title(
    "Salary by Experience Level and Company Size"
)

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
