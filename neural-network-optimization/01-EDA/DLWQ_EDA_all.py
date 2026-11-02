# -*- coding: utf-8 -*-
"""
Author: Aaron Niecestro
Created: October 19, 2026
Edited: November 1, 2026
Status: Completed

Description:
Exploratory Data Analysis on the combined Wine Quality dataset.

Purpose:
Regression Analysis and Predictive Modeling of Wine Quality.

Includes:
- Data loading and structural checks
- Summary statistics and profiling
- Numerical and categorical exploration
- Quality distribution analysis
- Correlation analysis
- Predictor-target relationship analysis
- Outlier analysis
- Skewness and transformation checks
- Multicollinearity diagnostics
- Regression readiness assessment

Quality is the target variable.
All chemistry measurements are candidate predictor variables.
"""

# =========================================================
# 0. Imports & Configuration
# =========================================================

from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    from ydata_profiling import ProfileReport
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False

sns.set_theme(style="whitegrid", context="talk")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)

RANDOM_STATE = 42

# =========================================================
# 1. Load Data
# =========================================================

file_path = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\00-data\winequality-final.csv"
)

output_dir = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\01-EDA\eda_tables_plots"
)

output_dir.mkdir(parents=True, exist_ok=True)

wqrw = pd.read_csv(file_path)

# Clean column names.
wqrw.columns = wqrw.columns.str.strip()

print("----- DATA SHAPE (MASTER) -----")
print(f"Rows: {wqrw.shape[0]:,}")
print(f"Columns: {wqrw.shape[1]}")

print("\n----- FIRST FIVE ROWS -----")
print(wqrw.head())

# =========================================================
# 2. Column Type Setup
# =========================================================

required_cols = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
    "type",
]

missing_cols = [
    col for col in required_cols
    if col not in wqrw.columns
]

if missing_cols:
    raise ValueError(
        f"Missing expected Wine Quality columns: {missing_cols}"
    )

# Categorical variable.
categorical_cols = ["type"]

# Numeric columns, including quality.
numeric_cols = wqrw.select_dtypes(
    include=["number"]
).columns.tolist()

# Regression target variable.
target = "quality"

# Predictor variables.
predictor_cols = [
    col for col in numeric_cols
    if col != target
]

print("\n----- NUMERIC COLUMNS -----")
print(numeric_cols)

print("\n----- CATEGORICAL COLUMNS -----")
print(categorical_cols)

print("\n----- PREDICTOR VARIABLES -----")
print(predictor_cols)

print("\n----- TARGET VARIABLE -----")
print(target)

# =========================================================
# 3. Structural Checks
# =========================================================

print("\n----- DATA TYPES -----")
print(wqrw.dtypes)

print("\n----- MISSING VALUES BY COLUMN -----")
print(wqrw.isnull().sum().sort_values(ascending=False))

print("\nTotal missing values:", wqrw.isnull().sum().sum())

print("\nDuplicate rows:", wqrw.duplicated().sum())

print("\n----- UNIQUE VALUES -----")
print(wqrw.nunique().sort_values())

print("\n----- WINE TYPE COUNTS -----")
print(wqrw["type"].value_counts(dropna=False))

print("\n----- QUALITY COUNTS (TARGET VARIABLE) -----")
print(
    wqrw[target]
    .value_counts(dropna=False)
    .sort_index()
)

# =========================================================
# Quality Variable Exploration
# =========================================================

print("\n----- QUALITY SUMMARY -----")

print(
    wqrw["quality"]
    .describe()
    .round(3)
)

plt.figure(figsize=(10, 6))

sns.histplot(
    wqrw["quality"],
    bins=len(wqrw["quality"].unique()),
    kde=True
)

plt.title("Distribution of Wine Quality")
plt.xlabel("Quality Score")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# =========================================================
# 4. R-Style Summary + Profiling
# =========================================================

def summary_r(df):
    """Display an R-style summary of each column."""

    print("\n----- R-STYLE SUMMARY -----")
    print(f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]}\n")

    for col in df.columns:
        print(f"--- {col} ---")

        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()

            for stat, label in [
                ("min", "Min"),
                ("25%", "1st Qu."),
                ("50%", "Median"),
                ("mean", "Mean"),
                ("75%", "3rd Qu."),
                ("max", "Max"),
            ]:
                print(
                    f"{label:<10} "
                    f"{desc.get(stat, np.nan):.4f}"
                )

        else:
            desc = df[col].describe()

            print("Type:      Categorical")
            print(f"Unique:    {desc.get('unique', np.nan)}")
            print(f"Top:       {desc.get('top', np.nan)}")
            print(f"Frequency: {desc.get('freq', np.nan)}")

        print()

summary_r(wqrw)

print("\n----- COMPLETE SUMMARY TABLE -----")
print(wqrw.describe(include="all").T)

# =========================================================
# 5. Optional HTML Profiling Report
# =========================================================

if PROFILING_AVAILABLE:

    profile = ProfileReport(
        wqrw,
        title="Wine Quality - Regression EDA",
        minimal=True
    )

    profile_path = (
        output_dir / "wine_quality_eda_profile.html"
    )

    profile.to_file(str(profile_path))

    print("\nProfiling report saved to:", profile_path)

else:
    print(
        "\nydata_profiling is not installed. "
        "Optional install: pip install ydata-profiling"
    )

# =========================================================
# 6. Univariate Numerical Exploration
# =========================================================

print("\n----- NUMERICAL SUMMARY -----")
print(wqrw[predictor_cols].describe().T)

# All histograms together in one plot

wqrw[predictor_cols].hist(
    bins=30,
    figsize=(16, 12),
    edgecolor="black"
)

plt.suptitle(
    "Distribution of Predictor Variables",
    y=1.02
)

plt.tight_layout()
plt.show()

# Individual histograms

for col in predictor_cols:

    plt.figure(figsize=(10, 6))

    sns.histplot(
        wqrw[col],
        bins=30,
        kde=True,
        edgecolor="black"
    )

    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()

# =========================================================
# 7. Categorical Exploration: Wine Type
# =========================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=wqrw,
    x="type",
    order=wqrw["type"].value_counts().index
)

plt.title("Distribution of Wine Types")
plt.xlabel("Wine Type")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

print("\n----- WINE TYPE PROPORTIONS -----")
print(wqrw["type"].value_counts(normalize=True).round(4))

print("\n----- QUALITY BY WINE TYPE -----")

print(
    wqrw.groupby("type")["quality"]
    .agg([
        "mean",
        "median",
        "std",
        "min",
        "max"
    ])
    .round(3)
)

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=wqrw,
    x="type",
    y="quality"
)

plt.title("Wine Quality by Type")
plt.xlabel("Wine Type")
plt.ylabel("Quality")

plt.tight_layout()
plt.show()

# =========================================================
# 8. Predictor Relationships with Quality
# =========================================================

selected_features = [
    "fixed acidity",
    "volatile acidity",
    "residual sugar",
    "chlorides",
    "density",
    "sulphates",
    "alcohol",
    "pH",
]

# ---------------------------------------------------------
# Summary statistics by quality score
# ---------------------------------------------------------

print("\n----- CHEMISTRY SUMMARY BY QUALITY -----")

print(
    wqrw.groupby("quality", observed=True)[predictor_cols]
    .agg(["mean", "median"])
    .round(3)
)

# ---------------------------------------------------------
# Boxplots by quality score
# ---------------------------------------------------------

fig, axes = plt.subplots(
    4,
    2,
    figsize=(15, 18)
)

for col, ax in zip(
    selected_features,
    axes.flatten()
):

    sns.boxplot(
        data=wqrw,
        x="quality",
        y=col,
        ax=ax
    )

    ax.set_title(
        f"{col.title()} by Quality Score"
    )

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Individual boxplots
# ---------------------------------------------------------

for col in selected_features:

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=wqrw,
        x="quality",
        y=col
    )

    plt.title(
        f"{col.title()} by Quality Score"
    )

    plt.xlabel("Quality")
    plt.ylabel(col.title())

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# Regression plots
# ---------------------------------------------------------

for col in selected_features:

    plt.figure(figsize=(10, 6))

    sns.regplot(
        data=wqrw,
        x=col,
        y="quality",
        scatter_kws={"alpha": 0.25},
        line_kws={"color": "red"}
    )

    plt.title(
        f"{col.title()} vs Quality"
    )

    plt.xlabel(col.title())
    plt.ylabel("Quality")

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# Boxplot + Violin Plot
# ---------------------------------------------------------

for col in selected_features:

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    sns.boxplot(
        data=wqrw,
        x="quality",
        y=col,
        ax=axes[0]
    )

    sns.violinplot(
        data=wqrw,
        x="quality",
        y=col,
        ax=axes[1]
    )

    axes[0].set_title(
        f"{col.title()} - Boxplot"
    )

    axes[1].set_title(
        f"{col.title()} - Violin Plot"
    )

    plt.tight_layout()
    plt.show()

# =========================================================
# 9. Pairplot of Selected Numerical Features
# =========================================================

pairplot_cols = [
    "alcohol",
    "volatile acidity",
    "residual sugar",
    "density",
    "sulphates",
]

pairplot_data = (
    wqrw[pairplot_cols + ["quality"]]
    .dropna()
)

sns.pairplot(
    pairplot_data,
    vars=pairplot_cols,
    hue="quality",
    diag_kind="hist",
    corner=True,
    plot_kws={
        "alpha": 0.40,
        "s": 20
    }
)

plt.suptitle(
    "Pairplot of Selected Predictors Colored by Quality",
    y=1.02
)

plt.show()

wqrw["quality_group"] = np.where(
    wqrw["quality"] >= 7,
    "High Quality",
    "Average/Low Quality"
)

pairplot_data = (
    wqrw[pairplot_cols + ["quality_group"]]
    .dropna()
)

sns.pairplot(
    pairplot_data,
    vars=pairplot_cols,
    hue="quality_group",
    diag_kind="hist",
    corner=True,
    plot_kws={
        "alpha": 0.40,
        "s": 20
    }
)

plt.suptitle(
    "Predictor Relationships by Quality Group",
    y=1.02
)

plt.show()

# =========================================================
# 10. Correlation Analysis
# =========================================================

corr = wqrw[predictor_cols].corr()

print("\n----- PREDICTOR CORRELATION MATRIX -----")
print(corr.round(3))

plt.figure(figsize=(13, 10))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True
)

plt.title(
    "Predictor Correlation Matrix"
)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Correlation with Quality
# ---------------------------------------------------------

print("\n----- CORRELATION WITH QUALITY -----")

quality_corr = (
    wqrw[numeric_cols]
    .corr()["quality"]
    .sort_values(ascending=False)
)

print(
    quality_corr.round(3)
)

plt.figure(figsize=(10, 8))

quality_corr.drop("quality").sort_values().plot(
    kind="barh"
)

plt.title(
    "Correlation of Predictors with Quality"
)

plt.xlabel("Correlation")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Highly Correlated Predictor Pairs
# ---------------------------------------------------------

corr_pairs = (
    corr.where(
        np.triu(
            np.ones(corr.shape),
            k=1
        ).astype(bool)
    )
    .stack()
    .reset_index()
)

corr_pairs.columns = [
    "feature_1",
    "feature_2",
    "correlation"
]

corr_pairs["absolute_correlation"] = (
    corr_pairs["correlation"].abs()
)

high_corr_pairs = corr_pairs[
    corr_pairs["absolute_correlation"] >= 0.80
].sort_values(
    "absolute_correlation",
    ascending=False
)

print(
    "\n----- HIGHLY CORRELATED PREDICTOR PAIRS (|r| >= 0.80) -----"
)

if high_corr_pairs.empty:

    print(
        "No predictor pairs meet the selected threshold."
    )

else:

    print(
        high_corr_pairs
        .round(3)
        .to_string(index=False)
    )

# =========================================================
# 11. Outlier Detection: IQR Method
# =========================================================

print("\n----- OUTLIER COUNTS (IQR METHOD) -----")

outlier_summary = []

for col in predictor_cols:

    series = wqrw[col].dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = (
        (series < lower) |
        (series > upper)
    ).sum()

    outlier_summary.append({
        "feature": col,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": int(count),
        "outlier_percent": (
            100 * count / len(series)
            if len(series) else np.nan
        ),
    })

outlier_df = pd.DataFrame(outlier_summary)

print(
    outlier_df
    .sort_values(
        "outlier_count",
        ascending=False
    )
    .round(3)
    .to_string(index=False)
)

# ---------------------------------------------------------
# Combined Boxplots
# ---------------------------------------------------------

fig, axes = plt.subplots(
    4,
    3,
    figsize=(17, 15)
)

axes = axes.flatten()

for col, ax in zip(
    predictor_cols,
    axes
):

    sns.boxplot(
        data=wqrw,
        x=col,
        ax=ax
    )

    ax.set_title(col.title())

for ax in axes[len(predictor_cols):]:
    ax.set_visible(False)

plt.suptitle(
    "Outlier Inspection: Predictor Variables",
    y=1.01
)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Individual Predictor Boxplots
# ---------------------------------------------------------

for col in predictor_cols:

    plt.figure(figsize=(10, 4))

    sns.boxplot(
        data=wqrw,
        x=col
    )

    plt.title(
        f"Outlier Inspection: {col.title()}"
    )

    plt.xlabel(col.title())

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# Histogram + Boxplot Combination
# ---------------------------------------------------------

for col in predictor_cols:

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        gridspec_kw={
            "height_ratios": [1, 4]
        }
    )

    sns.boxplot(
        x=wqrw[col],
        ax=axes[0]
    )

    sns.histplot(
        wqrw[col],
        bins=30,
        kde=True,
        ax=axes[1]
    )

    axes[0].set_title(
        f"{col.title()} - Outlier Check"
    )

    axes[1].set_title(
        f"{col.title()} - Distribution"
    )

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# Target Variable Outlier Inspection
# ---------------------------------------------------------

print("\n----- QUALITY OUTLIER INSPECTION -----")

plt.figure(figsize=(10, 4))

sns.boxplot(
    x=wqrw["quality"]
)

plt.title(
    "Outlier Inspection: Quality"
)

plt.xlabel("Quality Score")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

sns.histplot(
    wqrw["quality"],
    bins=len(
        wqrw["quality"].unique()
    ),
    kde=True
)

plt.title(
    "Distribution of Quality Scores"
)

plt.xlabel("Quality Score")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# =========================================================
# 12. Basic Consistency Checks
# =========================================================

print("\n----- BASIC CONSISTENCY CHECKS -----")

nonnegative_features = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "sulphates",
]

for col in nonnegative_features:

    print(
        f"Negative {col}:",
        (wqrw[col] < 0).sum()
    )

print(
    "Non-positive density:",
    (wqrw["density"] <= 0).sum()
)

print(
    "Non-positive alcohol:",
    (wqrw["alcohol"] <= 0).sum()
)

print(
    "Missing type values:",
    wqrw["type"].isna().sum()
)

print("\nObserved wine types:")

print(
    wqrw["type"]
    .value_counts(dropna=False)
)

print("\nObserved quality scores (target variable):")

print(
    sorted(
        wqrw["quality"]
        .dropna()
        .unique()
    )
)

# =========================================================
# 13. Skewness and Transformation Assessment
# =========================================================

skewness = (
    wqrw[predictor_cols]
    .skew()
    .sort_values(ascending=False)
)

print("\n----- PREDICTOR SKEWNESS -----")
print(skewness.round(3))

plt.figure(figsize=(10, 7))

skewness.sort_values().plot(
    kind="barh"
)

plt.title(
    "Skewness of Predictor Variables"
)

plt.xlabel("Skewness")
plt.ylabel("Predictor")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Evaluate Potential Log Transformations
# ---------------------------------------------------------

log_candidates = [
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
]

fig, axes = plt.subplots(
    len(log_candidates),
    2,
    figsize=(13, 4 * len(log_candidates))
)

for i, col in enumerate(log_candidates):

    values = wqrw[col].dropna()

    sns.histplot(
        values,
        bins=40,
        ax=axes[i, 0]
    )

    axes[i, 0].set_title(
        f"{col.title()} - Original"
    )

    if (values >= 0).all():

        sns.histplot(
            np.log1p(values),
            bins=40,
            ax=axes[i, 1]
        )

        axes[i, 1].set_title(
            f"{col.title()} - log1p"
        )

    else:

        axes[i, 1].set_visible(False)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Target Variable Skewness
# ---------------------------------------------------------

quality_skew = wqrw["quality"].skew()

print("\n----- QUALITY SKEWNESS -----")
print(round(quality_skew, 3))

# =========================================================
# 14. Multicollinearity: VIF
# =========================================================

print("\n----- VARIANCE INFLATION FACTOR -----")

# Evaluate multicollinearity among predictor variables.

vif_data = (
    wqrw[predictor_cols]
    .replace([np.inf, -np.inf], np.nan)
    .copy()
)

vif_data = vif_data.fillna(
    vif_data.median()
)

# Remove constant columns.

vif_data = vif_data.loc[
    :,
    vif_data.std() > 0
]

if vif_data.shape[1] >= 2:

    vif_scaled = (
        vif_data - vif_data.mean()
    ) / vif_data.std()

    vif_df = pd.DataFrame({
        "feature": vif_scaled.columns,
        "VIF": [
            variance_inflation_factor(
                vif_scaled.to_numpy(),
                i
            )
            for i in range(
                vif_scaled.shape[1]
            )
        ]
    }).sort_values(
        "VIF",
        ascending=False
    )

    print(
        vif_df
        .round(3)
        .to_string(index=False)
    )

    print("\nFeatures with VIF > 10:")

    print(
        vif_df[
            vif_df["VIF"] > 10
        ].round(3)
    )

else:

    vif_df = pd.DataFrame(
        columns=["feature", "VIF"]
    )

    print(
        "Insufficient nonconstant features for VIF."
    )

"""
VIF < 5     Low multicollinearity
VIF 5-10    Moderate concern
VIF > 10    High multicollinearity
"""

# =========================================================
# 15. Regression Readiness Snapshot
# =========================================================

print("\n----- REGRESSION READINESS SNAPSHOT -----")

print(f"Rows: {wqrw.shape,}")
print(f"Columns: {wqrw.shape[1]}")
print(f"Predictor Variables: {len(predictor_cols)}")

print(
    "\nMissing Values:",
    wqrw.isnull().sum().sum()
)

print(
    "Duplicate Rows:",
    wqrw.duplicated().sum()
)

print("\nTarget Variable:")
print(target)

print("\nQuality Counts:")

print(
    wqrw["quality"]
    .value_counts()
    .sort_index()
)

print("\nMost Skewed Predictors:")

print(
    skewness.head()
)

print("\nTop Quality Correlations:")

print(
    quality_corr
    .head(10)
)

print("\nStrongest Negative Quality Correlations:")

print(
    quality_corr
    .tail(10)
)

print("\nHigh VIF Predictors:")

if not vif_df.empty:
    print(
        vif_df[
            vif_df["VIF"] > 10
        ]
    )

# Exports

corr.to_csv(
    output_dir /
    "wine_quality_correlation_matrix.csv"
)

quality_corr.to_csv(
    output_dir /
    "wine_quality_target_correlations.csv"
)

outlier_df.to_csv(
    output_dir /
    "wine_quality_outlier_summary.csv",
    index=False
)

if not vif_df.empty:

    vif_df.to_csv(
        output_dir /
        "wine_quality_vif.csv",
        index=False
    )
    
# Wine Quality Counts

quality_counts = (
    wqrw["quality"]
    .value_counts(dropna=False)
    .sort_index()
    .rename("count")
    .to_frame()
)

quality_counts["percent"] = (
    quality_counts["count"] /
    quality_counts["count"].sum() * 100
).round(2)

print("\n----- QUALITY COUNTS -----")
print(quality_counts)

print("\n----- RECOMMENDED NEXT STEPS -----")

print("1. Create train/test split.")
print("2. Fit Linear Regression.")
print("3. Fit Ridge Regression.")
print("4. Fit Lasso Regression.")
print("5. Fit Elastic Net.")
print("6. Fit Random Forest Regressor.")
print("7. Fit Gradient Boosting Regressor.")
print("8. Compare RMSE.")
print("9. Compare MAE.")
print("10. Compare R-squared.")
print("11. Examine feature importance.")
print("12. Evaluate residual diagnostics.")