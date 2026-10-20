# -*- coding: utf-8 -*-
"""
Author: Aaron Niecestro
Created: October 19, 2026
Edited: October 20, 2026
Status: Completed

Description:
Exploratory Data Analysis on the combined Wine Quality dataset.

Purpose:
Unsupervised Machine Learning.

Includes:
- Data loading and structural checks
- Summary statistics and profiling
- Numerical and categorical exploration
- Outlier analysis
- Correlation analysis
- Skewness and transformation checks
- Feature scaling and PCA
- Clustering readiness

Quality is excluded from clustering features and reserved
for post-clustering interpretation.
"""

# =========================================================
# 0. Imports & Configuration
# =========================================================

from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
    r"C:\Users\aniec\Desktop\ibm-ml-project\unsupervised-ml\00-data\clean-data\winequality-final.csv"
)

output_dir = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\unsupervised-ml\01-EDA"
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

# Exclude quality from the clustering features.
target = "quality"

feature_cols = [
    col for col in numeric_cols
    if col != target
]

print("\n----- NUMERIC COLUMNS -----")
print(numeric_cols)

print("\n----- CATEGORICAL COLUMNS -----")
print(categorical_cols)

print("\n----- CLUSTERING FEATURES -----")
print(feature_cols)

print("\n----- EXCLUDED INTERPRETATION VARIABLE -----")
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

print("\n----- QUALITY COUNTS (INTERPRETATION ONLY) -----")
print(
    wqrw[target]
    .value_counts(dropna=False)
    .sort_index()
)

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
        title="Wine Quality - Unsupervised ML EDA",
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
print(wqrw[feature_cols].describe().T)

# All histograms together in one plot
wqrw[feature_cols].hist(
    bins=30,
    figsize=(16, 12),
    edgecolor="black"
)

plt.suptitle(
    "Distribution of Wine Chemistry Features",
    y=1.02
)

plt.tight_layout()
plt.show()

# Individual histograms as seperate plots
for col in feature_cols:

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

# =========================================================
# 8. Feature Distributions by Wine Type
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

# creates a grid with subplots()
fig, axes = plt.subplots(4, 2, figsize=(15, 18))

for col, ax in zip(selected_features, axes.flatten()):

    sns.boxplot(
        data=wqrw,
        x="type",
        y=col,
        ax=ax
    )

    ax.set_title(f"{col.title()} by Wine Type")

plt.tight_layout()
plt.show()

print("\n----- CHEMISTRY SUMMARY BY WINE TYPE -----")

print(
    wqrw.groupby("type", observed=True)[feature_cols]
    .agg(["mean", "median", "std"])
    .round(3)
)

# loop through the features and create a separate figure for each one
for col in selected_features:

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=wqrw,
        x="type",
        y=col
    )

    plt.title(f"{col.title()} by Wine Type")
    plt.xlabel("Wine Type")
    plt.ylabel(col.title())

    plt.tight_layout()
    plt.show()

print("\n----- CHEMISTRY SUMMARY BY WINE TYPE -----")

print(
    wqrw.groupby("type", observed=True)[feature_cols]
    .agg(["mean", "median", "std"])
    .round(3)
)

# boxplot + violin plot for each feature
for col in selected_features:

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(
        data=wqrw,
        x="type",
        y=col,
        ax=axes[0]
    )

    sns.violinplot(
        data=wqrw,
        x="type",
        y=col,
        ax=axes[1]
    )

    axes[0].set_title(f"{col.title()} - Boxplot")
    axes[1].set_title(f"{col.title()} - Violin Plot")

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

pairplot_data = wqrw[pairplot_cols + ["type"]].dropna()

sns.pairplot(
    pairplot_data,
    hue="type",
    diag_kind="hist",
    corner=True,
    plot_kws={"alpha": 0.35, "s": 18}
)

plt.suptitle(
    "Pairplot of Selected Wine Chemistry Features",
    y=1.02
)

plt.show()

# =========================================================
# 10. Correlation Analysis
# =========================================================

corr = wqrw[feature_cols].corr()

print("\n----- NUMERICAL FEATURE CORRELATION MATRIX -----")
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

plt.title("Wine Chemistry Correlation Matrix")
plt.tight_layout()
plt.show()

# Identify highly correlated feature pairs.
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

print("\n----- HIGHLY CORRELATED FEATURE PAIRS (|r| >= 0.80) -----")

if high_corr_pairs.empty:
    print("No feature pairs meet the selected threshold.")
else:
    print(high_corr_pairs.round(3).to_string(index=False))

# =========================================================
# 11. Outlier Detection: IQR Method
# =========================================================

print("\n----- OUTLIER COUNTS (IQR METHOD) -----")

outlier_summary = []

for col in feature_cols:

    series = wqrw[col].dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = (
        (series < lower) | (series > upper)
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
    outlier_df.sort_values(
        "outlier_count",
        ascending=False
    ).round(3).to_string(index=False)
)

fig, axes = plt.subplots(4, 3, figsize=(17, 15))
axes = axes.flatten()

for col, ax in zip(feature_cols, axes):

    sns.boxplot(
        data=wqrw,
        x=col,
        ax=ax
    )

    ax.set_title(col.title())

for ax in axes[len(feature_cols):]:
    ax.set_visible(False)

plt.suptitle(
    "Outlier Inspection: Wine Chemistry",
    y=1.01
)

plt.tight_layout()
plt.show()

outlier_summary = []

# Individual boxplots
for col in feature_cols:

    plt.figure(figsize=(10, 4))

    sns.boxplot(
        data=wqrw,
        x=col
    )

    plt.title(f"Outlier Inspection: {col.title()}")
    plt.xlabel(col.title())

    plt.tight_layout()
    plt.show()
    
# =========================================================
# Histogram and Boxplot Together
# =========================================================

for col in feature_cols:

    fig, axes = plt.subplots(
        2, 1,
        figsize=(10, 6),
        gridspec_kw={"height_ratios": [1, 4]}
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

    axes[0].set_title(f"{col.title()} - Outlier Check")
    axes[1].set_title(f"{col.title()} - Distribution")

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
print(wqrw["type"].value_counts(dropna=False))

print("\nObserved quality scores (not clustering inputs):")
print(sorted(wqrw["quality"].dropna().unique()))

# =========================================================
# 13. Skewness and Transformation Assessment
# =========================================================

skewness = (
    wqrw[feature_cols]
    .skew()
    .sort_values(ascending=False)
)

print("\n----- FEATURE SKEWNESS -----")
print(skewness.round(3))

plt.figure(figsize=(10, 7))

skewness.sort_values().plot(kind="barh")

plt.title("Skewness of Wine Chemistry Features")
plt.xlabel("Skewness")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()

# Compare raw and log1p distributions.
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

    axes[i, 0].set_title(f"{col.title()} - Original")

    if (values >= 0).all():

        sns.histplot(
            np.log1p(values),
            bins=40,
            ax=axes[i, 1]
        )

        axes[i, 1].set_title(f"{col.title()} - log1p")

    else:
        axes[i, 1].set_visible(False)

plt.tight_layout()
plt.show()

# =========================================================
# 14. Multicollinearity: VIF
# =========================================================

print("\n----- VARIANCE INFLATION FACTOR -----")

# Optional diagnostic for linear redundancy.
vif_data = (
    wqrw[feature_cols]
    .replace([np.inf, -np.inf], np.nan)
    .copy()
)

vif_data = vif_data.fillna(vif_data.median())

# Remove constant columns.
vif_data = vif_data.loc[:, vif_data.std() > 0]

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
            for i in range(vif_scaled.shape[1])
        ]
    }).sort_values(
        "VIF",
        ascending=False
    )

    print(vif_df.round(3).to_string(index=False))

    print("\nFeatures with VIF > 10:")

    print(
        vif_df[vif_df["VIF"] > 10].round(3)
    )

else:
    vif_df = pd.DataFrame(
        columns=["feature", "VIF"]
    )

    print("Insufficient nonconstant features for VIF.")

# =========================================================
# This here is the start of Unsupervised Machine Learning EDA
# =========================================================

# =========================================================
# 15. Feature Scaling Assessment
# =========================================================

print("\n----- FEATURE SCALE COMPARISON -----")

scale_summary = wqrw[feature_cols].agg(
    ["mean", "std", "min", "max"]
).T

scale_summary["range"] = (
    scale_summary["max"] - scale_summary["min"]
)

print(scale_summary.round(3))

plt.figure(figsize=(11, 7))

sns.barplot(
    data=scale_summary.reset_index(),
    x="range",
    y="index"
)

plt.title("Range of Wine Chemistry Features Before Scaling")
plt.xlabel("Maximum - Minimum")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()

# =========================================================
# 16. Prepare Features for PCA and Clustering
# =========================================================

# Use chemistry features only.
# Exclude quality and type from the initial feature matrix.
#
# Median imputation handles missing numerical values.
# StandardScaler places features on comparable scales.

X_raw = wqrw[feature_cols].replace(
    [np.inf, -np.inf],
    np.nan
)

imputer = SimpleImputer(strategy="median")

X_imputed = imputer.fit_transform(X_raw)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_imputed)

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=feature_cols,
    index=wqrw.index
)

print("\n----- SCALED FEATURE CHECK -----")
print(X_scaled_df.describe().round(3))

print("\nScaled feature means:")
print(X_scaled_df.mean().round(4))

print("\nScaled feature standard deviations:")
print(X_scaled_df.std(ddof=0).round(4))

# =========================================================
# 17. Principal Component Analysis (PCA)
# =========================================================

pca = PCA()

X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_

cumulative_variance = np.cumsum(
    explained_variance
)

pca_summary = pd.DataFrame({
    "Principal Component": [
        f"PC{i + 1}"
        for i in range(len(explained_variance))
    ],
    "Explained Variance": explained_variance,
    "Cumulative Variance": cumulative_variance
})

print("\n----- PCA EXPLAINED VARIANCE -----")

print(
    pca_summary.round(4).to_string(index=False)
)

# Individual explained variance.
plt.figure(figsize=(10, 6))

plt.bar(
    range(1, len(explained_variance) + 1),
    explained_variance
)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")

plt.title("PCA: Individual Explained Variance")

plt.xticks(
    range(1, len(explained_variance) + 1)
)

plt.tight_layout()
plt.show()

# Cumulative explained variance.
plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o"
)

plt.axhline(
    0.80,
    color="orange",
    linestyle="--",
    label="80% variance"
)

plt.axhline(
    0.90,
    color="red",
    linestyle="--",
    label="90% variance"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")

plt.title("PCA: Cumulative Explained Variance")

plt.xticks(
    range(1, len(cumulative_variance) + 1)
)

plt.legend()
plt.tight_layout()
plt.show()

for threshold in [0.80, 0.90, 0.95]:

    n_components = (
        np.searchsorted(
            cumulative_variance,
            threshold
        ) + 1
    )

    print(
        f"Components required for {threshold:.0%} variance: "
        f"{n_components}"
    )

# =========================================================
# 18. PCA Feature Loadings
# =========================================================

loadings = pd.DataFrame(
    pca.components_.T,
    index=feature_cols,
    columns=[
        f"PC{i + 1}"
        for i in range(len(feature_cols))
    ]
)

print("\n----- PCA LOADINGS: FIRST THREE COMPONENTS -----")

print(loadings.iloc[:, :3].round(3))

plt.figure(figsize=(12, 8))

sns.heatmap(
    loadings.iloc[:, :3],
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("PCA Loadings for First Three Components")

plt.tight_layout()
plt.show()

# =========================================================
# 19. PCA 2D Visualization
# =========================================================

pca_2d = PCA(n_components=2)

X_pca_2d = pca_2d.fit_transform(X_scaled)

pca_plot = pd.DataFrame({
    "PC1": X_pca_2d[:, 0],
    "PC2": X_pca_2d[:, 1],
    "type": wqrw["type"].values,
    "quality": wqrw["quality"].values,
})

print(
    "\nVariance explained by the first two PCs:",
    f"{pca_2d.explained_variance_ratio_.sum():.2%}"
)

# Color by wine type.
plt.figure(figsize=(11, 8))

sns.scatterplot(
    data=pca_plot,
    x="PC1",
    y="PC2",
    hue="type",
    alpha=0.45,
    s=30
)

plt.title("PCA Projection of Wine Chemistry")

plt.xlabel(
    f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)"
)

plt.ylabel(
    f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)"
)

plt.tight_layout()
plt.show()

# Color by quality for post-hoc interpretation only.
plt.figure(figsize=(11, 8))

sns.scatterplot(
    data=pca_plot,
    x="PC1",
    y="PC2",
    hue="quality",
    palette="viridis",
    alpha=0.55,
    s=30
)

plt.title("PCA Projection Colored by Quality (Post-hoc)")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.tight_layout()
plt.show()

# =========================================================
# 20. Post-Hoc Quality Exploration
# =========================================================

# Quality is not used to create the PCA or clustering features.
# These plots are for interpretation only.

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(
    data=wqrw,
    x="type",
    y="quality",
    ax=axes[0]
)

axes[0].set_title("Quality Distribution by Wine Type")
axes[0].set_xlabel("Wine Type")
axes[0].set_ylabel("Quality Score")

sns.countplot(
    data=wqrw,
    x="quality",
    hue="type",
    ax=axes[1]
)

axes[1].set_title("Quality Counts by Wine Type")
axes[1].set_xlabel("Quality Score")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()

# =========================================================
# 21. Model Readiness Snapshot
# =========================================================

print("\n----- UNSUPERVISED ML READINESS SNAPSHOT -----")

print(f"Rows: {wqrw.shape[0]:,}")
print(f"Columns: {wqrw.shape[1]}")
print(f"Clustering features: {len(feature_cols)}")
print(f"Categorical interpretation features: {len(categorical_cols)}")

print(
    "Missing values in original data:",
    wqrw.isnull().sum().sum()
)

print(
    "Duplicate rows:",
    wqrw.duplicated().sum()
)

print(
    "Missing clustering feature values:",
    X_raw.isnull().sum().sum()
)

print("\nMost skewed features:")
print(skewness.head())

print("\nHighly correlated pairs:")

if high_corr_pairs.empty:
    print("None at |r| >= 0.80.")
else:
    print(high_corr_pairs.round(3).to_string(index=False))

print("\nHigh VIF features (VIF > 10):")

if not vif_df.empty:
    print(vif_df[vif_df["VIF"] > 10].round(3))
else:
    print("VIF not available.")

print("\nPCA variance explained by first two components:")

print(
    f"{pca_2d.explained_variance_ratio_.sum():.2%}"
)

print("\nWine type distribution:")

print(
    wqrw["type"]
    .value_counts(normalize=True)
    .round(3)
)

print("\n----- RECOMMENDED NEXT STEPS -----")
print("1. Evaluate K-Means clustering.")
print("2. Evaluate hierarchical clustering.")
print("3. Consider DBSCAN if appropriate.")
print("4. Compare cluster configurations using silhouette score.")
print("5. Evaluate PCA-reduced and original scaled feature spaces.")
print("6. Interpret clusters using type and quality after fitting.")

# =========================================================
# 22. Save EDA Tables and Prepared Features
# =========================================================

corr.to_csv(
    output_dir / "wine_quality_correlation_matrix.csv"
)

outlier_df.to_csv(
    output_dir / "wine_quality_outlier_summary.csv",
    index=False
)

pca_summary.to_csv(
    output_dir / "wine_quality_pca_summary.csv",
    index=False
)

loadings.to_csv(
    output_dir / "wine_quality_pca_loadings.csv"
)

scale_summary.to_csv(
    output_dir / "wine_quality_scale_summary.csv"
)

if not vif_df.empty:

    vif_df.to_csv(
        output_dir / "wine_quality_vif.csv",
        index=False
    )

X_scaled_df.to_csv(
    output_dir / "wine_quality_scaled_features.csv",
    index=False
)

print("\nEDA outputs saved to:", output_dir)

print("\nWine Quality Unsupervised ML EDA completed.")

# =========================================================
# Unsupervised ML Wine EDA Key Insights
# =========================================================

'''
1. Dataset Quality and Structure

- Duplicate records = Yes
- The dataset contains no missing values and very few duplicate observations, 
making it suitable for clustering without heavy preprocessing.

2. Distribution of Wine Types

- White wines account for roughly 75% of observations while red wines account 
for 25%, indicating class imbalance by wine type.

3. Quality Distribution (Not Used for Clustering)

- Although quality is excluded from clustering inputs, the script explores it.
-Quality becomes the interpretation variable.
-Most wines have quality scores between 5 and 7, suggesting the dataset is 
heavily concentrated around average-quality wines.
- This becomes useful later: "Cluster 2 contains mostly wines with quality 
scores of 7 and above."

4. Feature Distributions

The histograms reveal the shape of each chemistry variable.

Findings

- Many wine chemistry variables are not normally distributed.
- Most wine chemistry features are positively skewed and may
require transformation before clustering.

5. Red vs White Wine Differences

- This is one of the most valuable sections.

White wines generally:
- Higher residual sugar
- Higher sulfur dioxide
- Slightly lower pH

Red wines generally:
- Higher volatile acidity
- Higher sulphates
- Higher fixed acidity
- Insight

- Wine type naturally separates many variables and may influence cluster formation.

This often predicts that clustering may discover: 
Red-wine-dominated clusters & 
White-wine-dominated clusters
without explicitly including wine type.

6. Pairwise Relationships

Key insight

- You can visually see whether natural groups already exist.
- White and red wines occupy different regions.
- Relationships appear nonlinear.
- No obvious cluster boundaries exist.
- Density tends to decrease as alcohol increases, suggesting a strong inverse 
relationship.

7. Correlation Analysis

One of the most important sections.

The heatmap identifies relationships among predictors.

Typical wine dataset findings
Strong Negative Correlation
alcohol vs density

Usually around:

A. r ≈ -0.70 to -0.80

Meaning:

Wines with higher alcohol generally have lower density.

Strong Positive Correlation
free sulfur dioxide
vs
total sulfur dioxide

Often:

B. r > 0.70

Meaning:

These variables capture similar information.

Moderate Relationships
residual sugar ↔ density

volatile acidity ↔ citric acid

Insight

Several features exhibit strong correlation and may introduce redundancy.

8. Highly Correlated Feature Pairs

This section specifically identifies:

|r| >= 0.80

Highly correlated features:

Contribute duplicate information.
Can dominate distance calculations.
May distort clustering.

Potential findings:

Feature 1	Feature 2Density	Residual Sugar
Alcohol	Density
Free SO₂	Total SO₂
Insight

Some variables may be candidates for dimensionality reduction through PCA.

9. Outlier Analysis

Largest outlier counts often appear in:

Residual sugar
Chlorides
Sulphates
Free sulfur dioxide
Insight

Several chemistry measurements contain extreme observations that may strongly influence centroid-based algorithms such as K-Means.

This helps determine:

Keep outliers?
Winsorize?
Transform?
Use robust clustering methods?

10. Data Integrity Checks

Nothing of note worth investigating deeper without expert

11. Skewness Assessment

Typical high-skew variables
Residual sugar
Chlorides
Total sulfur dioxide
Free sulfur dioxide
Insight

Several variables show strong positive skew (skewness > 1), indicating the need for log transformation before clustering.

12. Log Transformation Evaluation

The script compares:

original
vs
log1p

for skewed variables.

Log transformation substantially reduces skewness and produces more symmetric distributions.

This is often beneficial because distance-based clustering performs better when features are not extremely skewed.

13. Multicollinearity (VIF)

This section evaluates redundancy among predictors.

Typical interpretation:

VIF	Meaning< 5	Low concern
5-10	Moderate
>10	High
Potential findings

Features frequently showing elevated VIF:

Density
Alcohol
Residual Sugar
Insight

Density appears highly redundant with other chemistry measurements and 
may be contributing duplicate information.

14. Readiness for PCA

Although PCA is imported and not yet run, the EDA prepares for it.

The correlation and VIF analyses answer:

"Can PCA reduce dimensionality without losing much information?"

Likely answer:

Yes. Several chemistry variables exhibit substantial correlation and are likely to collapse into a smaller number of latent components.

15. Readiness for Clustering

The biggest business/data science conclusion from this EDA is:

Strengths

✅ Numerical variables available

✅ Few/no missing values

✅ Meaningful chemistry measurements

✅ Correlated variables suitable for PCA

✅ Wine types show measurable separation

Challenges

⚠ Outliers present

⚠ Strong skewness in multiple features

⚠ Multicollinearity among density, alcohol, sulfur measures, and sugar

⚠ Feature scaling required before K-Means

'''

'''
Executive Summary

- The dataset is generally clean and clustering-ready.
- White and red wines have distinct chemical profiles.
- Alcohol, density, sulfur dioxide, and residual sugar drive much of the 
variation in the dataset.
- Several variables are highly correlated, making PCA a valuable next step.
- Multiple features contain strong positive skew and outliers that should be 
transformed or standardized.
- Quality should remain excluded from clustering and used afterward to 
interpret cluster meaning and business value.
- The next logical step is Standard Scaling → PCA → K-Means/Hierarchical 
Clustering and evaluating whether clusters align with wine type and quality.

- This script is essentially a "cluster readiness assessment", and it does an 
excellent job of determining whether the wine chemistry variables contain 
meaningful structure before unsupervised modeling begins.
'''