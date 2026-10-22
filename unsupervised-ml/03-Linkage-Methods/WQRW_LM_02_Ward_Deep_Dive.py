# -*- coding: utf-8 -*-
"""
Author: Aaron Niecestro

Created:    October 19, 2026
Editted:    October 22, 2026
Status:     Completed

# =========================================================
# Ward Linkage Deep Dive
# =========================================================

Purpose

Investigates how the Ward hierarchy evolves as the dendrogram is cut
at deeper levels.

What the code does

• Compares Ward k=2, k=3, and k=4 solutions.
• Evaluates metrics, cluster balance, and cluster profiles.
• Compares clusters against known wine types.
• Tracks how clusters split as hierarchy depth increases.
• Examines feature differences and quality differences across clusters.

Why it matters

Ward Linkage not only recovers the dominant red-versus-white wine
structure, but also reveals meaningful subgroups such as dry and sweet
white wines and distinct red wine populations. The hierarchy develops
logically and remains balanced as additional clusters are introduced.

# =========================================================
# Features Driving the Ward Hierarchy
# =========================================================

Purpose

Determines which chemical variables contribute most strongly to the
Ward clustering hierarchy.

What the code does

• Calculates raw feature separation across Ward clusters.
• Standardizes cluster profiles for fair comparison.
• Measures the relative importance of each feature.
• Visualizes feature contributions with a ranked bar chart.

Why it matters

The analysis shows that Ward Linkage is driven by many variables
simultaneously rather than a single dominant feature. Alcohol, citric
acid, pH, quality, acidity measures, sulfur dioxide, and residual sugar
all contribute to the hierarchical structure discovered in the wine
dataset.
"""

# =========================================================
# 0. Imports & Configuration
# =========================================================

from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

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
    r"C:\Users\aniec\Desktop\ibm-ml-project\unsupervised-ml\01-EDA\output_tables"
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
# 3. Feature Matrix for Clustering
# =========================================================

from sklearn.preprocessing import StandardScaler

X = wqrw[feature_cols].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature matrix shape:", X_scaled.shape)

# =========================================================
# Output Directory
# =========================================================

output_dir = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\unsupervised-ml\03-Linkage-Methods\03-linkage-results"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# 4. Ward Solution Comparison
# =========================================================

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

ward_results = []

for k in [2, 3, 4]:

    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )

    labels = model.fit_predict(
        X_scaled
    )

    cluster_sizes = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
    )

    ward_results.append(
        {
            "clusters": k,
            "silhouette_score":
                silhouette_score(
                    X_scaled,
                    labels
                ),

            "davies_bouldin_index":
                davies_bouldin_score(
                    X_scaled,
                    labels
                ),

            "smallest_cluster":
                cluster_sizes.min(),

            "largest_cluster":
                cluster_sizes.max()
        }
    )

    print(
        f"\n----- WARD k={k} CLUSTER SIZES -----"
    )

    print(cluster_sizes)

ward_results_df = pd.DataFrame(
    ward_results
)

print(
    "\n----- WARD MODEL COMPARISON -----"
)

print(
    ward_results_df.round(4)
)

ward_colors = {
    2: "#1f77b4",   # Blue
    3: "#ff7f0e",   # Orange
    4: "#2ca02c"    # Green
}

# =========================================================
# 5. Ward Metric Comparison
# =========================================================

# K Color Palette

ward_colors = {
    2: "#1f77b4",   # Blue
    3: "#ff7f0e",   # Orange
    4: "#2ca02c"    # Green
}

# Plots
fig, axes = plt.subplots(
    1,
    2,
    figsize=(14,5)
)

sns.barplot(
    data=ward_results_df,
    x="clusters",
    y="silhouette_score",
    hue="clusters",
    palette=ward_colors,
    legend=False,
    ax=axes[0]
)

axes[0].set_title(
    "Ward Silhouette Score"
)

sns.barplot(
    data=ward_results_df,
    x="clusters",
    y="davies_bouldin_index",
    hue="clusters",
    palette=ward_colors,
    legend=False,
    ax=axes[1]
)

axes[1].set_title(
    "Ward Davies-Bouldin Index"
)

plt.tight_layout()

plt.show()

# =========================================================
# 6. Ward Cluster Size Comparison
# =========================================================

for k in [2, 3, 4]:

    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )

    labels = model.fit_predict(
        X_scaled
    )

    print(
        f"\n----- WARD k={k} BALANCE -----"
    )

    print(
        (
            pd.Series(labels)
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )
    )
    
# =========================================================
# 7. Ward k=2 Profiles
# =========================================================

ward_k2 = AgglomerativeClustering(
    n_clusters=2,
    linkage="ward"
)

wqrw["ward_k2"] = ward_k2.fit_predict(
    X_scaled
)

profile_k2 = (
    wqrw
    .groupby("ward_k2")
    [feature_cols + ["quality"]]
    .mean()
    .round(4)
)

print(profile_k2)

# =========================================================
# 8. Ward k=3 Profiles
# =========================================================

ward_k3 = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

wqrw["ward_k3"] = ward_k3.fit_predict(
    X_scaled
)

profile_k3 = (
    wqrw
    .groupby("ward_k3")
    [feature_cols + ["quality"]]
    .mean()
    .round(4)
)

print(profile_k3)

# =========================================================
# 9. Ward k=4 Profiles
# =========================================================

ward_k4 = AgglomerativeClustering(
    n_clusters=4,
    linkage="ward"
)

wqrw["ward_k4"] = ward_k4.fit_predict(
    X_scaled
)

profile_k4 = (
    wqrw
    .groupby("ward_k4")
    [feature_cols + ["quality"]]
    .mean()
    .round(4)
)

print(profile_k4)

# =========================================================
# Part 1 Insights
# =========================================================

"""
Ward k=2

Silhouette = 0.2607
DB Index = 1.7164
Balance = 72.2 / 27.8

Strengths:

Best separation
Simplest interpretation
Strongest overall clustering solution

Ward k=3 => Best Clustering Model:

Silhouette = 0.2145
DB Index = 1.5702
Balance = 50 / 28 / 22

Interesting finding:

The larger cluster from k=2 appears to split into:

Dry White Population
Sweet White Population
 
based on:

Residual Sugar
3.54
11.49

and

Alcohol 
11.08
9.58

This is actually a very meaningful hierarchy.

Ward k=4

Silhouette = 0.2229
DB Index = 1.5977
Balance = 50 / 22 / 15 / 13

The red-like cluster appears to split into:

Cluster 0
Higher acidity
Higher sulphates
Higher chlorides

Higher volatile acidity
Lower citric acid
Lower quality

So Ward is revealing additional structure inside the red-wine population.

- Most Interesting Hierarchical Story

Your results currently suggest:

Ward k=2
 
├── White-like wines
└── Red-like wines

then:

Ward k=3
 
├── Red-like wines
├── Dry Whites
└── Sweet Whites

then:

Ward k=4
 
├── Structured Reds
├── Volatile Reds
├── Dry Whites
└── Sweet Whites

If the wine-type cross-tabs confirm this, you'll have a very compelling 
hierarchical interpretation that goes beyond what you showed in the Agglomerative 
Clustering project.
"""

# =========================================================
# Part 2 Wine Type Comparison
# =========================================================

# =========================================================
# 10. Ward vs Wine Type
# =========================================================

for col in [
    "ward_k2",
    "ward_k3",
    "ward_k4"
]:

    print(
        f"\n----- {col.upper()} VS TYPE -----"
    )

    print(
        pd.crosstab(
            wqrw[col],
            wqrw["type"],
            normalize="index"
        )
        .mul(100)
        .round(2)
    )
    
"""
These results are really interesting and actually tell a much clearer story 
than the original Agglomerative analysis.

1. Ward k=2 vs Wine Type

Cluster 0 => 99.43% White
Cluster 1 => 90.46% Red
Interpretation:

Ward k=2 almost perfectly recovers the underlying wine-type structure despite 
never seeing the type variable during training.

This is strong evidence that the dominant latent structure in the dataset is:

White Wines
vs
Red Wines

The small amount of overlap likely reflects wines whose chemistry is atypical 
for their labeled category.

2. Ward k=3 vs Wine Type

Cluster 0 => 90.46% Red
Cluster 1 => 99.32% White
Cluster 2 => 99.66% White

Interpretation:

This is where the hierarchy becomes interesting.

The original White Wine cluster from k=2 has split into:

White Cluster 1
White Cluster 2

while:

Red Wines remain largely unchanged.

Looking at the profiles:

Cluster 1

Residual Sugar = 3.54
Alcohol = 11.08
Quality = 6.01

Cluster 2

Residual Sugar = 11.49
Alcohol = 9.58
Quality = 5.57

This suggests:

Dryer Higher-Alcohol Whites
vs
Sweeter Lower-Alcohol Whites

which is a very meaningful business segmentation.

3. Ward k=4 vs Wine Type

Cluster 1 → 99.32% White
Cluster 2 → 99.66% White

Cluster 0 → 86.57% Red
Cluster 3 → 93.82% Red
"""

"""
Now the hierarchy has gone one level deeper.

Previously:

White wines split

Now:

Red wines split as well.

Cluster 0

Fixed Acidity = 9.17
Sulphates = 0.71
Chlorides = 0.109
Quality = 5.82

Represents:

Structured / Heavier Reds

Cluster 3
Volatile Acidity = 0.62
Citric Acid = 0.14
Quality = 5.39

Represents:

Lower-Quality High-VA Reds

The Hierarchical Story

This is the coolest result from the entire Ward analysis.

The hierarchy evolves naturally:

k = 2

├── White-like Wines
└── Red-like Wines


↓

k = 3

├── Red-like Wines
├── Dry Whites
└── Sweet Whites


↓

k = 4

├── Structured Reds
├── Volatile Reds
├── Dry Whites
└── Sweet Whites


This is exactly what we hope to see from hierarchical clustering.

The clusters do not appear random.

Each additional split reveals meaningful chemical substructure.
"""

# ======================================================
# Ward Deep Dive: Key Insights
# ======================================================

"""
KEY INSIGHTS

• Ward Linkage consistently produced the only balanced
  clustering solutions among all evaluated linkage methods.

• Ward k=2 recovered the dominant wine-type structure,
  separating wines into primarily white and red populations.

• Ward k=2 achieved the strongest overall cluster separation:

      Silhouette Score = 0.2607
      Davies-Bouldin Index = 1.7164

• Ward k=3 revealed additional structure within the white
  wine population.

• The white-wine cluster separated into:

      Dryer, higher-alcohol whites

      Sweeter, lower-alcohol whites

• Ward k=4 further divided the red-wine population.

• The resulting hierarchy revealed:

      Structured red wines
      High volatile-acidity red wines
      Dry white wines
      Sweet white wines

• Cluster sizes remained balanced throughout the hierarchy,
  unlike Single, Complete, and Average Linkage.

• Wine-type comparisons demonstrated that Ward recovered
  the known red/white structure without access to type labels.

• The hierarchical progression from k=2 to k=4 produced
  increasingly detailed and interpretable wine segments.

• These results indicate that Ward Linkage successfully
  captures both the dominant population structure and
  meaningful nested subgroup structure within the
  Wine Quality dataset.


Honestly, this Ward Deep Dive ended up being one of the strongest analyses 
in the entire unsupervised project because it shows not just which clusters 
exist, but how the hierarchy unfolds as you cut the dendrogram deeper.
"""

# =========================================================
# Part 3 Ward Hierarchy Flow
# =========================================================

# =========================================================
# Cluster Transition Analysis
# =========================================================

print("\n----- k=2 -> k=3 -----")

print(
    pd.crosstab(
        wqrw["ward_k2"],
        wqrw["ward_k3"]
    )
)

print("\n----- k=3 -> k=4 -----")

print(
    pd.crosstab(
        wqrw["ward_k3"],
        wqrw["ward_k4"]
    )
)

# =========================================================
# Cluster Transition Analysis
# =========================================================

print("\n----- k=2 -> k=3 -----")

print(
    pd.crosstab(
        wqrw["ward_k2"],
        wqrw["ward_k3"]
    )
)

print("\n----- k=3 -> k=4 -----")

print(
    pd.crosstab(
        wqrw["ward_k3"],
        wqrw["ward_k4"]
    )
)


# =========================================================
# Part 4 Quality by Ward Cluster
# =========================================================

# =========================================================
# Quality Distribution
# =========================================================

plt.figure(figsize=(10,5))

sns.boxplot(
    data=wqrw,
    x="ward_k4",
    y="quality"
)

plt.title(
    "Wine Quality by Ward k=4 Cluster"
)

plt.show()

# Feature Heatmap

profile_k4_z = (
    profile_k4
    .apply(
        lambda x:
        (x - x.mean())
        / x.std()
    )
)

plt.figure(figsize=(12,6))

sns.heatmap(
    profile_k4_z,
    cmap="coolwarm",
    center=0
)

plt.title(
    "Ward k=4 Cluster Profiles"
)

plt.show()

# =========================================================
# Additional Ward Deep Dive Results & Interpretation
# =========================================================

"""
Transition analysis, cluster-profile evaluation, cluster heatmaps, and
quality-distribution comparisons provided additional insight into how the
Ward hierarchy evolves as the dendrogram is cut at progressively deeper
levels.

Transition tables revealed that the Ward hierarchy develops in a highly
structured and interpretable manner.

At k=2, the dataset separated into two dominant populations:

    Cluster 0 → predominantly white wines
    Cluster 1 → predominantly red wines

When the hierarchy was expanded to k=3, the red-wine cluster remained
intact while the white-wine cluster split into two distinct subgroups.

This indicates that the first major source of variation within the
dataset is the distinction between red and white wines, while the second
source of variation occurs within the white-wine population itself.

The cluster profiles showed that the white-wine split was primarily
driven by residual sugar, alcohol content, density, and sulfur dioxide
concentrations.

The resulting subgroups can be interpreted as:

    Dry, higher-alcohol white wines

    Sweeter, lower-alcohol white wines

When the hierarchy was expanded further to k=4, both white-wine
subgroups remained stable while the red-wine cluster divided into two
additional populations.

This indicates that Ward Linkage preserves previously discovered
structure and introduces new subdivisions only when sufficient evidence
exists within the data.

The resulting red-wine subgroups differed primarily in volatile acidity,
fixed acidity, sulphates, chlorides, and overall quality.

These clusters can be interpreted as:

    Structured red wines

    Higher volatile-acidity red wines

The Ward hierarchy therefore evolved as:

k = 2

    White Wines
    Red Wines

↓

k = 3

    Dry White Wines
    Sweet White Wines
    Red Wines

↓

k = 4

    Dry White Wines
    Sweet White Wines
    Structured Red Wines
    High Volatile-Acidity Red Wines

Cluster-type comparisons further strengthened this interpretation.

At k=2:

    Cluster 0 = 99.43% white wines
    Cluster 1 = 90.46% red wines

At k=3:

    Both white subgroups remained over 99% white wines.

At k=4:

    The two white clusters remained over 99% white wines.

    The two red clusters remained strongly red-dominated
    (86.57% and 93.82%).

These results demonstrate that Ward Linkage successfully recovered the
known wine-type structure without access to wine-type labels.

The heatmap visualization reinforced the profile analysis by showing that
the primary variables driving hierarchical separation were:

    Residual sugar
    Alcohol
    Volatile acidity
    Sulphates
    Chlorides
    Sulfur dioxide

The quality boxplot revealed meaningful quality differences across the
clusters.

Average quality ranking was approximately:

    Dry White Wines                (Highest)
    Structured Red Wines
    Sweet White Wines
    High Volatile-Acidity Reds     (Lowest)

This finding suggests that the Ward hierarchy is not merely separating
wines by chemistry, but is also uncovering meaningful differences in
wine quality.

Overall Interpretation

The Ward hierarchy displayed a logical and stable progression as
additional clusters were introduced. Rather than generating arbitrary
partitions, each successive split revealed chemically meaningful and
business-relevant subgroups while preserving previously discovered
structure.

These results provide strong evidence that Ward Linkage captures both the
dominant wine-type separation and important nested subgroup structure
within the Wine Quality dataset. The hierarchy progresses naturally from
broad wine populations to increasingly specialized wine profiles,
making Ward Linkage particularly valuable for exploratory analysis,
segmentation, and interpretation.
"""

# =========================================================
# Business Findings
# =========================================================

"""
k=2
    White Wines
    Red Wines

        ↓

k=3
    Dry Whites
    Sweet Whites
    Reds

        ↓

k=4
    Dry Whites
    Sweet Whites
    Structured Reds
    High-Volatile-Acidity Reds
"""

# =========================================================
# Physics of the Hierarchy
# =========================================================

# =========================================================
# Features Driving Ward Hierarchy
# =========================================================

feature_ranges = pd.DataFrame({
    "min": profile_k4.min(),
    "max": profile_k4.max()
})

feature_ranges["range"] = (
    feature_ranges["max"]
    - feature_ranges["min"]
)

feature_ranges = (
    feature_ranges
    .sort_values(
        by="range",
        ascending=False
    )
)

print(
    "\n----- FEATURES DRIVING WARD SPLITS -----"
)

print(
    feature_ranges.round(4)
)

# =========================================================
# Standardized Feature Separation
# =========================================================

profile_z = (
    profile_k4
    .apply(
        lambda x:
        (x - x.mean())
        / x.std()
    )
)

feature_sep = pd.DataFrame({
    "separation":
        profile_z.max()
        - profile_z.min()
})

feature_sep = (
    feature_sep
    .sort_values(
        by="separation",
        ascending=False
    )
)

print(
    "\n----- STANDARDIZED SEPARATION -----"
)

print(
    feature_sep.round(4)
)

# Visualization

plt.figure(figsize=(10,6))

sns.barplot(
    data=feature_sep.reset_index(),
    x="separation",
    y="index",
    color="steelblue"
)

plt.title(
    "Features Driving Ward Hierarchy"
)

plt.xlabel(
    "Standardized Separation"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

plt.show()

# Insights

"""
This is actually a fascinating result because it shows the difference between 
raw separation and standardized separation.

What surprised me

Your raw ranges said:

Total Sulfur Dioxide    119.4
Free Sulfur Dioxide      30.2
Residual Sugar            9.1

which initially made it look like sulfur dioxide was driving everything.

But that's mostly because those variables are measured on larger scales.

The standardized results are more interesting
Alcohol               2.41
Citric Acid           2.36
pH                    2.29
Quality               2.27
Chlorides             2.20
Fixed Acidity         2.16
Free Sulfur Dioxide   2.16
Volatile Acidity      2.16
Density               2.12
Residual Sugar        2.11
Sulphates             2.10
Total Sulfur Dioxide  2.07

Two things stand out:

1. Everything Matters

The biggest surprise is:

2.07 - 2.41

That's an incredibly narrow range.

Ward isn't being driven by:

one feature

It's being driven by:

many features simultaneously. Which is exactly what Ward is supposed to do.

Ward minimizes:

within-cluster variance across the entire feature space.

So instead of finding:

Residual Sugar clusters

or

Alcohol clusters

Ward creates groups that differ consistently across many variables.

2. Alcohol is #1
Alcohol = 2.41

That's interesting because alcohol repeatedly emerged throughout your project:

K-Means
GMM
Agglomerative
Ward hierarchy

Alcohol keeps appearing as one of the strongest separators of wine populations.

This aligns with the wine industry as well because alcohol content is often 
associated with:

Wine style
Body
Quality
Sweetness perception

3. Quality is Higher Than Expected
Quality = 2.27


This is very interesting.

Remember:

Quality was never used as a clustering feature.

Yet the Ward hierarchy still separated wines along quality dimensions.

That means:

The chemical structure of the wines naturally carries information about quality.

This is one of the strongest unsupervised-learning findings from the project.

4. The White Wine Split Makes Sense

Looking back:

Dry Whites
vs
Sweet Whites

was driven mainly by:

Alcohol
Density
Residual Sugar
Sulfur Dioxide

Exactly what we'd expect from white wine chemistry.

5. The Red Wine Split Makes Sense

Looking back:

Structured Reds
vs
High-VA Reds

was driven largely by:

Volatile Acidity
Citric Acid
Sulphates
Fixed Acidity

Again matching how red wine styles are typically differentiated.

My Personal Takeaway

For me, the most interesting thing isn't:

Ward found 4 clusters.


It's:

Ward recovered wine type,
then discovered meaningful subgroups,
and did so using nearly ALL variables.


The hierarchy wasn't dominated by one weird feature or one extreme measurement.

It was uncovering a genuinely multidimensional structure.

That's actually one of the strongest signals that the clustering is reflecting 
real patterns in the data rather than algorithmic artifacts.
"""

# Exports

ward_results_df.to_csv(
    output_dir / "ward_model_comparison.csv",
    index=False
)

profile_k2.to_csv(
    output_dir / "ward_k2_profile.csv"
)

profile_k3.to_csv(
    output_dir / "ward_k3_profile.csv"
)

profile_k4.to_csv(
    output_dir / "ward_k4_profile.csv"
)

ward_k2_type = pd.crosstab(
    wqrw["ward_k2"],
    wqrw["type"],
    normalize="index"
).mul(100).round(2)

ward_k3_type = pd.crosstab(
    wqrw["ward_k3"],
    wqrw["type"],
    normalize="index"
).mul(100).round(2)

ward_k4_type = pd.crosstab(
    wqrw["ward_k4"],
    wqrw["type"],
    normalize="index"
).mul(100).round(2)

ward_k2_type.to_csv(
    output_dir / "ward_k2_type_recovery.csv"
)

ward_k3_type.to_csv(
    output_dir / "ward_k3_type_recovery.csv"
)

ward_k4_type.to_csv(
    output_dir / "ward_k4_type_recovery.csv"
)

k2_k3_transition = pd.crosstab(
    wqrw["ward_k2"],
    wqrw["ward_k3"]
)

k3_k4_transition = pd.crosstab(
    wqrw["ward_k3"],
    wqrw["ward_k4"]
)

k2_k3_transition.to_csv(
    output_dir / "ward_k2_k3_transition.csv"
)

k3_k4_transition.to_csv(
    output_dir / "ward_k3_k4_transition.csv"
)

feature_ranges.to_csv(
    output_dir / "ward_feature_ranges.csv"
)

feature_sep.to_csv(
    output_dir / "ward_feature_separation.csv"
)